"""
Landed cost checked against a lane worked out by hand.

The numbers below were computed on paper first. If the module drifts, these
fail — which is the point. Asserting only "cost > 0" would catch nothing.

    SKU     $100/unit, 10 units/carton, 50x40x30 cm carton, 12 kg
    carton  0.06 m3, volumetric 10 kg -> chargeable 12 kg -> 1.2 kg/unit
    FEU     floor(67.7 x 0.80 / 0.06) = 902 cartons = 9,020 units
    lane    ocean fixed $1,150/container, air fixed $320,
            duty 4.3%, insurance 0.3%, WACC 9.5%
"""
import pytest

from src.landed_cost import Lane, SKU, air_cost, compare, ocean_cost, transit_capital_cost

LANE = Lane("hand-computed")
WIDGET = SKU("HAND-1", unit_value_usd=100.0, units_per_carton=10,
             carton_l_cm=50, carton_w_cm=40, carton_h_cm=30, carton_weight_kg=12.0)
UNITS = 1_000


class TestSKUGeometry:
    def test_units_per_container(self):
        assert WIDGET.units_per_container == 9_020

    def test_chargeable_kg_per_unit(self):
        # volumetric 10 kg vs actual 12 kg -> bills 12, over 10 units.
        assert WIDGET.kg_per_unit_chargeable == pytest.approx(1.2)


class TestTransitCapital:
    def test_matches_hand_calculation(self):
        # $100 x 1000 units x 9.5% x 25/365 = $650.68
        assert transit_capital_cost(100.0, 1_000, 25, 0.095) == pytest.approx(650.68, abs=0.01)

    def test_scales_linearly_with_days(self):
        a = transit_capital_cost(100.0, 1_000, 10, 0.095)
        b = transit_capital_cost(100.0, 1_000, 20, 0.095)
        assert b == pytest.approx(2 * a)

    def test_zero_days_costs_nothing(self):
        assert transit_capital_cost(100.0, 1_000, 0, 0.095) == 0.0


class TestOceanLane:
    """1,000 units at $3,000/FEU -> 1 container.

    freight 3000 + fixed 1150 + duty 4300 + insurance 300 + capital 650.68
    """
    def setup_method(self):
        self.b = ocean_cost(WIDGET, LANE, 3_000.0, UNITS, wacc=0.095)

    def test_one_container(self):
        assert self.b.components["containers"] == 1

    def test_components(self):
        assert self.b.freight == pytest.approx(3_000.0)
        assert self.b.fixed == pytest.approx(1_150.0)
        assert self.b.duty == pytest.approx(4_300.0)
        assert self.b.insurance == pytest.approx(300.0)
        assert self.b.transit_capital == pytest.approx(650.68, abs=0.01)

    def test_total_and_per_unit(self):
        assert self.b.total == pytest.approx(9_400.68, abs=0.01)
        assert self.b.per_unit == pytest.approx(9.40068, abs=0.0001)

    def test_second_container_when_volume_exceeds_one(self):
        b = ocean_cost(WIDGET, LANE, 3_000.0, 9_021, wacc=0.095)
        assert b.components["containers"] == 2
        assert b.freight == pytest.approx(6_000.0)
        assert b.fixed == pytest.approx(2_300.0), "fixed charges are per container"


class TestAirLane:
    """1,000 units at $6.00/chargeable kg -> 1,200 kg.

    freight 7200 + fixed 320 + duty 4300 + insurance 300 + capital 78.08
    """
    def setup_method(self):
        self.b = air_cost(WIDGET, LANE, 6.00, UNITS, wacc=0.095)

    def test_chargeable_weight(self):
        assert self.b.components["chargeable_kg"] == pytest.approx(1_200.0)

    def test_components(self):
        assert self.b.freight == pytest.approx(7_200.0)
        assert self.b.fixed == pytest.approx(320.0)
        assert self.b.transit_capital == pytest.approx(78.08, abs=0.01)

    def test_total_and_per_unit(self):
        assert self.b.total == pytest.approx(12_198.08, abs=0.01)
        assert self.b.per_unit == pytest.approx(12.19808, abs=0.0001)

    def test_air_fixed_charge_is_per_shipment_not_per_container(self):
        big = air_cost(WIDGET, LANE, 6.00, 50_000, wacc=0.095)
        assert big.fixed == pytest.approx(320.0)


class TestComparison:
    def test_duty_and_insurance_cancel(self):
        # Both are charged on customs value, so they cannot influence which
        # mode is cheaper -- a useful invariant, and easy to break.
        o = ocean_cost(WIDGET, LANE, 3_000.0, UNITS)
        a = air_cost(WIDGET, LANE, 6.00, UNITS)
        assert o.duty == pytest.approx(a.duty)
        assert o.insurance == pytest.approx(a.insurance)

    def test_ocean_cheaper_at_normal_rates(self):
        r = compare(WIDGET, LANE, 3_000.0, 6.00, UNITS)
        assert r["cheaper"] == "ocean"
        assert r["gap_per_unit"] == pytest.approx(12.19808 - 9.40068, abs=0.001)

    def test_air_wins_once_ocean_spikes(self):
        # The 2021-22 range. Air should take over.
        assert compare(WIDGET, LANE, 12_000.0, 6.00, UNITS)["cheaper"] == "air"

    def test_transit_capital_can_decide_a_close_call(self):
        """Ignoring the cost of capital flips the answer on a high-value good.

        This is the whole argument for carrying the term. On $4,000/unit goods,
        ocean's extra 22 days costs ~$22,900 in tied-up capital at 9.5% — far
        more than the ~$1,370 freight advantage ocean holds at $5,000/FEU. Drop
        the capital term and you would ship it by sea and be wrong.
        """
        pricey = SKU("HAND-2", unit_value_usd=4_000.0, units_per_carton=10,
                     carton_l_cm=50, carton_w_cm=40, carton_h_cm=30, carton_weight_kg=12.0)

        without = compare(pricey, LANE, 5_000.0, 6.00, UNITS, wacc=0.0)
        assert without["cheaper"] == "ocean", "on freight alone, ocean wins here"

        with_capital = compare(pricey, LANE, 5_000.0, 6.00, UNITS, wacc=0.095)
        assert with_capital["cheaper"] == "air", "costing transit time reverses it"
