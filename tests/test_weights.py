import pytest

from src import weights


class TestVolumetricWeight:
    def test_matches_worked_example(self):
        # A 100 x 50 x 60 cm carton is 300000 cm3; / 6000 = 50 kg volumetric.
        assert weights.volumetric_weight_kg(100, 50, 60) == pytest.approx(50.0)

    def test_scales_with_pieces(self):
        one = weights.volumetric_weight_kg(40, 30, 20)
        assert weights.volumetric_weight_kg(40, 30, 20, pieces=7) == pytest.approx(one * 7)

    def test_rejects_nonpositive_dimensions(self):
        with pytest.raises(ValueError):
            weights.volumetric_weight_kg(0, 30, 20)

    def test_rejects_zero_pieces(self):
        with pytest.raises(ValueError):
            weights.volumetric_weight_kg(40, 30, 20, pieces=0)


class TestChargeableWeight:
    def test_dense_freight_bills_actual(self):
        # Heavy and small: actual weight wins.
        assert weights.chargeable_weight_kg(120.0, 50.0) == 120.0

    def test_bulky_freight_bills_volumetric(self):
        # Light and bulky: volumetric wins. This is the whole point of the rule.
        assert weights.chargeable_weight_kg(12.0, 50.0) == 50.0

    def test_tie_is_stable(self):
        assert weights.chargeable_weight_kg(50.0, 50.0) == 50.0

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            weights.chargeable_weight_kg(-1.0, 50.0)


class TestContainerFill:
    def test_units_per_feu_is_cube_limited(self):
        # 50 x 40 x 30 cm = 0.06 m3. 67.7 * 0.80 = 54.16 m3 usable.
        # 54.16 / 0.06 = 902 cartons -> 902 * 12 units.
        assert weights.units_per_feu(50, 40, 30, units_per_carton=12) == 902 * 12

    def test_lower_utilization_fits_fewer(self):
        full = weights.units_per_feu(50, 40, 30, 12, utilization=0.80)
        half = weights.units_per_feu(50, 40, 30, 12, utilization=0.40)
        assert half < full

    def test_rejects_impossible_utilization(self):
        with pytest.raises(ValueError):
            weights.units_per_feu(50, 40, 30, 12, utilization=1.5)
