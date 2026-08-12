"""
Solver tests.

The important ones are the toy instances whose optimum can be worked out by
hand — if the model is wrong, matching a hand-computed answer is what catches
it. Asserting only that the solver returns "Optimal" proves nothing.
"""
import pytest

from src.allocate import Shipment, greedy_baseline, solve


def ship(ref, cost_ocean, cost_air, kg=100.0, due=30, units=1000):
    return Shipment(
        ref=ref, sku="TEST", units=units, days_until_due=due, chargeable_kg=kg,
        cost={"ocean": cost_ocean, "air": cost_air},
        transit_days={"ocean": 25, "air": 3},
    )


class TestBasicStructure:
    def test_every_shipment_gets_exactly_one_mode(self):
        s = [ship(f"S{i}", 1000, 1400) for i in range(6)]
        r = solve(s, air_capacity_kg=10_000, service_target=0.0)
        assert r.status == "Optimal"
        assert set(r.assignments) == {x.ref for x in s}
        assert set(r.assignments.values()) <= {"ocean", "air"}

    def test_picks_the_cheaper_mode_when_unconstrained(self):
        # Ocean cheaper everywhere, nothing binding -> all ocean.
        s = [ship(f"S{i}", 1000, 5000) for i in range(5)]
        r = solve(s, air_capacity_kg=1e9, service_target=0.0)
        assert set(r.assignments.values()) == {"ocean"}

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            solve([], air_capacity_kg=1000)

    def test_invalid_service_target_rejected(self):
        with pytest.raises(ValueError):
            solve([ship("S1", 1, 2)], air_capacity_kg=1, service_target=1.4)


class TestAirCapacityBinds:
    def test_capacity_caps_air_usage(self):
        # Air is cheaper for all five, but only two fit in the allocation.
        s = [ship(f"S{i}", 5000, 1000, kg=100.0) for i in range(5)]
        r = solve(s, air_capacity_kg=250.0, service_target=0.0)

        assert r.status == "Optimal"
        air = [ref for ref, m in r.assignments.items() if m == "air"]
        assert len(air) == 2, "250kg / 100kg per shipment = 2 shipments"
        assert r.air_kg_used <= r.air_capacity_kg

    def test_known_optimum_by_hand(self):
        # Three shipments, air saves a different amount on each, room for one.
        #   A: ocean 5000, air 1000  -> saves 4000
        #   B: ocean 3000, air 2000  -> saves 1000
        #   C: ocean 2500, air 2000  -> saves  500
        # One air slot, so put A on it. Optimal = 1000 + 3000 + 2500 = 6500.
        s = [ship("A", 5000, 1000), ship("B", 3000, 2000), ship("C", 2500, 2000)]
        r = solve(s, air_capacity_kg=100.0, service_target=0.0, late_penalty_usd=0.0)

        assert r.assignments == {"A": "air", "B": "ocean", "C": "ocean"}
        assert r.total_cost == pytest.approx(6500.0)


class TestServiceLevel:
    def test_service_floor_forces_air(self):
        # Due in 10 days: ocean (25d) is late, air (3d) is not. Ocean is
        # cheaper, so with no service floor the model would ship it all ocean.
        s = [ship(f"S{i}", 1000, 4000, due=10) for i in range(4)]

        loose = solve(s, air_capacity_kg=1e9, service_target=0.0, late_penalty_usd=0.0)
        assert set(loose.assignments.values()) == {"ocean"}
        assert loose.on_time_rate == 0.0

        strict = solve(s, air_capacity_kg=1e9, service_target=1.0, late_penalty_usd=0.0)
        assert set(strict.assignments.values()) == {"air"}
        assert strict.on_time_rate == 1.0

    def test_partial_service_target_buys_only_what_it_must(self):
        # 4 urgent shipments, 75% on-time required -> exactly 3 must fly.
        s = [ship(f"S{i}", 1000, 4000, due=10) for i in range(4)]
        r = solve(s, air_capacity_kg=1e9, service_target=0.75, late_penalty_usd=0.0)

        flown = sum(1 for m in r.assignments.values() if m == "air")
        assert flown == 3
        assert r.on_time_rate == pytest.approx(0.75)

    def test_infeasible_when_capacity_cannot_meet_service(self):
        # All urgent, all need air, but the air allocation fits none of them.
        s = [ship(f"S{i}", 1000, 4000, due=5, kg=500.0) for i in range(3)]
        r = solve(s, air_capacity_kg=100.0, service_target=1.0)
        assert r.status == "Infeasible"


class TestAgainstGreedy:
    def test_greedy_can_violate_capacity_where_solver_cannot(self):
        s = [ship(f"S{i}", 5000, 1000, kg=100.0) for i in range(5)]

        g = greedy_baseline(s)
        assert g["air_kg"] == 500.0, "greedy flies everything"

        r = solve(s, air_capacity_kg=250.0, service_target=0.0)
        assert r.air_kg_used <= 250.0
        # Respecting the constraint costs money; that gap is the honest
        # statement of what the constraint is worth.
        assert r.freight_cost > g["cost"]
