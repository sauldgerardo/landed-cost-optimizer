"""
Mode allocation as a mixed-integer program.

Choosing air vs ocean shipment-by-shipment on cost alone is a greedy rule, and
greedy is wrong as soon as anything is shared. Two things couple the decisions:

  * Air capacity. A weekly allocation is finite, so putting one shipment on a
    plane can push another off it.
  * A service-level target across the whole book, rather than per shipment.
    Some shipments are allowed to arrive late if enough others don't.

Both make this a knapsack-flavoured assignment problem, which is what a solver
is for. PuLP with the bundled CBC solver — no license, nothing to install.

    minimise  sum_i sum_m  cost[i][m] * x[i][m]  +  penalty * sum_i late[i]

    s.t.  sum_m x[i][m] = 1                     each shipment ships exactly once
          sum_i x[i][air] * kg[i] <= air_capacity_kg
          sum_i (1 - late[i]) >= ceil(service * N)
          late[i] >= x[i][m]   for every mode m that misses shipment i's due date
          x, late binary
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pulp


def _solver():
    """CBC, whichever way this PuLP version exposes it.

    PuLP 3.x deprecates PULP_CBC_CMD in favour of COIN_CMD but only ships the
    bundled binary under the old name. Prefer the new API when it can actually
    find a solver, fall back otherwise, so this keeps working across the 4.0
    boundary without pinning.
    """
    if hasattr(pulp, "COIN_CMD"):
        try:
            s = pulp.COIN_CMD(msg=0)
            if s.available():
                return s
        except Exception:
            pass
    return pulp.PULP_CBC_CMD(msg=0)


@dataclass
class Shipment:
    ref: str
    sku: str
    units: int
    days_until_due: int
    chargeable_kg: float
    cost: dict          # mode -> total landed cost, USD
    transit_days: dict  # mode -> door-to-door days

    def is_late(self, mode: str) -> bool:
        return self.transit_days[mode] > self.days_until_due


@dataclass
class Allocation:
    status: str
    assignments: dict          # ref -> mode
    total_cost: float
    freight_cost: float
    late_refs: list
    on_time_rate: float
    air_kg_used: float
    air_capacity_kg: float


def solve(shipments: list[Shipment],
          air_capacity_kg: float,
          service_target: float = 0.95,
          late_penalty_usd: float = 2_500.0,
          modes: tuple = ("ocean", "air")) -> Allocation:
    """Least-cost mode per shipment subject to air capacity and a service floor.

    `late_penalty_usd` prices a missed due date. It is a policy input, not a
    measured cost — raise it and the model buys more air. The service
    constraint is the hard floor; the penalty shapes behaviour above it.
    """
    if not shipments:
        raise ValueError("no shipments to allocate")
    if not 0 <= service_target <= 1:
        raise ValueError("service_target must be between 0 and 1")

    prob = pulp.LpProblem("mode_allocation", pulp.LpMinimize)

    x = {
        (s.ref, m): pulp.LpVariable(f"x_{s.ref}_{m}", cat="Binary")
        for s in shipments for m in modes
    }
    late = {s.ref: pulp.LpVariable(f"late_{s.ref}", cat="Binary") for s in shipments}

    prob += (
        pulp.lpSum(s.cost[m] * x[(s.ref, m)] for s in shipments for m in modes)
        + late_penalty_usd * pulp.lpSum(late.values())
    ), "total_cost_plus_late_penalty"

    for s in shipments:
        prob += pulp.lpSum(x[(s.ref, m)] for m in modes) == 1, f"one_mode_{s.ref}"
        # late[i] is forced on if the chosen mode can't make the due date.
        for m in modes:
            if s.is_late(m):
                prob += late[s.ref] >= x[(s.ref, m)], f"late_{s.ref}_{m}"

    prob += (
        pulp.lpSum(x[(s.ref, "air")] * s.chargeable_kg for s in shipments) <= air_capacity_kg
    ), "air_capacity"

    required_on_time = math.ceil(service_target * len(shipments))
    prob += (
        len(shipments) - pulp.lpSum(late.values()) >= required_on_time
    ), "service_level"

    prob.solve(_solver())
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        return Allocation(status, {}, float("nan"), float("nan"), [], 0.0, 0.0, air_capacity_kg)

    assignments = {
        s.ref: next(m for m in modes if x[(s.ref, m)].value() > 0.5)
        for s in shipments
    }
    freight = sum(s.cost[assignments[s.ref]] for s in shipments)
    late_refs = [s.ref for s in shipments if late[s.ref].value() > 0.5]
    air_kg = sum(s.chargeable_kg for s in shipments if assignments[s.ref] == "air")

    return Allocation(
        status=status,
        assignments=assignments,
        total_cost=pulp.value(prob.objective),
        freight_cost=freight,
        late_refs=late_refs,
        on_time_rate=1 - len(late_refs) / len(shipments),
        air_kg_used=air_kg,
        air_capacity_kg=air_capacity_kg,
    )


def greedy_baseline(shipments: list[Shipment], modes: tuple = ("ocean", "air")) -> dict:
    """Cheapest mode per shipment, ignoring capacity and service entirely.

    This is what a spreadsheet does. It exists so the solver's value can be
    stated as a number rather than asserted.
    """
    assignments = {s.ref: min(modes, key=lambda m: s.cost[m]) for s in shipments}
    return {
        "assignments": assignments,
        "cost": sum(s.cost[assignments[s.ref]] for s in shipments),
        "air_kg": sum(s.chargeable_kg for s in shipments if assignments[s.ref] == "air"),
        "late": [s.ref for s in shipments if s.is_late(assignments[s.ref])],
    }
