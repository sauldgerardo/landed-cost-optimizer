"""
Freight Mode & Landed Cost Optimizer — end to end.

    python main.py

Loads published freight PPIs, costs a book of shipments both ways, solves the
mode allocation, and writes the analyses to output/. Every number printed here
is reproducible on a clean clone: the rate data is a committed snapshot and the
shipment generator is seeded.
"""
from __future__ import annotations

import os

import pandas as pd

from src import allocate, ingest, report
from src.landed_cost import Lane, air_cost, ocean_cost
from src.shipments import BY_SKU, write_csv

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")

WACC = 0.095            # cost of capital used to price transit time
# Weekly air allocation. Deliberately below the ~116,000 kg it would take to
# fly every urgent shipment, so the capacity constraint actually binds — an
# allocation problem where nothing binds is just a lookup table.
AIR_CAPACITY_KG = 95_000
SERVICE_TARGET = 0.95
LATE_PENALTY = 2_500.0


def max_feasible_service(shipments, air_capacity_kg, late_penalty):
    """Highest on-time target that can actually be met at this air capacity.

    Worth knowing on its own: it's the honest answer to "why can't we promise
    95%", and it's a number you can take to a carrier negotiation.
    """
    best = None
    for pct in range(100, 49, -1):
        r = allocate.solve(shipments, air_capacity_kg=air_capacity_kg,
                           service_target=pct / 100, late_penalty_usd=late_penalty)
        if r.status == "Optimal":
            best = (pct / 100, r)
            break
    return best


def build_shipments(df: pd.DataFrame, usd_per_feu: float, usd_per_kg: float,
                    lane: Lane) -> list[allocate.Shipment]:
    out = []
    for row in df.itertuples():
        sku = BY_SKU[row.sku]
        o = ocean_cost(sku, lane, usd_per_feu, row.units, wacc=WACC)
        a = air_cost(sku, lane, usd_per_kg, row.units, wacc=WACC)
        out.append(allocate.Shipment(
            ref=row.ref, sku=row.sku, units=row.units,
            days_until_due=row.days_until_due,
            chargeable_kg=sku.kg_per_unit_chargeable * row.units,
            cost={"ocean": o.total, "air": a.total},
            transit_days={"ocean": o.transit_days, "air": a.transit_days},
        ))
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    lane = Lane("Shanghai -> Los Angeles")

    # ── rates ────────────────────────────────────────────────────────────────
    series = ingest.load_all()
    as_of = min(s.index.max() for s in series.values())
    usd_per_feu = ingest.rate_at("ocean", as_of, series["ocean"])
    usd_per_kg = ingest.rate_at("air", as_of, series["air"])

    print(f"Rates as of {as_of.date()}  (PPI-indexed off {ingest.ANCHORS['month'][:7]} anchors)")
    print(f"  ocean  ${usd_per_feu:,.0f} / 40ft")
    print(f"  air    ${usd_per_kg:,.2f} / chargeable kg")

    # ── shipment book ────────────────────────────────────────────────────────
    df = write_csv(os.path.join(DATA, "shipments.csv"))
    shipments = build_shipments(df, usd_per_feu, usd_per_kg, lane)
    print(f"\n{len(shipments)} shipments, {df.units.sum():,} units")

    # ── allocation ───────────────────────────────────────────────────────────
    result = allocate.solve(
        shipments,
        air_capacity_kg=AIR_CAPACITY_KG,
        service_target=SERVICE_TARGET,
        late_penalty_usd=LATE_PENALTY,
    )
    target_used = SERVICE_TARGET

    if result.status != "Optimal":
        # Not an error — the capacity genuinely can't buy that much service.
        # Report the binding target rather than printing NaN.
        found = max_feasible_service(shipments, AIR_CAPACITY_KG, LATE_PENALTY)
        print(f"\n  {SERVICE_TARGET:.0%} on-time is INFEASIBLE at "
              f"{AIR_CAPACITY_KG:,} kg of air capacity.")
        if found:
            target_used, result = found
            print(f"  Highest achievable target is {target_used:.0%}. Solving at that instead.")
        else:
            raise SystemExit("  no feasible allocation at any target — check the inputs")

    greedy = allocate.greedy_baseline(shipments)

    print(f"\nAllocation ({result.status}, target {target_used:.0%})")
    air_n = sum(1 for m in result.assignments.values() if m == "air")
    print(f"  air {air_n} / ocean {len(shipments) - air_n}")
    print(f"  freight + duty + capital   ${result.freight_cost:,.0f}")
    print(f"  on-time                    {result.on_time_rate:.0%}  (target {target_used:.0%})")
    print(f"  air allocation used        {result.air_kg_used:,.0f} / {AIR_CAPACITY_KG:,} kg")
    print(f"\n  cheapest-mode baseline     ${greedy['cost']:,.0f}"
          f"  using {greedy['air_kg']:,.0f} kg of air"
          f"  ({len(greedy['late'])} shipments late)")

    over = greedy["air_kg"] / AIR_CAPACITY_KG
    if over > 1:
        print(f"  -> that baseline needs {over:.1f}x the air allocation that exists,")
        print(f"     so it isn't a plan. The gap is what the constraint costs.")

    # ── analyses ─────────────────────────────────────────────────────────────
    print()
    findings = report.run_all(
        series=series, lane=lane, out_dir=OUT,
        shipments_df=df, wacc=WACC,
        air_capacity_kg=AIR_CAPACITY_KG, late_penalty=LATE_PENALTY,
    )
    for k, v in findings.items():
        print(f"  {k}: {v}")

    print(f"\nCharts written to {os.path.relpath(OUT, ROOT)}/")


if __name__ == "__main__":
    main()
