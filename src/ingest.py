"""
Freight rate data.

Two real, published series from FRED (St. Louis Fed), both free and without an
API key:

  PCU483111483111  PPI, Deep Sea Freight Transportation      1988-06 ->
  PCU481112481112  PPI, Scheduled Freight Air Transportation 2003-12 ->

WHAT IS REAL AND WHAT IS ASSUMED — read this before trusting any output.

  REAL     the *shape* of rate movement over time: when ocean spiked, how far,
           how air moved relative to it. That comes straight from the PPIs.

  ASSUMED  the *dollar level*. A PPI is an index, not a price, so it can't tell
           you what a container costs today. Each mode is anchored to a stated
           market level for one reference month (see ANCHORS) and indexed from
           there.

The anchors are the weakest input in this project, which is exactly why
report.py sweeps them. A conclusion that survives the sweep is worth something;
a point estimate off a single anchor is not.

The original BTS ocean spot-rate source used in an earlier version of this
analysis now returns 403 to non-browser clients, which is why this moved to
FRED. FRED is also better for the job: it covers both modes rather than one
ocean lane.
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

SERIES = {
    "ocean": "PCU483111483111",
    "air": "PCU481112481112",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

# ── Anchors ──────────────────────────────────────────────────────────────────
# Dollar level for one reference month, which the PPI is then indexed against.
# These are assumptions, not observations. Both sit inside commonly reported
# market ranges for the lane; neither is a quote. Change them and rerun — the
# sensitivity analysis exists because the answer should not hinge on them.
ANCHORS = {
    # Shanghai -> Los Angeles, USD per 40ft container.
    "ocean_usd_per_feu": 2200.0,
    # Shanghai -> US West Coast, USD per chargeable kg.
    "air_usd_per_kg": 5.00,
    # Month both anchors are quoted for.
    "month": "2024-01-01",
}

# Door-to-door transit. Stated assumptions, not measured.
TRANSIT_DAYS = {"ocean": 25, "air": 3}


def _cache_path(mode: str, stamp: str) -> str:
    return os.path.join(DATA_DIR, f"fred_{mode}_{SERIES[mode]}_{stamp}.csv")


def fetch_series(mode: str, refresh: bool = False, stamp: str | None = None) -> pd.Series:
    """Monthly PPI for `mode`, cached to data/raw as a dated snapshot.

    The snapshot is committed on purpose: it makes every number in the README
    reproducible. A live pull would quietly change results between runs.
    """
    if mode not in SERIES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(SERIES)}")

    os.makedirs(DATA_DIR, exist_ok=True)

    if stamp is None:
        existing = sorted(
            f for f in os.listdir(DATA_DIR)
            if f.startswith(f"fred_{mode}_") and f.endswith(".csv")
        )
        if existing and not refresh:
            path = os.path.join(DATA_DIR, existing[-1])
        else:
            path = _cache_path(mode, date.today().isoformat())
    else:
        path = _cache_path(mode, stamp)

    if refresh or not os.path.exists(path):
        df = pd.read_csv(FRED_CSV.format(series=SERIES[mode]))
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    df.columns = ["date", "index"]
    df["date"] = pd.to_datetime(df["date"])
    # FRED writes "." for missing observations.
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    return df.dropna().set_index("date")["index"]


def rate_at(mode: str, when, series: pd.Series | None = None) -> float:
    """Dollar rate for `mode` in the month containing `when`.

    index_then / index_anchor * anchor_level. Units are USD/FEU for ocean and
    USD/chargeable-kg for air.
    """
    s = fetch_series(mode) if series is None else series
    when = pd.Timestamp(when).normalize().replace(day=1)

    anchor_month = pd.Timestamp(ANCHORS["month"])
    if anchor_month not in s.index:
        raise ValueError(f"anchor month {anchor_month.date()} missing from the {mode} series")

    # Clamp to the series range rather than extrapolating off the end.
    if when < s.index.min():
        when = s.index.min()
    elif when > s.index.max():
        when = s.index.max()

    if when not in s.index:
        when = s.index[s.index.get_indexer([when], method="nearest")[0]]

    level = ANCHORS["ocean_usd_per_feu"] if mode == "ocean" else ANCHORS["air_usd_per_kg"]
    return float(s.loc[when] / s.loc[anchor_month] * level)


def load_all(refresh: bool = False) -> dict[str, pd.Series]:
    return {m: fetch_series(m, refresh=refresh) for m in SERIES}


if __name__ == "__main__":
    for mode, s in load_all().items():
        print(f"{mode:6} {SERIES[mode]}  {len(s)} obs  "
              f"{s.index.min().date()} -> {s.index.max().date()}  last={s.iloc[-1]:.1f}")
        print(f"       rate at anchor: {rate_at(mode, ANCHORS['month'], s):,.2f}")
        print(f"       rate at latest: {rate_at(mode, s.index.max(), s):,.2f}")
