"""
The three analyses, and the charts that go with them.

Charts are emitted in a dark and a light variant because they end up on a site
with a theme toggle and a PNG cannot follow one.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import allocate, ingest
from .landed_cost import Lane, air_cost, ocean_cost
from .shipments import BY_SKU

THEMES = {
    "dark": dict(bg="#0a0a0c", ink="#ffffff", mid="#ffffffb0", dim="#ffffff8c",
                 line="#ffffff1f", accent="#22d3ee", second="#ffffff66"),
    "light": dict(bg="#ffffff", ink="#0b0f12", mid="#46525a", dim="#5c6970",
                  line="#dde3e7", accent="#0e7490", second="#8a969e"),
}


def _style(t):
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["ink"], "axes.labelcolor": t["mid"],
        "xtick.color": t["dim"], "ytick.color": t["dim"],
        "axes.edgecolor": t["line"], "grid.color": t["line"],
        "font.family": "monospace", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linewidth": .7, "figure.dpi": 160,
    })


def _save(fig, out_dir, name, theme):
    p = os.path.join(out_dir, f"{name}-{theme}.png")
    fig.savefig(p, bbox_inches="tight", pad_inches=.28)
    plt.close(fig)
    return p


# ── 1. breakeven ─────────────────────────────────────────────────────────────

def breakeven(lane: Lane, out_dir: str, wacc: float, units: int = 2_000) -> dict:
    """At what ocean rate does air become the cheaper landed cost?

    Swept per SKU, because the answer depends entirely on value density — which
    is the actual insight. A cheap dense product never justifies air; a light
    valuable one flips at an ocean rate you routinely see.
    """
    air_rate = ingest.rate_at("air", ingest.ANCHORS["month"])
    feu_grid = np.linspace(800, 12_000, 220)
    crossings = {}

    for theme, t in THEMES.items():
        _style(t)
        fig, ax = plt.subplots(figsize=(7.6, 4.3))

        for i, sku in enumerate(BY_SKU.values()):
            a = air_cost(sku, lane, air_rate, units, wacc=wacc).per_unit
            o = [ocean_cost(sku, lane, f, units, wacc=wacc).per_unit for f in feu_grid]
            gap = np.array(o) - a

            # Highlight the cheapest SKU: it crosses first, which is the
            # counterintuitive part. Air competitiveness tracks chargeable
            # weight per unit, not unit value — so the legend shows both.
            colour = t["accent"] if i == 0 else t["second"]
            ax.plot(feu_grid, gap, lw=2.2 if i == 0 else 1.2, color=colour,
                    label=f"${sku.unit_value_usd:>6.0f}   {sku.kg_per_unit_chargeable:.2f} kg")

            sign = np.sign(gap)
            flip = np.where(np.diff(sign) > 0)[0]
            crossings[sku.sku] = float(feu_grid[flip[0]]) if len(flip) else None

        ax.axhline(0, color=t["ink"], lw=1, alpha=.5)
        ax.set_xlabel("OCEAN RATE  (USD / 40FT)", labelpad=10, fontsize=9)
        ax.set_ylabel("OCEAN MINUS AIR  (USD / UNIT)", labelpad=10, fontsize=9)
        ax.set_title("Above the line, air is the cheaper landed cost",
                     loc="left", pad=16, fontsize=11, color=t["ink"])
        leg = ax.legend(frameon=False, fontsize=8, loc="upper left",
                        title="unit value / chargeable kg per unit")
        leg.get_title().set_color(t["dim"])
        leg.get_title().set_fontsize(8)
        for x in leg.get_texts():
            x.set_color(t["mid"])
        ax.tick_params(length=0)
        fig.tight_layout()
        _save(fig, out_dir, "breakeven", theme)

    flipped = {k: v for k, v in crossings.items() if v}
    return {"breakeven_usd_per_feu": {k: round(v) for k, v in flipped.items()}}


# ── 2. anchor sensitivity ────────────────────────────────────────────────────

def anchor_sensitivity(lane: Lane, out_dir: str, wacc: float, units: int = 2_000) -> dict:
    """How much does the recommendation depend on the two assumed anchors?

    The anchors are the weakest input in the project, so the honest thing is to
    show the answer across a range rather than assert a point estimate.
    """
    ocean_grid = np.linspace(1_500, 12_000, 60)
    air_grid = np.linspace(3.0, 9.0, 60)
    sku = BY_SKU["SKU-1004"]  # the genuinely close call

    grid = np.zeros((len(air_grid), len(ocean_grid)))
    for i, ar in enumerate(air_grid):
        for j, orate in enumerate(ocean_grid):
            o = ocean_cost(sku, lane, orate, units, wacc=wacc).per_unit
            a = air_cost(sku, lane, ar, units, wacc=wacc).per_unit
            grid[i, j] = o - a

    share_air = float((grid > 0).mean())

    for theme, t in THEMES.items():
        _style(t)
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        lim = np.abs(grid).max()
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlBu_r",
                       vmin=-lim, vmax=lim,
                       extent=[ocean_grid[0], ocean_grid[-1], air_grid[0], air_grid[-1]])
        cs = ax.contour(ocean_grid, air_grid, grid, levels=[0], colors=[t["ink"]], linewidths=1.6)
        ax.clabel(cs, fmt={0: "break-even"}, fontsize=8)
        ax.set_xlabel("OCEAN ANCHOR  (USD / 40FT)", labelpad=10, fontsize=9)
        ax.set_ylabel("AIR ANCHOR  (USD / KG)", labelpad=10, fontsize=9)
        ax.set_title(f"{sku.sku} at ${sku.unit_value_usd:.0f}/unit — where air wins",
                     loc="left", pad=16, fontsize=11, color=t["ink"])
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("ocean minus air, USD/unit", color=t["mid"], fontsize=8)
        cb.ax.tick_params(colors=t["dim"], length=0)
        cb.outline.set_edgecolor(t["line"])
        ax.tick_params(length=0)
        fig.tight_layout()
        _save(fig, out_dir, "sensitivity", theme)

    return {"air_wins_share_of_anchor_grid": f"{share_air:.0%}"}


# ── 3. service level cost curve ──────────────────────────────────────────────

def service_curve(shipments_df: pd.DataFrame, lane: Lane, out_dir: str, wacc: float,
                  air_capacity_kg: float, late_penalty: float) -> dict:
    """What does each extra point of on-time delivery actually cost?

    Re-solves the MIP across service targets. The curve is convex for the same
    reason the safety-stock curve is: the last few points are bought with the
    most expensive shipments.
    """
    air_rate = ingest.rate_at("air", ingest.ANCHORS["month"])
    ocean_rate = ingest.rate_at("ocean", ingest.ANCHORS["month"])

    book = []
    for row in shipments_df.itertuples():
        sku = BY_SKU[row.sku]
        o = ocean_cost(sku, lane, ocean_rate, row.units, wacc=wacc)
        a = air_cost(sku, lane, air_rate, row.units, wacc=wacc)
        book.append(allocate.Shipment(
            ref=row.ref, sku=row.sku, units=row.units,
            days_until_due=row.days_until_due,
            chargeable_kg=sku.kg_per_unit_chargeable * row.units,
            cost={"ocean": o.total, "air": a.total},
            transit_days={"ocean": o.transit_days, "air": a.transit_days},
        ))

    targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.0]
    xs, ys = [], []
    for tgt in targets:
        r = allocate.solve(book, air_capacity_kg=air_capacity_kg,
                           service_target=tgt, late_penalty_usd=late_penalty)
        if r.status == "Optimal":
            xs.append(tgt * 100)
            ys.append(r.freight_cost)

    for theme, t in THEMES.items():
        _style(t)
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        ax.plot(xs, ys, color=t["accent"], lw=2.2, marker="o", ms=5)
        ax.set_xlabel("ON-TIME TARGET  (%)", labelpad=10, fontsize=9)
        ax.set_ylabel("TOTAL LANDED COST  (USD)", labelpad=10, fontsize=9)
        ax.set_title("Cost of buying service, solved at each target",
                     loc="left", pad=16, fontsize=11, color=t["ink"])
        ax.tick_params(length=0)
        fig.tight_layout()
        _save(fig, out_dir, "service-curve", theme)

    out = {}
    if len(xs) >= 2:
        base, top = ys[0], ys[-1]
        out[f"service_cost_{xs[0]:.0f}_to_{xs[-1]:.0f}pct"] = (
            f"${top - base:,.0f} ({(top / base - 1):.0%})")
        out["service_max_feasible"] = f"{xs[-1]:.0f}%"
    return out


def run_all(series, lane, out_dir, shipments_df, wacc, air_capacity_kg, late_penalty) -> dict:
    findings = {}
    findings.update(breakeven(lane, out_dir, wacc))
    findings.update(anchor_sensitivity(lane, out_dir, wacc))
    findings.update(service_curve(shipments_df, lane, out_dir, wacc,
                                  air_capacity_kg, late_penalty))
    return findings
