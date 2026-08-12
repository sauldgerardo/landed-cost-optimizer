"""
Per-unit landed cost, built from components rather than a single freight rate.

The point of this module is that freight is not the whole cost, and comparing
air to ocean on freight alone gets the answer wrong. Two things usually missing
from a naive comparison:

  1. Fixed per-shipment charges (customs entry, handling, drayage) that don't
     scale with volume and so hit small shipments hardest.

  2. The capital cost of transit time. Ocean's 25 days is 22 more days of
     inventory sitting on the water than air's 3. On a $40 unit at a 9.5% cost
     of capital that's about $0.23 — small per unit, real at volume, and it is
     the term that decides close calls.

Duty is charged on customs value, not on freight, under FOB-basis valuation,
which is why it's computed off unit_value rather than off the landed total.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import weights


@dataclass(frozen=True)
class Lane:
    """Fixed and variable charges for one origin-destination pair, per mode."""
    name: str
    # Fixed per shipment, USD.
    ocean_fixed: float = 1_150.0   # drayage both ends, terminal handling, customs entry
    air_fixed: float = 320.0       # airport handling, customs entry, delivery
    # Duty as a fraction of customs value.
    duty_rate: float = 0.043
    # Insurance as a fraction of customs value.
    insurance_rate: float = 0.003


@dataclass(frozen=True)
class SKU:
    """A product being shipped. Dimensions are per carton, in cm."""
    sku: str
    unit_value_usd: float
    units_per_carton: int
    carton_l_cm: float
    carton_w_cm: float
    carton_h_cm: float
    carton_weight_kg: float

    @property
    def units_per_container(self) -> int:
        return weights.units_per_feu(
            self.carton_l_cm, self.carton_w_cm, self.carton_h_cm, self.units_per_carton
        )

    @property
    def kg_per_unit_chargeable(self) -> float:
        """Chargeable kg attributable to one unit."""
        vol = weights.volumetric_weight_kg(self.carton_l_cm, self.carton_w_cm, self.carton_h_cm)
        chargeable = weights.chargeable_weight_kg(self.carton_weight_kg, vol)
        return chargeable / self.units_per_carton


@dataclass
class CostBreakdown:
    mode: str
    freight: float = 0.0
    fixed: float = 0.0
    duty: float = 0.0
    insurance: float = 0.0
    transit_capital: float = 0.0
    units: int = 0
    transit_days: int = 0
    components: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.freight + self.fixed + self.duty + self.insurance + self.transit_capital

    @property
    def per_unit(self) -> float:
        return self.total / self.units if self.units else float("inf")


def transit_capital_cost(unit_value_usd: float, units: int, days: int, wacc: float) -> float:
    """Capital tied up in goods while they're in transit.

    Simple carrying cost: value x rate x (days / 365). Not discounting a cash
    flow — it's the opportunity cost of inventory in motion, which is the
    standard way this is charged in a landed-cost model.
    """
    return unit_value_usd * units * wacc * (days / 365.0)


def ocean_cost(sku: SKU, lane: Lane, usd_per_feu: float, units: int,
               wacc: float = 0.095, transit_days: int = 25) -> CostBreakdown:
    containers = max(1, -(-units // sku.units_per_container))  # ceil
    customs_value = sku.unit_value_usd * units

    b = CostBreakdown(mode="ocean", units=units, transit_days=transit_days)
    b.freight = usd_per_feu * containers
    b.fixed = lane.ocean_fixed * containers
    b.duty = customs_value * lane.duty_rate
    b.insurance = customs_value * lane.insurance_rate
    b.transit_capital = transit_capital_cost(sku.unit_value_usd, units, transit_days, wacc)
    b.components = {"containers": containers, "units_per_container": sku.units_per_container}
    return b


def air_cost(sku: SKU, lane: Lane, usd_per_kg: float, units: int,
             wacc: float = 0.095, transit_days: int = 3) -> CostBreakdown:
    chargeable_kg = sku.kg_per_unit_chargeable * units
    customs_value = sku.unit_value_usd * units

    b = CostBreakdown(mode="air", units=units, transit_days=transit_days)
    b.freight = usd_per_kg * chargeable_kg
    b.fixed = lane.air_fixed
    b.duty = customs_value * lane.duty_rate
    b.insurance = customs_value * lane.insurance_rate
    b.transit_capital = transit_capital_cost(sku.unit_value_usd, units, transit_days, wacc)
    b.components = {"chargeable_kg": chargeable_kg}
    return b


def compare(sku: SKU, lane: Lane, usd_per_feu: float, usd_per_kg: float, units: int,
            wacc: float = 0.095) -> dict:
    o = ocean_cost(sku, lane, usd_per_feu, units, wacc)
    a = air_cost(sku, lane, usd_per_kg, units, wacc)
    return {
        "ocean": o,
        "air": a,
        "cheaper": "ocean" if o.per_unit <= a.per_unit else "air",
        "gap_per_unit": abs(o.per_unit - a.per_unit),
    }
