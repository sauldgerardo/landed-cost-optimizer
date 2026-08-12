"""
Shipment book.

These records are SYNTHETIC and seeded. There is no public dataset of real
company shipment-level data, so the alternative to generating them is not using
real ones — it's having nothing to allocate. The SKU dimensions and values are
chosen to span the range where the air/ocean decision is genuinely close, which
is the interesting region.

Everything downstream of here is real: the rate movements come from published
PPIs, and the cost maths is the maths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .landed_cost import SKU

# A spread of consumer goods: dense/cheap through light/valuable. The last two
# are where air starts winning, which is the point of including them.
CATALOG = [
    SKU("SKU-1001", unit_value_usd=12.50, units_per_carton=24,
        carton_l_cm=50, carton_w_cm=40, carton_h_cm=30, carton_weight_kg=14.0),
    SKU("SKU-1002", unit_value_usd=34.00, units_per_carton=12,
        carton_l_cm=45, carton_w_cm=35, carton_h_cm=28, carton_weight_kg=9.5),
    SKU("SKU-1003", unit_value_usd=78.00, units_per_carton=6,
        carton_l_cm=40, carton_w_cm=30, carton_h_cm=25, carton_weight_kg=5.2),
    SKU("SKU-1004", unit_value_usd=145.00, units_per_carton=4,
        carton_l_cm=38, carton_w_cm=28, carton_h_cm=22, carton_weight_kg=3.1),
    SKU("SKU-1005", unit_value_usd=260.00, units_per_carton=2,
        carton_l_cm=35, carton_w_cm=25, carton_h_cm=20, carton_weight_kg=1.8),
]

BY_SKU = {s.sku: s for s in CATALOG}


def generate(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """A book of `n` shipments with varied sizes and urgency."""
    rng = np.random.default_rng(seed)

    skus = rng.choice([s.sku for s in CATALOG], size=n)
    # Order sizes span roughly a third of a container up to several.
    units = rng.integers(400, 9_000, size=n)
    # Urgency mix: most comfortable, a meaningful minority tight enough that
    # ocean's 25 days cannot make it.
    days_until_due = rng.choice(
        [7, 12, 18, 25, 30, 45, 60],
        size=n,
        p=[0.08, 0.10, 0.14, 0.18, 0.20, 0.18, 0.12],
    )

    return pd.DataFrame({
        "ref": [f"SH-{i:04d}" for i in range(1, n + 1)],
        "sku": skus,
        "units": units,
        "days_until_due": days_until_due,
    })


def write_csv(path: str, n: int = 60, seed: int = 42) -> pd.DataFrame:
    df = generate(n=n, seed=seed)
    df.to_csv(path, index=False)
    return df
