"""
Chargeable weight.

Air freight bills on the *greater* of actual weight and volumetric weight, so a
light bulky shipment pays for the space it occupies. Volumetric weight in the
metric convention is (L x W x H in cm) / 6000.

Reference for both the 6000 divisor and the "higher of" rule:
https://www.maersk.com/logistics-explained/transportation-and-freight/2025/03/10/air-cargo-chargeable-weight

Ocean is billed per container, not per kg, so chargeable weight doesn't apply
there — what matters for ocean is how many units fit in a 40ft box, which is
the cube calculation below.
"""
from __future__ import annotations

# Interior capacity of a standard 40ft dry container. Nominal is ~67.7 m3;
# usable is lower once you account for pallet footprints and stacking limits.
FEU_CUBIC_M = 67.7

# Fraction of that cube actually filled. Real loads rarely exceed ~80% because
# cartons don't tessellate perfectly and weight limits bind first on dense
# freight. Stated here rather than buried in a formula.
DEFAULT_UTILIZATION = 0.80

VOLUMETRIC_DIVISOR = 6000.0  # cm^3 per kg, metric air convention


def volumetric_weight_kg(length_cm: float, width_cm: float, height_cm: float,
                         pieces: int = 1, divisor: float = VOLUMETRIC_DIVISOR) -> float:
    """Volumetric (dimensional) weight for `pieces` cartons of these dimensions."""
    if min(length_cm, width_cm, height_cm) <= 0:
        raise ValueError("carton dimensions must be positive")
    if pieces < 1:
        raise ValueError("pieces must be at least 1")
    return (length_cm * width_cm * height_cm) / divisor * pieces


def chargeable_weight_kg(actual_weight_kg: float, volumetric_kg: float) -> float:
    """The greater of actual and volumetric — what the airline actually bills."""
    if actual_weight_kg < 0 or volumetric_kg < 0:
        raise ValueError("weights cannot be negative")
    return max(actual_weight_kg, volumetric_kg)


def carton_cubic_m(length_cm: float, width_cm: float, height_cm: float) -> float:
    return (length_cm * width_cm * height_cm) / 1_000_000.0


def units_per_feu(length_cm: float, width_cm: float, height_cm: float,
                  units_per_carton: int, utilization: float = DEFAULT_UTILIZATION) -> int:
    """How many saleable units fit in one 40ft container.

    Cube-limited only. For dense freight the 26-tonne payload limit binds first;
    this project's SKUs are consumer goods well under that, and the README says
    so rather than the code silently pretending otherwise.
    """
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    if units_per_carton < 1:
        raise ValueError("units_per_carton must be at least 1")

    cartons = int((FEU_CUBIC_M * utilization) // carton_cubic_m(length_cm, width_cm, height_cm))
    return cartons * units_per_carton
