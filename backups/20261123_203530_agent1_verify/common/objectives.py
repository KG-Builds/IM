"""The six SSA objective variants.

Formulas are preserved exactly from the starter implementation:

O1 Original                 : Spread_A + 1.5 * Blocked_B
O2 Competition-aware        : Spread_A - Spread_B
O3 Explicit blocking        : Blocked_B
O4 Combined influence-block : Spread_A + 2.0 * Blocked_B
O5 Gain_A + blocking        : (Spread_A - Spread_B) + 2.0 * Blocked_B
O6 Budget-aware             : alpha(K)*Spread_A + beta(K)*Blocked_B

O4 and O5 are different and must never be conflated.

Validity: O2 does not depend on Blocked_B and is fully validated by the
project's explicit tests.  O1/O3/O4/O5/O6 depend on the Blocked_B protocol,
which has no reference notebook in this workspace and is therefore marked
``provisional_blocked_b``.  Results for those variants are stored and reported
separately; they are never presented as a validated six-objective result.
"""
from __future__ import annotations

from collections import OrderedDict

BUDGET_COEFFICIENTS = {
    10: (0.75, 2.25),
    20: (0.75, 2.25),
    30: (1.00, 1.75),
    40: (1.00, 1.75),
    50: (1.25, 1.50),
}

VARIANTS = OrderedDict(
    [
        (
            "O1",
            {
                "id": "O1",
                "name": "Original SSA",
                "type": "weighted",
                "alpha": 1.0,
                "beta": 1.5,
                "formula": "Spread_A + 1.5 * Blocked_B",
                "blocked_b_required": True,
                "validity": "provisional_blocked_b",
            },
        ),
        (
            "O2",
            {
                "id": "O2",
                "name": "Competition-aware",
                "type": "net",
                "formula": "Spread_A - Spread_B",
                "blocked_b_required": False,
                "validity": "validated",
            },
        ),
        (
            "O3",
            {
                "id": "O3",
                "name": "Explicit blocking",
                "type": "blocking",
                "formula": "Blocked_B",
                "blocked_b_required": True,
                "validity": "provisional_blocked_b",
            },
        ),
        (
            "O4",
            {
                "id": "O4",
                "name": "Combined influence-blocking",
                "type": "weighted",
                "alpha": 1.0,
                "beta": 2.0,
                "formula": "Spread_A + 2.0 * Blocked_B",
                "blocked_b_required": True,
                "validity": "provisional_blocked_b",
            },
        ),
        (
            "O5",
            {
                "id": "O5",
                "name": "Gain_A + blocking",
                "type": "gain_blocking",
                "beta": 2.0,
                "formula": "(Spread_A - Spread_B) + 2.0 * Blocked_B",
                "blocked_b_required": True,
                "validity": "provisional_blocked_b",
            },
        ),
        (
            "O6",
            {
                "id": "O6",
                "name": "Budget-aware",
                "type": "budget",
                "formula": "alpha(K)*Spread_A + beta(K)*Blocked_B",
                "blocked_b_required": True,
                "validity": "provisional_blocked_b",
            },
        ),
    ]
)

VARIANT_IDS = list(VARIANTS.keys())


def resolve_variant(variant: str) -> dict:
    """Accept an id (``O2``) or a name (``Competition-aware``)."""
    if variant in VARIANTS:
        return VARIANTS[variant]
    for spec in VARIANTS.values():
        if spec["name"] == variant:
            return spec
    raise ValueError(f"Unknown objective variant {variant!r}; expected {VARIANT_IDS} or a known name")


def budget_coefficients(k: int) -> tuple[float, float]:
    k = int(k)
    if k not in BUDGET_COEFFICIENTS:
        raise ValueError(f"No budget coefficients defined for K={k}")
    return BUDGET_COEFFICIENTS[k]


def objective_score(metrics: dict, variant: str, k: int) -> float:
    """Evaluate one objective from a metrics dictionary.

    Required metric keys: ``spread_A``, ``spread_B``, ``net_advantage`` and,
    for Blocked_B-dependent variants, ``blocked_B``.
    """
    spec = resolve_variant(variant)
    t = spec["type"]
    if t == "net":
        return float(metrics["net_advantage"])
    if t == "blocking":
        return float(metrics["blocked_B"])
    if t == "weighted":
        return float(spec["alpha"]) * float(metrics["spread_A"]) + float(spec["beta"]) * float(metrics["blocked_B"])
    if t == "gain_blocking":
        return float(metrics["net_advantage"]) + float(spec["beta"]) * float(metrics["blocked_B"])
    if t == "budget":
        alpha, beta = budget_coefficients(k)
        return alpha * float(metrics["spread_A"]) + beta * float(metrics["blocked_B"])
    raise ValueError(f"Unknown objective type {t!r}")


def objective_validity(variant: str) -> str:
    return resolve_variant(variant)["validity"]


def requires_blocked_b(variant: str) -> bool:
    return bool(resolve_variant(variant)["blocked_b_required"])


def variant_metadata() -> list[dict]:
    return [dict(spec) for spec in VARIANTS.values()]
