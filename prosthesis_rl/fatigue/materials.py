"""Material + manufacturing fatigue database for the arm's loaded joints.

`fatigue.estimate` used to hard-code one material (FDM PA12-CF) as module
constants. This makes the material a first-class, swappable input so the
durability recommender can ask "what if this joint were MJF nylon / CNC
aluminium / titanium instead?" and recompute the lifespan honestly through the
*same* Basquin S-N math.

Each `Material` carries the Basquin coefficients (sigma_f', b), an endurance
limit (amplitude below which life is effectively infinite), density (for the
mass trade-off the recommender reports), and a **process knockdown** — the
fraction of bulk fatigue strength a given manufacturing route actually delivers
(FDM inter-layer adhesion is the big one, ~0.6; CNC/wrought ~1.0).

Values are literature-order, cross-checked by the `material-fatigue-db` research
pass — good enough to *rank* design choices, NOT a durability certification.
The baseline `pa12_cf_fdm` reproduces the original `fatigue.estimate` constants
exactly, so existing estimates are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

MPa = 1e6


@dataclass(frozen=True)
class Material:
    """One material + manufacturing route, as the fatigue model sees it."""

    key: str
    name: str
    sigma_f_prime_pa: float      # Basquin fatigue strength coefficient (Pa)
    basquin_b: float             # Basquin fatigue strength exponent (negative)
    endurance_limit_pa: float    # amplitude below which life ~ infinite (Pa)
    density_kg_m3: float
    process: str                 # "FDM" | "SLS/MJF" | "CNC" | "wrought"
    process_knockdown: float     # fraction of bulk fatigue strength this route delivers
    rel_cost: float = 1.0        # rough per-part cost+effort, FDM PA12-CF = 1.0
    note: str = ""

    @property
    def effective_sigma_f_pa(self) -> float:
        """Fatigue strength coefficient after the manufacturing knockdown."""
        return self.sigma_f_prime_pa * self.process_knockdown

    @property
    def effective_sigma_f_mpa(self) -> float:
        return self.effective_sigma_f_pa / MPa

    def to_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name,
            "sigma_f_prime_mpa": self.sigma_f_prime_pa / MPa,
            "basquin_b": self.basquin_b,
            "endurance_limit_mpa": self.endurance_limit_pa / MPa,
            "effective_sigma_f_mpa": self.effective_sigma_f_mpa,
            "density_kg_m3": self.density_kg_m3,
            "process": self.process, "process_knockdown": self.process_knockdown,
            "rel_cost": self.rel_cost, "note": self.note,
        }


# --------------------------------------------------------------------------- #
# The database. pa12_cf_fdm MUST stay == the original fatigue.estimate constants
# (sigma_f'=90 MPa, b=-0.10, endurance=18 MPa, knockdown=0.60) so existing
# lifespan numbers don't move. Other rows are literature-order alternatives the
# recommender can swap in. Refined by the material-fatigue-db research pass.
# --------------------------------------------------------------------------- #
# Values + citations from the `material-fatigue-db` research pass (literature-order,
# reconciled for S-N self-consistency). effective σ_f' = σ_f'·knockdown sets the
# ranking: PETG-CF 21 < Onyx 29 < MJF-PA12 33 < PA12-CF(FDM) 80 << Al 500 << Ti 1928 MPa.
# So the carbon-filled FDM baseline is the strongest *printed* option — only metal beats it.
_MATERIAL_LIST: list[Material] = [
    Material("pa12_cf_fdm", "PA12-CF (FDM)", 160 * MPa, -0.10, 30 * MPa, 1060,
             "FDM", 0.50, 1.0,
             "Chopped-CF nylon, FDM. Inter-layer Z-weakness is the limiter (the baseline)."),
    Material("petg_cf_fdm", "PETG-CF (FDM)", 43 * MPa, -0.10, 8 * MPa, 1290,
             "FDM", 0.50, 0.65,
             "Cheaper/tougher but markedly weaker in fatigue than PA12-CF."),
    Material("pa12_mjf", "PA12 nylon (MJF/SLS)", 39 * MPa, -0.056, 15.8 * MPa, 1010,
             "SLS/MJF", 0.85, 1.1,
             "Near-isotropic powder process, but plain (unfilled) — weaker than CF-FDM nylon."),
    Material("onyx_cf", "Onyx micro-CF nylon (FDM)", 52 * MPa, -0.065, 17.9 * MPa, 1200,
             "FDM", 0.55, 1.6,
             "Markforged Onyx (micro-CF); continuous-fibre option would raise this a lot."),
    Material("al_6061_t6", "Aluminium 6061-T6 (CNC)", 527 * MPa, -0.085, 96.5 * MPa, 2700,
             "CNC", 0.95, 3.5,
             "Machined light metal; ~6x the effective fatigue strength of printed nylon."),
    Material("ti_6al_4v", "Titanium Ti-6Al-4V (CNC)", 2030 * MPa, -0.095, 530 * MPa, 4430,
             "CNC", 0.95, 12.0,
             "Highest specific fatigue strength; heavy on cost, light on mass."),
]

MATERIALS: dict[str, Material] = {m.key: m for m in _MATERIAL_LIST}
DEFAULT_MATERIAL_KEY = "pa12_cf_fdm"
DEFAULT_MATERIAL = MATERIALS[DEFAULT_MATERIAL_KEY]


def get_material(key: str | Material | None) -> Material:
    """Resolve a key (or pass-through Material/None) to a Material."""
    if key is None:
        return DEFAULT_MATERIAL
    if isinstance(key, Material):
        return key
    try:
        return MATERIALS[key]
    except KeyError:
        raise KeyError(f"unknown material {key!r}; have {sorted(MATERIALS)}") from None


def all_materials() -> list[Material]:
    return list(_MATERIAL_LIST)
