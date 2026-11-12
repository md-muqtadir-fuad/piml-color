from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from piml_color_recipe.preprocessing.normalization import looks_like_pantone, looks_like_recipe, looks_like_sample


@dataclass(frozen=True)
class CmcLayoutCandidate:
    layout_name: str
    sample_id: Any
    target_id: Any
    recipe_no: Any


def candidate_score(candidate: CmcLayoutCandidate) -> int:
    return int(looks_like_sample(candidate.sample_id)) + int(looks_like_pantone(candidate.target_id)) + int(looks_like_recipe(candidate.recipe_no))


def select_best_cmc_layout(sample_id: Any, target_id: Any, recipe_no: Any) -> tuple[CmcLayoutCandidate, int]:
    """Select the most plausible CMC schema layout.

    Handles both raw shifted rows and manually corrected-but-swapped rows.
    Expected valid layout is:
        sample_id = trial code with letter, e.g. 00k0001a
        target_id = PANTONE xx-xxxx TCX
        recipe_no = base recipe code, e.g. 00k0001
    """
    candidates = [
        CmcLayoutCandidate("normal", sample_id, target_id, recipe_no),
        CmcLayoutCandidate("sample_recipe_swapped_target_ok", recipe_no, target_id, sample_id),
        CmcLayoutCandidate("target_recipe_shifted", target_id, recipe_no, sample_id),
        CmcLayoutCandidate("sample_target_recipe_rotated", recipe_no, sample_id, target_id),
        CmcLayoutCandidate("recipe_sample_target_rotated", target_id, sample_id, recipe_no),
    ]
    scored = [(candidate_score(c), 1 if c.layout_name == "normal" else 0, c) for c in candidates]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    score, _, best = scored[0]
    return best, score
