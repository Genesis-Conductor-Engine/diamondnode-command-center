"""Load and validate the agent model card.

Two things happen here that plain ``yaml.safe_load`` will not do:

1. **Duplicate keys are rejected.** PyYAML silently keeps the *last* value for a
   repeated mapping key. In a card where a section can disable a control, a
   duplicated ``prohibited_persistence:`` or ``never_disable:`` would quietly
   discard the first list — the file would still look correct on a skim.
2. **Unknown fields are rejected**, by the strict models in ``types``. A
   misspelled key is a control that is not in force, not an extension point.

The loader is deliberately the only place that reads the card from disk, so
every consumer works with a frozen, validated :class:`AgentModelCard`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .types import AgentModelCard

DEFAULT_CARD_PATH = (
    Path(__file__).resolve().parents[2]
    / "torx_contextual_sovereignty_agent.model-card.yaml"
)

# Keys the card writes in ``a-b`` form but Python must reach as ``a_b``. Renaming
# them in the YAML would break the published card, so the loader normalises them
# on the way in and the models declare only the Python spelling.
_KEY_ALIASES = {
    "recent_high-confidence_overrides_old-low-confidence": (
        "recent_high_confidence_overrides_old_low_confidence"
    ),
    "single-message-high-impact-inference_forbidden": (
        "single_message_high_impact_inference_forbidden"
    ),
}


class ModelCardError(ValueError):
    """Raised when the card cannot be loaded or does not satisfy its contract."""


class _NoDuplicateKeyLoader(yaml.SafeLoader):
    """SafeLoader that treats a repeated mapping key as an error."""


def _construct_mapping(loader: _NoDuplicateKeyLoader, node: yaml.MappingNode) -> dict:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            mark = key_node.start_mark
            raise ModelCardError(
                f"duplicate key {key!r} at line {mark.line + 1}, "
                f"column {mark.column + 1}: a repeated key silently overwrites "
                "the earlier value and can disable a control"
            )
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_NoDuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _normalise_keys(node: Any) -> Any:
    """Rewrite hyphenated aliases to their Python spelling, recursively."""
    if isinstance(node, dict):
        return {
            _KEY_ALIASES.get(k, k): _normalise_keys(v) for k, v in node.items()
        }
    if isinstance(node, list):
        return [_normalise_keys(v) for v in node]
    return node


def parse_model_card(text: str, *, source: str = "<string>") -> AgentModelCard:
    """Parse card YAML from a string.

    Separated from :func:`load_model_card` so tests can feed mutated cards
    without touching the filesystem.
    """
    try:
        raw = yaml.load(text, Loader=_NoDuplicateKeyLoader)
    except ModelCardError:
        raise
    except yaml.YAMLError as exc:
        raise ModelCardError(f"{source}: YAML did not parse: {exc}") from exc

    if not isinstance(raw, dict):
        raise ModelCardError(
            f"{source}: expected a mapping at the top level, got "
            f"{type(raw).__name__}"
        )

    try:
        return AgentModelCard.model_validate(_normalise_keys(raw))
    except ValidationError as exc:
        raise ModelCardError(f"{source}: card failed validation:\n{exc}") from exc


def load_model_card(path: Path | str | None = None) -> AgentModelCard:
    """Load, validate, and return the model card at ``path``.

    ``path`` defaults to the card shipped alongside this package.
    """
    card_path = Path(path) if path is not None else DEFAULT_CARD_PATH
    try:
        text = card_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelCardError(f"cannot read model card at {card_path}: {exc}") from exc
    return parse_model_card(text, source=str(card_path))


@lru_cache(maxsize=4)
def cached_model_card(path: str | None = None) -> AgentModelCard:
    """Process-wide cached card.

    The card is immutable and read on every compile; re-parsing ~900 lines of
    YAML per request would show up directly in
    ``gc_effective_agent_compile_seconds``.
    """
    return load_model_card(path)
