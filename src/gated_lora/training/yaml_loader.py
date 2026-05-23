"""YAML config loader with extends/overrides resolution.

Loading order:
1. Load the requested YAML file.
2. Recursively load all paths in its `extends:` list (relative to configs/).
3. Deep-merge: later entries in `extends` override earlier ones.
4. Apply the file's own top-level keys on top.
5. Apply `overrides:` blocks from any ablation YAML in the chain.

Returns a plain dict — pass to `ExperimentConfig.from_dict()` to build a typed
config object, or use directly when only a subset of fields is needed.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict

import yaml


CONFIGS_ROOT = Path(__file__).resolve().parents[3] / "configs"


class _Yaml12FloatLoader(yaml.SafeLoader):
    """SafeLoader patched with the YAML 1.2 float resolver.

    PyYAML's default SafeLoader implements YAML 1.1, which requires a leading
    sign or decimal point in scientific notation (``2.0e-4`` or ``+2e-4``).
    Without it, ``learning_rate: 2e-4`` is parsed as the string ``"2e-4"`` and
    quietly breaks downstream (e.g. ``torch.optim.AdamW`` raises
    ``TypeError: '<=' not supported between instances of 'float' and 'str'``).
    Switching to YAML 1.2's float regex makes ``2e-4`` parse as a float.
    """


_Yaml12FloatLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
        [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
       |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
       |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
       |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
       |[-+]?\.(?:inf|Inf|INF)
       |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base` (returns a new dict)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_path(ref: str | Path) -> Path:
    """Resolve a config reference to an absolute path.

    Accepts:
      - absolute paths,
      - paths relative to repo root (e.g. ``configs/models/phi2.yaml``),
      - paths relative to configs/ (e.g. ``models/phi2.yaml``, used inside ``extends:``).
    """
    p = Path(ref)
    if p.is_absolute():
        return p

    if p.exists():
        return p.resolve()

    repo_root = CONFIGS_ROOT.parent
    candidate = (repo_root / p).resolve()
    if candidate.exists():
        return candidate

    return (CONFIGS_ROOT / p).resolve()


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a config YAML, resolving `extends:` and `overrides:`.

    Args:
        path: Path to the YAML (absolute, or relative to configs/).

    Returns:
        Fully-resolved configuration dictionary.
    """
    full_path = _resolve_path(path)
    with open(full_path) as f:
        raw = yaml.load(f, Loader=_Yaml12FloatLoader) or {}

    extends = raw.pop("extends", [])
    overrides = raw.pop("overrides", None)
    ablation_meta = raw.pop("ablation", None)

    # Start from the merged base (left-to-right precedence).
    merged: Dict[str, Any] = {}
    for ref in extends:
        loaded = load_config(ref)
        merged = _deep_merge(merged, loaded)

    # Apply the current file's own keys.
    merged = _deep_merge(merged, raw)

    # Apply ablation overrides (highest precedence — must take effect last).
    if overrides:
        merged = _deep_merge(merged, overrides)

    # Preserve ablation metadata for traceability.
    if ablation_meta is not None:
        merged.setdefault("ablation", ablation_meta)

    return merged
