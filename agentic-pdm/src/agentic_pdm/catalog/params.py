"""Typed, bounded parameters for catalog entries.

An agent never passes free-form values into the harness: every tunable
knob is a ParamSpec with a type and bounds, and validate_params() either
returns a complete, normalized parameter dict (defaults filled in) or a
list of errors an agent can act on. Out-of-range values are rejected, not
clamped — silently changing what the agent asked for would make its
experiment log lie.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ParamSpec:
    type: str  # "int" | "float" | "choice" | "bool"
    default: Any
    description: str
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[tuple] = None
    log_scale: bool = False  # a hint for search: sample this on a log scale

    def describe(self) -> dict:
        out = {"type": self.type, "default": self.default, "description": self.description}
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.choices is not None:
            out["choices"] = list(self.choices)
        if self.log_scale:
            out["log_scale"] = True
        return out

    def check(self, name: str, value: Any) -> tuple[Any, Optional[str]]:
        """(normalized value, error or None)."""
        if self.type == "bool":
            if isinstance(value, bool):
                return value, None
            return value, f"{name} must be true or false, got {value!r}"
        if self.type == "choice":
            if value in self.choices:
                return value, None
            return value, f"{name} must be one of {list(self.choices)}, got {value!r}"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value, f"{name} must be a number, got {value!r}"
        if self.type == "int":
            if float(value) != int(value):
                return value, f"{name} must be a whole number, got {value!r}"
            value = int(value)
        else:
            value = float(value)
        if self.min is not None and value < self.min:
            return value, f"{name} must be >= {self.min}, got {value}"
        if self.max is not None and value > self.max:
            return value, f"{name} must be <= {self.max}, got {value}"
        return value, None


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    kind: str  # "architecture" | "augmentation" | "schedule" | "loss"
    description: str
    params: dict[str, ParamSpec]
    # Cross-parameter rule: returns an error string or None.
    constraint: Optional[Callable[[dict], Optional[str]]] = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Turns validated params into the thing itself (a model, an augmentation
    # function, ...). Its signature depends on `kind`; see each catalog module.
    builder: Optional[Callable] = field(default=None, compare=False, repr=False)

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "description": self.description,
                "params": sorted(self.params)}

    def describe(self) -> dict:
        out = {"id": self.id, "kind": self.kind, "description": self.description,
               "params": {name: spec.describe() for name, spec in self.params.items()}}
        if self.notes:
            out["notes"] = list(self.notes)
        return out


def validate_params(entry: CatalogEntry, given: Optional[dict], where: str) -> tuple[dict, list[str]]:
    """Fills defaults and checks every value. Unknown parameter names are
    errors (a typo must not silently fall back to a default)."""
    given = dict(given or {})
    errors = []
    unknown = sorted(set(given) - set(entry.params))
    if unknown:
        errors.append(f"{where}: unknown parameter(s) {unknown} for {entry.id!r}; "
                      f"valid: {sorted(entry.params)}")
    out = {}
    for name, spec in entry.params.items():
        value, error = spec.check(f"{where}.{name}", given.get(name, spec.default))
        if error:
            errors.append(error)
        out[name] = value
    if not errors and entry.constraint is not None:
        error = entry.constraint(out)
        if error:
            errors.append(f"{where}: {error}")
    return out, errors
