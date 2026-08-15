"""
backend/core/schema_registry/base.py

The contract every textile production process must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FieldSpec:
    """One canonical field a process module requires in cleaned data."""
    name: str
    dtype: str
    required: bool
    description: str

    # If True, a missing value in this field is a legitimate real-world
    # state (e.g. a loom idle after its order was already fulfilled),
    # not a data-quality problem - the Validation Agent must not flag
    # it as an error just because it's blank.
    null_is_meaningful: bool = False

    # Fraction of a column's distinct values that a proposed cleaning
    # rule must correctly parse before the Verifier auto-approves it
    # without a flag. Below this, the rule still applies (pipeline
    # doesn't stop) but the field is marked low-confidence in the
    # final report.
    coverage_threshold: float = 0.98


class ProcessModule(ABC):
    """Abstract interface. Every concrete module (WeavingModule, later
    SpinningModule) subclasses this and implements every member below."""

    @property
    @abstractmethod
    def process_name(self) -> str:
        ...

    @property
    @abstractmethod
    def required_fields(self) -> list[FieldSpec]:
        ...

    @property
    @abstractmethod
    def stoppage_categories(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def domain_context(self) -> str:
        ...

    @abstractmethod
    def compute_kpis(self, df: pd.DataFrame) -> dict:
        ...


_REGISTRY: dict[str, ProcessModule] = {}


def register_module(module: ProcessModule) -> ProcessModule:
    _REGISTRY[module.process_name] = module
    return module


def get_process_module(process_type: str) -> ProcessModule:
    if process_type not in _REGISTRY:
        raise ValueError(
            f"Unknown process type: {process_type!r}. "
            f"Registered: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[process_type]