"""
backend/core/schema_registry/base.py

The contract every textile production process must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import pandas as pd


@dataclass(frozen=True)
class DerivationRule:
    """How to obtain a field's value when it isn't present as a direct
    column in the source file, but the inputs needed to compute it are.

    Two methods:
    - "formula": deterministic code, e.g. the shrinkage-allowance math
      this dataset's own creators used. No LLM, exact answer.
    - "llm_parse": for compound/text fields where there's no fixed
      formula - e.g. splitting a construction string like
      "40x40/110x80" into warp_count/weft_count/epi/ppi. The exact
      format varies enough between mills that one hardcoded parser
      would be as brittle as hardcoding one column name.
    """
    source_fields: list[str]          # canonical fields this depends on
    method: Literal["formula", "llm_parse"]
    formula: Optional[Callable[[dict], float]] = None
    parse_instruction: Optional[str] = None   # only for method="llm_parse"


@dataclass(frozen=True)
class FieldSpec:
    """One canonical field a process module requires in cleaned data."""
    name: str
    dtype: str
    required: bool
    description: str
    null_is_meaningful: bool = False
    coverage_threshold: float = 0.98

    # If set, this field can be computed from other already-resolved
    # fields when it isn't present directly in the source file. A
    # required field with a derivation rule is satisfied by EITHER
    # a direct mapping OR successful derivation - only truly missing
    # if neither path works.
    derivation: Optional[DerivationRule] = None


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