"""
backend/core/schema_registry/base.py

The contract every textile production process must implement.

No agent in the pipeline ever imports a specific process module (like
weaving) directly - every agent only ever calls get_process_module()
and interacts with whatever comes back through this interface. That is
the entire mechanism that makes adding spinning or dyeing later a
matter of writing one new file, not editing every agent.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FieldSpec:
    """One canonical field a process module requires in cleaned data."""
    name: str          # canonical field name, e.g. "loom_id"
    dtype: str          # "str" | "int" | "float" | "date" - used for validation
    required: bool
    description: str    # plain-English meaning - given to the LLM during
                         # column mapping, so it knows what it's matching to


class ProcessModule(ABC):
    """Abstract interface. Every concrete module (WeavingModule, later
    SpinningModule) subclasses this and implements every member below."""

    @property
    @abstractmethod
    def process_name(self) -> str:
        """Short identifier used everywhere: "weaving", "spinning", etc."""
        ...

    @property
    @abstractmethod
    def required_fields(self) -> list[FieldSpec]:
        """Canonical fields this process's cleaned data must contain."""
        ...

    @property
    @abstractmethod
    def stoppage_categories(self) -> list[str]:
        """Valid downtime-cause categories the Validation Agent classifies
        the operator's free-text stoppage reason into."""
        ...

    @property
    @abstractmethod
    def domain_context(self) -> str:
        """Plain-English description of this process's machinery and
        terminology, injected into every LLM prompt (ingestion mapping,
        analysis, verification) so the model knows what it's reasoning
        about, without every prompt hardcoding "loom" or "spindle" itself."""
        ...

    @abstractmethod
    def compute_kpis(self, df: pd.DataFrame) -> dict:
        """Pure-code KPI computation, owned entirely by the module. This
        is the one part of the whole interface explicitly forbidden from
        calling an LLM - every metric here has exactly one correct answer."""
        ...


# --- registry: maps a process_type string to its concrete module ---

_REGISTRY: dict[str, ProcessModule] = {}


def register_module(module: ProcessModule) -> ProcessModule:
    """Called once by each concrete module file, at the bottom of that
    file, e.g.: weaving_module = register_module(WeavingModule())"""
    _REGISTRY[module.process_name] = module
    return module


def get_process_module(process_type: str) -> ProcessModule:
    """Every agent calls this - never imports a concrete module directly."""
    if process_type not in _REGISTRY:
        raise ValueError(
            f"Unknown process type: {process_type!r}. "
            f"Registered: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[process_type]