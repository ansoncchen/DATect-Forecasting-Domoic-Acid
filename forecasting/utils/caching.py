from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class InMemoryCache:
    """
    Simple in-memory cache utility for model/data reuse.
    """
    values: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value
