"""Generic name -> class registry used by every extension point.

Why it exists
-------------
Judges, stages, metrics and reporters are all looked up by name. A
``Registry`` lets a new implementation register itself with a decorator, so
adding one never requires editing existing code (open-closed principle).

Usage
-----
    judges = Registry[BaseJudge]("judge")

    @judges.register("groq")
    class GroqJudge(BaseJudge): ...

    cls = judges.get("groq")
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Ordered mapping of ``name -> class`` with decorator registration."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a class decorator registering the class under ``name``."""

        def decorator(cls: T) -> T:
            if name in self._items and self._items[name] is not cls:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"Unknown {self.kind} '{name}'. Registered: {known}") from None

    def names(self) -> list[str]:
        return list(self._items)

    def items(self) -> Iterator[tuple[str, T]]:
        return iter(self._items.items())

    def __contains__(self, name: object) -> bool:
        return name in self._items
