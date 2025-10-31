#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resultado funcional mínimo para el sistema de errores
Provee Result, Ok, Err y utilidades usadas por functional_errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, List, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass
class Result(Generic[T, E]):
    ok: bool
    value: T | None = None
    error: E | None = None

    def is_ok(self) -> bool:
        return self.ok

    def is_err(self) -> bool:
        return not self.ok

    def unwrap(self) -> T:
        if not self.ok:
            raise RuntimeError(f"Tried to unwrap Err: {self.error}")
        return self.value  # type: ignore

    def unwrap_err(self) -> E:
        if self.ok:
            raise RuntimeError("Tried to unwrap_err on Ok result")
        return self.error  # type: ignore


def Ok(value: T) -> Result[T, E]:  # type: ignore[override]
    return Result(True, value=value, error=None)


def Err(error: E) -> Result[T, E]:  # type: ignore[override]
    return Result(False, value=None, error=error)


def create_ok(value: T) -> Result[T, E]:  # alias
    return Ok(value)


def create_err(error: E) -> Result[T, E]:  # alias
    return Err(error)


def is_ok(r: Result[Any, Any]) -> bool:
    return r.is_ok()


def is_err(r: Result[Any, Any]) -> bool:
    return r.is_err()


def catch(fn: Callable[[], T]) -> Result[T, Exception]:
    try:
        return Ok(fn())
    except Exception as e:
        return Err(e)


def traverse_results(results: List[Result[T, E]]) -> Result[List[T], E]:
    values: List[T] = []
    for r in results:
        if r.is_err():
            return Err(r.error)  # type: ignore
        values.append(r.value)  # type: ignore
    return Ok(values)
