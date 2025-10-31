#!/usr/bin/env python3
# Re-export explícito desde utils.functional_errors para compatibilidad
from ..utils import functional_errors as _fe

SheilyError = _fe.SheilyError
ErrorCategory = _fe.ErrorCategory
ErrorSeverity = _fe.ErrorSeverity
ErrorContext = _fe.ErrorContext
RecoveryStrategy = _fe.RecoveryStrategy
RetryStrategy = _fe.RetryStrategy
FallbackStrategy = _fe.FallbackStrategy
CircuitBreakerStrategy = _fe.CircuitBreakerStrategy
Result = _fe.Result
Ok = _fe.Ok
Err = _fe.Err
ContextualResult = _fe.ContextualResult
with_error_handling = _fe.with_error_handling
async_with_error_handling = _fe.async_with_error_handling
error_monitor = _fe.error_monitor
create_error = _fe.create_error
create_memory_error = _fe.create_memory_error
safe_pipe = _fe.safe_pipe
async_safe_pipe = _fe.async_safe_pipe

__all__ = [
    "SheilyError",
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorContext",
    "RecoveryStrategy",
    "Result",
    "Ok",
    "Err",
    "ContextualResult",
    "with_error_handling",
    "async_with_error_handling",
    "error_monitor",
    "create_error",
    "create_memory_error",
    "safe_pipe",
    "async_safe_pipe",
    "RetryStrategy",
    "FallbackStrategy",
    "CircuitBreakerStrategy",
]
