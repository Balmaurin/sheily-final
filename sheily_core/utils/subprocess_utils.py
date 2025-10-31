#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades Seguras para Subprocess
===================================
Funciones helper para ejecutar comandos de forma segura.
"""

import logging
import re
import subprocess
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def validate_command_args(args: List[str]) -> bool:
    """
    Validar argumentos de comando para prevenir inyección.

    Args:
        args: Lista de argumentos del comando

    Returns:
        True si los argumentos son seguros

    Raises:
        ValueError: Si se detectan caracteres peligrosos
    """
    # Caracteres peligrosos que pueden permitir inyección
    dangerous_chars = [";", "|", "&", "$", "`", "\n", "\r", "||", "&&"]

    for arg in args:
        arg_str = str(arg)
        for char in dangerous_chars:
            if char in arg_str:
                raise ValueError(f"Carácter potencialmente peligroso '{char}' detectado en: {arg_str[:50]}")

    return True


def safe_subprocess_run(
    cmd: List[str],
    timeout: int = 30,
    check: bool = False,
    capture_output: bool = True,
    text: bool = True,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """
    Ejecutar subprocess de forma segura con validación.

    Args:
        cmd: Lista de comandos (NO string)
        timeout: Timeout en segundos
        check: Si True, lanza excepción si returncode != 0
        capture_output: Si True, captura stdout y stderr
        text: Si True, decodifica output como texto
        env: Variables de entorno adicionales
        cwd: Directorio de trabajo
        **kwargs: Argumentos adicionales para subprocess.run

    Returns:
        subprocess.CompletedProcess

    Raises:
        ValueError: Si la validación falla
        subprocess.TimeoutExpired: Si se excede el timeout
        subprocess.CalledProcessError: Si check=True y returncode != 0
    """
    # Validar que es una lista
    if not isinstance(cmd, list):
        raise ValueError(f"Command debe ser una lista, no {type(cmd)}")

    # Validar que no está vacía
    if not cmd:
        raise ValueError("Command no puede estar vacío")

    # Validar argumentos
    validate_command_args(cmd)

    # NUNCA usar shell=True
    if kwargs.get("shell", False):
        logger.warning("Intentando usar shell=True - forzando a False por seguridad")
        kwargs["shell"] = False

    # Log del comando (para debugging)
    logger.debug(f"Ejecutando comando: {' '.join(str(x) for x in cmd)}")

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            check=check,
            capture_output=capture_output,
            text=text,
            env=env,
            cwd=cwd,
            shell=False,  # CRÍTICO: siempre False
            **kwargs,
        )

        logger.debug(f"Comando completado con código: {result.returncode}")
        return result

    except subprocess.TimeoutExpired as e:
        logger.error(f"Timeout ejecutando comando después de {timeout}s: {cmd[0]}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Error ejecutando comando (código {e.returncode}): {cmd[0]}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado ejecutando comando: {e}")
        raise


def safe_subprocess_popen(
    cmd: List[str],
    stdout: Optional[Any] = None,
    stderr: Optional[Any] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    **kwargs,
) -> subprocess.Popen:
    """
    Crear proceso subprocess de forma segura con validación.

    Args:
        cmd: Lista de comandos (NO string)
        stdout: Configuración de stdout
        stderr: Configuración de stderr
        env: Variables de entorno
        cwd: Directorio de trabajo
        **kwargs: Argumentos adicionales para subprocess.Popen

    Returns:
        subprocess.Popen

    Raises:
        ValueError: Si la validación falla
    """
    # Validar que es una lista
    if not isinstance(cmd, list):
        raise ValueError(f"Command debe ser una lista, no {type(cmd)}")

    # Validar argumentos
    validate_command_args(cmd)

    # NUNCA usar shell=True
    if kwargs.get("shell", False):
        logger.warning("Intentando usar shell=True - forzando a False por seguridad")
        kwargs["shell"] = False

    logger.debug(f"Iniciando proceso: {' '.join(str(x) for x in cmd)}")

    return subprocess.Popen(
        cmd, stdout=stdout, stderr=stderr, env=env, cwd=cwd, shell=False, **kwargs  # CRÍTICO: siempre False
    )


# Exports
__all__ = [
    "validate_command_args",
    "safe_subprocess_run",
    "safe_subprocess_popen",
]
