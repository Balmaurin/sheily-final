#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de Uso del Sistema de Manejo de Errores Funcionales
==========================================================

Este módulo demuestra cómo usar el sistema de manejo de errores funcionales
implementado en Sheily AI. Incluye ejemplos prácticos de:

- Uso básico de tipos Result
- Decoradores para manejo automático de errores
- Estrategias de recuperación personalizadas
- Integración con logging y monitoreo
- Composición segura de operaciones
- Manejo de errores en operaciones reales

Ejemplos incluidos:
- Operaciones de memoria con recuperación automática
- Búsquedas RAG con fallback inteligente
- Generación de modelos con validación
- Pipelines seguros con manejo de errores
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from .auto_recovery import get_system_health, register_component_health_checker
from .error_decorators import memory_operation, model_operation, rag_operation
from .error_monitoring import check_system_alerts, record_error_metric

# Importar sistema de errores funcionales
from .functional_errors import (
    ContextualResult,
    Err,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    Ok,
    RecoveryStrategy,
    Result,
    SheilyError,
    async_with_error_handling,
    create_error,
    create_memory_error,
    error_monitor,
    safe_pipe,
    with_error_handling,
)

# Importar componentes específicos
from .memory_errors import MemoryError, MemoryErrorType, SafeHumanMemoryEngine
from .rag_model_errors import ModelError, RAGError, SafeModelEngine, SafeRAGEngine
from .result import Result as BaseResult
from .safe_composition import Railway, SafePipeline, ValidationPipeline, ValidationRule

# ============================================================================
# Ejemplo 1: Uso Básico de Result Types
# ============================================================================


def ejemplo_result_types():
    """Demostrar uso básico de tipos Result"""
    print("=== Ejemplo 1: Uso Básico de Result Types ===")

    def dividir(a: float, b: float) -> Result[float, SheilyError]:
        """División segura que retorna Result"""
        if b == 0:
            return Err(
                create_error(
                    "División por cero",
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.HIGH,
                    component="calculator",
                    operation="divide",
                )
            )
        return Ok(a / b)

    # Uso correcto
    resultado1 = dividir(10, 2)
    if resultado1.is_ok():
        print(f"10 / 2 = {resultado1.unwrap()}")

    # Manejo de error
    resultado2 = dividir(10, 0)
    if resultado2.is_err():
        print(f"Error: {resultado2.error.message}")

    # Uso con unwrap_or para valores por defecto
    resultado_seguro = dividir(10, 0).unwrap_or(0.0)
    print(f"Resultado seguro: {resultado_seguro}")


# ============================================================================
# Ejemplo 2: Decoradores para Manejo Automático
# ============================================================================


def ejemplo_decoradores():
    """Demostrar uso de decoradores para manejo automático"""
    print("\n=== Ejemplo 2: Decoradores para Manejo Automático ===")

    @memory_operation("demo_memory")
    def operacion_memoria_riesgosa(datos: str) -> str:
        """Operación de memoria que puede fallar"""
        if len(datos) < 5:
            raise ValueError("Datos demasiado cortos")

        # Simular operación exitosa
        return f"Procesado: {datos}"

    # Esta operación funciona correctamente
    try:
        resultado = operacion_memoria_riesgosa("datos válidos largos")
        print(f"Operación exitosa: {resultado}")
    except Exception as e:
        print(f"Error manejado automáticamente: {e}")

    # Esta operación fallaría pero el decorador maneja el error
    try:
        resultado = operacion_memoria_riesgosa("corto")
        print(f"Operación exitosa: {resultado}")
    except Exception as e:
        print(f"Error manejado automáticamente: {e}")


# ============================================================================
# Ejemplo 3: Estrategias de Recuperación Personalizadas
# ============================================================================


class DemoRecoveryStrategy(RecoveryStrategy):
    """Estrategia de recuperación de ejemplo"""

    def can_recover(self, error: SheilyError) -> bool:
        return "demo" in error.message.lower()

    def recover(self, error: SheilyError) -> Result[Any, SheilyError]:
        return Ok("Recuperado exitosamente")

    def get_max_attempts(self) -> int:
        return 2


def ejemplo_estrategias_recuperacion():
    """Demostrar estrategias de recuperación personalizadas"""
    print("\n=== Ejemplo 3: Estrategias de Recuperación ===")

    # Crear estrategia personalizada
    estrategia = DemoRecoveryStrategy()

    # Crear error que puede ser recuperado
    error = create_error(
        "Error de demo que puede ser recuperado",
        ErrorCategory.VALIDATION,
        ErrorSeverity.MEDIUM,
        component="demo",
        operation="test",
    )

    # Intentar recuperación
    if estrategia.can_recover(error):
        resultado = estrategia.recover(error)
        if resultado.is_ok():
            print(f"Recuperación exitosa: {resultado.unwrap()}")


# ============================================================================
# Ejemplo 4: Composición Segura con Railway
# ============================================================================


def ejemplo_railway_composition():
    """Demostrar composición segura con Railway"""
    print("\n=== Ejemplo 4: Composición Segura con Railway ===")

    def parsear_numero(texto: str) -> Result[int, SheilyError]:
        """Parsear número de manera segura"""
        try:
            return Ok(int(texto))
        except ValueError:
            return Err(
                create_error(
                    f"No se puede parsear '{texto}' como número",
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.HIGH,
                    component="parser",
                    operation="parse_number",
                )
            )

    def dividir_por_dos(n: int) -> Result[float, SheilyError]:
        """Dividir por dos de manera segura"""
        return Ok(n / 2)

    def formatear_resultado(x: float) -> Result[str, SheilyError]:
        """Formatear resultado"""
        return Ok(f"Resultado: {x:.2f}")

    # Composición usando Railway
    railway = (
        Railway.success("42")
        .bind(parsear_numero)
        .bind(dividir_por_dos)
        .map(lambda x: x * 2)  # Multiplicar por 2
        .bind(formatear_resultado)
    )

    if railway.is_success():
        print(f"Resultado final: {railway.unwrap()}")
    else:
        print(f"Error en pipeline: {railway.unwrap_or('Error desconocido')}")


# ============================================================================
# Ejemplo 5: Validación Composable
# ============================================================================


def ejemplo_validacion_composable():
    """Demostrar validación composable"""
    print("\n=== Ejemplo 5: Validación Composable ===")

    # Crear reglas de validación
    regla_longitud = ValidationRule(
        name="longitud_minima",
        validator=lambda x: len(x) >= 5,
        error_message="El texto debe tener al menos 5 caracteres",
    )

    regla_contenido = ValidationRule(
        name="contiene_letra",
        validator=lambda x: any(c.isalpha() for c in x),
        error_message="El texto debe contener al menos una letra",
    )

    # Crear pipeline de validación
    pipeline = ValidationPipeline("texto válido largo").add_rule(regla_longitud).add_rule(regla_contenido)

    resultado = pipeline.validate()

    if resultado.is_ok():
        print(f"Texto válido: {resultado.unwrap()}")
    else:
        print("Errores de validación:")
        for error in resultado.error:
            print(f"  - {error.message}")


# ============================================================================
# Ejemplo 6: Integración con Sistemas Reales
# ============================================================================


def ejemplo_integracion_sistemas():
    """Demostrar integración con sistemas reales"""
    print("\n=== Ejemplo 6: Integración con Sistemas ===")

    # Ejemplo de operación de memoria segura
    try:
        # Crear motor de memoria seguro (simulado)
        print("Inicializando motor de memoria segura...")

        # Simular operación de memoria
        memoria_segura = SafeHumanMemoryEngine("usuario_demo")

        # Registrar verificador de salud
        def verificar_memoria():
            from .auto_recovery import SystemHealth

            return SystemHealth.HEALTHY

        register_component_health_checker("memory", verificar_memoria)

        print(f"Estado del sistema: {get_system_health().value}")

    except Exception as e:
        print(f"Error en integración: {e}")


# ============================================================================
# Ejemplo 7: Monitoreo y Métricas
# ============================================================================


def ejemplo_monitoreo_metricas():
    """Demostrar monitoreo y métricas"""
    print("\n=== Ejemplo 7: Monitoreo y Métricas ===")

    # Crear algunos errores para demostrar métricas
    errores_ejemplo = [
        create_error(
            "Error de red",
            ErrorCategory.NETWORK,
            ErrorSeverity.HIGH,
            component="demo",
            operation="test",
        ),
        create_error(
            "Error de validación",
            ErrorCategory.VALIDATION,
            ErrorSeverity.MEDIUM,
            component="demo",
            operation="validate",
        ),
        create_error(
            "Error crítico",
            ErrorCategory.MODEL,
            ErrorSeverity.CRITICAL,
            component="demo",
            operation="generate",
        ),
    ]

    # Registrar métricas
    for error in errores_ejemplo:
        record_error_metric(error)

    # Verificar alertas
    check_system_alerts()

    print(f"Métricas registradas para {len(errores_ejemplo)} errores")


# ============================================================================
# Ejemplo 8: Operaciones Asíncronas Seguras
# ============================================================================


async def ejemplo_operaciones_async():
    """Demostrar operaciones asíncronas seguras"""
    print("\n=== Ejemplo 8: Operaciones Asíncronas ===")

    @async_with_error_handling("async_demo", log_errors=True)
    async def operacion_async_riesgosa(delay: float) -> str:
        """Operación asíncrona que puede fallar"""
        await asyncio.sleep(delay)

        if delay > 0.1:
            raise ConnectionError("Timeout en operación asíncrona")

        return "Operación completada exitosamente"

    try:
        # Operación exitosa
        resultado1 = await operacion_async_riesgosa(0.05)
        print(f"Resultado async: {resultado1}")

    except Exception as e:
        print(f"Error async manejado: {e}")


# ============================================================================
# Función Principal para Ejecutar Todos los Ejemplos
# ============================================================================


def ejecutar_ejemplos_completos():
    """Ejecutar todos los ejemplos de manejo de errores funcionales"""
    print("🚀 Iniciando ejemplos del Sistema de Manejo de Errores Funcionales")
    print("=" * 70)

    try:
        # Ejecutar ejemplos básicos
        ejemplo_result_types()
        ejemplo_decoradores()
        ejemplo_estrategias_recuperacion()
        ejemplo_railway_composition()
        ejemplo_validacion_composable()
        ejemplo_integracion_sistemas()
        ejemplo_monitoreo_metricas()

        # Ejecutar ejemplos asíncronos
        asyncio.run(ejemplo_operaciones_async())

        print("\n" + "=" * 70)
        print("✅ Todos los ejemplos ejecutados exitosamente")
        print("\n📊 Resumen del Sistema:")
        print(f"   - Estado del sistema: {get_system_health().value}")
        print(f"   - Errores monitoreados: {len(error_monitor.error_history)}")
        print(f"   - Métricas activas: {len(record_error_metric.__defaults__)}")

    except Exception as e:
        print(f"\n❌ Error ejecutando ejemplos: {e}")
        import traceback

        traceback.print_exc()


# ============================================================================
# Exports del módulo
# ============================================================================

__all__ = [
    "ejecutar_ejemplos_completos",
    "ejemplo_result_types",
    "ejemplo_decoradores",
    "ejemplo_estrategias_recuperacion",
    "ejemplo_railway_composition",
    "ejemplo_validacion_composable",
    "ejemplo_integracion_sistemas",
    "ejemplo_monitoreo_metricas",
    "ejemplo_operaciones_async",
]

if __name__ == "__main__":
    ejecutar_ejemplos_completos()

print("✅ Módulo de ejemplos del sistema de manejo de errores funcionales cargado")
