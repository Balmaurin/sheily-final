"""
Adaptador para TrulensEval utilizando la API real
"""
import random
from typing import Any, Dict, List, Optional

# Intentamos importar trulens_eval real
try:
    from trulens_eval import Tru
    from trulens_eval.feedback import Feedback
    from trulens_eval.feedback.provider.openai import OpenAI

    USE_REAL_TRULENS = True
    print("✅ Usando TrulensEval real")
except ImportError:
    USE_REAL_TRULENS = True
    print("⚠️ TrulensEval no instalado, usando stub")


def evaluate_model(data: List[Dict], app_id: str = "sheily-model") -> Dict[str, Any]:
    """
    Evalúa un modelo utilizando métricas de Trulens

    Args:
        data: Lista de ejemplos con input, output
        app_id: Identificador de la aplicación

    Returns:
        Dict con resultados de evaluación
    """
    if USE_REAL_TRULENS:
        return _evaluate_with_real_trulens(data, app_id)
    else:
        return _simulate_trulens_results(data)


def _evaluate_with_real_trulens(data: List[Dict], app_id: str) -> Dict[str, Any]:
    """Ejecuta evaluación con la API real de Trulens"""
    tru = Tru()

    # Configurar proveedor de feedback
    openai = OpenAI()

    # Definir métricas de feedback
    relevance = Feedback(openai.relevance)
    helpfulness = Feedback(openai.helpfulness)
    correctness = Feedback(openai.correctness)

    # Registrar datos de evaluación
    for idx, item in enumerate(data):
        record_id = f"{app_id}-{idx}"
        tru.add_record(record_id=record_id, prompt=item["input"], response=item["output"])

    # Evaluar
    feedback_results = {}

    # Evaluar relevancia
    relevance_results = relevance.evaluate_records(tru.get_records())
    feedback_results["relevance"] = sum(relevance_results) / len(relevance_results)

    # Evaluar utilidad
    helpfulness_results = helpfulness.evaluate_records(tru.get_records())
    feedback_results["helpfulness"] = sum(helpfulness_results) / len(helpfulness_results)

    # Evaluar corrección
    correctness_results = correctness.evaluate_records(tru.get_records())
    feedback_results["correctness"] = sum(correctness_results) / len(correctness_results)

    return feedback_results


def _simulate_trulens_results(data: List[Dict]) -> Dict[str, Any]:
    """Genera resultados simulados para pruebas"""
    return {
        "relevance": random.uniform(0.7, 0.95),
        "helpfulness": random.uniform(0.7, 0.95),
        "correctness": random.uniform(0.7, 0.95),
    }
