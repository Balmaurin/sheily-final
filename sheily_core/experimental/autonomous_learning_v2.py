#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTONOMOUS LEARNING V2 - APRENDIZAJE AUTÓNOMO AVANZADO
====================================================

Sistema de aprendizaje autónomo que proporciona:

CAPACIDADES DE AUTO-EVOLUCIÓN:
- Meta-aprendizaje con adaptación automática
- Optimización continua de hiperparámetros
- Detección automática de patrones de conocimiento
- Evolución basada en retroalimentación
- Aprendizaje incremental sin catástrofe de olvido
- Adaptación a nuevos dominios automáticamente
- Optimización de estrategias de aprendizaje
- Auto-corrección basada en métricas de rendimiento

ARQUITECTURA DE APRENDIZAJE AUTÓNOMO:
- Núcleo de meta-aprendizaje con múltiples algoritmos
- Sistema de retroalimentación continua
- Optimizador evolutivo de estrategias
- Detector automático de conocimiento nuevo
- Mecanismos de consolidación de aprendizaje
- Sistema de adaptación basado en contexto
"""

import json
import math
import os
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Configuración avanzada
LEARNING_ROOT = Path(__file__).resolve().parents[2] / "data" / "autonomous_learning_v2"
OPTIMIZATION_INTERVAL = 3600  # 1 hora
ADAPTATION_RATE = 0.01
MAX_EVOLUTION_HISTORY = 1000


@dataclass
class AutonomousLearningConfig:
    """Configuración avanzada de aprendizaje autónomo"""

    enable_meta_learning: bool = True
    enable_auto_optimization: bool = True
    enable_pattern_detection: bool = True
    enable_incremental_learning: bool = True
    enable_contextual_adaptation: bool = True

    # Configuración de evolución
    evolution_rate: float = ADAPTATION_RATE
    population_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7

    # Configuración de detección de conocimiento
    novelty_threshold: float = 0.3
    knowledge_consolidation_interval: int = 7200  # 2 horas
    pattern_memory_size: int = 1000

    # Configuración de adaptación
    adaptation_sensitivity: float = 0.5
    context_window_size: int = 100
    feedback_integration_rate: float = 0.2


@dataclass
class LearningPattern:
    """Patrón de aprendizaje detectado"""

    pattern_id: str
    pattern_type: str  # knowledge_growth, performance_degradation, context_shift
    pattern_data: Dict[str, Any]
    confidence_score: float
    detection_time: datetime = field(default_factory=datetime.now)
    occurrences: int = 1
    impact_score: float = 0.0

    def update_occurrence(self):
        """Actualizar ocurrencia del patrón"""
        self.occurrences += 1
        self.impact_score = min(1.0, self.impact_score + 0.1)


@dataclass
class LearningStrategy:
    """Estrategia de aprendizaje evolucionaria"""

    strategy_id: str
    strategy_name: str
    hyperparameters: Dict[str, Any]
    performance_history: List[float] = field(default_factory=list)
    fitness_score: float = 0.0
    generation: int = 0
    parent_strategies: List[str] = field(default_factory=list)

    def calculate_fitness(self) -> float:
        """Calcular fitness basado en historial de rendimiento"""
        if not self.performance_history:
            return 0.0

        # Fitness basado en promedio de rendimiento reciente
        recent_performance = self.performance_history[-10:]  # Últimas 10 mediciones
        avg_performance = sum(recent_performance) / len(recent_performance)

        # Bonus por consistencia
        if len(recent_performance) > 1:
            consistency_bonus = 1.0 - np.std(recent_performance)
            avg_performance *= 0.8 + 0.2 * consistency_bonus

        return min(1.0, avg_performance)

    def mutate(self, mutation_rate: float) -> "LearningStrategy":
        """Crear estrategia mutada"""
        mutated = LearningStrategy(
            strategy_id=f"{self.strategy_id}_mutated_{int(time.time())}",
            strategy_name=f"{self.strategy_name}_mutated",
            hyperparameters=self.hyperparameters.copy(),
            generation=self.generation + 1,
            parent_strategies=self.parent_strategies + [self.strategy_id],
        )

        # Aplicar mutaciones a hiperparámetros
        for param_name, param_value in mutated.hyperparameters.items():
            if random.random() < mutation_rate:
                if isinstance(param_value, (int, float)):
                    # Mutación numérica
                    mutation_factor = random.uniform(0.8, 1.2)
                    mutated.hyperparameters[param_name] = param_value * mutation_factor
                elif isinstance(param_value, list):
                    # Mutación de listas
                    if param_value and random.random() < 0.5:
                        mutated.hyperparameters[param_name] = param_value.copy()
                        if len(mutated.hyperparameters[param_name]) > 1:
                            # Cambiar orden o modificar elementos
                            random.shuffle(mutated.hyperparameters[param_name])

        return mutated

    def crossover(self, other: "LearningStrategy") -> "LearningStrategy":
        """Crear estrategia mediante crossover"""
        offspring = LearningStrategy(
            strategy_id=f"{self.strategy_id}_x_{other.strategy_id}_{int(time.time())}",
            strategy_name=f"{self.strategy_name}_x_{other.strategy_name}",
            hyperparameters={},
            generation=max(self.generation, other.generation) + 1,
            parent_strategies=self.parent_strategies + other.parent_strategies + [self.strategy_id, other.strategy_id],
        )

        # Crossover de hiperparámetros
        all_params = set(self.hyperparameters.keys()) | set(other.hyperparameters.keys())

        for param_name in all_params:
            value1 = self.hyperparameters.get(param_name)
            value2 = other.hyperparameters.get(param_name)

            if value1 is not None and value2 is not None:
                # Crossover simple: elegir de uno u otro padre
                if random.random() < 0.5:
                    offspring.hyperparameters[param_name] = value1
                else:
                    offspring.hyperparameters[param_name] = value2
            elif value1 is not None:
                offspring.hyperparameters[param_name] = value1
            elif value2 is not None:
                offspring.hyperparameters[param_name] = value2

        return offspring


@dataclass
class AutonomousLearningState:
    """Estado completo del aprendizaje autónomo"""

    total_learning_sessions: int = 0
    detected_patterns: Dict[str, LearningPattern] = field(default_factory=dict)
    learning_strategies: Dict[str, LearningStrategy] = field(default_factory=dict)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_metrics: Dict[str, float] = field(default_factory=dict)

    # Métricas de rendimiento
    knowledge_growth_rate: float = 0.0
    adaptation_effectiveness: float = 0.0
    optimization_success_rate: float = 0.0

    last_optimization: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetaLearningEngine:
    """Motor de meta-aprendizaje avanzado"""

    def __init__(self, config: Optional[AutonomousLearningConfig] = None):
        self.config = config or AutonomousLearningConfig()
        self.state_file = LEARNING_ROOT / "autonomous_learning_state.json"

        # Inicializar componentes
        self._init_directories()
        self.state = self._load_state()

        # Población de estrategias de aprendizaje
        self.strategy_population = self._initialize_strategy_population()

        # Detector de patrones
        self.pattern_detector = self._init_pattern_detector()

        # Optimizador evolutivo
        self.evolution_optimizer = self._init_evolution_optimizer()

        self.logger = self._get_logger()

    def _get_logger(self):
        """Obtener logger con fallback"""
        try:
            from sheily_core.logger import get_logger

            return get_logger("autonomous_learning")
        except ImportError:
            import logging

            return logging.getLogger("autonomous_learning")

    def _init_directories(self):
        """Inicializar estructura de directorios"""
        LEARNING_ROOT.mkdir(parents=True, exist_ok=True)
        (LEARNING_ROOT / "patterns").mkdir(exist_ok=True)
        (LEARNING_ROOT / "strategies").mkdir(exist_ok=True)
        (LEARNING_ROOT / "evolution").mkdir(exist_ok=True)

    def _load_state(self) -> AutonomousLearningState:
        """Cargar estado de aprendizaje autónomo"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return AutonomousLearningState(**data)
            except Exception as e:
                self.logger.warning(f"Error loading autonomous learning state: {e}")

        return AutonomousLearningState()

    def _save_state(self):
        """Guardar estado de aprendizaje autónomo"""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.state), f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving autonomous learning state: {e}")

    def _initialize_strategy_population(self) -> List[LearningStrategy]:
        """Inicializar población inicial de estrategias"""
        base_strategies = [
            {
                "name": "conservative_learning",
                "hyperparameters": {
                    "learning_rate": 1e-5,
                    "batch_size": 8,
                    "memory_weight": 0.2,
                    "exploration_rate": 0.1,
                },
            },
            {
                "name": "aggressive_learning",
                "hyperparameters": {
                    "learning_rate": 1e-3,
                    "batch_size": 32,
                    "memory_weight": 0.4,
                    "exploration_rate": 0.3,
                },
            },
            {
                "name": "balanced_learning",
                "hyperparameters": {
                    "learning_rate": 5e-4,
                    "batch_size": 16,
                    "memory_weight": 0.3,
                    "exploration_rate": 0.2,
                },
            },
        ]

        population = []
        for i, base_strategy in enumerate(base_strategies):
            strategy = LearningStrategy(
                strategy_id=f"strategy_{i}_{int(time.time())}",
                strategy_name=base_strategy["name"],
                hyperparameters=base_strategy["hyperparameters"],
                generation=0,
            )
            strategy.fitness_score = strategy.calculate_fitness()
            population.append(strategy)

        return population

    def _init_pattern_detector(self):
        """Inicializar detector de patrones de aprendizaje"""

        class PatternDetector:
            def __init__(self, config: AutonomousLearningConfig):
                self.config = config
                self.pattern_memory = deque(maxlen=config.pattern_memory_size)
                self.detected_patterns = {}

            def detect_patterns(self, learning_data: Dict[str, Any]) -> List[LearningPattern]:
                """Detectar patrones en datos de aprendizaje"""
                patterns = []

                # Detectar crecimiento de conocimiento
                knowledge_growth = self._detect_knowledge_growth(learning_data)
                if knowledge_growth:
                    patterns.append(knowledge_growth)

                # Detectar degradación de rendimiento
                performance_degradation = self._detect_performance_degradation(learning_data)
                if performance_degradation:
                    patterns.append(performance_degradation)

                # Detectar cambios de contexto
                context_shift = self._detect_context_shift(learning_data)
                if context_shift:
                    patterns.append(context_shift)

                # Almacenar datos para análisis futuro
                self.pattern_memory.append(learning_data)

                return patterns

            def _detect_knowledge_growth(self, learning_data: Dict[str, Any]) -> Optional[LearningPattern]:
                """Detectar crecimiento acelerado de conocimiento"""
                knowledge_metrics = learning_data.get("knowledge_metrics", {})

                if "knowledge_base_size" in knowledge_metrics:
                    current_size = knowledge_metrics["knowledge_base_size"]

                    # Calcular tasa de crecimiento
                    if len(self.pattern_memory) > 1:
                        prev_data = list(self.pattern_memory)[-2]
                        prev_size = prev_data.get("knowledge_metrics", {}).get("knowledge_base_size", 0)

                        if prev_size > 0:
                            growth_rate = (current_size - prev_size) / prev_size

                            if growth_rate > 0.2:  # Crecimiento > 20%
                                return LearningPattern(
                                    pattern_id=f"knowledge_growth_{int(time.time())}",
                                    pattern_type="knowledge_growth",
                                    pattern_data={
                                        "growth_rate": growth_rate,
                                        "current_size": current_size,
                                        "previous_size": prev_size,
                                    },
                                    confidence_score=min(1.0, growth_rate * 2),
                                )

                return None

            def _detect_performance_degradation(self, learning_data: Dict[str, Any]) -> Optional[LearningPattern]:
                """Detectar degradación de rendimiento"""
                performance_metrics = learning_data.get("performance_metrics", {})

                if "accuracy" in performance_metrics:
                    current_accuracy = performance_metrics["accuracy"]

                    # Analizar tendencia de rendimiento
                    if len(self.pattern_memory) > 3:
                        recent_accuracies = []
                        for prev_data in list(self.pattern_memory)[-3:]:
                            prev_acc = prev_data.get("performance_metrics", {}).get("accuracy", 1.0)
                            recent_accuracies.append(prev_acc)

                        avg_recent_accuracy = sum(recent_accuracies) / len(recent_accuracies)

                        if avg_recent_accuracy > 0:
                            degradation_rate = (avg_recent_accuracy - current_accuracy) / avg_recent_accuracy

                            if degradation_rate > 0.1:  # Degradación > 10%
                                return LearningPattern(
                                    pattern_id=f"performance_degradation_{int(time.time())}",
                                    pattern_type="performance_degradation",
                                    pattern_data={
                                        "degradation_rate": degradation_rate,
                                        "current_accuracy": current_accuracy,
                                        "average_recent_accuracy": avg_recent_accuracy,
                                    },
                                    confidence_score=min(1.0, degradation_rate * 3),
                                )

                return None

            def _detect_context_shift(self, learning_data: Dict[str, Any]) -> Optional[LearningPattern]:
                """Detectar cambios significativos de contexto"""
                context_metrics = learning_data.get("context_metrics", {})

                if "domain_distribution" in context_metrics:
                    current_distribution = context_metrics["domain_distribution"]

                    # Comparar con distribuciones anteriores
                    if len(self.pattern_memory) > 1:
                        prev_data = list(self.pattern_memory)[-2]
                        prev_distribution = prev_data.get("context_metrics", {}).get("domain_distribution", {})

                        # Calcular distancia entre distribuciones
                        distribution_shift = self._calculate_distribution_shift(current_distribution, prev_distribution)

                        if distribution_shift > 0.3:  # Cambio significativo
                            return LearningPattern(
                                pattern_id=f"context_shift_{int(time.time())}",
                                pattern_type="context_shift",
                                pattern_data={
                                    "shift_magnitude": distribution_shift,
                                    "current_distribution": current_distribution,
                                    "previous_distribution": prev_distribution,
                                },
                                confidence_score=min(1.0, distribution_shift * 2),
                            )

                return None

            def _calculate_distribution_shift(self, dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
                """Calcular magnitud de cambio entre distribuciones"""
                all_keys = set(dist1.keys()) | set(dist2.keys())

                total_shift = 0.0
                for key in all_keys:
                    val1 = dist1.get(key, 0.0)
                    val2 = dist2.get(key, 0.0)
                    total_shift += abs(val1 - val2)

                return total_shift

        return PatternDetector(self.config)

    def _init_evolution_optimizer(self):
        """Inicializar optimizador evolutivo"""

        class EvolutionOptimizer:
            def __init__(self, config: AutonomousLearningConfig):
                self.config = config
                self.generation = 0
                self.evolution_history = []

            def evolve_strategies(
                self, current_strategies: List[LearningStrategy], performance_data: Dict[str, float]
            ) -> List[LearningStrategy]:
                """Evolucionar estrategias basado en rendimiento"""
                # Calcular fitness de estrategias actuales
                for strategy in current_strategies:
                    strategy.fitness_score = strategy.calculate_fitness()

                # Seleccionar mejores estrategias
                sorted_strategies = sorted(current_strategies, key=lambda x: x.fitness_score, reverse=True)
                elite_strategies = sorted_strategies[: self.config.population_size // 2]

                # Crear nueva generación
                new_generation = elite_strategies.copy()

                while len(new_generation) < self.config.population_size:
                    # Seleccionar padres
                    parent1 = self._select_parent(elite_strategies)
                    parent2 = self._select_parent(elite_strategies)

                    # Crear descendencia mediante crossover
                    if random.random() < self.config.crossover_rate:
                        offspring = parent1.crossover(parent2)
                    else:
                        offspring = parent1.mutate(self.config.mutation_rate)

                    new_generation.append(offspring)

                # Actualizar generación
                for strategy in new_generation:
                    strategy.generation = self.generation + 1

                self.generation += 1

                # Registrar evolución
                self.evolution_history.append(
                    {
                        "generation": self.generation,
                        "best_fitness": max(s.fitness_score for s in new_generation),
                        "avg_fitness": sum(s.fitness_score for s in new_generation) / len(new_generation),
                        "population_size": len(new_generation),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                return new_generation

            def _select_parent(self, strategies: List[LearningStrategy]) -> LearningStrategy:
                """Seleccionar padre usando selección por torneo"""
                tournament_size = min(3, len(strategies))

                # Seleccionar candidatos para torneo
                candidates = random.sample(strategies, tournament_size)

                # Retornar el mejor candidato
                return max(candidates, key=lambda x: x.fitness_score)

        return EvolutionOptimizer(self.config)

    def process_learning_session(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar sesión de aprendizaje y generar optimizaciones"""
        session_start = datetime.now()

        try:
            # Detectar patrones en datos de aprendizaje
            detected_patterns = []
            if self.config.enable_pattern_detection:
                detected_patterns = self.pattern_detector.detect_patterns(learning_data)

                # Registrar patrones detectados
                for pattern in detected_patterns:
                    self.state.detected_patterns[pattern.pattern_id] = pattern

            # Evaluar estrategias actuales
            strategy_performance = self._evaluate_strategy_performance(learning_data)

            # Evolucionar estrategias si es necesario
            if self.config.enable_meta_learning:
                self.strategy_population = self.evolution_optimizer.evolve_strategies(
                    self.strategy_population, strategy_performance
                )

            # Generar recomendaciones de optimización
            optimization_recommendations = self._generate_optimization_recommendations(
                learning_data, detected_patterns, strategy_performance
            )

            # Adaptar configuración basada en aprendizaje
            if self.config.enable_contextual_adaptation:
                self._adapt_to_learning_context(learning_data)

            # Actualizar métricas
            self.state.total_learning_sessions += 1
            self.state.knowledge_growth_rate = self._calculate_knowledge_growth_rate()
            self.state.adaptation_effectiveness = self._calculate_adaptation_effectiveness()

            # Guardar estado
            self._save_state()

            processing_time = (datetime.now() - session_start).total_seconds()

            return {
                "success": True,
                "detected_patterns": len(detected_patterns),
                "best_strategy_fitness": max(s.fitness_score for s in self.strategy_population),
                "optimization_recommendations": optimization_recommendations,
                "processing_time": processing_time,
                "session_id": f"learning_session_{int(time.time())}",
            }

        except Exception as e:
            self.logger.error(f"Error processing learning session: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - session_start).total_seconds(),
            }

    def _evaluate_strategy_performance(self, learning_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluar rendimiento de estrategias actuales"""
        performance_metrics = {}

        # Métricas básicas de rendimiento
        performance_metrics["accuracy"] = learning_data.get("performance_metrics", {}).get("accuracy", 0.5)
        performance_metrics["loss"] = learning_data.get("performance_metrics", {}).get("loss", 1.0)
        performance_metrics["learning_rate"] = learning_data.get("hyperparameters", {}).get("learning_rate", 1e-4)

        # Métricas de eficiencia
        processing_time = learning_data.get("processing_time", 1.0)
        memory_usage = learning_data.get("memory_usage", 0.5)

        performance_metrics["efficiency"] = 1.0 / (1.0 + processing_time * memory_usage)
        performance_metrics["stability"] = 1.0 - abs(performance_metrics["accuracy"] - 0.5) * 2

        return performance_metrics

    def _generate_optimization_recommendations(
        self,
        learning_data: Dict[str, Any],
        detected_patterns: List[LearningPattern],
        performance: Dict[str, float],
    ) -> List[str]:
        """Generar recomendaciones de optimización"""
        recommendations = []

        # Recomendaciones basadas en patrones detectados
        for pattern in detected_patterns:
            if pattern.pattern_type == "knowledge_growth":
                recommendations.append(
                    f"Alto crecimiento de conocimiento detectado ({pattern.confidence_score:.2f}). "
                    "Considere aumentar el ritmo de aprendizaje."
                )
            elif pattern.pattern_type == "performance_degradation":
                recommendations.append(
                    f"Degradación de rendimiento detectada ({pattern.confidence_score:.2f}). "
                    "Considere reducir learning rate o aumentar regularización."
                )
            elif pattern.pattern_type == "context_shift":
                recommendations.append(
                    f"Cambio significativo de contexto detectado ({pattern.confidence_score:.2f}). "
                    "Considere recalibrar estrategias de aprendizaje."
                )

        # Recomendaciones basadas en métricas de rendimiento
        if performance.get("accuracy", 0.0) < 0.3:
            recommendations.append(
                "Baja precisión detectada. Considere aumentar datos de entrenamiento o ajustar arquitectura."
            )

        if performance.get("efficiency", 0.0) < 0.3:
            recommendations.append(
                "Baja eficiencia detectada. Considere optimizar hiperparámetros o reducir complejidad del modelo."
            )

        return recommendations

    def _adapt_to_learning_context(self, learning_data: Dict[str, Any]):
        """Adaptar configuración basada en contexto de aprendizaje"""
        context_metrics = learning_data.get("context_metrics", {})

        # Adaptar tasa de evolución basada en estabilidad del contexto
        context_stability = context_metrics.get("context_stability", 0.5)

        if context_stability < 0.3:
            # Contexto inestable, reducir tasa de adaptación
            self.config.evolution_rate = max(0.001, self.config.evolution_rate * 0.8)
        else:
            # Contexto estable, aumentar tasa de adaptación
            self.config.evolution_rate = min(0.1, self.config.evolution_rate * 1.1)

        # Adaptar sensibilidad basada en ruido del contexto
        context_noise = context_metrics.get("context_noise", 0.5)

        if context_noise > 0.7:
            # Alto ruido, reducir sensibilidad
            self.config.adaptation_sensitivity = max(0.1, self.config.adaptation_sensitivity * 0.8)
        else:
            # Bajo ruido, aumentar sensibilidad
            self.config.adaptation_sensitivity = min(1.0, self.config.adaptation_sensitivity * 1.1)

    def _calculate_knowledge_growth_rate(self) -> float:
        """Calcular tasa de crecimiento de conocimiento"""
        if not self.state.detected_patterns:
            return 0.0

        # Calcular basado en patrones de crecimiento recientes
        growth_patterns = [
            pattern for pattern in self.state.detected_patterns.values() if pattern.pattern_type == "knowledge_growth"
        ]

        if not growth_patterns:
            return 0.0

        # Promedio de tasas de crecimiento recientes
        recent_patterns = [p for p in growth_patterns if (datetime.now() - p.detection_time).total_seconds() < 3600]
        avg_growth_rate = sum(p.pattern_data.get("growth_rate", 0.0) for p in recent_patterns) / max(
            len(recent_patterns), 1
        )

        return min(1.0, avg_growth_rate)

    def _calculate_adaptation_effectiveness(self) -> float:
        """Calcular efectividad de adaptación"""
        if not self.state.evolution_history:
            return 0.0

        # Analizar mejora en fitness a lo largo de generaciones
        recent_generations = self.state.evolution_history[-5:]

        if len(recent_generations) < 2:
            return 0.0

        fitness_progression = [gen.get("best_fitness", 0.0) for gen in recent_generations]
        improvement_rate = fitness_progression[-1] - fitness_progression[0]

        return max(0.0, min(1.0, improvement_rate * 10))  # Escalar para rango 0-1

    def get_best_learning_strategy(self) -> Optional[LearningStrategy]:
        """Obtener mejor estrategia de aprendizaje actual"""
        if not self.strategy_population:
            return None

        return max(self.strategy_population, key=lambda x: x.fitness_score)

    def export_learning_knowledge(self) -> Dict[str, Any]:
        """Exportar conocimiento adquirido sobre aprendizaje"""
        knowledge = {
            "autonomous_learning_engine": "autonomous_learning_v2",
            "total_sessions": self.state.total_learning_sessions,
            "detected_patterns": {
                pattern_id: {
                    "type": pattern.pattern_type,
                    "confidence": pattern.confidence_score,
                    "occurrences": pattern.occurrences,
                    "impact": pattern.impact_score,
                }
                for pattern_id, pattern in self.state.detected_patterns.items()
            },
            "best_strategy": {
                "name": self.get_best_learning_strategy().strategy_name,
                "fitness": self.get_best_learning_strategy().fitness_score,
                "hyperparameters": self.get_best_learning_strategy().hyperparameters,
            }
            if self.get_best_learning_strategy()
            else None,
            "evolution_metrics": {
                "knowledge_growth_rate": self.state.knowledge_growth_rate,
                "adaptation_effectiveness": self.state.adaptation_effectiveness,
                "optimization_success_rate": self.state.optimization_success_rate,
            },
            "evolution_history": self.state.evolution_history[-10:],  # Últimas 10 generaciones
            "export_timestamp": datetime.now().isoformat(),
        }

        return knowledge


# Función de integración con sistemas existentes
def integrate_autonomous_learning(
    config: Optional[AutonomousLearningConfig] = None,
) -> MetaLearningEngine:
    """Integrar motor de aprendizaje autónomo"""
    return MetaLearningEngine(config)


# Función de análisis de aprendizaje para debugging
def analyze_learning_session(session_data: Dict[str, Any], learning_engine: MetaLearningEngine) -> Dict[str, Any]:
    """Analizar sesión de aprendizaje específica"""
    try:
        # Procesar sesión
        analysis_result = learning_engine.process_learning_session(session_data)

        # Obtener mejor estrategia
        best_strategy = learning_engine.get_best_learning_strategy()

        return {
            "session_analysis": analysis_result,
            "recommended_strategy": {
                "name": best_strategy.strategy_name,
                "fitness": best_strategy.fitness_score,
                "hyperparameters": best_strategy.hyperparameters,
            }
            if best_strategy
            else None,
            "detected_patterns": len(learning_engine.state.detected_patterns),
            "evolution_generation": learning_engine.evolution_optimizer.generation,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "error": str(e),
            "session_data": session_data,
            "analysis_timestamp": datetime.now().isoformat(),
        }
