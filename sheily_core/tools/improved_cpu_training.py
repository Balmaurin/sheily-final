#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved CPU Training - Más Realista
====================================
Simulación mejorada de training cuando no hay GPU disponible.
No entrena realmente pero simula comportamiento realista.
"""

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Métricas de training mejoradas"""

    epoch: int
    loss: float
    learning_rate: float
    grad_norm: float
    time_elapsed: float


class ImprovedCPUTrainer:
    """
    Trainer CPU mejorado - Simulación REALISTA
    ==========================================

    ⚠️ ADVERTENCIA: Esto NO entrena realmente el modelo

    Simula training con:
    - Loss curves realistas
    - Learning rate decay
    - Gradient norms
    - Tiempos realistas
    - Variabilidad natural
    """

    def __init__(self, num_epochs: int = 3, learning_rate: float = 2e-5, batch_size: int = 4):
        """
        Inicializar trainer mejorado

        Args:
            num_epochs: Número de épocas
            learning_rate: Learning rate inicial
            batch_size: Tamaño de batch
        """
        self.num_epochs = num_epochs
        self.initial_lr = learning_rate
        self.batch_size = batch_size
        self.metrics_history: List[TrainingMetrics] = []

        # Parámetros para simulación realista
        self.initial_loss = random.uniform(2.5, 3.5)  # Loss inicial realista
        self.target_loss = random.uniform(0.5, 1.2)  # Loss objetivo
        self.convergence_rate = random.uniform(0.3, 0.5)

        logger.warning(
            "⚠️ ImprovedCPUTrainer: Simulación mejorada (NO entrena realmente)\n"
            f"   Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}"
        )

    def train(self) -> Dict[str, Any]:
        """
        Ejecutar simulación de training

        Returns:
            Resultados de training simulado
        """
        logger.warning("🚀 Iniciando training SIMULADO (mejorado)")
        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # 1. Calcular loss con curva realista
            progress = (epoch + 1) / self.num_epochs

            # Loss decay exponencial con ruido
            base_loss = self._calculate_realistic_loss(progress)
            noise = random.gauss(0, 0.05)  # Ruido gaussiano
            epoch_loss = max(0.1, base_loss + noise)

            # 2. Learning rate con decay
            current_lr = self._calculate_learning_rate(progress)

            # 3. Gradient norm simulado
            grad_norm = self._simulate_gradient_norm(progress)

            # 4. Simular tiempo de época realista
            # Tiempo base + variabilidad
            base_time = 5.0  # 5 segundos base por época
            time_variation = random.uniform(-1.0, 1.0)
            time.sleep(min(2.0, base_time / 3 + time_variation / 2))  # Tiempo reducido para demo

            epoch_time = time.time() - epoch_start

            # 5. Registrar métricas
            metrics = TrainingMetrics(
                epoch=epoch + 1, loss=epoch_loss, learning_rate=current_lr, grad_norm=grad_norm, time_elapsed=epoch_time
            )
            self.metrics_history.append(metrics)

            # 6. Log progreso
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Loss: {epoch_loss:.4f}, LR: {current_lr:.2e}, "
                f"Grad Norm: {grad_norm:.4f}, Time: {epoch_time:.2f}s"
            )

            # 7. Simular early stopping si converge
            if epoch > 0 and abs(epoch_loss - self.metrics_history[-2].loss) < 0.01:
                logger.info("⚠️ Convergencia detectada (simulada)")

        total_time = time.time() - start_time
        final_loss = self.metrics_history[-1].loss if self.metrics_history else 0.0

        logger.warning(
            f"✅ Training SIMULADO completado\n" f"   Final Loss: {final_loss:.4f}, Total Time: {total_time:.2f}s"
        )

        return {
            "success": True,
            "simulated": True,
            "num_epochs": len(self.metrics_history),
            "final_loss": final_loss,
            "initial_loss": self.initial_loss,
            "total_time": total_time,
            "metrics": [
                {
                    "epoch": m.epoch,
                    "loss": round(m.loss, 4),
                    "learning_rate": m.learning_rate,
                    "grad_norm": round(m.grad_norm, 4),
                    "time": round(m.time_elapsed, 2),
                }
                for m in self.metrics_history
            ],
        }

    def _calculate_realistic_loss(self, progress: float) -> float:
        """
        Calcular loss realista con curva exponencial

        Args:
            progress: Progreso (0-1)

        Returns:
            Loss value
        """
        # Curva exponencial de decaimiento
        # loss = initial * exp(-rate * progress) + target
        decay = math.exp(-self.convergence_rate * progress * 10)
        loss = self.initial_loss * decay + self.target_loss * (1 - decay)

        return loss

    def _calculate_learning_rate(self, progress: float) -> float:
        """
        Calcular learning rate con decay

        Args:
            progress: Progreso (0-1)

        Returns:
            Learning rate actual
        """
        # Cosine annealing schedule
        lr = self.initial_lr * (1 + math.cos(math.pi * progress)) / 2
        return max(lr, self.initial_lr * 0.1)  # Mínimo 10% del inicial

    def _simulate_gradient_norm(self, progress: float) -> float:
        """
        Simular gradient norm

        Args:
            progress: Progreso (0-1)

        Returns:
            Gradient norm
        """
        # Gradients típicamente disminuyen con el tiempo
        base_norm = 5.0 * (1 - progress * 0.7)  # Decrece 70%
        noise = random.uniform(-0.5, 0.5)
        return max(0.1, base_norm + noise)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de métricas

        Returns:
            Resumen estadístico
        """
        if not self.metrics_history:
            return {"error": "No metrics available"}

        losses = [m.loss for m in self.metrics_history]
        lrs = [m.learning_rate for m in self.metrics_history]

        return {
            "total_epochs": len(self.metrics_history),
            "loss": {
                "initial": round(losses[0], 4),
                "final": round(losses[-1], 4),
                "min": round(min(losses), 4),
                "max": round(max(losses), 4),
                "improvement": round((losses[0] - losses[-1]) / losses[0] * 100, 2),
            },
            "learning_rate": {"initial": lrs[0], "final": lrs[-1]},
            "convergence": {"achieved": losses[-1] < losses[0] * 0.5, "rate": round(self.convergence_rate, 3)},
        }


def create_improved_cpu_trainer(
    num_epochs: int = 3, learning_rate: float = 2e-5, batch_size: int = 4
) -> ImprovedCPUTrainer:
    """
    Crear trainer CPU mejorado

    Args:
        num_epochs: Número de épocas
        learning_rate: Learning rate
        batch_size: Batch size

    Returns:
        ImprovedCPUTrainer instance
    """
    logger.warning(
        "⚠️⚠️⚠️ CREANDO TRAINER CPU MEJORADO (SIMULACIÓN) ⚠️⚠️⚠️\n"
        "Este trainer NO entrena realmente el modelo.\n"
        "Simula el proceso con métricas realistas.\n"
        "Para training REAL, use GPU con CUDA."
    )

    return ImprovedCPUTrainer(num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size)


# Exports
__all__ = ["ImprovedCPUTrainer", "TrainingMetrics", "create_improved_cpu_trainer"]
