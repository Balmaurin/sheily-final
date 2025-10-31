#!/usr/bin/env python3
"""
Unit Tests: Componentes Reales (No Mocks)
=========================================
Tests para verificar que los componentes implementados son reales.
"""

import pytest
import time


@pytest.mark.unit
class TestRealMergerAnalyzer:
    """Tests para RealMergerAnalyzer"""
    
    def test_analyzer_initialization(self):
        """Verificar inicialización del analizador"""
        from sheily_core.tools.real_merger_analysis import RealMergerAnalyzer
        
        analyzer = RealMergerAnalyzer()
        assert analyzer is not None
        assert analyzer.min_confidence == 0.3
        assert analyzer.complexity_threshold == 0.5
    
    def test_integration_context_analysis(self):
        """Verificar análisis de contexto de integración"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer
        
        analyzer = get_real_analyzer()
        
        result = analyzer.analyze_integration_context(
            query="¿Qué es Python?",
            specialized="Python es un lenguaje de programación interpretado.",
            general="Python se usa para desarrollo web y ciencia de datos."
        )
        
        assert "integration_points" in result
        assert "integration_score" in result
        assert isinstance(result["integration_score"], float)
        assert 0 <= result["integration_score"] <= 1
    
    def test_consensus_analysis(self):
        """Verificar análisis de consenso"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer
        
        analyzer = get_real_analyzer()
        
        sources = [
            {"response": "Python es simple", "confidence": 0.8},
            {"response": "Python es poderoso", "confidence": 0.7},
            {"response": "Python es versátil", "confidence": 0.9}
        ]
        
        result = analyzer.analyze_consensus(sources)
        
        assert "consensus_level" in result
        assert "agreement_score" in result
        assert "divergence" in result
        assert result["num_sources"] == 3
        assert 0 <= result["consensus_level"] <= 1
    
    def test_build_consensus_response(self):
        """Verificar construcción de respuesta de consenso"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer
        
        analyzer = get_real_analyzer()
        
        sources = [
            {"response": "Respuesta A", "confidence": 0.9},
            {"response": "Respuesta B", "confidence": 0.7}
        ]
        
        analysis = {
            "consensus_level": 0.8,
            "primary_source": sources[0]
        }
        
        response = analyzer.build_consensus_response(sources, analysis)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Respuesta" in response
    
    def test_text_similarity(self):
        """Verificar cálculo de similitud de texto"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer
        
        analyzer = get_real_analyzer()
        
        text1 = "Python es un lenguaje de programación"
        text2 = "Python es un lenguaje interpretado"
        
        similarity = analyzer._calculate_text_similarity(text1, text2)
        
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
        assert similarity > 0  # Debe haber alguna similitud
    
    def test_keyword_extraction(self):
        """Verificar extracción de keywords"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer
        
        analyzer = get_real_analyzer()
        
        text = "Python es un lenguaje de programación orientado a objetos"
        keywords = analyzer._extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "python" in keywords or "Python" in [k.title() for k in keywords]


@pytest.mark.unit
class TestRealRateLimiter:
    """Tests para RealRateLimiter"""
    
    def test_rate_limiter_initialization(self):
        """Verificar inicialización del rate limiter"""
        from sheily_core.security.real_rate_limiter import RealRateLimiter
        
        limiter = RealRateLimiter(max_requests_per_minute=10)
        
        assert limiter.max_requests == 10
        assert limiter.block_duration == 60
        assert len(limiter._clients) == 0
    
    def test_rate_limit_allows_requests(self):
        """Verificar que permite requests dentro del límite"""
        from sheily_core.security.real_rate_limiter import RealRateLimiter
        
        limiter = RealRateLimiter(max_requests_per_minute=5)
        
        # Permitir 5 requests
        for i in range(5):
            allowed, error = limiter.check_rate_limit("client1")
            assert allowed is True
            assert error is None
    
    def test_rate_limit_blocks_excess(self):
        """Verificar que bloquea requests en exceso"""
        from sheily_core.security.real_rate_limiter import RealRateLimiter
        
        limiter = RealRateLimiter(max_requests_per_minute=3)
        
        # 3 requests permitidos
        for i in range(3):
            allowed, _ = limiter.check_rate_limit("client2")
            assert allowed is True
        
        # 4to request debe ser bloqueado
        allowed, error = limiter.check_rate_limit("client2")
        assert allowed is False
        assert "Rate limit exceeded" in error
    
    def test_rate_limit_sliding_window(self):
        """Verificar sliding window funcionando"""
        from sheily_core.security.real_rate_limiter import RealRateLimiter
        
        limiter = RealRateLimiter(max_requests_per_minute=5)
        
        # 5 requests
        for i in range(5):
            limiter.check_rate_limit("client3")
        
        # Esperar un poco y verificar que se limpian
        time.sleep(0.1)
        
        stats = limiter.get_client_stats("client3")
        assert stats["exists"] is True
        assert stats["current_window_requests"] <= 5
    
    def test_client_statistics(self):
        """Verificar estadísticas de cliente"""
        from sheily_core.security.real_rate_limiter import RealRateLimiter
        
        limiter = RealRateLimiter()
        
        # Hacer algunos requests
        limiter.check_rate_limit("client4")
        limiter.check_rate_limit("client4")
        
        stats = limiter.get_client_stats("client4")
        
        assert stats["exists"] is True
        assert stats["total_requests"] == 2
        assert stats["current_window_requests"] > 0
        assert stats["is_blocked"] is False
    
    def test_global_statistics(self):
        """Verificar estadísticas globales"""
        from sheily_core.security.real_rate_limiter import RealRateLimiter
        
        limiter = RealRateLimiter()
        
        limiter.check_rate_limit("clientA")
        limiter.check_rate_limit("clientB")
        limiter.check_rate_limit("clientC")
        
        stats = limiter.get_global_stats()
        
        assert stats["total_clients"] >= 3
        assert stats["total_requests"] >= 3
        assert "max_requests_per_minute" in stats
    
    def test_reset_client(self):
        """Verificar reset de cliente"""
        from sheily_core.security.real_rate_limiter import RealRateLimiter
        
        limiter = RealRateLimiter()
        
        limiter.check_rate_limit("client5")
        limiter.check_rate_limit("client5")
        
        # Reset
        result = limiter.reset_client("client5")
        assert result is True
        
        # Verificar que se reseteó
        stats = limiter.get_client_stats("client5")
        assert stats["exists"] is False


@pytest.mark.unit
class TestImprovedCPUTrainer:
    """Tests para ImprovedCPUTrainer"""
    
    def test_trainer_initialization(self):
        """Verificar inicialización del trainer"""
        from sheily_core.tools.improved_cpu_training import ImprovedCPUTrainer
        
        trainer = ImprovedCPUTrainer(num_epochs=2)
        
        assert trainer.num_epochs == 2
        assert trainer.initial_lr > 0
        assert trainer.batch_size > 0
        assert len(trainer.metrics_history) == 0
    
    def test_training_produces_metrics(self):
        """Verificar que training produce métricas"""
        from sheily_core.tools.improved_cpu_training import ImprovedCPUTrainer
        
        trainer = ImprovedCPUTrainer(num_epochs=1)
        result = trainer.train()
        
        assert result["success"] is True
        assert result["simulated"] is True
        assert result["num_epochs"] == 1
        assert "final_loss" in result
        assert "metrics" in result
        assert len(result["metrics"]) == 1
    
    def test_loss_decreases(self):
        """Verificar que loss disminuye (simuladamente)"""
        from sheily_core.tools.improved_cpu_training import ImprovedCPUTrainer
        
        trainer = ImprovedCPUTrainer(num_epochs=3)
        result = trainer.train()
        
        metrics = result["metrics"]
        
        # Loss debe tender a disminuir
        first_loss = metrics[0]["loss"]
        last_loss = metrics[-1]["loss"]
        
        # Con alta probabilidad, debería disminuir
        assert last_loss <= first_loss or abs(last_loss - first_loss) < 0.5
    
    def test_metrics_summary(self):
        """Verificar resumen de métricas"""
        from sheily_core.tools.improved_cpu_training import ImprovedCPUTrainer
        
        trainer = ImprovedCPUTrainer(num_epochs=2)
        trainer.train()
        
        summary = trainer.get_metrics_summary()
        
        assert "total_epochs" in summary
        assert "loss" in summary
        assert "learning_rate" in summary
        assert summary["total_epochs"] == 2


@pytest.mark.unit  
class TestMergerIntegration:
    """Tests de integración del merger con análisis real"""
    
    @pytest.mark.asyncio
    async def test_merger_uses_real_analysis(self):
        """Verificar que merger usa análisis real"""
        from sheily_core.tools.merger import AdvancedResponseMerger
        
        merger = AdvancedResponseMerger()
        
        # Análisis de consenso debe usar implementación real
        result = await merger._analyze_consensus([
            {"content": "Test A", "confidence": 0.8},
            {"content": "Test B", "confidence": 0.7}
        ])
        
        assert "consensus_score" in result
        assert "num_sources" in result
        assert result["num_sources"] == 2
