#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Branch Manager - Gestor de Ramas Especializadas
===============================================

Coordina y gestiona todas las ramas especializadas del sistema Sheily-AI,
permitiendo especialización dinámica y fusión de conocimientos.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .merger import BranchMerger
from .specialization import SpecializationEngine

logger = logging.getLogger(__name__)


class BranchType(Enum):
    """Tipos de ramas disponibles"""

    ACADEMIC = "academic"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"


class BranchStatus(Enum):
    """Estados de las ramas"""

    ACTIVE = "active"
    LOADING = "loading"
    READY = "ready"
    UPDATING = "updating"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class BranchInfo:
    """Información de una rama especializada"""

    name: str
    type: BranchType
    language: str
    domain: str
    description: str
    keywords: List[str] = field(default_factory=list)
    status: BranchStatus = BranchStatus.LOADING
    creation_time: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    usage_count: int = 0
    accuracy_score: float = 0.0
    specialization_level: float = 0.0
    corpus_size: int = 0
    model_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchQuery:
    """Consulta para una rama específica"""

    branch_name: str
    query: str
    context: Optional[Dict] = None
    require_specialization: bool = True
    merge_with_general: bool = False


@dataclass
class BranchResponse:
    """Respuesta de una rama"""

    branch_name: str
    response: str
    confidence: float
    specialization_used: bool
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BranchManager:
    """
    Gestor central de ramas especializadas
    """

    def __init__(self, config: Dict):
        """
        Inicializar el gestor de ramas

        Args:
            config: Configuración del gestor
        """
        self.config = config
        self.branches_path = Path(config.get("branches_path", "./branches"))
        self.corpus_es_path = Path(config.get("corpus_es_path", "./corpus_ES"))
        self.corpus_en_path = Path(config.get("corpus_EN_path", "./corpus_EN"))

        # Gestión de ramas
        self.branches: Dict[str, BranchInfo] = {}
        self.active_branches: Set[str] = set()

        # Configuración
        self.max_concurrent_branches = config.get("max_concurrent_branches", 10)
        self.branch_timeout = config.get("branch_timeout", 30.0)
        self.auto_unload_minutes = config.get("auto_unload_minutes", 30)
        self.enable_specialization = config.get("enable_specialization", True)
        self.enable_merging = config.get("enable_merging", True)

        # Componentes especializados
        self.specialization_engine = (
            SpecializationEngine(config) if self.enable_specialization else None
        )
        self.branch_merger = BranchMerger(config) if self.enable_merging else None

        # Cache de respuestas
        self.response_cache: Dict[str, BranchResponse] = {}
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_size = config.get("cache_size", 1000)

        # Estadísticas
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "cache_hits": 0,
            "branch_loads": 0,
            "specialization_requests": 0,
            "merge_requests": 0,
            "branches_by_usage": {},
            "average_response_time": 0.0,
        }

        logger.info("BranchManager inicializado")

    async def initialize(self) -> bool:
        """
        Inicializar el gestor de ramas

        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Descubrir ramas disponibles
            await self._discover_available_branches()

            # Inicializar componentes
            if self.specialization_engine:
                await self.specialization_engine.initialize()

            if self.branch_merger:
                await self.branch_merger.initialize()

            # Cargar ramas prioritarias
            await self._load_priority_branches()

            # Iniciar procesos de mantenimiento
            asyncio.create_task(self._maintenance_loop())

            logger.info(f"BranchManager inicializado con {len(self.branches)} ramas disponibles")
            return True

        except Exception as e:
            logger.error(f"Error inicializando BranchManager: {e}")
            return False

    async def query_branch(self, query: BranchQuery) -> BranchResponse:
        """
        Ejecutar consulta en una rama específica

        Args:
            query: Consulta para la rama

        Returns:
            Respuesta de la rama
        """
        start_time = time.time()

        try:
            # Verificar cache primero
            cache_key = self._generate_cache_key(query)

            if self.cache_enabled and cache_key in self.response_cache:
                self.stats["cache_hits"] += 1
                cached_response = self.response_cache[cache_key]
                logger.debug(f"Cache hit para consulta en rama {query.branch_name}")
                return cached_response

            # Verificar que la rama existe
            if query.branch_name not in self.branches:
                return BranchResponse(
                    branch_name=query.branch_name,
                    response=f"Rama '{query.branch_name}' no encontrada",
                    confidence=0.0,
                    specialization_used=False,
                    processing_time=time.time() - start_time,
                    metadata={"error": "branch_not_found"},
                )

            # Cargar rama si no está activa
            if query.branch_name not in self.active_branches:
                await self._load_branch(query.branch_name)

            # Ejecutar consulta en la rama
            response = await self._execute_branch_query(query)

            # Aplicar especialización si se solicita
            if query.require_specialization and self.specialization_engine:
                response = await self._apply_specialization(query, response)

            # Fusionar con rama general si se solicita
            if query.merge_with_general and self.branch_merger:
                response = await self._merge_with_general(query, response)

            # Actualizar estadísticas
            processing_time = time.time() - start_time
            response.processing_time = processing_time

            await self._update_branch_stats(query.branch_name, response, processing_time)

            # Guardar en cache
            if self.cache_enabled and response.confidence > 0.7:
                self._cache_response(cache_key, response)

            return response

        except Exception as e:
            logger.error(f"Error en consulta de rama {query.branch_name}: {e}")

            return BranchResponse(
                branch_name=query.branch_name,
                response=f"Error procesando consulta: {str(e)}",
                confidence=0.0,
                specialization_used=False,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)},
            )

    async def _discover_available_branches(self):
        """Descubrir ramas disponibles en el sistema"""
        discovered_branches = []

        # Escanear corpus español
        if self.corpus_es_path.exists():
            for domain_path in self.corpus_es_path.iterdir():
                if domain_path.is_dir() and not domain_path.name.startswith("."):
                    discovered_branches.append(
                        {
                            "name": domain_path.name,
                            "language": "es",
                            "domain": domain_path.name,
                            "corpus_path": domain_path,
                        }
                    )

        # Escanear corpus inglés
        if self.corpus_en_path.exists():
            for domain_path in self.corpus_en_path.iterdir():
                if domain_path.is_dir() and not domain_path.name.startswith("."):
                    # Evitar duplicados con español
                    english_name = f"{domain_path.name}_en"
                    discovered_branches.append(
                        {
                            "name": english_name,
                            "language": "en",
                            "domain": domain_path.name,
                            "corpus_path": domain_path,
                        }
                    )

        # Escanear ramas definidas manualmente
        if self.branches_path.exists():
            for branch_path in self.branches_path.iterdir():
                if branch_path.is_dir() and not branch_path.name.startswith("."):
                    config_file = branch_path / "config.json"
                    if config_file.exists():
                        try:
                            with open(config_file, "r", encoding="utf-8") as f:
                                branch_config = json.load(f)

                            discovered_branches.append(
                                {
                                    "name": branch_path.name,
                                    "language": branch_config.get("language", "es"),
                                    "domain": branch_config.get("domain", branch_path.name),
                                    "config": branch_config,
                                    "branch_path": branch_path,
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Error cargando config de rama {branch_path.name}: {e}")

        # Crear BranchInfo para cada rama descubierta
        for branch_data in discovered_branches:
            branch_info = await self._create_branch_info(branch_data)
            self.branches[branch_info.name] = branch_info

        logger.info(f"Descubiertas {len(discovered_branches)} ramas")

    async def _create_branch_info(self, branch_data: Dict) -> BranchInfo:
        """
        Crear información de rama desde datos descubiertos

        Args:
            branch_data: Datos de la rama descubierta

        Returns:
            Información de la rama
        """
        name = branch_data["name"]
        domain = branch_data["domain"]
        language = branch_data["language"]

        # Determinar tipo de rama basado en dominio
        branch_type = self._classify_branch_type(domain)

        # Generar descripción
        description = self._generate_branch_description(domain, language)

        # Extraer keywords del dominio
        keywords = self._extract_domain_keywords(domain, language)

        # Calcular tamaño del corpus si existe
        corpus_size = 0
        if "corpus_path" in branch_data and Path(branch_data["corpus_path"]).exists():
            corpus_size = await self._calculate_corpus_size(Path(branch_data["corpus_path"]))

        # Configuración específica de rama
        config = branch_data.get("config", {})

        return BranchInfo(
            name=name,
            type=branch_type,
            language=language,
            domain=domain,
            description=description,
            keywords=keywords,
            status=BranchStatus.READY,
            corpus_size=corpus_size,
            config=config,
        )

    def _classify_branch_type(self, domain: str) -> BranchType:
        """Clasificar tipo de rama basado en dominio"""
        domain_lower = domain.lower()

        # Dominios académicos
        academic_domains = {
            "matemáticas",
            "mathematics",
            "física",
            "physics",
            "química",
            "chemistry",
            "biología",
            "biology",
            "historia",
            "history",
            "filosofía",
            "philosophy",
            "literatura",
            "literature",
            "geografía",
            "geography",
            "sociología",
            "sociology",
        }

        # Dominios técnicos
        technical_domains = {
            "programación",
            "programming",
            "inteligencia artificial",
            "artificial_intelligence",
            "ingeniería",
            "engineering",
            "computación",
            "computer_science",
            "robótica",
            "robotics",
            "ciberseguridad",
            "cybersecurity",
            "blockchain",
            "redes",
            "networks",
        }

        # Dominios creativos
        creative_domains = {
            "arte",
            "art",
            "música",
            "music",
            "diseño",
            "design",
            "fotografía",
            "photography",
            "artes visuales",
            "visual_arts",
            "cine",
            "cinema",
            "teatro",
            "theater",
        }

        # Dominios analíticos
        analytical_domains = {
            "economía",
            "economics",
            "finanzas",
            "finance",
            "estadística",
            "statistics",
            "psicología",
            "psychology",
            "neurociencia",
            "neuroscience",
            "investigación",
            "research",
        }

        if any(d in domain_lower for d in academic_domains):
            return BranchType.ACADEMIC
        elif any(d in domain_lower for d in technical_domains):
            return BranchType.TECHNICAL
        elif any(d in domain_lower for d in creative_domains):
            return BranchType.CREATIVE
        elif any(d in domain_lower for d in analytical_domains):
            return BranchType.ANALYTICAL
        else:
            return BranchType.CONVERSATIONAL

    def _generate_branch_description(self, domain: str, language: str) -> str:
        """Generar descripción de rama"""
        if language == "es":
            return f"Rama especializada en {domain} con conocimientos avanzados y contexto específico del dominio"
        else:
            return f"Specialized branch in {domain} with advanced knowledge and domain-specific context"

    def _extract_domain_keywords(self, domain: str, language: str) -> List[str]:
        """Extraer keywords relevantes del dominio"""
        # Mapeo de dominios a keywords
        domain_keywords = {
            "matemáticas": ["ecuación", "álgebra", "geometría", "cálculo", "estadística", "número"],
            "mathematics": ["equation", "algebra", "geometry", "calculus", "statistics", "number"],
            "programación": ["código", "algoritmo", "función", "variable", "clase", "método"],
            "programming": ["code", "algorithm", "function", "variable", "class", "method"],
            "medicina": ["diagnóstico", "tratamiento", "síntoma", "enfermedad", "paciente"],
            "medicine": ["diagnosis", "treatment", "symptom", "disease", "patient"],
            "física": ["energía", "fuerza", "velocidad", "masa", "gravedad"],
            "physics": ["energy", "force", "velocity", "mass", "gravity"],
        }

        # Obtener keywords específicos o generar basado en dominio
        keywords = domain_keywords.get(domain.lower(), [])

        if not keywords:
            # Keywords genéricos basados en el nombre del dominio
            keywords = [domain.lower()]
            if " " in domain:
                keywords.extend(domain.lower().split())

        return keywords

    async def _calculate_corpus_size(self, corpus_path: Path) -> int:
        """Calcular tamaño del corpus en archivos"""
        try:
            size = 0
            for file_path in corpus_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in [".txt", ".md", ".json"]:
                    size += 1
            return size
        except Exception as e:
            logger.warning(f"Error calculando tamaño de corpus {corpus_path}: {e}")
            return 0

    async def _load_priority_branches(self):
        """Cargar ramas de alta prioridad"""
        priority_branches = ["general", "programación", "programming", "inteligencia artificial"]

        loaded_count = 0
        for branch_name in priority_branches:
            if branch_name in self.branches and loaded_count < 3:
                await self._load_branch(branch_name)
                loaded_count += 1

        logger.info(f"Cargadas {loaded_count} ramas prioritarias")

    async def _load_branch(self, branch_name: str) -> bool:
        """
        Cargar una rama específica

        Args:
            branch_name: Nombre de la rama a cargar

        Returns:
            bool: True si se cargó exitosamente
        """
        if branch_name not in self.branches:
            logger.warning(f"Rama {branch_name} no existe")
            return False

        if branch_name in self.active_branches:
            logger.debug(f"Rama {branch_name} ya está cargada")
            return True

        try:
            # Verificar límite de ramas concurrentes
            if len(self.active_branches) >= self.max_concurrent_branches:
                # Descargar rama menos usada
                await self._unload_least_used_branch()

            branch_info = self.branches[branch_name]
            branch_info.status = BranchStatus.LOADING

            # Simular carga de rama (en implementación real cargaría modelo/índices)
            await asyncio.sleep(0.1)  # Simular tiempo de carga

            # Marcar como activa
            self.active_branches.add(branch_name)
            branch_info.status = BranchStatus.ACTIVE
            branch_info.last_access = time.time()

            self.stats["branch_loads"] += 1

            logger.info(f"Rama {branch_name} cargada exitosamente")
            return True

        except Exception as e:
            if branch_name in self.branches:
                self.branches[branch_name].status = BranchStatus.ERROR

            logger.error(f"Error cargando rama {branch_name}: {e}")
            return False

    async def _unload_branch(self, branch_name: str) -> bool:
        """
        Descargar una rama de memoria

        Args:
            branch_name: Nombre de la rama a descargar

        Returns:
            bool: True si se descargó exitosamente
        """
        if branch_name not in self.active_branches:
            return True

        try:
            # Limpiar recursos de la rama
            self.active_branches.discard(branch_name)

            if branch_name in self.branches:
                self.branches[branch_name].status = BranchStatus.READY

            logger.info(f"Rama {branch_name} descargada")
            return True

        except Exception as e:
            logger.error(f"Error descargando rama {branch_name}: {e}")
            return False

    async def _unload_least_used_branch(self):
        """Descargar la rama menos usada"""
        if not self.active_branches:
            return

        # Encontrar rama con menor uso reciente
        least_used = None
        oldest_access = float("inf")

        for branch_name in self.active_branches:
            if branch_name in self.branches:
                branch_info = self.branches[branch_name]
                if branch_info.last_access < oldest_access:
                    oldest_access = branch_info.last_access
                    least_used = branch_name

        if least_used:
            await self._unload_branch(least_used)
            logger.info(f"Rama menos usada '{least_used}' descargada automáticamente")

    async def _execute_branch_query(self, query: BranchQuery) -> BranchResponse:
        """
        Ejecutar consulta en una rama específica

        Args:
            query: Consulta para la rama

        Returns:
            Respuesta de la rama
        """
        # En implementación real, ejecutaría el modelo/sistema de la rama

        # Simular procesamiento de la rama
        await asyncio.sleep(0.05)  # Simular tiempo de procesamiento

        branch_info = self.branches[query.branch_name]

        # Generar respuesta basada en especialización de la rama
        if branch_info.type == BranchType.TECHNICAL:
            response_text = (
                f"[Respuesta técnica especializada en {branch_info.domain}] {query.query}"
            )
        elif branch_info.type == BranchType.ACADEMIC:
            response_text = f"[Análisis académico en {branch_info.domain}] {query.query}"
        elif branch_info.type == BranchType.CREATIVE:
            response_text = f"[Enfoque creativo en {branch_info.domain}] {query.query}"
        elif branch_info.type == BranchType.ANALYTICAL:
            response_text = f"[Análisis especializado en {branch_info.domain}] {query.query}"
        else:
            response_text = f"[Respuesta de {branch_info.domain}] {query.query}"

        # Calcular confianza basada en match de keywords
        confidence = self._calculate_query_confidence(query, branch_info)

        return BranchResponse(
            branch_name=query.branch_name,
            response=response_text,
            confidence=confidence,
            specialization_used=False,  # Se actualizará si se aplica especialización
            processing_time=0.0,  # Se actualizará después
            metadata={
                "branch_type": branch_info.type.value,
                "domain": branch_info.domain,
                "language": branch_info.language,
            },
        )

    def _calculate_query_confidence(self, query: BranchQuery, branch_info: BranchInfo) -> float:
        """Calcular confianza de la consulta para la rama"""
        query_lower = query.query.lower()
        confidence = 0.5  # Base confidence

        # Bonus por keywords de dominio
        keyword_matches = 0
        for keyword in branch_info.keywords:
            if keyword.lower() in query_lower:
                keyword_matches += 1

        if keyword_matches > 0:
            confidence += min(keyword_matches * 0.15, 0.4)

        # Bonus por tamaño del corpus
        if branch_info.corpus_size > 100:
            confidence += 0.1
        elif branch_info.corpus_size > 50:
            confidence += 0.05

        # Bonus por especialización
        if branch_info.specialization_level > 0:
            confidence += branch_info.specialization_level * 0.2

        return min(confidence, 1.0)

    async def _apply_specialization(
        self, query: BranchQuery, response: BranchResponse
    ) -> BranchResponse:
        """Aplicar especialización a la respuesta"""
        if not self.specialization_engine:
            return response

        try:
            specialized_response = await self.specialization_engine.specialize_response(
                query.query, response.response, query.branch_name, query.context
            )

            response.response = specialized_response.enhanced_response
            response.confidence = max(response.confidence, specialized_response.confidence)
            response.specialization_used = True
            response.metadata["specialization_applied"] = True

            self.stats["specialization_requests"] += 1

            return response

        except Exception as e:
            logger.error(f"Error aplicando especialización: {e}")
            return response

    async def _merge_with_general(
        self, query: BranchQuery, response: BranchResponse
    ) -> BranchResponse:
        """Fusionar respuesta con conocimiento general"""
        if not self.branch_merger:
            return response

        try:
            merged_response = await self.branch_merger.merge_responses(
                specialized_response=response, query=query.query, context=query.context
            )

            response.response = merged_response.merged_response
            response.confidence = merged_response.confidence
            response.metadata["merged_with_general"] = True

            self.stats["merge_requests"] += 1

            return response

        except Exception as e:
            logger.error(f"Error fusionando con general: {e}")
            return response

    async def _update_branch_stats(
        self, branch_name: str, response: BranchResponse, processing_time: float
    ):
        """Actualizar estadísticas de rama"""
        # Estadísticas globales
        self.stats["total_queries"] += 1

        if response.confidence > 0.5:
            self.stats["successful_queries"] += 1

        # Actualizar tiempo promedio
        total_queries = self.stats["total_queries"]
        current_avg = self.stats["average_response_time"]
        self.stats["average_response_time"] = (
            current_avg * (total_queries - 1) + processing_time
        ) / total_queries

        # Estadísticas por rama
        if branch_name not in self.stats["branches_by_usage"]:
            self.stats["branches_by_usage"][branch_name] = 0
        self.stats["branches_by_usage"][branch_name] += 1

        # Actualizar info de la rama
        if branch_name in self.branches:
            branch_info = self.branches[branch_name]
            branch_info.last_access = time.time()
            branch_info.usage_count += 1

            # Actualizar accuracy score
            if branch_info.usage_count == 1:
                branch_info.accuracy_score = response.confidence
            else:
                # Promedio móvil
                branch_info.accuracy_score = (branch_info.accuracy_score * 0.9) + (
                    response.confidence * 0.1
                )

    def _generate_cache_key(self, query: BranchQuery) -> str:
        """Generar clave de cache para la consulta"""
        import hashlib

        cache_content = f"{query.branch_name}|{query.query}|{query.require_specialization}|{query.merge_with_general}"
        return hashlib.md5(cache_content.encode()).hexdigest()

    def _cache_response(self, cache_key: str, response: BranchResponse):
        """Guardar respuesta en cache"""
        if len(self.response_cache) >= self.cache_size:
            # Remover el más antiguo
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]

        self.response_cache[cache_key] = response

    async def _maintenance_loop(self):
        """Loop de mantenimiento del gestor"""
        maintenance_interval = 300.0  # 5 minutos

        while True:
            try:
                await asyncio.sleep(maintenance_interval)

                # Descargar ramas inactivas
                await self._unload_inactive_branches()

                # Limpiar cache viejo
                await self._cleanup_old_cache()

                # Actualizar métricas de ramas
                await self._update_branch_metrics()

            except Exception as e:
                logger.error(f"Error en maintenance loop: {e}")

    async def _unload_inactive_branches(self):
        """Descargar ramas que no se han usado recientemente"""
        current_time = time.time()
        inactive_threshold = self.auto_unload_minutes * 60

        inactive_branches = []

        for branch_name in list(self.active_branches):
            if branch_name in self.branches:
                branch_info = self.branches[branch_name]
                if current_time - branch_info.last_access > inactive_threshold:
                    inactive_branches.append(branch_name)

        for branch_name in inactive_branches:
            await self._unload_branch(branch_name)
            logger.info(f"Rama inactiva '{branch_name}' descargada automáticamente")

    async def _cleanup_old_cache(self):
        """Limpiar cache antiguo"""
        # Mantener solo las respuestas más recientes
        if len(self.response_cache) > self.cache_size * 0.8:
            # Remover 20% del cache más antiguo
            items_to_remove = int(len(self.response_cache) * 0.2)

            for _ in range(items_to_remove):
                if self.response_cache:
                    oldest_key = next(iter(self.response_cache))
                    del self.response_cache[oldest_key]

    async def _update_branch_metrics(self):
        """Actualizar métricas de todas las ramas"""
        for branch_info in self.branches.values():
            # Calcular nivel de especialización basado en uso y accuracy
            if branch_info.usage_count > 10:
                usage_factor = min(branch_info.usage_count / 100.0, 1.0)
                accuracy_factor = branch_info.accuracy_score

                branch_info.specialization_level = (usage_factor + accuracy_factor) / 2.0

    # Métodos públicos para gestión de ramas

    def get_available_branches(self) -> List[BranchInfo]:
        """Obtener lista de ramas disponibles"""
        return list(self.branches.values())

    def get_active_branches(self) -> List[BranchInfo]:
        """Obtener lista de ramas actualmente cargadas"""
        return [self.branches[name] for name in self.active_branches if name in self.branches]

    def get_branch_info(self, branch_name: str) -> Optional[BranchInfo]:
        """Obtener información de una rama específica"""
        return self.branches.get(branch_name)

    async def preload_branch(self, branch_name: str) -> bool:
        """Pre-cargar una rama específica"""
        return await self._load_branch(branch_name)

    async def unload_branch_manual(self, branch_name: str) -> bool:
        """Descargar una rama manualmente"""
        return await self._unload_branch(branch_name)

    def get_stats(self) -> Dict:
        """Obtener estadísticas del gestor"""
        return {
            **self.stats,
            "total_branches": len(self.branches),
            "active_branches": len(self.active_branches),
            "cache_size": len(self.response_cache),
            "branches_info": {
                name: {
                    "status": info.status.value,
                    "usage_count": info.usage_count,
                    "accuracy_score": info.accuracy_score,
                    "specialization_level": info.specialization_level,
                    "last_access": info.last_access,
                }
                for name, info in self.branches.items()
            },
        }

    async def health_check(self) -> Dict:
        """Verificar estado de salud del gestor"""
        healthy_branches = len(
            [b for b in self.branches.values() if b.status == BranchStatus.ACTIVE]
        )
        total_branches = len(self.branches)

        return {
            "status": "healthy" if healthy_branches > 0 else "warning",
            "total_branches": total_branches,
            "active_branches": len(self.active_branches),
            "healthy_branches": healthy_branches,
            "specialization_enabled": self.enable_specialization,
            "merging_enabled": self.enable_merging,
            "stats": self.get_stats(),
        }

    async def shutdown(self):
        """Cerrar gestor y limpiar recursos"""
        logger.info("Cerrando BranchManager")

        # Descargar todas las ramas activas
        for branch_name in list(self.active_branches):
            await self._unload_branch(branch_name)

        # Cerrar componentes
        if self.specialization_engine:
            await self.specialization_engine.shutdown()

        if self.branch_merger:
            await self.branch_merger.shutdown()

        # Limpiar cache
        self.response_cache.clear()

        # Log estadísticas finales
        final_stats = self.get_stats()
        logger.info(f"Estadísticas finales del BranchManager: {final_stats}")

        logger.info("BranchManager cerrado")
