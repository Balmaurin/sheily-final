#!/usr/bin/env python3
"""
specialized_branches_loader.py - Sistema de carga de ramas especializadas
========================================================================

Carga automática de todas las ramas especializadas de entrenamiento
para el sistema RAG híbrido con modelo GGUF.

FUNCIONALIDADES:
- Carga automática de datasets de todas las ramas
- Indexación especializada por dominio
- Router inteligente para selección de rama
- Embeddings híbridos con modelo entrenado
- Sistema de cache para rápido acceso
"""

import os

# 🛡️ ACTIVACIÓN DEPSWITCH
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheily_core.depswitch import activate_secure

activate_secure()

import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SpecializedBranchesLoader:
    """Cargador de ramas especializadas para RAG híbrido"""

    def __init__(
        self, branches_root: str, corpus_root: str, cache_db: str = "specialized_branches.db"
    ):
        self.branches_root = Path(branches_root)
        self.corpus_root = Path(corpus_root)
        self.cache_db = cache_db

        # Estructura de ramas especializadas
        self.specialized_branches = {}
        self.domain_router = {}
        self.embeddings_cache = {}

        # Base de datos para cache
        self._init_cache_db()

        print(f"🌳 SpecializedBranchesLoader inicializado")
        print(f"   • Ramas: {self.branches_root}")
        print(f"   • Corpus: {self.corpus_root}")
        print(f"   • Cache: {self.cache_db}")

    def _init_cache_db(self):
        """Inicializar base de datos de cache"""
        self.conn = sqlite3.connect(self.cache_db)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS branch_metadata (
                branch_name TEXT PRIMARY KEY,
                domain TEXT,
                dataset_path TEXT,
                config_path TEXT,
                adapter_path TEXT,
                last_updated INTEGER,
                document_count INTEGER,
                specialization_keywords TEXT
            )
        """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS domain_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                branch_name TEXT,
                domain TEXT,
                content TEXT,
                metadata TEXT,
                embedding_vector BLOB,
                specialization_score REAL,
                last_indexed INTEGER
            )
        """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS domain_router (
                domain TEXT PRIMARY KEY,
                keywords TEXT,
                weight REAL,
                branch_priority TEXT,
                routing_rules TEXT
            )
        """
        )

        self.conn.commit()

    def scan_all_branches(self) -> Dict[str, Dict]:
        """Escanear todas las ramas disponibles"""
        print("🔍 Escaneando ramas especializadas...")

        branches_found = {}

        # Escanear directorio de branches
        if self.branches_root.exists():
            for branch_dir in self.branches_root.iterdir():
                if branch_dir.is_dir() and not branch_dir.name.startswith("."):
                    branch_info = self._analyze_branch(branch_dir)
                    if branch_info:
                        branches_found[branch_dir.name] = branch_info

        print(f"✅ Encontradas {len(branches_found)} ramas especializadas")
        return branches_found

    def _analyze_branch(self, branch_path: Path) -> Optional[Dict]:
        """Analizar una rama específica"""
        try:
            branch_info = {
                "name": branch_path.name,
                "path": str(branch_path),
                "domain": branch_path.name.replace("_", " ").replace("-", " "),
                "config": None,
                "dataset": None,
                "adapter": None,
                "specialization_keywords": [],
                "document_count": 0,
            }

            # Buscar archivos de configuración
            config_file = branch_path / "config.json"
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    branch_info["config"] = config

            # Buscar dataset
            dataset_file = branch_path / "dataset.jsonl"
            if dataset_file.exists():
                branch_info["dataset"] = str(dataset_file)
                # Contar documentos en dataset
                with open(dataset_file, "r", encoding="utf-8") as f:
                    branch_info["document_count"] = sum(1 for _ in f)

            # Buscar adapter
            adapter_dir = branch_path / "adapter"
            if adapter_dir.exists():
                branch_info["adapter"] = str(adapter_dir)

            # Extraer keywords especializadas del dominio
            branch_info["specialization_keywords"] = self._extract_domain_keywords(branch_path.name)

            return branch_info

        except Exception as e:
            print(f"⚠️  Error analizando rama {branch_path.name}: {e}")
            return None

    def _extract_domain_keywords(self, branch_name: str) -> List[str]:
        """Extraer keywords especializadas por dominio"""
        domain_keywords = {
            "inteligencia_artificial": [
                "ai",
                "ml",
                "machine learning",
                "deep learning",
                "neural",
                "algoritmo",
                "modelo",
                "entrenamiento",
            ],
            "medicina": [
                "médico",
                "salud",
                "diagnóstico",
                "tratamiento",
                "síntoma",
                "enfermedad",
                "paciente",
                "clínico",
            ],
            "programacion": [
                "código",
                "python",
                "javascript",
                "desarrollo",
                "software",
                "programación",
                "algoritmo",
                "función",
            ],
            "biologia": [
                "biología",
                "célula",
                "gen",
                "proteína",
                "adn",
                "evolución",
                "ecosistema",
                "organismo",
            ],
            "matematicas": [
                "matemática",
                "ecuación",
                "función",
                "cálculo",
                "geometría",
                "algebra",
                "estadística",
                "probabilidad",
            ],
            "fisica": [
                "física",
                "energía",
                "fuerza",
                "velocidad",
                "onda",
                "partícula",
                "cuántico",
                "relatividad",
            ],
            "quimica": [
                "química",
                "elemento",
                "átomo",
                "molécula",
                "reacción",
                "compuesto",
                "enlace",
                "ion",
            ],
            "historia": [
                "historia",
                "histórico",
                "época",
                "civilización",
                "guerra",
                "revolución",
                "cultura",
                "sociedad",
            ],
            "filosofia": [
                "filosofía",
                "ética",
                "moral",
                "existencia",
                "conocimiento",
                "verdad",
                "realidad",
                "conciencia",
            ],
            "psicologia": [
                "psicología",
                "mente",
                "comportamiento",
                "cognitivo",
                "emocional",
                "terapia",
                "diagnóstico",
                "trastorno",
            ],
        }

        # Normalizar nombre de rama
        normalized = branch_name.lower().replace(" ", "_").replace("-", "_")

        # Buscar coincidencias
        keywords = []
        for domain, domain_kw in domain_keywords.items():
            if domain in normalized or any(kw.replace(" ", "_") in normalized for kw in domain_kw):
                keywords.extend(domain_kw)

        # Agregar keywords derivadas del nombre
        words = re.split(r"[_\-\s]+", branch_name.lower())
        keywords.extend([w for w in words if len(w) > 3])

        return list(set(keywords))

    def load_corpus_documents(self) -> Dict[str, List[Dict]]:
        """Cargar documentos del corpus por dominio"""
        print("📚 Cargando documentos del corpus...")

        corpus_docs = defaultdict(list)

        if self.corpus_root.exists():
            for domain_dir in self.corpus_root.iterdir():
                if domain_dir.is_dir() and not domain_dir.name.startswith("."):
                    docs = self._load_domain_documents(domain_dir)
                    if docs:
                        corpus_docs[domain_dir.name] = docs

        total_docs = sum(len(docs) for docs in corpus_docs.values())
        print(f"✅ Cargados {total_docs} documentos en {len(corpus_docs)} dominios")

        return dict(corpus_docs)

    def _load_domain_documents(self, domain_path: Path) -> List[Dict]:
        """Cargar documentos de un dominio específico"""
        documents = []

        try:
            # Buscar archivos de texto
            for file_path in domain_path.rglob("*.txt"):
                if file_path.is_file():
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            doc = {
                                "id": f"{domain_path.name}_{file_path.stem}",
                                "path": str(file_path),
                                "domain": domain_path.name,
                                "content": content,
                                "metadata": {
                                    "file_size": file_path.stat().st_size,
                                    "modified": file_path.stat().st_mtime,
                                    "type": "corpus_document",
                                },
                            }
                            documents.append(doc)

            # Buscar archivos JSONL consolidados
            for jsonl_file in domain_path.glob("*.jsonl"):
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        try:
                            data = json.loads(line.strip())
                            if "text" in data or "content" in data:
                                content = data.get("text", data.get("content", ""))
                                if content.strip():
                                    doc = {
                                        "id": f"{domain_path.name}_{jsonl_file.stem}_{i}",
                                        "path": str(jsonl_file),
                                        "domain": domain_path.name,
                                        "content": content,
                                        "metadata": data,
                                    }
                                    documents.append(doc)
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            print(f"⚠️  Error cargando documentos de {domain_path.name}: {e}")

        return documents

    def build_domain_router(self) -> Dict[str, Dict]:
        """Construir router de dominios inteligente"""
        print("🧠 Construyendo router de dominios...")

        router = {}

        # Cargar información de ramas
        branches = self.scan_all_branches()

        for branch_name, branch_info in branches.items():
            domain = branch_info["domain"]
            keywords = branch_info["specialization_keywords"]

            # Calcular peso basado en documentos disponibles
            weight = min(1.0, branch_info["document_count"] / 1000.0)

            router[domain] = {
                "branch_name": branch_name,
                "keywords": keywords,
                "weight": weight,
                "priority": self._calculate_priority(branch_info),
                "routing_rules": self._generate_routing_rules(keywords),
            }

            # Guardar en base de datos
            self.conn.execute(
                """
                INSERT OR REPLACE INTO domain_router 
                (domain, keywords, weight, branch_priority, routing_rules)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    domain,
                    json.dumps(keywords),
                    weight,
                    json.dumps(router[domain]["priority"]),
                    json.dumps(router[domain]["routing_rules"]),
                ),
            )

        self.conn.commit()
        print(f"✅ Router construido para {len(router)} dominios")
        return router

    def _calculate_priority(self, branch_info: Dict) -> Dict:
        """Calcular prioridad de rama"""
        priority = {
            "base_score": 0.5,
            "document_bonus": min(0.3, branch_info["document_count"] / 1000.0),
            "keyword_bonus": min(0.2, len(branch_info["specialization_keywords"]) / 20.0),
            "config_bonus": 0.1 if branch_info["config"] else 0.0,
            "adapter_bonus": 0.1 if branch_info["adapter"] else 0.0,
        }

        priority["total"] = sum(priority.values())
        return priority

    def _generate_routing_rules(self, keywords: List[str]) -> Dict:
        """Generar reglas de routing"""
        return {
            "exact_match": keywords[:5],  # Top 5 keywords para match exacto
            "partial_match": keywords[5:10],  # Keywords para match parcial
            "semantic_boost": keywords[:3],  # Keywords para boost semántico
            "threshold": 0.3,  # Umbral mínimo de similitud
        }

    def route_query_to_domain(self, query: str) -> Tuple[str, float]:
        """Enrutar consulta a dominio más apropiado"""
        query_lower = query.lower()
        best_domain = "general"
        best_score = 0.0

        # Cargar router desde DB si no está en memoria
        if not self.domain_router:
            cursor = self.conn.execute(
                "SELECT domain, keywords, weight, routing_rules FROM domain_router"
            )
            for row in cursor:
                domain, keywords_json, weight, rules_json = row
                self.domain_router[domain] = {
                    "keywords": json.loads(keywords_json),
                    "weight": weight,
                    "routing_rules": json.loads(rules_json),
                }

        # Evaluar cada dominio
        for domain, router_info in self.domain_router.items():
            score = self._calculate_domain_score(query_lower, router_info)
            if score > best_score:
                best_score = score
                best_domain = domain

        return best_domain, best_score

    def _calculate_domain_score(self, query: str, router_info: Dict) -> float:
        """Calcular score de dominio para una consulta"""
        score = 0.0
        keywords = router_info["keywords"]
        rules = router_info["routing_rules"]
        weight = router_info["weight"]

        # Match exacto
        exact_matches = sum(1 for kw in rules.get("exact_match", []) if kw in query)
        score += exact_matches * 0.4

        # Match parcial
        partial_matches = sum(
            1 for kw in rules.get("partial_match", []) if any(part in query for part in kw.split())
        )
        score += partial_matches * 0.2

        # Boost semántico (keywords importantes)
        semantic_matches = sum(1 for kw in rules.get("semantic_boost", []) if kw in query)
        score += semantic_matches * 0.3

        # Aplicar peso del dominio
        score *= weight

        # Normalizar
        max_possible = (
            len(rules.get("exact_match", [])) * 0.4
            + len(rules.get("partial_match", [])) * 0.2
            + len(rules.get("semantic_boost", [])) * 0.3
        )

        if max_possible > 0:
            score = score / max_possible

        return min(1.0, score)

    def get_specialized_documents(self, domain: str, limit: int = 100) -> List[Dict]:
        """Obtener documentos especializados de un dominio"""
        cursor = self.conn.execute(
            """
            SELECT content, metadata, specialization_score 
            FROM domain_documents 
            WHERE domain = ? OR branch_name LIKE ?
            ORDER BY specialization_score DESC 
            LIMIT ?
        """,
            (domain, f"%{domain}%", limit),
        )

        documents = []
        for row in cursor:
            content, metadata_json, score = row
            doc = {
                "content": content,
                "metadata": json.loads(metadata_json) if metadata_json else {},
                "specialization_score": score,
                "domain": domain,
            }
            documents.append(doc)

        return documents

    def index_all_documents(self):
        """Indexar todos los documentos especializados"""
        print("🔄 Indexando documentos especializados...")

        branches = self.scan_all_branches()
        corpus_docs = self.load_corpus_documents()

        total_indexed = 0

        # Indexar documentos por rama
        for branch_name, branch_info in branches.items():
            if branch_info["dataset"]:
                docs_indexed = self._index_branch_dataset(branch_name, branch_info)
                total_indexed += docs_indexed

        # Indexar documentos del corpus
        for domain, docs in corpus_docs.items():
            docs_indexed = self._index_corpus_domain(domain, docs)
            total_indexed += docs_indexed

        print(f"✅ Indexados {total_indexed} documentos especializados")

    def _index_branch_dataset(self, branch_name: str, branch_info: Dict) -> int:
        """Indexar dataset de una rama específica"""
        indexed_count = 0

        try:
            dataset_path = branch_info["dataset"]
            domain = branch_info["domain"]
            keywords = branch_info["specialization_keywords"]

            with open(dataset_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        content = (
                            data.get("text", data.get("instruction", ""))
                            + " "
                            + data.get("output", "")
                        )

                        if content.strip():
                            # Calcular score de especialización
                            spec_score = self._calculate_specialization_score(content, keywords)

                            # Insertar en base de datos
                            self.conn.execute(
                                """
                                INSERT OR REPLACE INTO domain_documents 
                                (branch_name, domain, content, metadata, specialization_score, last_indexed)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    branch_name,
                                    domain,
                                    content,
                                    json.dumps(data),
                                    spec_score,
                                    int(time.time()),
                                ),
                            )

                            indexed_count += 1

                    except json.JSONDecodeError:
                        continue

            self.conn.commit()

        except Exception as e:
            print(f"⚠️  Error indexando rama {branch_name}: {e}")

        return indexed_count

    def _index_corpus_domain(self, domain: str, documents: List[Dict]) -> int:
        """Indexar documentos de un dominio del corpus"""
        indexed_count = 0

        # Obtener keywords del dominio
        domain_keywords = self._extract_domain_keywords(domain)

        for doc in documents:
            content = doc["content"]
            spec_score = self._calculate_specialization_score(content, domain_keywords)

            self.conn.execute(
                """
                INSERT OR REPLACE INTO domain_documents 
                (branch_name, domain, content, metadata, specialization_score, last_indexed)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    f"corpus_{domain}",
                    domain,
                    content,
                    json.dumps(doc["metadata"]),
                    spec_score,
                    int(time.time()),
                ),
            )

            indexed_count += 1

        self.conn.commit()
        return indexed_count

    def _calculate_specialization_score(self, content: str, keywords: List[str]) -> float:
        """Calcular score de especialización de un documento"""
        content_lower = content.lower()

        # Contar ocurrencias de keywords
        keyword_count = sum(content_lower.count(kw.lower()) for kw in keywords)

        # Normalizar por longitud del contenido
        content_length = len(content.split())
        if content_length == 0:
            return 0.0

        # Score base por densidad de keywords
        density_score = min(1.0, keyword_count / content_length)

        # Bonus por keywords únicas encontradas
        unique_keywords = sum(1 for kw in keywords if kw.lower() in content_lower)
        uniqueness_score = min(1.0, unique_keywords / len(keywords) if keywords else 0)

        # Score combinado
        final_score = (density_score * 0.6) + (uniqueness_score * 0.4)

        return final_score

    def get_stats(self) -> Dict:
        """Obtener estadísticas del sistema"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM domain_documents")
        total_docs = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(DISTINCT domain) FROM domain_documents")
        total_domains = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(DISTINCT branch_name) FROM domain_documents")
        total_branches = cursor.fetchone()[0]

        cursor = self.conn.execute(
            """
            SELECT domain, COUNT(*), AVG(specialization_score) 
            FROM domain_documents 
            GROUP BY domain 
            ORDER BY COUNT(*) DESC
        """
        )

        domain_stats = []
        for row in cursor:
            domain, count, avg_score = row
            domain_stats.append(
                {
                    "domain": domain,
                    "document_count": count,
                    "avg_specialization": round(avg_score, 3),
                }
            )

        return {
            "total_documents": total_docs,
            "total_domains": total_domains,
            "total_branches": total_branches,
            "domain_breakdown": domain_stats,
            "database_file": self.cache_db,
        }


def main():
    """Función principal para testing"""
    loader = SpecializedBranchesLoader(branches_root="branches", corpus_root="corpus_ES")

    print("🔄 Iniciando carga completa del sistema...")

    # Indexar todos los documentos
    loader.index_all_documents()

    # Construir router
    router = loader.build_domain_router()

    # Mostrar estadísticas
    stats = loader.get_stats()
    print("\n📊 ESTADÍSTICAS DEL SISTEMA:")
    print(f"   • Documentos totales: {stats['total_documents']}")
    print(f"   • Dominios: {stats['total_domains']}")
    print(f"   • Ramas: {stats['total_branches']}")

    print("\n🏆 TOP DOMINIOS:")
    for domain_stat in stats["domain_breakdown"][:10]:
        print(
            f"   • {domain_stat['domain']}: {domain_stat['document_count']} docs (score: {domain_stat['avg_specialization']})"
        )

    # Test de routing
    test_queries = [
        "¿Qué es machine learning?",
        "Síntomas de la gripe",
        "Algoritmo de ordenamiento en Python",
        "Guerra Mundial historia",
        "Mecánica cuántica física",
    ]

    print("\n🧪 TEST DE ROUTING:")
    for query in test_queries:
        domain, score = loader.route_query_to_domain(query)
        print(f"   • '{query}' → {domain} (score: {score:.3f})")


if __name__ == "__main__":
    main()
