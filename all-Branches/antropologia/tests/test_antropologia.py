#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌳 TESTS REALES DE BRANCHES - SHEILY AI

Tests comprehensivos del sistema completo de branches/ramas:
- Gestión y validación de configuraciones
- Sistema de ramas dinámicas y especializadas
- Validación de estructura y contenidos
- Métricas de cobertura de dominios
- Sistema de routing inteligente
- Análisis de consistencia y calidad

TODO REAL - SISTEMA COMPLETO DE BRANCHES MANAGEMENT
"""

import json
import re
import shutil
import tempfile
import unittest
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class BranchDefinition:
    """Definición completa de una rama especializada"""

    id: str
    name: str
    description: str
    keywords: List[str]
    specializations: List[str]
    related_domains: List[str]
    complexity_level: str  # basic, intermediate, advanced, expert
    corpus_path: Optional[str] = None
    config_path: Optional[str] = None
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.corpus_path:
            self.corpus_path = f"corpus_ES/{self.id.replace(' ', '_')}"
        if not self.config_path:
            self.config_path = f"branches/{self.id.replace(' ', '_')}"


class BranchManager:
    """
    Gestor completo y real del sistema de ramas especializadas
    """

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.branches: Dict[str, BranchDefinition] = {}
        self.domain_mapping: Dict[str, List[str]] = {}
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self.statistics: Dict[str, Any] = {}

    def load_base_branches(self, config_path: Optional[Path] = None) -> int:
        """Cargar definiciones base de ramas"""
        if config_path is None:
            # Probar múltiples ubicaciones
            possible_paths = [
                self.base_path / "branches" / "base_branches.json",
                self.base_path / "config" / "branches.yaml",
                self.base_path / "config" / "branches_config.yaml",
            ]
        else:
            possible_paths = [config_path]

        loaded_count = 0

        for path in possible_paths:
            if path.exists():
                try:
                    if path.suffix == ".json":
                        loaded_count += self._load_json_config(path)
                    elif path.suffix in [".yaml", ".yml"]:
                        loaded_count += self._load_yaml_config(path)
                except Exception as e:
                    print(f"Error cargando {path}: {e}")

        self._build_indexes()
        self._update_statistics()
        return loaded_count

    def _load_json_config(self, path: Path) -> int:
        """Cargar configuración desde JSON"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_count = 0

        if isinstance(data, dict):
            if "domains" in data:
                # Formato con dominios agrupados
                for domain_name, domain_info in data["domains"].items():
                    branch = self._create_branch_from_dict(domain_name, domain_info)
                    if branch:
                        self.branches[branch.id] = branch
                        loaded_count += 1
            elif "branches" in data:
                # Formato con ramas explícitas
                for branch_info in data["branches"]:
                    branch = self._create_branch_from_dict(branch_info.get("name", ""), branch_info)
                    if branch:
                        self.branches[branch.id] = branch
                        loaded_count += 1
        elif isinstance(data, list):
            # Lista directa de ramas
            for i, branch_info in enumerate(data):
                branch = self._create_branch_from_dict(branch_info.get("name", f"branch_{i}"), branch_info)
                if branch:
                    self.branches[branch.id] = branch
                    loaded_count += 1

        return loaded_count

    def _load_yaml_config(self, path: Path) -> int:
        """Cargar configuración desde YAML"""
        try:
            import yaml
        except ImportError:
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self._load_json_config_data(data)

    def _load_json_config_data(self, data: Any) -> int:
        """Procesar datos de configuración (común para JSON y YAML)"""
        loaded_count = 0

        if isinstance(data, dict):
            if "branches" in data:
                for branch_info in data["branches"]:
                    branch = self._create_branch_from_dict(branch_info.get("name", ""), branch_info)
                    if branch:
                        self.branches[branch.id] = branch
                        loaded_count += 1
            else:
                # Tratar como diccionario de ramas
                for branch_name, branch_info in data.items():
                    branch = self._create_branch_from_dict(branch_name, branch_info)
                    if branch:
                        self.branches[branch.id] = branch
                        loaded_count += 1

        return loaded_count

    def _create_branch_from_dict(self, name: str, info: Dict[str, Any]) -> Optional[BranchDefinition]:
        """Crear definición de rama desde diccionario"""
        if not name or not isinstance(info, dict):
            return None

        # Extractar información con valores por defecto
        description = info.get("description", f"Rama especializada en {name}")
        keywords = info.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [kw.strip() for kw in keywords.split(",")]

        specializations = info.get("specializations", [])
        if isinstance(specializations, str):
            specializations = [spec.strip() for spec in specializations.split(",")]

        related_domains = info.get("related_domains", info.get("related", []))
        if isinstance(related_domains, str):
            related_domains = [dom.strip() for dom in related_domains.split(",")]

        complexity_level = info.get("complexity", info.get("level", "intermediate"))

        # Crear ID único
        branch_id = name.lower().replace(" ", "_").replace("-", "_")
        branch_id = re.sub(r"[^\w_]", "", branch_id)

        return BranchDefinition(
            id=branch_id,
            name=name,
            description=description,
            keywords=keywords,
            specializations=specializations,
            related_domains=related_domains,
            complexity_level=complexity_level,
            corpus_path=info.get("corpus_path"),
            config_path=info.get("config_path"),
            is_active=info.get("active", info.get("enabled", True)),
            metadata=info.get("metadata", {}),
        )

    def add_branch(self, branch: BranchDefinition) -> bool:
        """Agregar nueva rama"""
        if branch.id in self.branches:
            return False

        self.branches[branch.id] = branch
        self._build_indexes()
        self._update_statistics()
        return True

    def remove_branch(self, branch_id: str) -> bool:
        """Remover rama"""
        if branch_id not in self.branches:
            return False

        del self.branches[branch_id]
        self._build_indexes()
        self._update_statistics()
        return True

    def find_branches_by_keyword(self, keyword: str) -> List[BranchDefinition]:
        """Encontrar ramas por palabra clave"""
        keyword_lower = keyword.lower()
        matching_branch_ids = self.keyword_index.get(keyword_lower, set())
        return [self.branches[branch_id] for branch_id in matching_branch_ids if branch_id in self.branches]

    def find_branches_by_domain(self, domain: str) -> List[BranchDefinition]:
        """Encontrar ramas por dominio"""
        domain_lower = domain.lower()
        matching_branches = []

        for branch in self.branches.values():
            # Buscar en dominios relacionados
            if any(domain_lower in related.lower() for related in branch.related_domains):
                matching_branches.append(branch)
            # Buscar en nombre y descripción
            elif domain_lower in branch.name.lower() or domain_lower in branch.description.lower():
                matching_branches.append(branch)

        return matching_branches

    def get_branch_by_id(self, branch_id: str) -> Optional[BranchDefinition]:
        """Obtener rama por ID"""
        return self.branches.get(branch_id)

    def get_branch_by_name(self, name: str) -> Optional[BranchDefinition]:
        """Obtener rama por nombre"""
        for branch in self.branches.values():
            if branch.name.lower() == name.lower():
                return branch
        return None

    def route_query_to_branch(self, query: str, context: Optional[str] = None) -> List[Tuple[BranchDefinition, float]]:
        """Enrutar consulta a ramas apropiadas con scoring"""
        query_lower = query.lower()
        context_lower = context.lower() if context else ""

        branch_scores = []

        for branch in self.branches.values():
            if not branch.is_active:
                continue

            score = 0.0

            # Score por keywords
            for keyword in branch.keywords:
                if keyword.lower() in query_lower:
                    score += 2.0
                if context and keyword.lower() in context_lower:
                    score += 1.0

            # Score por nombre de rama
            if branch.name.lower() in query_lower:
                score += 3.0

            # Score por especializations
            for spec in branch.specializations:
                if spec.lower() in query_lower:
                    score += 1.5
                if context and spec.lower() in context_lower:
                    score += 0.5

            # Score por dominios relacionados
            for domain in branch.related_domains:
                if domain.lower() in query_lower:
                    score += 1.0

            # Bonus por complejidad apropiada
            complexity_keywords = {
                "basic": ["simple", "básico", "fácil", "principiante"],
                "intermediate": ["intermedio", "medio", "normal"],
                "advanced": ["avanzado", "complejo", "difícil"],
                "expert": ["experto", "profesional", "especializado"],
            }

            for level_keyword in complexity_keywords.get(branch.complexity_level, []):
                if level_keyword in query_lower:
                    score += 0.5

            if score > 0:
                branch_scores.append((branch, score))

        # Ordenar por score descendente
        branch_scores.sort(key=lambda x: x[1], reverse=True)
        return branch_scores

    def validate_branches(self) -> Dict[str, Any]:
        """Validar todas las ramas y su configuración"""
        validation_report = {
            "total_branches": len(self.branches),
            "active_branches": sum(1 for b in self.branches.values() if b.is_active),
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        for branch_id, branch in self.branches.items():
            branch_issues = []
            branch_warnings = []

            # Validar ID
            if not branch.id or not branch.id.strip():
                branch_issues.append("ID vacío")

            # Validar nombre
            if not branch.name or not branch.name.strip():
                branch_issues.append("Nombre vacío")

            # Validar descripción
            if not branch.description or len(branch.description.strip()) < 10:
                branch_warnings.append("Descripción muy corta")

            # Validar keywords
            if not branch.keywords:
                branch_warnings.append("Sin keywords definidas")
            elif len(branch.keywords) < 3:
                branch_warnings.append("Pocas keywords (menos de 3)")

            # Validar paths si existen
            if branch.corpus_path:
                corpus_path = self.base_path / branch.corpus_path
                if not corpus_path.exists():
                    branch_warnings.append(f"Corpus path no existe: {branch.corpus_path}")

            if branch.config_path:
                config_path = self.base_path / branch.config_path
                if not config_path.exists():
                    branch_warnings.append(f"Config path no existe: {branch.config_path}")

            # Validar complejidad
            valid_complexities = ["basic", "intermediate", "advanced", "expert"]
            if branch.complexity_level not in valid_complexities:
                branch_issues.append(f"Nivel de complejidad inválido: {branch.complexity_level}")

            if branch_issues:
                validation_report["issues"].append(
                    {"branch_id": branch_id, "branch_name": branch.name, "issues": branch_issues}
                )

            if branch_warnings:
                validation_report["warnings"].append(
                    {
                        "branch_id": branch_id,
                        "branch_name": branch.name,
                        "warnings": branch_warnings,
                    }
                )

        # Estadísticas de validación
        validation_report["statistics"] = {
            "branches_with_issues": len(validation_report["issues"]),
            "branches_with_warnings": len(validation_report["warnings"]),
            "complexity_distribution": Counter(b.complexity_level for b in self.branches.values()),
            "avg_keywords_per_branch": sum(len(b.keywords) for b in self.branches.values()) / len(self.branches)
            if self.branches
            else 0,
            "total_keywords": len(self.keyword_index),
            "domains_covered": len(set(domain for b in self.branches.values() for domain in b.related_domains)),
        }

        return validation_report

    def analyze_coverage(self) -> Dict[str, Any]:
        """Analizar cobertura de dominios y áreas"""
        all_domains = set()
        all_keywords = set()
        complexity_distribution = Counter()

        for branch in self.branches.values():
            all_domains.update(branch.related_domains)
            all_keywords.update(kw.lower() for kw in branch.keywords)
            complexity_distribution[branch.complexity_level] += 1

        # Detectar gaps potenciales
        common_domains = {
            "programación",
            "medicina",
            "matemáticas",
            "física",
            "química",
            "biología",
            "historia",
            "filosofía",
            "economía",
            "psicología",
            "educación",
            "arte",
            "música",
            "literatura",
            "geografía",
            "astronomía",
            "geología",
            "ecología",
            "política",
            "sociología",
            "antropología",
            "lingüística",
            "arqueología",
        }

        covered_domains = {d.lower() for d in all_domains}
        missing_domains = common_domains - covered_domains

        return {
            "total_domains": len(all_domains),
            "total_keywords": len(all_keywords),
            "complexity_distribution": dict(complexity_distribution),
            "domains_covered": sorted(all_domains),
            "missing_common_domains": sorted(missing_domains),
            "coverage_percentage": len(covered_domains & common_domains) / len(common_domains) * 100,
            "branches_per_complexity": dict(complexity_distribution),
            "avg_domains_per_branch": len(all_domains) / len(self.branches) if self.branches else 0,
        }

    def export_config(self, format_type: str = "json") -> Dict[str, Any]:
        """Exportar configuración de ramas"""
        if format_type == "json":
            return {
                "branches": [asdict(branch) for branch in self.branches.values()],
                "metadata": {
                    "total_branches": len(self.branches),
                    "export_timestamp": datetime.now().isoformat(),
                    "statistics": self.statistics,
                },
            }
        elif format_type == "yaml_compatible":
            return {
                "branches": {
                    branch.id: {
                        "name": branch.name,
                        "description": branch.description,
                        "keywords": branch.keywords,
                        "specializations": branch.specializations,
                        "related_domains": branch.related_domains,
                        "complexity": branch.complexity_level,
                        "active": branch.is_active,
                    }
                    for branch in self.branches.values()
                }
            }
        else:
            return {"error": f"Formato {format_type} no soportado"}

    def save_config(self, path: Path, format_type: str = "json") -> bool:
        """Guardar configuración a archivo"""
        try:
            config_data = self.export_config(format_type)

            with open(path, "w", encoding="utf-8") as f:
                if format_type == "json":
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                elif format_type == "yaml_compatible":
                    try:
                        import yaml

                        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                    except ImportError:
                        return False

            return True
        except Exception:
            return False

    def _build_indexes(self) -> None:
        """Construir índices internos"""
        self.keyword_index.clear()
        self.domain_mapping.clear()

        for branch_id, branch in self.branches.items():
            # Índice de keywords
            for keyword in branch.keywords:
                self.keyword_index[keyword.lower()].add(branch_id)

            # Mapeo de dominios
            for domain in branch.related_domains:
                if domain not in self.domain_mapping:
                    self.domain_mapping[domain] = []
                self.domain_mapping[domain].append(branch_id)

    def _update_statistics(self) -> None:
        """Actualizar estadísticas internas"""
        if not self.branches:
            self.statistics = {}
            return

        active_branches = [b for b in self.branches.values() if b.is_active]

        self.statistics = {
            "total_branches": len(self.branches),
            "active_branches": len(active_branches),
            "complexity_distribution": dict(Counter(b.complexity_level for b in active_branches)),
            "total_keywords": sum(len(b.keywords) for b in active_branches),
            "total_domains": len(set(d for b in active_branches for d in b.related_domains)),
            "avg_keywords_per_branch": sum(len(b.keywords) for b in active_branches) / len(active_branches)
            if active_branches
            else 0,
            "last_updated": datetime.now().isoformat(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas actuales"""
        return self.statistics.copy()


class TestBranchesReal(unittest.TestCase):
    """Tests reales y comprehensivos del sistema de branches"""

    def setUp(self):
        """Configuración para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.branch_manager = BranchManager(self.temp_path)

        # Crear estructura de directorios
        (self.temp_path / "branches").mkdir(exist_ok=True)
        (self.temp_path / "config").mkdir(exist_ok=True)
        (self.temp_path / "corpus_ES").mkdir(exist_ok=True)

        # Datos de prueba realistas
        self.sample_branches_json = {
            "domains": {
                "programación": {
                    "description": "Desarrollo de software y programación",
                    "keywords": [
                        "código",
                        "algoritmos",
                        "desarrollo",
                        "software",
                        "python",
                        "javascript",
                    ],
                    "specializations": ["web development", "data science", "machine learning"],
                    "related_domains": ["matemáticas", "lógica", "algoritmos"],
                    "complexity": "intermediate",
                },
                "medicina": {
                    "description": "Ciencias médicas y de la salud",
                    "keywords": ["salud", "diagnóstico", "tratamiento", "síntomas", "medicina"],
                    "specializations": ["cardiología", "neurología", "pediatría"],
                    "related_domains": ["biología", "química", "anatomía"],
                    "complexity": "expert",
                },
                "matemáticas": {
                    "description": "Ciencias matemáticas y exactas",
                    "keywords": ["números", "ecuaciones", "cálculo", "álgebra", "geometría"],
                    "specializations": ["análisis", "álgebra abstracta", "geometría diferencial"],
                    "related_domains": ["física", "estadística", "lógica"],
                    "complexity": "advanced",
                },
            }
        }

    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_branch_definition_creation_real(self):
        """Test real de creación de definiciones de ramas"""
        branch = BranchDefinition(
            id="test_programming",
            name="Programación",
            description="Rama especializada en programación",
            keywords=["python", "código", "desarrollo"],
            specializations=["web", "data science"],
            related_domains=["matemáticas", "lógica"],
            complexity_level="intermediate",
        )

        self.assertEqual(branch.id, "test_programming")
        self.assertEqual(branch.name, "Programación")
        self.assertTrue(branch.is_active)
        self.assertIsNotNone(branch.metadata)
        self.assertIn("corpus_ES/test_programming", branch.corpus_path)

    def test_json_config_loading_real(self):
        """Test real de carga de configuración JSON"""
        # Crear archivo de configuración
        config_path = self.temp_path / "branches" / "base_branches.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.sample_branches_json, f, indent=2)

        # Cargar configuración
        loaded_count = self.branch_manager.load_base_branches()

        self.assertEqual(loaded_count, 3)
        self.assertEqual(len(self.branch_manager.branches), 3)

        # Verificar rama específica
        prog_branch = self.branch_manager.get_branch_by_id("programación")
        self.assertIsNotNone(prog_branch)
        self.assertEqual(prog_branch.name, "programación")
        self.assertIn("python", prog_branch.keywords)
        self.assertEqual(prog_branch.complexity_level, "intermediate")

    def test_yaml_config_loading_real(self):
        """Test real de carga de configuración YAML"""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML no disponible")

        yaml_config = {
            "branches": {
                "física": {
                    "name": "Física",
                    "description": "Ciencias físicas",
                    "keywords": ["energía", "movimiento", "ondas"],
                    "complexity": "advanced",
                    "active": True,
                }
            }
        }

        config_path = self.temp_path / "config" / "branches.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_config, f, allow_unicode=True)

        loaded_count = self.branch_manager.load_base_branches()

        self.assertGreater(loaded_count, 0)
        fisica_branch = self.branch_manager.get_branch_by_name("Física")
        self.assertIsNotNone(fisica_branch)

    def test_branch_management_operations_real(self):
        """Test real de operaciones de gestión de ramas"""
        # Crear rama
        new_branch = BranchDefinition(
            id="test_branch",
            name="Test Branch",
            description="Branch for testing",
            keywords=["test", "example"],
            specializations=["unit testing"],
            related_domains=["software quality"],
            complexity_level="basic",
        )

        # Agregar rama
        result = self.branch_manager.add_branch(new_branch)
        self.assertTrue(result)
        self.assertIn("test_branch", self.branch_manager.branches)

        # Intentar agregar duplicado
        duplicate_result = self.branch_manager.add_branch(new_branch)
        self.assertFalse(duplicate_result)

        # Obtener rama por ID
        retrieved_branch = self.branch_manager.get_branch_by_id("test_branch")
        self.assertIsNotNone(retrieved_branch)
        self.assertEqual(retrieved_branch.name, "Test Branch")

        # Obtener rama por nombre
        retrieved_by_name = self.branch_manager.get_branch_by_name("Test Branch")
        self.assertIsNotNone(retrieved_by_name)
        self.assertEqual(retrieved_by_name.id, "test_branch")

        # Remover rama
        remove_result = self.branch_manager.remove_branch("test_branch")
        self.assertTrue(remove_result)
        self.assertNotIn("test_branch", self.branch_manager.branches)

        # Intentar remover rama inexistente
        remove_nonexistent = self.branch_manager.remove_branch("nonexistent")
        self.assertFalse(remove_nonexistent)

    def test_keyword_and_domain_search_real(self):
        """Test real de búsqueda por keywords y dominios"""
        # Cargar datos de prueba
        config_path = self.temp_path / "branches" / "base_branches.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.sample_branches_json, f, indent=2)
        self.branch_manager.load_base_branches()

        # Búsqueda por keyword
        python_branches = self.branch_manager.find_branches_by_keyword("python")
        self.assertGreater(len(python_branches), 0)

        found_programming = any(b.id == "programación" for b in python_branches)
        self.assertTrue(found_programming)

        # Búsqueda por dominio
        math_branches = self.branch_manager.find_branches_by_domain("matemáticas")
        self.assertGreater(len(math_branches), 0)

        # Búsqueda case-insensitive
        upper_case_search = self.branch_manager.find_branches_by_keyword("PYTHON")
        self.assertEqual(len(upper_case_search), len(python_branches))

    def test_query_routing_real(self):
        """Test real de enrutamiento de consultas"""
        # Cargar datos de prueba
        config_path = self.temp_path / "branches" / "base_branches.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.sample_branches_json, f, indent=2)
        self.branch_manager.load_base_branches()

        # Test consulta de programación
        query = "¿Cómo programar en Python?"
        results = self.branch_manager.route_query_to_branch(query)

        self.assertGreater(len(results), 0)

        # El resultado más relevante debe ser programación
        top_branch, top_score = results[0]
        self.assertEqual(top_branch.id, "programación")
        self.assertGreater(top_score, 0)

        # Test consulta médica
        medical_query = "¿Cuáles son los síntomas de la gripe?"
        medical_results = self.branch_manager.route_query_to_branch(medical_query)

        self.assertGreater(len(medical_results), 0)
        medical_branch, medical_score = medical_results[0]
        self.assertEqual(medical_branch.id, "medicina")

        # Test con contexto
        context = "Estoy estudiando desarrollo web"
        context_results = self.branch_manager.route_query_to_branch("¿Qué lenguaje usar?", context)

        if context_results:
            context_branch, context_score = context_results[0]
            self.assertEqual(context_branch.id, "programación")

    def test_branch_validation_real(self):
        """Test real de validación de ramas"""
        # Agregar ramas con diferentes problemas
        valid_branch = BranchDefinition(
            id="valid_branch",
            name="Valid Branch",
            description="This is a properly configured branch with sufficient description",
            keywords=["test", "valid", "example", "good"],
            specializations=["testing", "validation"],
            related_domains=["quality assurance", "software development"],
            complexity_level="intermediate",
        )

        invalid_branch = BranchDefinition(
            id="",  # ID vacío
            name="",  # Nombre vacío
            description="Short",  # Descripción muy corta
            keywords=[],  # Sin keywords
            specializations=[],
            related_domains=[],
            complexity_level="invalid_level",  # Nivel inválido
        )

        self.branch_manager.add_branch(valid_branch)
        self.branch_manager.add_branch(invalid_branch)

        validation_report = self.branch_manager.validate_branches()

        self.assertEqual(validation_report["total_branches"], 2)
        self.assertGreater(len(validation_report["issues"]), 0)
        self.assertGreater(len(validation_report["warnings"]), 0)

        # Verificar estadísticas de validación
        self.assertIn("branches_with_issues", validation_report["statistics"])
        self.assertIn("complexity_distribution", validation_report["statistics"])

    def test_coverage_analysis_real(self):
        """Test real de análisis de cobertura"""
        # Cargar datos de prueba
        config_path = self.temp_path / "branches" / "base_branches.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.sample_branches_json, f, indent=2)
        self.branch_manager.load_base_branches()

        coverage_analysis = self.branch_manager.analyze_coverage()

        self.assertGreater(coverage_analysis["total_domains"], 0)
        self.assertGreater(coverage_analysis["total_keywords"], 0)
        self.assertIn("complexity_distribution", coverage_analysis)
        self.assertIn("coverage_percentage", coverage_analysis)
        self.assertIn("missing_common_domains", coverage_analysis)

        # Verificar que detecta dominios cubiertos
        self.assertIn("matemáticas", coverage_analysis["domains_covered"])
        self.assertIn("programación", coverage_analysis["domains_covered"])

        # Verificar porcentaje de cobertura
        self.assertGreaterEqual(coverage_analysis["coverage_percentage"], 0)
        self.assertLessEqual(coverage_analysis["coverage_percentage"], 100)

    def test_config_export_and_save_real(self):
        """Test real de exportación y guardado de configuración"""
        # Cargar y agregar datos
        config_path = self.temp_path / "branches" / "base_branches.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.sample_branches_json, f, indent=2)
        self.branch_manager.load_base_branches()

        # Export JSON
        json_config = self.branch_manager.export_config("json")

        self.assertIn("branches", json_config)
        self.assertIn("metadata", json_config)
        self.assertEqual(len(json_config["branches"]), 3)

        # Export YAML compatible
        yaml_config = self.branch_manager.export_config("yaml_compatible")

        self.assertIn("branches", yaml_config)
        self.assertEqual(len(yaml_config["branches"]), 3)

        # Save to file
        save_path = self.temp_path / "exported_config.json"
        save_result = self.branch_manager.save_config(save_path, "json")

        self.assertTrue(save_result)
        self.assertTrue(save_path.exists())

        # Verificar contenido guardado
        with open(save_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        self.assertIn("branches", saved_data)
        self.assertEqual(len(saved_data["branches"]), 3)

    def test_statistics_tracking_real(self):
        """Test real de seguimiento de estadísticas"""
        # Estadísticas iniciales
        initial_stats = self.branch_manager.get_statistics()
        self.assertEqual(initial_stats, {})

        # Cargar datos y verificar estadísticas
        config_path = self.temp_path / "branches" / "base_branches.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.sample_branches_json, f, indent=2)
        self.branch_manager.load_base_branches()

        stats = self.branch_manager.get_statistics()

        self.assertEqual(stats["total_branches"], 3)
        self.assertEqual(stats["active_branches"], 3)
        self.assertIn("complexity_distribution", stats)
        self.assertGreater(stats["total_keywords"], 0)
        self.assertGreater(stats["total_domains"], 0)
        self.assertIn("last_updated", stats)

    def test_edge_cases_and_robustness_real(self):
        """Test real de casos edge y robustez"""
        # Cargar configuración vacía
        empty_config = {"domains": {}}
        config_path = self.temp_path / "branches" / "empty.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(empty_config, f)

        loaded_count = self.branch_manager.load_base_branches(config_path)
        self.assertEqual(loaded_count, 0)

        # Búsqueda en manager vacío
        empty_search = self.branch_manager.find_branches_by_keyword("test")
        self.assertEqual(len(empty_search), 0)

        # Routing en manager vacío
        empty_routing = self.branch_manager.route_query_to_branch("test query")
        self.assertEqual(len(empty_routing), 0)

        # Validación de manager vacío
        empty_validation = self.branch_manager.validate_branches()
        self.assertEqual(empty_validation["total_branches"], 0)

        # Cobertura de manager vacío
        empty_coverage = self.branch_manager.analyze_coverage()
        self.assertEqual(empty_coverage["total_domains"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
