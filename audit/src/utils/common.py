"""
Utilidades de Logging para el Sistema de Auditoría Sheily

Proporciona funcionalidades comunes de logging, formateo y manejo de logs
para todos los módulos del sistema.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class SheilyLogger:
    """Sistema de logging unificado para el proyecto Sheily"""

    def __init__(self, module_name: str, log_dir: str = "logs"):
        self.module_name = module_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, action: str, details: Optional[Dict[str, Any]] = None, level: str = "INFO"):
        """Registrar una acción en el log"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "module": self.module_name,
            "level": level,
            "action": action,
            "details": details or {},
        }

        # Crear archivo de log específico del módulo
        log_file = self.log_dir / f"{self.module_name}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # También imprimir en consola con formato amigable
        print(f"[{level}] {self.module_name}: {action}")
        if details:
            print(f"      Detalles: {json.dumps(details, indent=2, ensure_ascii=False)}")

    def info(self, action: str, details: Optional[Dict[str, Any]] = None):
        """Log de información"""
        self.log(action, details, "INFO")

    def warning(self, action: str, details: Optional[Dict[str, Any]] = None):
        """Log de advertencia"""
        self.log(action, details, "WARNING")

    def error(self, action: str, details: Optional[Dict[str, Any]] = None):
        """Log de error"""
        self.log(action, details, "ERROR")

    def success(self, action: str, details: Optional[Dict[str, Any]] = None):
        """Log de éxito"""
        self.log(action, details, "SUCCESS")


class FileManager:
    """Utilidades para manejo de archivos comunes"""

    @staticmethod
    def ensure_directory(path: Path):
        """Asegurar que un directorio existe"""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_copy(src: Path, dst: Path):
        """Copiar archivo de forma segura"""
        try:
            import shutil

            shutil.copy2(src, dst)
            return True
        except Exception as e:
            print(f"Error copiando {src} a {dst}: {e}")
            return False

    @staticmethod
    def get_file_size_mb(path: Path) -> float:
        """Obtener tamaño de archivo en MB"""
        if not path.exists():
            return 0.0
        return path.stat().st_size / (1024 * 1024)


class DataValidator:
    """Utilidades para validación de datos"""

    @staticmethod
    def is_valid_jsonl(file_path: Path) -> bool:
        """Validar que un archivo JSONL tenga formato correcto"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        json.loads(line)
            return True
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error en línea {i} de {file_path}: {e}")
            return False
        except Exception as e:
            print(f"Error leyendo {file_path}: {e}")
            return False
