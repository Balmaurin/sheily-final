#!/usr/bin/env python3
"""
Configuración centralizada de imports para tests
Resuelve problemas de importación y path management
"""

import os
import sys
from pathlib import Path

# Configurar path del proyecto
def setup_test_environment():
    """Configura el entorno de testing con paths correctos"""
    
    # Obtener directorio raíz del proyecto
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    
    # Agregar rutas al sys.path si no están presentes
    paths_to_add = [
        str(project_root),
        str(project_root / "sheily_core"),
        str(project_root / "sheily_train"),
        str(project_root / "tests"),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Configurar variables de entorno para testing
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("SECRET_KEY", "test_secret_key_for_testing_12345678901234567890")
    os.environ.setdefault("DEBUG", "False")
    os.environ.setdefault("LOG_LEVEL", "WARNING")  # Reducir logs en tests
    
    return project_root

# Configurar automáticamente al importar
PROJECT_ROOT = setup_test_environment()


def safe_import(module_name, fallback_class=None):
    """
    Importa un módulo de forma segura con fallback
    
    Args:
        module_name: Nombre del módulo a importar
        fallback_class: Clase mock de fallback si el import falla
    
    Returns:
        El módulo importado o la clase fallback
    """
    try:
        parts = module_name.split('.')
        if len(parts) == 1:
            return __import__(module_name)
        
        module = __import__(module_name, fromlist=[parts[-1]])
        return getattr(module, parts[-1]) if hasattr(module, parts[-1]) else module
        
    except ImportError as e:
        if fallback_class:
            return fallback_class
        
        # Crear mock dinámico basado en el nombre del módulo
        class MockClass:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                
            def __call__(self, *args, **kwargs):
                return MockClass(*args, **kwargs)
                
            def __getattr__(self, name):
                return MockClass()
        
        return MockClass


def get_test_config():
    """Configuración estándar para tests"""
    return {
        "model_name": "test-model",
        "device": "cpu",
        "max_length": 512,
        "batch_size": 2,
        "learning_rate": 0.001,
        "debug": True,
        "testing": True
    }


def create_temp_file(content="", suffix=".tmp"):
    """Crea archivo temporal para tests"""
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name


def cleanup_temp_files(*file_paths):
    """Limpia archivos temporales"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except (OSError, PermissionError):
            pass  # Ignorar errores de cleanup


# Mocks comunes para módulos que podrían no estar disponibles
class MockLogger:
    """Mock para el módulo Logger"""
    def __init__(self, name="test_logger", level=20):
        self.name = name
        self.level = level
    
    def info(self, message): pass
    def error(self, message): pass
    def warning(self, message): pass
    def debug(self, message): pass


class MockConfig:
    """Mock para el módulo Config"""
    def __init__(self, **kwargs):
        self.data = kwargs or get_test_config()
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def set(self, key, value):
        self.data[key] = value
    
    def save(self): return True
    def load(self): return True


class MockHealth:
    """Mock para el módulo Health"""
    def __init__(self):
        self.status = "healthy"
        self.checks = {}
    
    def add_check(self, name, status, details=None):
        self.checks[name] = {"status": status, "details": details or {}}
    
    def is_healthy(self):
        return self.status == "healthy"


# Importaciones seguras predefinidas
def import_sheily_modules():
    """Importa módulos de Sheily de forma segura"""
    modules = {}
    
    # Logger
    try:
        from sheily_core.logger import Logger
        modules['Logger'] = Logger
    except ImportError:
        modules['Logger'] = MockLogger
    
    # Config
    try:
        from sheily_core.config import Config
        modules['Config'] = Config
    except ImportError:
        modules['Config'] = MockConfig
    
    # Health
    try:
        from sheily_core.health import HealthChecker, SystemHealth
        modules['HealthChecker'] = HealthChecker
        modules['SystemHealth'] = SystemHealth
    except ImportError:
        modules['HealthChecker'] = MockHealth
        modules['SystemHealth'] = MockHealth
    
    return modules


# Auto-setup cuando se importa este módulo
SHEILY_MODULES = import_sheily_modules()

# Exportar para uso fácil en tests
Logger = SHEILY_MODULES['Logger']
Config = SHEILY_MODULES['Config'] 
HealthChecker = SHEILY_MODULES['HealthChecker']
SystemHealth = SHEILY_MODULES['SystemHealth']

__all__ = [
    'setup_test_environment',
    'safe_import', 
    'get_test_config',
    'create_temp_file',
    'cleanup_temp_files',
    'PROJECT_ROOT',
    'Logger',
    'Config',
    'HealthChecker', 
    'SystemHealth',
    'SHEILY_MODULES'
]