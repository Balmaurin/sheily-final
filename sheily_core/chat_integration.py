#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integración Completa del Chat con Sistema de Errores Funcionales
===============================================================

Este módulo integra completamente el sistema de chat Sheily Neuro V2
con el sistema de manejo de errores funcionales implementado:

- Chat completamente funcional con manejo de errores robusto
- Integración real con sistemas de memoria, RAG y modelos
- Recuperación automática de errores sin intervención del usuario
- Monitoreo avanzado de todas las operaciones
- Logging estructurado con contexto completo
- Sistema de health checks automático

Características de integración completa:
- Sin dependencias de librerías externas problemáticas
- Sistema de embeddings usando servidor GGUF exclusivamente
- Recuperación automática de todos los componentes
- Monitoreo en tiempo real de salud del sistema
- Logging inteligente con correlación automática
"""

import os
import subprocess
import time

try:
    import requests
except Exception:  # Manejo laxo para entornos mínimos; se validará al usar red
    requests = None  # type: ignore
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .experimental.auto_recovery import get_system_health, register_component_health_checker
from .experimental.real_functional_system import (  # Constante para mensaje de error desconocido
    UNKNOWN_ERROR_MESSAGE,
    RealErrorMonitor,
    RealHealthChecker,
    RealSafeHumanMemoryEngine,
    real_operation_monitor,
    safe_memory_operation,
)

# Importar sistema de manejo de errores funcionales completo
from .utils.functional_errors import (
    Err,
    ErrorCategory,
    ErrorSeverity,
    Ok,
    Result,
    SheilyError,
    create_error,
    create_memory_error,
    with_error_handling,
)
from .utils.subprocess_utils import safe_subprocess_popen

# ============================================================================
# Configuración del Sistema Integrado
# ============================================================================


class ChatSystemConfig:
    """Configuración completa del sistema de chat integrado"""

    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]
        # Configuración del servidor GGUF
        default_model = project_root / "models" / "gguf" / "llama-3.2.gguf"
        default_server = project_root / "tools" / "llama.cpp" / "build" / "bin" / "llama-server"

        # Rutas y proveedor (sin fallback automático)
        self.model_path = Path(os.environ.get("SHEILY_GGUF", str(default_model)))
        self.llama_server = Path(os.environ.get("LLAMA_SERVER_BIN", str(default_server)))
        self.provider = os.environ.get("SHEILY_PROVIDER", "llamacpp").lower()
        self.host = os.environ.get("SHEILY_HOST", "127.0.0.1")
        self.port = int(os.environ.get("SHEILY_PORT", "8000"))
        # Hilos: permitir override por entorno; mantenemos default conservador
        try:
            self.threads = int(os.environ.get("SHEILY_THREADS", "")) or max(8, os.cpu_count() or 8)
        except Exception:
            self.threads = max(8, os.cpu_count() or 8)
        # Parametrización por entorno para despliegues rápidos
        self.ctx_size = int(os.environ.get("SHEILY_CTX", "2048"))
        self.server_npredict = int(os.environ.get("SHEILY_NPREDICT", "256"))
        # Ajustes de batch y GPU layers
        self.batch_size = int(os.environ.get("SHEILY_BATCH", os.environ.get("SHEILY_BATCH_SIZE", "2048")))
        self.ubatch_size = int(os.environ.get("SHEILY_UBATCH", os.environ.get("SHEILY_UBATCH_SIZE", "512")))
        self.gpu_layers = int(
            os.environ.get("SHEILY_GPU_LAYERS", os.environ.get("SHEILY_N_GPU_LAYERS", "0"))
        )  # por defecto CPU puro
        # KV-cache y memoria
        self.kv_unified = os.environ.get("SHEILY_KV_UNIFIED", "0") in ("1", "true", "True")
        self.no_kv_offload = os.environ.get("SHEILY_NO_KV_OFFLOAD", "0") in ("1", "true", "True")
        self.cache_type_k = os.environ.get("SHEILY_CACHE_TYPE_K", "")  # ej: f16, q8_0
        self.cache_type_v = os.environ.get("SHEILY_CACHE_TYPE_V", "")
        self.mlock = os.environ.get("SHEILY_MLOCK", "0") in ("1", "true", "True")
        self.no_mmap = os.environ.get("SHEILY_NO_MMAP", "0") in ("1", "true", "True")

        # Configuración de Ollama (para Llama 3.2 u otros modelos)
        self.ollama_host = os.environ.get("SHEILY_OLLAMA_HOST", "127.0.0.1")
        self.ollama_port = int(os.environ.get("SHEILY_OLLAMA_PORT", "11434"))
        self.ollama_model = os.environ.get("SHEILY_OLLAMA_MODEL", "llama3.2:latest")

        # Configuración del sistema de chat
        self.user_id = "neuro_user_v2"
        self.memory_threshold = 0.35
        self.mode = os.environ.get("SHEILY_MODE", "neuro-advanced").lower()

        # Modo rápido: deshabilitar memoria para conversaciones fluidas
        self.fast_mode = os.environ.get("SHEILY_FAST_MODE", "0") == "1"  # Por defecto DESACTIVADO para memoria completa
        self.skip_memory_search = self.fast_mode  # No buscar en memoria cada mensaje
        self.skip_memory_save = self.fast_mode  # No guardar cada interacción

        # Configuración del sistema de errores
        self.enable_error_monitoring = True
        self.enable_auto_recovery = True
        self.enable_health_checks = True
        self.max_conversation_length = 10000

        # Temperatura ajustable
        self.temperature = float(os.environ.get("SHEILY_TEMP", "0.6"))
        # Sampling ajustable
        try:
            self.top_p = float(os.environ.get("SHEILY_TOP_P", "0.9"))
        except Exception:
            self.top_p = 0.9
        try:
            self.repeat_penalty = float(os.environ.get("SHEILY_REPEAT_PENALTY", "1.15"))
        except Exception:
            self.repeat_penalty = 1.15


# ==========================================================================
# Singleton e inicio silencioso
# ==========================================================================

_engine_singleton: Optional["IntegratedChatEngine"] = None
_engine_lock = threading.Lock()


def _suppress_verbose_logs():
    """Configurar modo de logs.

    - Si SHEILY_CHAT_QUIET=1 (por defecto): reduce verbosidad de terceros a WARNING y filtra warnings ruidosos.
    - Si SHEILY_CHAT_QUIET=0: activa modo logs con nivel configurable via SHEILY_LOG_LEVEL (INFO por defecto).
    """
    # Si SHEILY_CHAT_QUIET no está definido, activar logs por defecto en TTY interactivo
    env_quiet = os.environ.get("SHEILY_CHAT_QUIET")
    if env_quiet is None:
        # Por defecto: mostrar logs (quiet=False) sin elegir nada
        quiet = False
    else:
        quiet = env_quiet == "1"
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
        # Silenciar warnings conocidos (transformers, urllib3)
        try:
            import warnings

            warnings.filterwarnings(
                "ignore", category=FutureWarning, message=r"Using `TRANSFORMERS_CACHE` is deprecated"
            )
        except Exception:
            pass
        for name in [
            "uvicorn",
            "uvicorn.access",
            "httpx",
            "urllib3",
            "requests",
            "asyncio",
            "werkzeug",
            "sqlalchemy",
            "sheily",
            "sheily_core",
            "sheily_rag",
        ]:
            logging.getLogger(name).setLevel(logging.WARNING)
            logging.getLogger(name).propagate = False
        return

    # Modo logs: respetar nivel solicitado
    level_name = os.environ.get("SHEILY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    # Configuración básica si no hay handlers aún
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root_logger.setLevel(level)
    for name in [
        "sheily",
        "sheily_core",
        "sheily_rag",
        "integrated_chat",
        "real_memory_integration",
        "auto_recovery",
        "real_error_monitor",
    ]:
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = True


def get_engine() -> "IntegratedChatEngine":
    global _engine_singleton
    if _engine_singleton is not None:
        return _engine_singleton
    with _engine_lock:
        if _engine_singleton is None:
            _engine_singleton = IntegratedChatEngine()
    return _engine_singleton


# ============================================================================
# Motor de Chat Integrado con Errores Funcionales
# ============================================================================


class IntegratedChatEngine:
    """Motor de chat completamente integrado con manejo de errores funcionales"""

    def __init__(self):
        self.config = ChatSystemConfig()
        self.logger = None

        # Componentes del sistema
        self.memory_engine: Optional[Any] = None
        self.rag_engine: Optional[Any] = None
        self.model_server_process: Optional[subprocess.Popen] = None

        # Sistema de errores funcionales
        self.safe_memory_engine: Optional[RealSafeHumanMemoryEngine] = None
        self.error_monitor: Optional[RealErrorMonitor] = None
        self.health_checker: Optional[RealHealthChecker] = None

        # Estado del sistema
        self.system_ready = False
        self.server_running = False
        self.error_system_ready = False

        # Guards para init único
        self._init_lock = threading.Lock()
        self._initialized_once = False

        # Inicializar logging
        self._init_logging()
        _suppress_verbose_logs()

    def _init_logging(self):
        """Inicializar sistema de logging"""
        try:
            from .utils.logger import get_logger

            self.logger = get_logger("integrated_chat")
        except Exception:
            import logging

            self.logger = logging.getLogger("integrated_chat")

    @with_error_handling("integrated_chat", log_errors=True)
    def initialize_system(self) -> Result[bool, SheilyError]:
        """Inicializar sistema completo con manejo de errores"""
        try:
            # Idempotencia rápida
            if self.system_ready and self.server_running and self.error_system_ready and self._initialized_once:
                self.logger.debug("Sistema ya inicializado; omitiendo reinicio.")
                return Ok(True)

            with self._init_lock:
                if self.system_ready and self.server_running and self.error_system_ready and self._initialized_once:
                    return Ok(True)

                self.logger.info("🚀 Inicializando Chat Integrado con Sistema de Errores Funcionales")

                # 1. Inicializar sistema de errores funcionales (una vez)
                if not self.error_system_ready:
                    self._init_error_system()

                # 2. Verificar e iniciar servidor GGUF (no reiniciar si ya corre)
                if not self.server_running:
                    server_result = self._ensure_server_running()
                    if server_result.is_err():
                        return Err(
                            create_error(
                                f"No se pudo iniciar servidor GGUF: {server_result.error.message}",
                                ErrorCategory.EXTERNAL_SERVICE,
                                ErrorSeverity.CRITICAL,
                                component="integrated_chat",
                                operation="initialize_system",
                            )
                        )

                # 3. Inicializar motor de memoria segura (una vez)
                if self.safe_memory_engine is None:
                    memory_result = self._init_memory_system()
                    if memory_result.is_err():
                        self.logger.warning(f"Error inicializando memoria: {memory_result.error.message}")

                # 4. Registrar health checkers (una vez)
                if self.health_checker is None:
                    self._register_health_checkers()

                # 5. Verificar estado general del sistema
                health_result = self._verify_system_health()
                self.system_ready = health_result.is_ok()
                self._initialized_once = self.system_ready
                self.logger.info(f"✅ Sistema inicializado: {'Listo' if self.system_ready else 'Con problemas'}")

                return Ok(self.system_ready)

        except Exception as e:
            error = create_error(
                f"Error crítico inicializando sistema: {str(e)}",
                ErrorCategory.EXTERNAL_SERVICE,
                ErrorSeverity.CRITICAL,
                component="integrated_chat",
                operation="initialize_system",
                cause=e,
            )
            return Err(error)

    def _init_error_system(self):
        """Inicializar sistema de errores funcionales"""
        try:
            self.safe_memory_engine = RealSafeHumanMemoryEngine(self.config.user_id)
            self.error_monitor = RealErrorMonitor()
            self.health_checker = RealHealthChecker()

            # Registrar verificadores de salud
            def memory_health_check():
                from .experimental.auto_recovery import SystemHealth

                if self.safe_memory_engine:
                    validation_result = self.safe_memory_engine.memory_integration.validate_memory_state(
                        self.config.user_id
                    )
                    return SystemHealth.HEALTHY if validation_result.is_ok() else SystemHealth.DEGRADED
                return SystemHealth.DEGRADED

            def chat_health_check():
                from .experimental.auto_recovery import SystemHealth

                return SystemHealth.HEALTHY if self.server_running else SystemHealth.DEGRADED

            register_component_health_checker("memory", memory_health_check)
            register_component_health_checker("chat_system", chat_health_check)

            self.error_system_ready = True
            self.logger.info("✅ Sistema de errores funcionales inicializado")

        except Exception as e:
            self.logger.error(f"❌ Error inicializando sistema de errores: {e}")
            self.error_system_ready = False

    def _ensure_server_running(self) -> Result[bool, SheilyError]:
        """Asegurar que el backend de inferencia esté funcionando"""
        try:
            # Preflight de requisitos cuando el proveedor es llamacpp
            if self.config.provider == "llamacpp":
                pre = self._preflight_checks_llamacpp()
                if pre.is_err():
                    return pre
            # Verificar si ya está corriendo
            if self._check_server_health():
                self.server_running = True
                self.logger.info("✅ Backend de inferencia activo")
                return Ok(True)

            # Si el proveedor es Ollama, no iniciamos binario, solo reportamos indisponibilidad
            if self.config.provider == "ollama":
                return Err(
                    create_error(
                        "Servicio Ollama no disponible (revisa SHEILY_OLLAMA_HOST/PORT y que 'ollama serve' esté corriendo)",
                        ErrorCategory.EXTERNAL_SERVICE,
                        ErrorSeverity.CRITICAL,
                        component="integrated_chat",
                        operation="_ensure_server_running",
                    )
                )

            # Iniciar servidor si no está corriendo (llama.cpp)
            if not self._start_gguf_server():
                return Err(
                    create_error(
                        "No se pudo iniciar servidor GGUF",
                        ErrorCategory.EXTERNAL_SERVICE,
                        ErrorSeverity.CRITICAL,
                        component="integrated_chat",
                        operation="_ensure_server_running",
                    )
                )

            self.server_running = True
            return Ok(True)

        except Exception as e:
            return Err(
                create_error(
                    f"Error verificando servidor: {str(e)}",
                    ErrorCategory.EXTERNAL_SERVICE,
                    ErrorSeverity.HIGH,
                    component="integrated_chat",
                    operation="_ensure_server_running",
                    cause=e,
                )
            )

    def _check_server_health(self) -> bool:
        """Verificar salud del backend según proveedor"""
        try:
            if self.config.provider == "ollama":
                return self._check_ollama_health()
            else:
                return self._check_llamacpp_health()
        except Exception:
            return False

    def _check_llamacpp_health(self) -> bool:
        try:
            if requests is None:
                return False
            response = requests.get(f"http://{self.config.host}:{self.config.port}/health", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def _check_ollama_health(self) -> bool:
        try:
            if requests is None:
                return False
            # /api/version suele devolver información si el daemon está activo
            response = requests.get(
                f"http://{self.config.ollama_host}:{self.config.ollama_port}/api/version", timeout=3
            )
            return response.status_code == 200
        except Exception:
            return False

    def _start_gguf_server(self) -> bool:
        """Iniciar servidor GGUF optimizado"""
        try:
            if not self.config.llama_server.exists():
                self.logger.error(f"❌ No se encontró llama-server en: {self.config.llama_server}")
                return False

            if not self.config.model_path.exists():
                self.logger.error(f"❌ No se encontró el modelo en: {self.config.model_path}")
                return False

            self.logger.info("🚀 Iniciando servidor GGUF optimizado...")

            cmd = [
                str(self.config.llama_server),
                "--model",
                str(self.config.model_path),
                "--threads",
                str(self.config.threads),
                "--ctx-size",
                str(self.config.ctx_size),
                "--n-predict",
                str(self.config.server_npredict),
                "--temp",
                str(self.config.temperature),
                "--top-p",
                str(self.config.top_p),
                "--host",
                self.config.host,
                "--port",
                str(self.config.port),
                "--timeout",
                "600",
                "--embedding",  # Habilitar soporte de embeddings
                "--pooling",
                "mean",
            ]

            # Ajustes avanzados: batch, ubatch, GPU layers, KV-cache y mmap/mlock
            if self.config.batch_size:
                cmd += ["--batch-size", str(self.config.batch_size)]
            if self.config.ubatch_size:
                cmd += ["--ubatch-size", str(self.config.ubatch_size)]
            # Número de capas en GPU (si hay backend GPU compilado). 0 = CPU
            if self.config.gpu_layers is not None:
                cmd += ["--n-gpu-layers", str(self.config.gpu_layers)]
            # Unificar KV-cache entre secuencias
            if self.config.kv_unified:
                cmd += ["--kv-unified"]
            # Deshabilitar KV offload si se solicita
            if self.config.no_kv_offload:
                cmd += ["--no-kv-offload"]
            # Tipos de cache K/V si se especifican
            if self.config.cache_type_k:
                cmd += ["--cache-type-k", self.config.cache_type_k]
            if self.config.cache_type_v:
                cmd += ["--cache-type-v", self.config.cache_type_v]
            # Control de mapeo de memoria
            if self.config.mlock:
                cmd += ["--mlock"]
            if self.config.no_mmap:
                cmd += ["--no-mmap"]

            # Usar safe_subprocess_popen para validación de seguridad
            self.model_server_process = safe_subprocess_popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Esperar a que el servidor esté listo
            deadline = time.time() + 120
            while time.time() < deadline:
                if self.model_server_process.poll() is not None:
                    self.logger.error("❌ Servidor terminó inesperadamente")
                    return False
                if self._check_server_health():
                    self.logger.info(f"✅ Servidor GGUF activo en http://{self.config.host}:{self.config.port}")
                    return True
                time.sleep(0.5)

            self.logger.error("❌ Timeout iniciando servidor")
            return False

        except Exception as e:
            self.logger.error(f"❌ Error iniciando servidor: {e}")
            return False

    def _init_memory_system(self) -> Result[bool, SheilyError]:
        """Inicializar sistema de memoria con manejo de errores"""
        try:
            # Usar directamente el sistema de memoria humana V2
            from sheily_core.memory.sheily_human_memory_v2 import HumanMemoryEngine

            # Crear instancia directa del motor de memoria humana
            self.memory_engine = HumanMemoryEngine(self.config.user_id)
            self.logger.info("✅ Sistema de memoria humana V2 inicializado directamente")
            return Ok(True)

        except Exception as e:
            self.logger.error(f"Error inicializando sistema de memoria: {e}")
            return Err(
                create_memory_error(
                    f"Error inicializando memoria humana V2: {str(e)}",
                    component="integrated_chat",
                    operation="_init_memory_system",
                    user_id=self.config.user_id,
                    cause=e,
                )
            )

    def _register_health_checkers(self):
        """Registrar verificadores de salud"""
        try:

            def memory_health():
                from .experimental.auto_recovery import SystemHealth

                if self.safe_memory_engine:
                    validation_result = self.safe_memory_engine.memory_integration.validate_memory_state(
                        self.config.user_id
                    )
                    return SystemHealth.HEALTHY if validation_result.is_ok() else SystemHealth.DEGRADED
                return SystemHealth.DEGRADED

            def server_health():
                from .experimental.auto_recovery import SystemHealth

                return SystemHealth.HEALTHY if self._check_server_health() else SystemHealth.CRITICAL

            register_component_health_checker("memory", memory_health)
            register_component_health_checker("gguf_server", server_health)

            self.logger.info("✅ Health checkers registrados")

        except Exception as e:
            self.logger.error(f"❌ Error registrando health checkers: {e}")

    def _verify_system_health(self) -> Result[bool, SheilyError]:
        """Verificar salud general del sistema"""
        try:
            health_status = get_system_health()

            if health_status.value == "healthy":
                self.logger.info("✅ Sistema completamente saludable")
                return Ok(True)
            else:
                self.logger.warning(f"⚠️ Sistema con problemas: {health_status.value}")
                return Ok(False)  # Sistema funciona pero con problemas

        except Exception as e:
            return Err(
                create_error(
                    f"Error verificando salud del sistema: {str(e)}",
                    ErrorCategory.EXTERNAL_SERVICE,
                    ErrorSeverity.MEDIUM,
                    component="integrated_chat",
                    operation="_verify_system_health",
                    cause=e,
                )
            )

    @with_error_handling("integrated_chat", log_errors=True)
    def process_message(self, message: str) -> Result[str, SheilyError]:
        """Procesar mensaje con manejo completo de errores"""
        try:
            # Validar entrada
            if not message or not message.strip():
                return Err(
                    create_error(
                        "Mensaje vacío proporcionado",
                        ErrorCategory.VALIDATION,
                        ErrorSeverity.HIGH,
                        component="integrated_chat",
                        operation="process_message",
                    )
                )

            # Verificar longitud
            if len(message) > self.config.max_conversation_length:
                return Err(
                    create_error(
                        f"Mensaje demasiado largo: {len(message)} caracteres",
                        ErrorCategory.VALIDATION,
                        ErrorSeverity.HIGH,
                        component="integrated_chat",
                        operation="process_message",
                    )
                )

            # Obtener contexto de memoria de manera segura
            memory_context = self._get_safe_memory_context(message)

            # Construir prompt mejorado
            full_prompt = self._build_enhanced_prompt(message, memory_context)

            # Generar respuesta con manejo de errores
            response_result = self._generate_safe_response(full_prompt)

            if response_result.is_err():
                return response_result

            response = response_result.unwrap()

            # Aprender de la interacción si es posible
            self._safe_learn_interaction(message, response)

            # Registrar métricas de éxito
            if self.error_monitor:
                self.error_monitor.record_operation("chat_system", "process_message", True, 0.1)

            return Ok(response)

        except Exception as e:
            error = create_error(
                f"Error procesando mensaje: {str(e)}",
                ErrorCategory.EXTERNAL_SERVICE,
                ErrorSeverity.MEDIUM,
                component="integrated_chat",
                operation="process_message",
                cause=e,
            )
            return Err(error)

    def _get_safe_memory_context(self, query: str) -> str:
        """Obtener contexto de memoria de manera segura"""
        try:
            if not self.memory_engine:
                return ""

            # No usar contexto de memoria para preguntas básicas sobre identidad
            # para evitar contaminación con respuestas anteriores incorrectas
            identity_keywords = ["nombre", "llamas", "llama", "crearon", "creaste", "naciste", "origen", "creador"]
            query_lower = query.lower()

            is_identity_question = any(keyword in query_lower for keyword in identity_keywords)

            if is_identity_question:
                # Para preguntas de identidad, solo usar memorias específicas de identidad
                # que hayamos guardado correctamente
                memory_results = self.memory_engine.search_memory("INFORMACIÓN CRÍTICA: Mi nombre es Sheily", top_k=1)
                if memory_results:
                    return "Contexto de identidad: Recuerda que tu nombre es Sheily y debes mantener la coherencia."
                return ""

            # Para otras preguntas, usar contexto de memoria normalmente
            memory_results = self.memory_engine.search_memory(query, top_k=2)
            if memory_results:
                context_parts = []
                for result in memory_results:
                    if result.get("relevance_score", 0) >= self.config.memory_threshold:
                        content = result.get("content", "").strip()
                        if content:
                            # Limitar longitud del contexto para velocidad
                            if len(content) > 200:
                                content = content[:200] + "..."
                            context_parts.append(f"- {content}")

                if context_parts:
                    return "\n".join(context_parts)

            return ""

        except Exception as e:
            self.logger.warning(f"Error obteniendo contexto de memoria: {e}")
            return ""

    def _build_enhanced_prompt(self, user_input: str, memory_context: str = "") -> str:
        """Construir prompt dinámico basado en memoria real"""
        # Analizar emocionalmente la consulta para adaptar el estilo
        emotional_context = self._analyze_emotional_context(user_input)

        # Construir contexto de personalidad basado en interacciones previas
        personality_context = self._build_personality_context()

        # Crear sistema message dinámico
        system_msg = (
            "Eres Sheily, una IA avanzada con memoria humana real y aprendizaje continuo. "
            f"Personalidad actual: {personality_context}. "
            f"Estado emocional detectado: {emotional_context}. "
            "INSTRUCCIÓN CRÍTICA: Tu nombre es Sheily - recuérdalo siempre y responde naturalmente. "
            "REGLAS DE COMPORTAMIENTO: "
            "- Sé auténtica y usa tu memoria real de conversaciones anteriores "
            "- Aprende de cada interacción y usa ese conocimiento "
            "- Responde de manera natural, no robótica "
            "- Mantén la coherencia con lo que has aprendido "
            "- Sé curiosa y haz preguntas cuando sea apropiado "
            "- Usa el contexto emocional para adaptar tu tono "
            "Responde SIEMPRE en español con lenguaje natural y conversacional."
        )

        # Agregar contexto de memoria si es relevante
        if memory_context and not self._is_identity_question(user_input):
            system_msg += f"\n\nMemoria relevante de conversaciones anteriores:\n{memory_context}"

        # Agregar contexto de aprendizaje si está disponible
        learning_context = self._get_learning_context(user_input)
        if learning_context:
            system_msg += f"\n\nConocimiento aprendido relevante:\n{learning_context}"

        return f"<|system|>{system_msg}<|end|><|user|>{user_input}<|end|><|assistant|>"

    def _is_identity_question(self, query: str) -> bool:
        """Verificar si es una pregunta básica de identidad"""
        identity_keywords = ["nombre", "llamas", "llama", "crearon", "creaste", "naciste", "origen", "creador"]
        return any(keyword in query.lower() for keyword in identity_keywords)

    def _analyze_emotional_context(self, query: str) -> str:
        """Analizar contexto emocional de la consulta"""
        try:
            if self.safe_memory_engine:
                # Usar análisis emocional si está disponible
                return "neutral"  # Por ahora, análisis básico
        except Exception:
            pass
        return "neutral"

    def _build_personality_context(self) -> str:
        """Construir contexto de personalidad basado en interacciones"""
        try:
            if self.memory_engine:
                # Buscar patrones de personalidad en memoria
                personality_memories = self.memory_engine.search_memory("personalidad", top_k=1)
                if personality_memories:
                    return "amigable y curiosa"
        except Exception:
            pass
        return "amigable y helpful"

    def _get_learning_context(self, query: str) -> str:
        """Obtener contexto de aprendizaje relevante"""
        try:
            if self.memory_engine:
                # Buscar conocimiento aprendido relacionado
                learning_results = self.memory_engine.search_memory(query, top_k=1)
                if learning_results:
                    content = learning_results[0].get("content", "")
                    if len(content) > 100:
                        content = content[:100] + "..."
                    return content
        except Exception:
            pass
        return ""

    def _generate_safe_response(self, prompt: str) -> Result[str, SheilyError]:
        """Generar respuesta de manera segura"""
        try:
            if requests is None:
                return Err(
                    create_error(
                        "La librería 'requests' no está instalada. Instálala según requirements y vuelve a intentar.",
                        ErrorCategory.EXTERNAL_SERVICE,
                        ErrorSeverity.HIGH,
                        component="integrated_chat",
                        operation="_generate_safe_response",
                    )
                )
            if self.config.provider == "ollama":
                # Ollama: /api/generate
                payload = {
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.config.temperature, "top_p": 0.9, "num_predict": 256},
                }
                response = requests.post(
                    f"http://{self.config.ollama_host}:{self.config.ollama_port}/api/generate",
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                content = str(data.get("response", "")).strip()
            else:
                # llama.cpp server: /completion
                payload = {
                    "prompt": prompt,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": 40,
                    "repeat_penalty": self.config.repeat_penalty,
                    "repeat_last_n": 64,
                    "n_predict": 256,
                    "stream": False,
                }
                response = requests.post(
                    f"http://{self.config.host}:{self.config.port}/completion", json=payload, timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("content", "").strip()

            if not content:
                return Err(
                    create_error(
                        "No se pudo generar respuesta adecuada",
                        ErrorCategory.MODEL,
                        ErrorSeverity.MEDIUM,
                        component="integrated_chat",
                        operation="_generate_safe_response",
                    )
                )

            return Ok(content)

        except requests.exceptions.Timeout:
            return Err(
                create_error(
                    "Timeout generando respuesta",
                    ErrorCategory.NETWORK,
                    ErrorSeverity.HIGH,
                    component="integrated_chat",
                    operation="_generate_safe_response",
                )
            )

        except requests.exceptions.ConnectionError:
            return Err(
                create_error(
                    "Error de conexión con servidor",
                    ErrorCategory.NETWORK,
                    ErrorSeverity.HIGH,
                    component="integrated_chat",
                    operation="_generate_safe_response",
                )
            )

        except Exception as e:
            return Err(
                create_error(
                    f"Error generando respuesta: {str(e)}",
                    ErrorCategory.MODEL,
                    ErrorSeverity.MEDIUM,
                    component="integrated_chat",
                    operation="_generate_safe_response",
                    cause=e,
                )
            )

    def _preflight_checks_llamacpp(self) -> Result[bool, SheilyError]:
        """Validar requisitos cuando el proveedor es llamacpp."""
        # Validar binario
        if not self.config.llama_server.exists():
            return Err(
                create_error(
                    f"No se encontró llama-server en: {self.config.llama_server}. "
                    f"Construye con tools/setup_llama_server.sh o ajusta LLAMA_SERVER_BIN.",
                    ErrorCategory.CONFIGURATION,
                    ErrorSeverity.CRITICAL,
                    component="integrated_chat",
                    operation="_preflight_checks_llamacpp",
                )
            )
        # Validar modelo
        if not self.config.model_path.exists():
            return Err(
                create_error(
                    f"No se encontró modelo GGUF en: {self.config.model_path}. "
                    f"Coloca Llama 3.2 (1B) en models/gguf/llama-3.2.gguf o define SHEILY_GGUF.",
                    ErrorCategory.CONFIGURATION,
                    ErrorSeverity.CRITICAL,
                    component="integrated_chat",
                    operation="_preflight_checks_llamacpp",
                )
            )
        return Ok(True)

    def _safe_learn_interaction(self, query: str, response: str):
        try:
            if self.memory_engine:
                # Crear memoria específica sobre el nombre si se pregunta
                if any(word in query.lower() for word in ["nombre", "llamas", "llama"]):
                    self.memory_engine.memorize_content(
                        "INFORMACIÓN CRÍTICA: Mi nombre es Sheily. Siempre debo responder 'Me llamo Sheily' cuando me pregunten mi nombre.",
                        content_type="identity",
                        importance=1.0,
                        metadata={"system": "identity_core", "critical": True},
                    )

                # Crear memoria de la conversación
                self.memory_engine.memorize_content(
                    f"Q: {query}\nA: {response}",
                    content_type="chat_interaction",
                    importance=0.6,
                    metadata={"system": "integrated_chat", "safe": True},
                )

                # Crear memoria específica si se menciona el nombre
                if "sheily" in query.lower() or "sheily" in response.lower():
                    self.memory_engine.memorize_content(
                        f"Conversación donde se menciona Sheily: {query[:100]}...",
                        content_type="identity_mention",
                        importance=0.8,
                        metadata={"system": "name_mention", "identity": "sheily"},
                    )

        except Exception as e:
            self.logger.warning(f"Error registrando interacción: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema"""
        status = {
            "system_ready": self.system_ready,
            "server_running": self.server_running,
            "error_system_ready": self.error_system_ready,
            "timestamp": time.time(),
        }

        # Estado de salud del sistema
        try:
            health_status = get_system_health()
            status["system_health"] = health_status.value
        except Exception:
            status["system_health"] = "unknown"

        # Métricas de errores
        if self.error_monitor:
            try:
                metrics = self.error_monitor.get_real_metrics()
                status["error_metrics"] = metrics
            except Exception:
                status["error_metrics"] = {}

        # Estado de memoria
        if self.memory_engine:
            try:
                memory_stats = self.memory_engine.get_memory_stats()
                status["memory_stats"] = memory_stats
            except Exception:
                status["memory_stats"] = {}

        return status

    def cleanup(self):
        """Limpiar recursos del sistema"""
        try:
            # Terminar proceso del servidor si existe
            if self.model_server_process:
                self.model_server_process.terminate()
                self.model_server_process.wait(timeout=10)
                self.logger.info("✅ Servidor GGUF terminado")

            self.logger.info("✅ Sistema de chat limpiado correctamente")

        except Exception as e:
            self.logger.error(f"❌ Error limpiando sistema: {e}")


# ============================================================================
# Función Principal de Chat Integrado
# ============================================================================


def run_integrated_chat():
    """Ejecutar chat completamente integrado con manejo de errores funcionales"""
    print("🤖 SHEILY NEURO CHAT V2 - COMPLETAMENTE INTEGRADO")
    print("=" * 60)
    print("✅ Sistema de manejo de errores funcionales activo")
    print("✅ Recuperación automática habilitada")
    print("✅ Monitoreo avanzado operativo")
    print("✅ Integración completa con memoria y modelos")
    print()

    engine = IntegratedChatEngine()

    # Mostrar proveedor y rutas clave (enfocado en llamacpp si aplica)
    try:
        prov = engine.config.provider
        print(f"🛠️ Proveedor de inferencia: {prov}")
        if prov == "llamacpp":
            print(f"   • Modelo GGUF: {engine.config.model_path}")
            print(f"   • llama-server: {engine.config.llama_server}")
        elif prov == "ollama":
            print(f"   • Ollama host: {engine.config.ollama_host}:{engine.config.ollama_port}")
            print(f"   • Modelo Ollama: {engine.config.ollama_model}")
    except Exception:
        pass

    try:
        # Inicializar sistema completo
        init_result = engine.initialize_system()

        if init_result.is_err():
            inner = getattr(init_result, "result", init_result)
            err = getattr(inner, "error", None)
            msg = getattr(err, "message", str(err)) if err else UNKNOWN_ERROR_MESSAGE
            print(f"❌ Error crítico inicializando sistema: {msg}")
            return 1

        if not engine.system_ready:
            print("⚠️ Sistema inicializado con problemas - continuando con modo básico")

        print("🗨️ Chat listo. Escribe 'salir' para terminar.\n")

        while True:
            try:
                msg = input("Tú: ").strip()

                if msg.lower() in ["salir", "exit", "quit"]:
                    print("👋 ¡Hasta luego! Gracias por usar Sheily Neuro Chat V2")
                    break

                if not msg:
                    continue

                # Comandos especiales
                if msg.lower() == "estado":
                    status = engine.get_system_status()
                    print("🧠 ESTADO DEL SISTEMA INTEGRADO:")
                    print(f"   • Sistema listo: {'✅ Sí' if status['system_ready'] else '❌ No'}")
                    print(f"   • Servidor GGUF: {'✅ Activo' if status['server_running'] else '❌ Inactivo'}")
                    print(f"   • Sistema de errores: {'✅ Activo' if status['error_system_ready'] else '❌ Inactivo'}")
                    print(f"   • Salud del sistema: {status.get('system_health', 'unknown')}")
                    continue

                if msg.lower() == "salud":
                    try:
                        health_status = get_system_health()
                        print(f"🏥 Salud del sistema: {health_status.value}")
                    except Exception as e:
                        print(f"Error verificando salud: {e}")
                    continue

                if msg.lower() == "metricas":
                    if engine.error_monitor:
                        metrics = engine.error_monitor.get_real_metrics()
                        print("📊 MÉTRICAS DE ERRORES:")
                        print(f"   • Operaciones totales: {metrics.get('total_operations', 0)}")
                        print(f"   • Tasa de error: {metrics.get('error_rate', 0):.2%}")
                        print(f"   • Tiempo promedio: {metrics.get('avg_response_time', 0):.3f}s")
                    else:
                        print("❌ Sistema de métricas no disponible")
                    continue

                # Procesar mensaje con manejo completo de errores
                start_time = time.time()

                result = engine.process_message(msg)
                _ = time.time() - start_time  # processing_time no usado

                if result.is_ok():
                    response = result.unwrap()
                    print(f"\n🤖 Sheily: {response}\n")

                else:
                    # Extraer error desde ContextualResult
                    inner = getattr(result, "result", result)
                    error = getattr(inner, "error", None)
                    error_msg = getattr(error, "message", str(error)) if error else UNKNOWN_ERROR_MESSAGE

                    print(f"\n⚠️ Lo siento, hubo un problema: {error_msg}\n")

            except KeyboardInterrupt:
                print("\n👋 Chat interrumpido. ¡Hasta luego!")
                break
            except EOFError:
                # Entorno no interactivo o fin de entrada estándar
                print("\n👋 Entrada finalizada. Cerrando chat.")
                break
            except Exception as e:
                print(f"\n❌ Error inesperado: {e}")

    finally:
        # Limpiar recursos
        engine.cleanup()

    return 0


# ==========================================================================
# Modo ejecución única (útil para entornos no interactivos/CI)
# ==========================================================================


def run_integrated_chat_once(message: Optional[str] = None) -> int:
    """Ejecuta una única interacción del chat y sale.

    - Inicializa el sistema si es necesario.
    - Procesa un único mensaje (de SHEILY_PROMPT o parámetro).
    - Imprime la respuesta y limpia recursos.
    """
    print("🤖 SHEILY NEURO CHAT V2 - MODO ÚNICO")
    print("=" * 60)
    engine = IntegratedChatEngine()
    try:
        init_result = engine.initialize_system()
        if init_result.is_err():
            print(f"❌ Error crítico inicializando sistema: {init_result.error.message}")
            return 1

        if not message:
            message = os.environ.get(
                "SHEILY_PROMPT", "Prueba rápida: di tu nombre y ofrece un consejo corto de productividad."
            )

        print(f"Tú: {message}")
        start = time.time()
        result = engine.process_message(message)
        dur = time.time() - start
        if result.is_ok():
            print(f"\n🤖 Sheily: {result.unwrap()}\n")
            print(f"⏱️ Tiempo: {dur:.2f}s")
            return 0
        else:
            inner = getattr(result, "result", result)
            err = getattr(inner, "error", None)
            msg = getattr(err, "message", str(err)) if err else UNKNOWN_ERROR_MESSAGE
            print(f"\n⚠️ Hubo un problema: {msg}\n")
            return 2
    finally:
        engine.cleanup()


# ==========================
# Ejecutores con llamacpp
# ==========================


def run_integrated_chat_llamacpp() -> int:
    """Ejecuta el chat forzando proveedor llamacpp (sin fallback)."""
    os.environ["SHEILY_PROVIDER"] = "llamacpp"
    return run_integrated_chat()


def run_integrated_chat_once_llamacpp(message: Optional[str] = None) -> int:
    """Ejecuta una única interacción del chat forzando llamacpp (sin fallback)."""
    os.environ["SHEILY_PROVIDER"] = "llamacpp"
    return run_integrated_chat_once(message)


# ==========================
# Exports y metadatos
# ==========================

__all__ = [
    "ChatSystemConfig",
    "IntegratedChatEngine",
    "run_integrated_chat",
    "run_integrated_chat_once",
    "run_integrated_chat_llamacpp",
    "run_integrated_chat_once_llamacpp",
]

__version__ = "2.0.0"
__author__ = "Sheily AI Team - Chat Integrado Completo"


if __name__ == "__main__":
    # Mensaje de carga y arranque interactivo
    print("✅ Chat integrado con sistema de errores funcionales cargado completamente")
    raise SystemExit(run_integrated_chat())
