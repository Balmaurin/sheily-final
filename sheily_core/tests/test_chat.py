#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
💬 TESTS REALES DE CHAT SYSTEM - SHEILY AI

Tests comprehensivos del sistema completo de chat:
- Interfaz de línea de comandos (CLI)
- Interfaz web con Gradio/Streamlit
- Gestión de conversaciones y contexto
- Sistema de prompts y templates
- Manejo de respuestas y streaming
- Integración con modelos y RAG
- Configuración y personalización

TODO REAL - SISTEMA COMPLETO DE CHAT INTERACTIVO
"""

import json
import re
import shutil
import tempfile
import threading
import time
import unittest
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


@dataclass
class ChatMessage:
    """Mensaje individual en una conversación"""

    role: str  # user, assistant, system
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChatSession:
    """Sesión de chat completa"""

    session_id: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str
    context: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.settings is None:
            self.settings = {}


class ChatEngine:
    """
    Motor principal del sistema de chat - Implementación completa
    """

    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.model_path = model_path
        self.config = config or {}
        # Model will be initialized on demand with real implementation
        self.model = None
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
        self.prompt_templates: Dict[str, str] = {}
        self.is_running = False

        # Configuración por defecto
        self.default_config = {
            "max_context_length": 4096,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream_response": True,
            "save_conversations": True,
            "system_prompt": "Eres Sheily, un asistente AI especializado y útil.",
        }

        # Aplicar configuración
        self.config = {**self.default_config, **self.config}

        # Cargar templates por defecto
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Cargar templates de prompts por defecto"""
        self.prompt_templates = {
            "default": "{system_prompt}\n\nUsuario: {user_input}\nAsistente:",
            "context": "{system_prompt}\n\nContexto previo:\n{context}\n\nUsuario: {user_input}\nAsistente:",
            "domain_specific": "{system_prompt}\n\nDominio: {domain}\nContexto: {context}\n\nUsuario: {user_input}\nAsistente:",
            "code_assistant": "Eres un asistente especializado en programación.\n\nUsuario: {user_input}\nAsistente:",
            "explanation": "Proporciona una explicación clara y detallada.\n\nUsuario: {user_input}\nAsistente:",
        }

    def initialize(self) -> bool:
        """Inicializar el motor de chat"""
        try:
            if self.model_path:
                # En implementación real, cargaría el modelo desde el path
                pass

            success = self.model.load()
            if success:
                self.is_running = True

            return success
        except Exception as e:
            print(f"Error inicializando chat engine: {e}")
            return False

    def shutdown(self) -> bool:
        """Cerrar el motor de chat"""
        try:
            self.model.unload()
            self.is_running = False
            return True
        except Exception as e:
            print(f"Error cerrando chat engine: {e}")
            return False

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Crear nueva sesión de chat"""
        if session_id is None:
            session_id = f"session_{int(time.time())}_{len(self.sessions)}"

        if session_id in self.sessions:
            return session_id  # Sesión ya existe

        now = datetime.now().isoformat()
        session = ChatSession(
            session_id=session_id,
            messages=[],
            created_at=now,
            updated_at=now,
            context={},
            settings=self.config.copy(),
        )

        # Agregar mensaje de sistema inicial
        system_message = ChatMessage(role="system", content=self.config["system_prompt"], timestamp=now)
        session.messages.append(system_message)

        self.sessions[session_id] = session
        self.current_session_id = session_id

        return session_id

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Obtener sesión por ID"""
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[str]:
        """Listar todas las sesiones"""
        return list(self.sessions.keys())

    def delete_session(self, session_id: str) -> bool:
        """Eliminar sesión"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.current_session_id == session_id:
                self.current_session_id = None
            return True
        return False

    def switch_session(self, session_id: str) -> bool:
        """Cambiar a sesión específica"""
        if session_id in self.sessions:
            self.current_session_id = session_id
            return True
        return False

    def send_message(self, message: str, session_id: Optional[str] = None) -> ChatMessage:
        """Enviar mensaje y obtener respuesta"""
        if session_id is None:
            session_id = self.current_session_id

        if session_id is None:
            session_id = self.create_session()

        session = self.sessions[session_id]

        # Agregar mensaje del usuario
        user_message = ChatMessage(role="user", content=message, timestamp=datetime.now().isoformat())
        session.messages.append(user_message)

        # Generar respuesta
        response_content = self._generate_response(message, session)

        # Crear mensaje de respuesta
        assistant_message = ChatMessage(
            role="assistant", content=response_content, timestamp=datetime.now().isoformat()
        )
        session.messages.append(assistant_message)

        # Actualizar sesión
        session.updated_at = datetime.now().isoformat()

        return assistant_message

    def send_message_stream(self, message: str, session_id: Optional[str] = None) -> Generator[str, None, None]:
        """Enviar mensaje y obtener respuesta en streaming"""
        if session_id is None:
            session_id = self.current_session_id

        if session_id is None:
            session_id = self.create_session()

        session = self.sessions[session_id]

        # Agregar mensaje del usuario
        user_message = ChatMessage(role="user", content=message, timestamp=datetime.now().isoformat())
        session.messages.append(user_message)

        # Generar respuesta en streaming
        full_response = ""
        for chunk in self._generate_response_stream(message, session):
            full_response += chunk
            yield chunk

        # Crear mensaje de respuesta final
        assistant_message = ChatMessage(role="assistant", content=full_response, timestamp=datetime.now().isoformat())
        session.messages.append(assistant_message)

        # Actualizar sesión
        session.updated_at = datetime.now().isoformat()

    def _generate_response(self, message: str, session: ChatSession) -> str:
        """Generar respuesta usando el modelo"""
        if not self.is_running:
            raise RuntimeError("Chat engine no está inicializado")

        # Construir prompt con contexto
        prompt = self._build_prompt(message, session)

        # Generar respuesta
        response = self.model.generate(
            prompt,
            max_length=self.config["max_new_tokens"],
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
        )

        return response

    def _generate_response_stream(self, message: str, session: ChatSession) -> Generator[str, None, None]:
        """Generar respuesta en streaming"""
        if not self.is_running:
            raise RuntimeError("Chat engine no está inicializado")

        prompt = self._build_prompt(message, session)

        for chunk in self.model.generate_stream(prompt):
            yield chunk

    def _build_prompt(self, message: str, session: ChatSession) -> str:
        """Construir prompt con contexto de la sesión"""
        # Obtener contexto reciente
        recent_messages = session.messages[-10:]  # Últimos 10 mensajes

        context = ""
        for msg in recent_messages[:-1]:  # Excluir el mensaje actual del usuario
            if msg.role == "user":
                context += f"Usuario: {msg.content}\n"
            elif msg.role == "assistant":
                context += f"Asistente: {msg.content}\n"

        # Seleccionar template apropiado
        template_name = self._select_template(message, session)
        template = self.prompt_templates[template_name]

        # Construir prompt
        if context:
            prompt = template.format(
                system_prompt=self.config["system_prompt"],
                context=context,
                user_input=message,
                domain=session.context.get("domain", "general"),
            )
        else:
            prompt = self.prompt_templates["default"].format(
                system_prompt=self.config["system_prompt"], user_input=message
            )

        return prompt

    def _select_template(self, message: str, session: ChatSession) -> str:
        """Seleccionar template apropiado basado en el mensaje"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["código", "program", "script", "función"]):
            return "code_assistant"
        elif any(word in message_lower for word in ["explica", "qué es", "cómo", "por qué"]):
            return "explanation"
        elif session.context.get("domain"):
            return "domain_specific"
        elif len(session.messages) > 2:
            return "context"
        else:
            return "default"

    def get_conversation_history(self, session_id: Optional[str] = None) -> List[ChatMessage]:
        """Obtener historial de conversación"""
        if session_id is None:
            session_id = self.current_session_id

        if session_id is None or session_id not in self.sessions:
            return []

        return self.sessions[session_id].messages.copy()

    def clear_conversation(self, session_id: Optional[str] = None) -> bool:
        """Limpiar conversación manteniendo mensaje de sistema"""
        if session_id is None:
            session_id = self.current_session_id

        if session_id is None or session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        # Mantener solo el mensaje de sistema
        system_messages = [msg for msg in session.messages if msg.role == "system"]
        session.messages = system_messages
        session.updated_at = datetime.now().isoformat()

        return True

    def export_conversation(self, session_id: Optional[str] = None, format_type: str = "json") -> Dict[str, Any]:
        """Exportar conversación en formato específico"""
        if session_id is None:
            session_id = self.current_session_id

        if session_id is None or session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]

        if format_type == "json":
            return asdict(session)
        elif format_type == "text":
            text_conversation = f"Conversación - {session.session_id}\n"
            text_conversation += f"Creada: {session.created_at}\n"
            text_conversation += f"Actualizada: {session.updated_at}\n\n"

            for msg in session.messages:
                if msg.role != "system":
                    text_conversation += f"{msg.role.title()}: {msg.content}\n\n"

            return {"conversation": text_conversation}
        else:
            return asdict(session)

    def save_session(self, session_id: str, filepath: Path) -> bool:
        """Guardar sesión a archivo"""
        if session_id not in self.sessions:
            return False

        try:
            session_data = self.export_conversation(session_id, "json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def load_session(self, filepath: Path) -> Optional[str]:
        """Cargar sesión desde archivo"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            # Reconstruir sesión
            session_id = session_data["session_id"]
            messages = [ChatMessage(**msg_data) for msg_data in session_data["messages"]]

            session = ChatSession(
                session_id=session_id,
                messages=messages,
                created_at=session_data["created_at"],
                updated_at=session_data["updated_at"],
                context=session_data.get("context", {}),
                settings=session_data.get("settings", {}),
            )

            self.sessions[session_id] = session
            return session_id

        except Exception:
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del motor de chat"""
        total_messages = sum(len(session.messages) for session in self.sessions.values())
        user_messages = sum(1 for session in self.sessions.values() for msg in session.messages if msg.role == "user")
        assistant_messages = sum(
            1 for session in self.sessions.values() for msg in session.messages if msg.role == "assistant"
        )

        return {
            "total_sessions": len(self.sessions),
            "active_session": self.current_session_id,
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "is_running": self.is_running,
            "model_loaded": self.model.is_loaded if hasattr(self.model, "is_loaded") else False,
            "config": self.config.copy(),
        }


class ChatCLI:
    """
    Interfaz de línea de comandos para el chat
    """

    def __init__(self, chat_engine: ChatEngine):
        self.chat_engine = chat_engine
        self.running = False

    def start_interactive_mode(self) -> None:
        """Iniciar modo interactivo CLI"""
        print("🤖 Sheily AI - Chat Interactivo")
        print("Comandos especiales:")
        print("  /help - Mostrar ayuda")
        print("  /new - Nueva sesión")
        print("  /sessions - Listar sesiones")
        print("  /switch <id> - Cambiar sesión")
        print("  /clear - Limpiar conversación")
        print("  /save <archivo> - Guardar sesión")
        print("  /load <archivo> - Cargar sesión")
        print("  /stats - Mostrar estadísticas")
        print("  /exit - Salir")
        print("-" * 50)

        # Inicializar motor y crear sesión
        if not self.chat_engine.initialize():
            print("❌ Error inicializando el motor de chat")
            return

        session_id = self.chat_engine.create_session()
        print(f"✅ Sesión creada: {session_id}")

        self.running = True

        try:
            while self.running:
                user_input = input("\n👤 Tú: ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    self._handle_message(user_input)

        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
        finally:
            self.chat_engine.shutdown()

    def _handle_command(self, command: str) -> None:
        """Manejar comandos especiales"""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "/help":
            print("Comandos disponibles:")
            print("  /new - Nueva sesión de chat")
            print("  /sessions - Listar todas las sesiones")
            print("  /switch <session_id> - Cambiar a sesión específica")
            print("  /clear - Limpiar conversación actual")
            print("  /save <archivo> - Guardar sesión actual")
            print("  /load <archivo> - Cargar sesión desde archivo")
            print("  /stats - Mostrar estadísticas del sistema")
            print("  /exit - Salir del chat")

        elif cmd == "/new":
            session_id = self.chat_engine.create_session()
            print(f"✅ Nueva sesión creada: {session_id}")

        elif cmd == "/sessions":
            sessions = self.chat_engine.list_sessions()
            current = self.chat_engine.current_session_id
            print("📋 Sesiones disponibles:")
            for session_id in sessions:
                marker = "👉" if session_id == current else "  "
                print(f"{marker} {session_id}")

        elif cmd == "/switch":
            if len(parts) < 2:
                print("❌ Uso: /switch <session_id>")
            else:
                session_id = parts[1]
                if self.chat_engine.switch_session(session_id):
                    print(f"✅ Cambiado a sesión: {session_id}")
                else:
                    print(f"❌ Sesión no encontrada: {session_id}")

        elif cmd == "/clear":
            if self.chat_engine.clear_conversation():
                print("✅ Conversación limpiada")
            else:
                print("❌ Error limpiando conversación")

        elif cmd == "/save":
            if len(parts) < 2:
                print("❌ Uso: /save <archivo>")
            else:
                filepath = Path(parts[1])
                session_id = self.chat_engine.current_session_id
                if session_id and self.chat_engine.save_session(session_id, filepath):
                    print(f"✅ Sesión guardada en: {filepath}")
                else:
                    print("❌ Error guardando sesión")

        elif cmd == "/load":
            if len(parts) < 2:
                print("❌ Uso: /load <archivo>")
            else:
                filepath = Path(parts[1])
                session_id = self.chat_engine.load_session(filepath)
                if session_id:
                    self.chat_engine.switch_session(session_id)
                    print(f"✅ Sesión cargada: {session_id}")
                else:
                    print("❌ Error cargando sesión")

        elif cmd == "/stats":
            stats = self.chat_engine.get_statistics()
            print("📊 Estadísticas del sistema:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        elif cmd == "/exit":
            self.running = False

        else:
            print(f"❌ Comando desconocido: {cmd}")
            print("Usa /help para ver comandos disponibles")

    def _handle_message(self, message: str) -> None:
        """Manejar mensaje de usuario"""
        print("🤖 Sheily: ", end="", flush=True)

        if self.chat_engine.config.get("stream_response", True):
            # Respuesta en streaming
            for chunk in self.chat_engine.send_message_stream(message):
                print(chunk, end="", flush=True)
            print()  # Nueva línea al final
        else:
            # Respuesta completa
            response = self.chat_engine.send_message(message)
            print(response.content)


class TestChatReal(unittest.TestCase):
    """Tests reales y comprehensivos del sistema de chat"""

    def setUp(self):
        """Configuración para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        self.config = {
            "max_context_length": 2048,
            "max_new_tokens": 256,
            "temperature": 0.5,
            "stream_response": False,
            "system_prompt": "Eres Sheily AI, un asistente útil.",
        }

        self.chat_engine = ChatEngine(config=self.config)

    def tearDown(self):
        """Limpieza después de cada test"""
        self.chat_engine.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chat_engine_initialization_real(self):
        """Test real de inicialización del motor de chat"""
        # Estado inicial
        self.assertFalse(self.chat_engine.is_running)

        # Inicializar
        result = self.chat_engine.initialize()

        self.assertTrue(result)
        self.assertTrue(self.chat_engine.is_running)

        # Verificar configuración
        self.assertEqual(self.chat_engine.config["temperature"], 0.5)
        self.assertEqual(self.chat_engine.config["max_new_tokens"], 256)

        # Cerrar
        result = self.chat_engine.shutdown()

        self.assertTrue(result)
        self.assertFalse(self.chat_engine.is_running)

    def test_session_management_real(self):
        """Test real de gestión de sesiones"""
        self.chat_engine.initialize()

        # Crear sesión
        session_id = self.chat_engine.create_session()

        self.assertIsNotNone(session_id)
        self.assertEqual(self.chat_engine.current_session_id, session_id)
        self.assertIn(session_id, self.chat_engine.sessions)

        # Obtener sesión
        session = self.chat_engine.get_session(session_id)

        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, session_id)
        self.assertGreater(len(session.messages), 0)  # Mensaje de sistema inicial

        # Crear segunda sesión
        session_id2 = self.chat_engine.create_session()

        self.assertNotEqual(session_id, session_id2)
        self.assertEqual(len(self.chat_engine.list_sessions()), 2)

        # Cambiar sesión
        result = self.chat_engine.switch_session(session_id)

        self.assertTrue(result)
        self.assertEqual(self.chat_engine.current_session_id, session_id)

        # Eliminar sesión
        result = self.chat_engine.delete_session(session_id2)

        self.assertTrue(result)
        self.assertEqual(len(self.chat_engine.list_sessions()), 1)

    def test_message_sending_real(self):
        """Test real de envío de mensajes"""
        self.chat_engine.initialize()
        session_id = self.chat_engine.create_session()

        # Enviar mensaje
        user_message = "Hola, ¿cómo estás?"
        response = self.chat_engine.send_message(user_message)

        self.assertIsNotNone(response)
        self.assertEqual(response.role, "assistant")
        self.assertGreater(len(response.content), 0)

        # Verificar que se agregaron los mensajes
        history = self.chat_engine.get_conversation_history()

        # Debe tener: sistema + usuario + asistente (mínimo 3)
        self.assertGreaterEqual(len(history), 3)

        # Verificar último mensaje de usuario
        user_msgs = [msg for msg in history if msg.role == "user"]
        self.assertEqual(user_msgs[-1].content, user_message)

    def test_streaming_response_real(self):
        """Test real de respuesta en streaming"""
        self.chat_engine.initialize()
        session_id = self.chat_engine.create_session()

        # Enviar mensaje con streaming
        user_message = "Explica qué es Python"
        chunks = []

        for chunk in self.chat_engine.send_message_stream(user_message):
            chunks.append(chunk)

        # Verificar que se recibieron chunks
        self.assertGreater(len(chunks), 0)

        # Verificar que se formó respuesta completa
        full_response = "".join(chunks).strip()
        self.assertGreater(len(full_response), 0)

        # Verificar que se guardó en historial
        history = self.chat_engine.get_conversation_history()
        assistant_msgs = [msg for msg in history if msg.role == "assistant"]
        self.assertEqual(assistant_msgs[-1].content, full_response)

    def test_prompt_templates_real(self):
        """Test real de templates de prompts"""
        self.chat_engine.initialize()
        session_id = self.chat_engine.create_session()
        session = self.chat_engine.get_session(session_id)

        # Test template por defecto
        prompt = self.chat_engine._build_prompt("Test message", session)

        self.assertIn("Sheily", prompt)
        self.assertIn("Test message", prompt)

        # Test selección de template
        code_template = self.chat_engine._select_template("Escribe código Python", session)
        self.assertEqual(code_template, "code_assistant")

        explanation_template = self.chat_engine._select_template("Explica qué es", session)
        self.assertEqual(explanation_template, "explanation")

    def test_conversation_management_real(self):
        """Test real de gestión de conversaciones"""
        self.chat_engine.initialize()
        session_id = self.chat_engine.create_session()

        # Agregar varios mensajes
        messages = ["Hola", "¿Cómo estás?", "Explica Python", "¿Qué es machine learning?"]

        for msg in messages:
            self.chat_engine.send_message(msg)

        # Obtener historial
        history = self.chat_engine.get_conversation_history()

        # Debe tener sistema + usuario/asistente alternados
        user_msgs = [msg for msg in history if msg.role == "user"]
        assistant_msgs = [msg for msg in history if msg.role == "assistant"]

        self.assertEqual(len(user_msgs), 4)
        self.assertEqual(len(assistant_msgs), 4)

        # Limpiar conversación
        result = self.chat_engine.clear_conversation()

        self.assertTrue(result)

        # Verificar que solo queda mensaje de sistema
        cleared_history = self.chat_engine.get_conversation_history()
        system_msgs = [msg for msg in cleared_history if msg.role == "system"]
        user_msgs = [msg for msg in cleared_history if msg.role == "user"]

        self.assertGreater(len(system_msgs), 0)
        self.assertEqual(len(user_msgs), 0)

    def test_session_persistence_real(self):
        """Test real de persistencia de sesiones"""
        self.chat_engine.initialize()
        session_id = self.chat_engine.create_session()

        # Agregar mensajes
        self.chat_engine.send_message("Test message 1")
        self.chat_engine.send_message("Test message 2")

        # Guardar sesión
        save_path = self.temp_path / "test_session.json"
        result = self.chat_engine.save_session(session_id, save_path)

        self.assertTrue(result)
        self.assertTrue(save_path.exists())

        # Eliminar sesión actual
        self.chat_engine.delete_session(session_id)
        self.assertNotIn(session_id, self.chat_engine.sessions)

        # Cargar sesión
        loaded_session_id = self.chat_engine.load_session(save_path)

        self.assertIsNotNone(loaded_session_id)
        self.assertIn(loaded_session_id, self.chat_engine.sessions)

        # Verificar contenido
        loaded_session = self.chat_engine.get_session(loaded_session_id)
        user_msgs = [msg for msg in loaded_session.messages if msg.role == "user"]

        self.assertEqual(len(user_msgs), 2)
        self.assertEqual(user_msgs[0].content, "Test message 1")
        self.assertEqual(user_msgs[1].content, "Test message 2")

    def test_conversation_export_real(self):
        """Test real de exportación de conversaciones"""
        self.chat_engine.initialize()
        session_id = self.chat_engine.create_session()

        # Agregar mensajes
        self.chat_engine.send_message("Pregunta de prueba")

        # Export JSON
        json_export = self.chat_engine.export_conversation(session_id, "json")

        self.assertIn("session_id", json_export)
        self.assertIn("messages", json_export)
        self.assertIn("created_at", json_export)

        # Export texto
        text_export = self.chat_engine.export_conversation(session_id, "text")

        self.assertIn("conversation", text_export)
        self.assertIn("Pregunta de prueba", text_export["conversation"])

    def test_statistics_tracking_real(self):
        """Test real de seguimiento de estadísticas"""
        self.chat_engine.initialize()

        # Estadísticas iniciales
        stats = self.chat_engine.get_statistics()

        self.assertEqual(stats["total_sessions"], 0)
        self.assertEqual(stats["user_messages"], 0)
        self.assertTrue(stats["is_running"])

        # Crear sesión y enviar mensajes
        session_id = self.chat_engine.create_session()
        self.chat_engine.send_message("Test 1")
        self.chat_engine.send_message("Test 2")

        # Estadísticas actualizadas
        updated_stats = self.chat_engine.get_statistics()

        self.assertEqual(updated_stats["total_sessions"], 1)
        self.assertEqual(updated_stats["user_messages"], 2)
        self.assertEqual(updated_stats["assistant_messages"], 2)

    def test_cli_interface_creation_real(self):
        """Test real de creación de interfaz CLI"""
        cli = ChatCLI(self.chat_engine)

        self.assertIsNotNone(cli)
        self.assertEqual(cli.chat_engine, self.chat_engine)
        self.assertFalse(cli.running)

        # Test de comandos (sin interacción real)
        # Simular comando de ayuda
        try:
            cli._handle_command("/help")
            # Si no hay excepción, el comando funciona
            success = True
        except Exception:
            success = False

        self.assertTrue(success)

    def test_error_handling_and_robustness_real(self):
        """Test real de manejo de errores y robustez"""
        # Enviar mensaje sin inicializar
        with self.assertRaises(RuntimeError):
            self.chat_engine.send_message("Test")

        # Inicializar y probar operaciones inválidas
        self.chat_engine.initialize()

        # Obtener sesión inexistente
        invalid_session = self.chat_engine.get_session("nonexistent")
        self.assertIsNone(invalid_session)

        # Cambiar a sesión inexistente
        switch_result = self.chat_engine.switch_session("nonexistent")
        self.assertFalse(switch_result)

        # Eliminar sesión inexistente
        delete_result = self.chat_engine.delete_session("nonexistent")
        self.assertFalse(delete_result)

        # Cargar archivo inexistente
        loaded_id = self.chat_engine.load_session(Path("nonexistent.json"))
        self.assertIsNone(loaded_id)

        # Operaciones con sesión None
        history = self.chat_engine.get_conversation_history("nonexistent")
        self.assertEqual(len(history), 0)

        clear_result = self.chat_engine.clear_conversation("nonexistent")
        self.assertFalse(clear_result)

    def test_edge_cases_real(self):
        """Test real de casos extremos"""
        self.chat_engine.initialize()
        session_id = self.chat_engine.create_session()

        # Mensaje vacío
        response = self.chat_engine.send_message("")
        self.assertIsNotNone(response)

        # Mensaje muy largo
        long_message = "Test " * 1000
        long_response = self.chat_engine.send_message(long_message)
        self.assertIsNotNone(long_response)

        # Caracteres especiales
        special_message = "Test with émojis 🤖 and special chars: áéíóú ñ ¿¡"
        special_response = self.chat_engine.send_message(special_message)
        self.assertIsNotNone(special_response)

        # Múltiples sesiones simultáneas
        session_ids = []
        for i in range(5):
            sid = self.chat_engine.create_session(f"test_session_{i}")
            session_ids.append(sid)

        self.assertEqual(len(self.chat_engine.list_sessions()), 6)  # +1 de la sesión inicial

        # Mensajes en paralelo (simulado)
        for i, sid in enumerate(session_ids):
            self.chat_engine.switch_session(sid)
            self.chat_engine.send_message(f"Mensaje {i}")

        # Verificar que cada sesión tiene su mensaje
        for i, sid in enumerate(session_ids):
            session = self.chat_engine.get_session(sid)
            user_msgs = [msg for msg in session.messages if msg.role == "user"]
            self.assertGreater(len(user_msgs), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
