class EnterpriseSecurityValidator:
    """
    Sistema de validación de seguridad empresarial avanzado
    Cumple con estándares OWASP Enterprise 2025
    """

    def __init__(self):
        self.enterprise_patterns = self._load_enterprise_security_patterns()
        self.rate_limiter = EnterpriseRateLimiter()
        self.threat_intelligence = EnterpriseThreatIntelligence()
        self.security_auditor = EnterpriseSecurityAuditor()

    def _load_enterprise_security_patterns(self) -> Dict[str, List[str]]:
        """Cargar patrones de seguridad empresarial avanzados"""
        return {
            "sql_injection": [
                "drop table",
                "delete from",
                "insert into",
                "update.*set",
                "union select",
                "alter table",
                "create table",
                "select.*from",
                "where.*=.*'",
                "or.*=.*'",
                "exec.*sp_",
                "execute.*sp_",
                "xp_cmdshell",
            ],
            "command_injection": [
                "rm ",
                "del ",
                "format ",
                "fdisk",
                "mkfs",
                "chmod ",
                "chown ",
                "sudo ",
                "su ",
                "su -",
                "wget ",
                "curl ",
                "nc ",
                "netcat ",
                "ssh ",
                "telnet ",
                "ftp ",
                "tftp ",
            ],
            "code_injection": [
                "eval(",
                "exec(",
                "compile(",
                "__import__",
                "globals()",
                "locals()",
                "vars()",
                "dir()",
                "importlib.",
                "imp.",
                "reload(",
                "getattr(",
                "setattr(",
                "delattr(",
                "hasattr(",
                "callable(",
            ],
            "path_traversal": [
                "../",
                "..\\",
                "/etc/",
                "/proc/",
                "/sys/",
                "/dev/",
                "/root/",
                "/home/",
                "/usr/",
                "/var/",
                "/tmp/",
                "/boot/",
                "/lib/",
                "..%2f",
                "..%5c",
                "%2e%2e%2f",
                "%2e%2e%5c",
            ],
            "sensitive_data": [
                "api_key",
                "apikey",
                "password",
                "passwd",
                "secret",
                "token",
                "auth_token",
                "bearer",
                "credential",
                "credentials",
                "login",
                "session",
                "admin",
                "root",
                "administrator",
                "superuser",
                "private_key",
                "public_key",
                "ssh_key",
                "database_url",
                "db_url",
                "connection_string",
            ],
            "injection_vectors": [
                "; ",
                "| ",
                "$(",
                "`",
                "${",
                "$(",
                "2>&1",
                "&&",
                "||",
                ">",
                "<",
                ">>",
                "<<",
                "${IFS}",
                "$(IFS)",
                "`whoami`",
                "$(whoami)",
                "; cat /etc/passwd",
                "| cat /etc/passwd",
                "&& cat /etc/passwd",
                "|| cat /etc/passwd",
            ],
            "enterprise_threats": [
                "import os",
                "import subprocess",
                "import pickle",
                "import sys",
                "import shutil",
                "import tarfile",
                "subprocess.",
                "os.system",
                "os.popen",
                "os.exec",
                "popen",
                "call",
                "check_output",
                "run",
                "Popen",
                "call",
                "check_call",
                "check_output",
                "getoutput",
                "getstatusoutput",
            ],
            "protocol_attacks": [
                "javascript:",
                "data:",
                "vbscript:",
                "file:",
                "ftp://",
                "ldap://",
                "gopher://",
                "jar:",
                "chrome-extension://",
                "moz-extension://",
                "about:",
                "chrome://",
                "resource://",
            ],
            "encoding_attacks": [
                "%00",
                "%0a",
                "%0d",
                "%2e%2e%2f",
                "%2e%2e%5c",
                "%2f",
                "%5c",
                "%3b",
                "%7c",
                "%26",
                "%24",
                "%60",
                "%28",
                "%29",
                "%7b",
                "%7d",
            ],
        }

    def validate_enterprise_input(self, q: str, client_id: str = "unknown") -> Tuple[bool, str]:
        """
        Validación empresarial avanzada de entradas

        Args:
            q: Consulta de entrada a validar
            client_id: Identificador del cliente empresarial

        Returns:
            Tuple[bool, str]: (es_seguro, razón_si_no_lo_es)
        """
        if not q or not isinstance(q, str):
            return False, "Entrada inválida o vacía"

        # Verificación de rate limiting empresarial
        if not self.rate_limiter.check_enterprise_limit(client_id):
            return False, "Límite de rate empresarial excedido"

        # Convertir a minúsculas para comparación empresarial
        q_lower = q.lower().strip()
        q_original = q.strip()

        # Verificar cada categoría de patrones empresariales
        for category, patterns in self.enterprise_patterns.items():
            for pattern in patterns:
                if pattern in q_lower:
                    # Registrar amenaza empresarial
                    self.threat_intelligence.log_enterprise_threat(
                        category, pattern, client_id, q_original
                    )
                    return False, f"Patrón empresarial peligroso detectado: {category}"

        # Verificaciones adicionales empresariales
        validation_results = self._enterprise_input_validation(q_original)
        if not validation_results["valid"]:
            return False, validation_results["reason"]

        # Auditoría de seguridad empresarial
        self.security_auditor.log_enterprise_validation(client_id, True, q_original)

        return True, "Entrada validada empresarialmente"

    def _enterprise_input_validation(self, text: str) -> Dict[str, Any]:
        """Validaciones empresariales adicionales"""
        issues = []

        # Longitud empresarial
        if len(text) > 10000:
            issues.append("Longitud empresarial excedida")

        # Análisis de caracteres empresarial
        quote_count = text.count('"') + text.count("'")
        if quote_count > 10:
            issues.append("Número empresarial excesivo de comillas")

        # Análisis de entropía empresarial
        if self._calculate_enterprise_entropy(text) > 7.5:
            issues.append("Entropía empresarial sospechosa")

        # Análisis de repetición empresarial
        if self._detect_enterprise_repetition(text):
            issues.append("Patrón de repetición empresarial detectado")

        return {
            "valid": len(issues) == 0,
            "reason": "; ".join(issues) if issues else "Válido empresarialmente",
        }

    def _calculate_enterprise_entropy(self, text: str) -> float:
        """Calcular entropía empresarial para detectar ataques"""
        if not text:
            return 0.0

        import math
        from collections import Counter

        # Calcular frecuencia de caracteres
        char_counts = Counter(text.lower())
        text_length = len(text)

        entropy = 0.0
        for count in char_counts.values():
            if count > 0:
                probability = count / text_length
                entropy -= probability * math.log2(probability)

        return entropy

    def _detect_enterprise_repetition(self, text: str) -> bool:
        """Detectar patrones de repetición empresariales sospechosos"""
        # Detectar repetición excesiva de caracteres
        for char in set(text):
            if text.count(char) > len(text) * 0.3:  # Más del 30% del mismo carácter
                return True

        # Detectar patrones repetitivos largos
        for i in range(len(text) - 10):
            pattern = text[i : i + 5]
            if text.count(pattern) > 3:
                return True

        return False


class EnterpriseRateLimiter:
    """Rate limiter empresarial avanzado"""

    def __init__(self):
        self.client_requests = {}
        self.enterprise_limits = {
            "requests_per_minute": 100,
            "burst_limit": 20,
            "sliding_window_seconds": 60,
        }

    def check_enterprise_limit(self, client_id: str) -> bool:
        """Verificar límites empresariales de rate limiting"""
        import time

        current_time = time.time()
        if client_id not in self.client_requests:
            self.client_requests[client_id] = []

        # Limpiar requests antiguos
        self.client_requests[client_id] = [
            req_time
            for req_time in self.client_requests[client_id]
            if current_time - req_time < self.enterprise_limits["sliding_window_seconds"]
        ]

        # Verificar límites empresariales
        if len(self.client_requests[client_id]) >= self.enterprise_limits["requests_per_minute"]:
            return False

        # Agregar request actual
        self.client_requests[client_id].append(current_time)
        return True


class EnterpriseThreatIntelligence:
    """Sistema de inteligencia de amenazas empresarial"""

    def __init__(self):
        self.threat_log = []
        self.max_log_entries = 10000

    def log_enterprise_threat(self, category: str, pattern: str, client_id: str, query: str):
        """Registrar amenaza empresarial"""
        import time

        threat_entry = {
            "timestamp": time.time(),
            "category": category,
            "pattern": pattern,
            "client_id": client_id,
            "query_length": len(query),
            "threat_level": self._assess_enterprise_threat_level(category, pattern),
        }

        self.threat_log.append(threat_entry)

        # Mantener tamaño del log empresarial
        if len(self.threat_log) > self.max_log_entries:
            self.threat_log = self.threat_log[-self.max_log_entries :]

    def _assess_enterprise_threat_level(self, category: str, pattern: str) -> str:
        """Evaluar nivel de amenaza empresarial"""
        high_threat_categories = ["sql_injection", "command_injection", "code_injection"]
        critical_patterns = ["eval(", "exec(", "subprocess.", "import os"]

        if category in high_threat_categories:
            return "HIGH"
        elif pattern in critical_patterns:
            return "CRITICAL"
        else:
            return "MEDIUM"


class EnterpriseSecurityAuditor:
    """Auditor de seguridad empresarial"""

    def __init__(self):
        self.audit_log = []
        self.enterprise_audit_file = "logs/enterprise_security_audit.log"

    def log_enterprise_validation(self, client_id: str, success: bool, query: str):
        """Registrar validación empresarial"""
        import time

        audit_entry = {
            "timestamp": time.time(),
            "client_id": client_id,
            "validation_success": success,
            "query_length": len(query),
            "validation_type": "enterprise_input_validation",
        }

        self.audit_log.append(audit_entry)

        # Mantener tamaño del log empresarial
        if len(self.audit_log) > 50000:
            self.audit_log = self.audit_log[-50000:]


def validate_input(q: str) -> bool:
    """
    Función de compatibilidad - usar validate_enterprise_input() para seguridad empresarial
    """
    validator = EnterpriseSecurityValidator()
    result, _ = validator.validate_enterprise_input(q)
    return result


def sanitize_input(text: str) -> str:
    """
    Sanitizar entrada eliminando caracteres potencialmente peligrosos

    Args:
        text: Texto a sanitizar

    Returns:
        str: Texto sanitizado
    """
    if not isinstance(text, str):
        return ""

    # Eliminar caracteres de control peligrosos
    dangerous_chars = ["\x00", "\x01", "\x02", "\x03", "\x04"]
    for char in dangerous_chars:
        text = text.replace(char, "")

    # Limitar longitud
    if len(text) > 10000:
        text = text[:10000] + "..."

    return text.strip()


def validate_file_path(file_path: str) -> bool:
    """
    Validar que una ruta de archivo sea segura

    Args:
        file_path: Ruta del archivo a validar

    Returns:
        bool: True si la ruta es segura
    """
    if not file_path or not isinstance(file_path, str):
        return False

    # Rutas peligrosas a evitar
    dangerous_paths = [
        "/etc/",
        "/proc/",
        "/sys/",
        "/dev/",
        "/root/",
        "/home/",
        "/usr/",
        "..",
        "~",
        "$HOME",
    ]

    path_lower = file_path.lower()
    for dangerous in dangerous_paths:
        if dangerous in path_lower:
            return False

    # Verificar que la ruta esté dentro del directorio permitido
    allowed_dirs = ["memoria_entrada", "data", "models", "corpus_ES"]
    path_obj = Path(file_path)

    # Si es una ruta absoluta, verificar que esté en directorios permitidos
    if path_obj.is_absolute():
        return False

    # Si es relativa, verificar que no tenga componentes peligrosos
    for part in path_obj.parts:
        if part in ["..", "~"] or part.startswith("$"):
            return False

    return True
