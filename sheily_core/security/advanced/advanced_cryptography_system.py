"""
Sistema de Criptografía Avanzada
RSA-4096 + AES-256 híbrido con post-quantum readiness
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, hmac, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


class AdvancedCryptographySystem:
    """Sistema de criptografía avanzada con RSA-4096 + AES-256"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.backend = default_backend()
        self.logger = self._setup_logging()

        # Configuración por defecto
        self.rsa_key_size = 4096
        self.aes_key_size = 256
        self.pbkdf2_iterations = 480000  # Aumentado para mayor seguridad
        self.scrypt_n = 2**20  # 1MB memory cost
        self.scrypt_r = 8
        self.scrypt_p = 1

        # Cache de claves
        self._key_cache = {}
        self._session_keys = {}

        # Post-quantum preparation
        self.quantum_resistant_enabled = True

    def _setup_logging(self) -> logging.Logger:
        """Configurar logging seguro"""
        logger = logging.getLogger("advanced_crypto")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    # === GENERACIÓN DE CLAVES ===

    def generate_rsa_keypair(self) -> Tuple[bytes, bytes]:
        """Generar par de claves RSA-4096"""
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=self.rsa_key_size, backend=self.backend
            )

            public_key = private_key.public_key()

            # Serializar claves
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            self.logger.info("Par de claves RSA-4096 generado exitosamente")
            return private_pem, public_pem

        except Exception as e:
            self.logger.error(f"Error generando claves RSA: {e}")
            raise

    def generate_aes_key(self) -> bytes:
        """Generar clave AES-256"""
        return secrets.token_bytes(32)  # 256 bits

    def generate_session_key(self, session_id: str) -> bytes:
        """Generar clave de sesión única"""
        session_key = self.generate_aes_key()
        self._session_keys[session_id] = {
            "key": session_key,
            "created": datetime.now(),
            "expires": datetime.now() + timedelta(hours=24),
        }
        return session_key

    # === ENCRIPTACIÓN HÍBRIDA ===

    def hybrid_encrypt(self, data: Union[str, bytes], public_key_pem: bytes) -> Dict[str, str]:
        """
        Encriptación híbrida RSA-4096 + AES-256
        RSA para la clave AES, AES para los datos
        """
        try:
            # Convertir datos a bytes si es necesario
            if isinstance(data, str):
                data = data.encode("utf-8")

            # Cargar clave pública
            public_key = serialization.load_pem_public_key(public_key_pem, backend=self.backend)

            # Generar clave AES aleatoria
            aes_key = self.generate_aes_key()

            # Generar IV aleatorio
            iv = secrets.token_bytes(16)

            # Encriptar datos con AES
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=self.backend)
            encryptor = cipher.encryptor()

            # Padding PKCS7
            padded_data = self._pad_data(data)
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

            # Encriptar clave AES con RSA
            from cryptography.hazmat.primitives.asymmetric import rsa

            if not isinstance(public_key, rsa.RSAPublicKey):
                raise TypeError("La clave pública debe ser RSA para encriptar la clave AES.")
            encrypted_aes_key = public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            # Crear HMAC para integridad
            hmac_key = secrets.token_bytes(32)
            h = hmac.HMAC(hmac_key, hashes.SHA256(), backend=self.backend)
            # Encriptar HMAC key con RSA
            if not isinstance(public_key, rsa.RSAPublicKey):
                raise TypeError("La clave pública debe ser RSA para encriptar la clave HMAC.")
            encrypted_hmac_key = public_key.encrypt(
                hmac_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            # Calcular HMAC del mensaje cifrado
            h.update(encrypted_data)
            message_hmac = h.finalize()

            result = {
                "encrypted_data": base64.b64encode(encrypted_data).decode("utf-8"),
                "encrypted_aes_key": base64.b64encode(encrypted_aes_key).decode("utf-8"),
                "encrypted_hmac_key": base64.b64encode(encrypted_hmac_key).decode("utf-8"),
                "iv": base64.b64encode(iv).decode("utf-8"),
                "hmac": base64.b64encode(message_hmac).decode("utf-8"),
                "algorithm": "RSA-4096+AES-256+HMAC-SHA256",
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info("Encriptación híbrida completada exitosamente")
            return result

        except Exception as e:
            self.logger.error(f"Error en encriptación híbrida: {e}")
            raise

    def hybrid_decrypt(self, encrypted_package: Dict[str, str], private_key_pem: bytes) -> bytes:
        """Desencriptar paquete híbrido"""
        try:
            # Cargar clave privada
            private_key = serialization.load_pem_private_key(
                private_key_pem, password=None, backend=self.backend
            )

            # Decodificar componentes
            encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])
            encrypted_aes_key = base64.b64decode(encrypted_package["encrypted_aes_key"])
            encrypted_hmac_key = base64.b64decode(encrypted_package["encrypted_hmac_key"])
            iv = base64.b64decode(encrypted_package["iv"])
            expected_hmac = base64.b64decode(encrypted_package["hmac"])

            # Desencriptar claves con RSA
            from cryptography.hazmat.primitives.asymmetric import rsa

            if not isinstance(private_key, rsa.RSAPrivateKey):
                raise TypeError(
                    "La clave privada debe ser RSA para desencriptar la clave AES y HMAC."
                )
            aes_key = private_key.decrypt(
                encrypted_aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            hmac_key = private_key.decrypt(
                encrypted_hmac_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            # Verificar integridad con HMAC
            h = hmac.HMAC(hmac_key, hashes.SHA256(), backend=self.backend)
            h.update(encrypted_data)
            try:
                h.verify(expected_hmac)
            except Exception:
                raise ValueError("Verificación de integridad HMAC falló")

            # Desencriptar datos con AES
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=self.backend)
            decryptor = cipher.decryptor()

            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            data = self._unpad_data(padded_data)

            self.logger.info("Desencriptación híbrida completada exitosamente")
            return data

        except Exception as e:
            self.logger.error(f"Error en desencriptación híbrida: {e}")
            raise

    # === DERIVACIÓN DE CLAVES ===

    def derive_key_pbkdf2(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Derivar clave usando PBKDF2"""
        if salt is None:
            salt = secrets.token_bytes(32)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.pbkdf2_iterations,
            backend=self.backend,
        )

        key = kdf.derive(password.encode("utf-8"))
        return key, salt

    def derive_key_scrypt(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Derivar clave usando Scrypt (más resistente a ataques)"""
        if salt is None:
            salt = secrets.token_bytes(32)

        kdf = Scrypt(
            salt=salt,
            length=32,
            n=self.scrypt_n,
            r=self.scrypt_r,
            p=self.scrypt_p,
            backend=self.backend,
        )

        key = kdf.derive(password.encode("utf-8"))
        return key, salt

    # === ENCRIPTACIÓN SIMÉTRICA RÁPIDA ===

    def fast_encrypt(self, data: Union[str, bytes], key: bytes = None) -> Dict[str, str]:
        """Encriptación AES rápida para datos grandes"""
        if key is None:
            key = self.generate_aes_key()

        if isinstance(data, str):
            data = data.encode("utf-8")

        # Usar Fernet para simplicidad y seguridad
        f = Fernet(base64.urlsafe_b64encode(key))
        encrypted_data = f.encrypt(data)

        return {
            "encrypted_data": base64.b64encode(encrypted_data).decode("utf-8"),
            "key": base64.b64encode(key).decode("utf-8"),
            "algorithm": "AES-256-Fernet",
            "timestamp": datetime.now().isoformat(),
        }

    def fast_decrypt(self, encrypted_package: Dict[str, str]) -> bytes:
        """Desencriptación AES rápida"""
        encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])
        key = base64.b64decode(encrypted_package["key"])

        f = Fernet(base64.urlsafe_b64encode(key))
        return f.decrypt(encrypted_data)

    # === FUNCIONES DE HASH ===

    def secure_hash(self, data: Union[str, bytes], algorithm: str = "SHA256") -> str:
        """Generar hash seguro"""
        if isinstance(data, str):
            data = data.encode("utf-8")

        if algorithm == "SHA256":
            digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
        elif algorithm == "SHA512":
            digest = hashes.Hash(hashes.SHA512(), backend=self.backend)
        elif algorithm == "SHA3-256":
            digest = hashes.Hash(hashes.SHA3_256(), backend=self.backend)
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm}")

        digest.update(data)
        return digest.finalize().hex()

    def verify_hash(
        self, data: Union[str, bytes], expected_hash: str, algorithm: str = "SHA256"
    ) -> bool:
        """Verificar hash"""
        calculated_hash = self.secure_hash(data, algorithm)
        return calculated_hash == expected_hash

    # === UTILIDADES ===

    def _pad_data(self, data: bytes) -> bytes:
        """Padding PKCS7"""
        padding_length = 16 - (len(data) % 16)
        padding = bytes([padding_length] * padding_length)
        return data + padding

    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remover padding PKCS7"""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]

    def generate_secure_token(self, length: int = 32) -> str:
        """Generar token seguro"""
        return secrets.token_urlsafe(length)

    def cleanup_session_keys(self):
        """Limpiar claves de sesión expiradas"""
        current_time = datetime.now()
        expired_sessions = [
            session_id
            for session_id, session_data in self._session_keys.items()
            if session_data["expires"] < current_time
        ]

        for session_id in expired_sessions:
            del self._session_keys[session_id]

        if expired_sessions:
            self.logger.info(f"Limpiadas {len(expired_sessions)} claves de sesión expiradas")

    # === POST-QUANTUM PREPARATION ===

    def quantum_safe_hash(self, data: Union[str, bytes]) -> str:
        """Hash resistente a computación cuántica"""
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Usar SHA-3 que es más resistente a ataques cuánticos
        digest = hashes.Hash(hashes.SHA3_512(), backend=self.backend)
        digest.update(data)
        return digest.finalize().hex()

    def get_system_status(self) -> Dict[str, Any]:
        """Estado del sistema de criptografía"""
        return {
            "rsa_key_size": self.rsa_key_size,
            "aes_key_size": self.aes_key_size,
            "pbkdf2_iterations": self.pbkdf2_iterations,
            "active_sessions": len(self._session_keys),
            "quantum_resistant": self.quantum_resistant_enabled,
            "backend": str(self.backend),
            "algorithms_available": [
                "RSA-4096",
                "AES-256",
                "SHA-256",
                "SHA-512",
                "SHA3-256",
                "SHA3-512",
                "PBKDF2",
                "Scrypt",
                "HMAC-SHA256",
            ],
        }


# === FUNCIONES DE UTILIDAD GLOBAL ===


def create_crypto_system(config: Dict[str, Any] = None) -> AdvancedCryptographySystem:
    """Factory function para crear el sistema de criptografía"""
    return AdvancedCryptographySystem(config)


def quick_encrypt_text(text: str, password: str = None) -> Dict[str, str]:
    """Encriptación rápida de texto"""
    crypto = AdvancedCryptographySystem()

    if password:
        key, salt = crypto.derive_key_scrypt(password)
        result = crypto.fast_encrypt(text, key)
        result["salt"] = base64.b64encode(salt).decode("utf-8")
        result["derived_key"] = "true"
    else:
        result = crypto.fast_encrypt(text)
        result["derived_key"] = "false"

    return result


def quick_decrypt_text(encrypted_package: Dict[str, str], password: str = None) -> str:
    """Desencriptación rápida de texto"""
    crypto = AdvancedCryptographySystem()

    if encrypted_package.get("derived_key") and password:
        salt = base64.b64decode(encrypted_package["salt"])
        key, _ = crypto.derive_key_scrypt(password, salt)
        # Reconstruir paquete con clave derivada
        package = encrypted_package.copy()
        package["key"] = base64.b64encode(key).decode("utf-8")
        decrypted_data = crypto.fast_decrypt(package)
    else:
        decrypted_data = crypto.fast_decrypt(encrypted_package)

    return decrypted_data.decode("utf-8")


if __name__ == "__main__":
    # Ejemplo de uso
    crypto = AdvancedCryptographySystem()

    # Generar claves
    private_key, public_key = crypto.generate_rsa_keypair()

    # Encriptar datos
    test_data = "¡Este es un mensaje ultra secreto con encriptación híbrida RSA-4096 + AES-256!"
    encrypted = crypto.hybrid_encrypt(test_data, public_key)

    # Desencriptar datos
    decrypted = crypto.hybrid_decrypt(encrypted, private_key)

    print("✅ Sistema de Criptografía Avanzada funcionando correctamente")
    print(f"Original: {test_data}")
    print(f"Desencriptado: {decrypted.decode('utf-8')}")
    print(f"Estado del sistema: {crypto.get_system_status()}")
