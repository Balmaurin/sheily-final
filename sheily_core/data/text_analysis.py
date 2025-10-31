#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Análisis de Texto - Funciones de análisis y procesamiento de texto
Extraído de main.py para mejorar la organización del código
"""

import math
import re
from typing import Dict, List, Tuple


class TextAnalyzer:
    """Clase para análisis completo de texto"""

    def __init__(self, rag_pipeline=None):
        self.rag_pipeline = rag_pipeline

    def analyze_text(self, text: str, analysis_type: str = "complete") -> Dict:
        """Análisis completo de texto"""
        analysis_result = {
            "input_text": text,
            "analysis_type": analysis_type,
            "basic_stats": self._analyze_text_basic(text),
            "timestamp": self._get_timestamp(),
        }

        if analysis_type in ["complete", "advanced"]:
            analysis_result.update(
                {
                    "linguistic_analysis": self._analyze_text_linguistic(text),
                    "similarity_to_corpus": self._analyze_corpus_similarity(text),
                    "topic_analysis": self._analyze_text_topics(text),
                }
            )

        if analysis_type == "complete":
            analysis_result.update(
                {
                    "readability_metrics": self._analyze_readability(text),
                    "entity_extraction": self._extract_entities(text),
                    "sentiment_analysis": self._analyze_sentiment(text),
                }
            )

        return analysis_result

    def analyze_similarity(
        self, text1: str, text2: str, similarity_types: List[str] = None
    ) -> Dict:
        """Análisis de similitud entre textos"""
        if similarity_types is None:
            similarity_types = ["cosine", "jaccard", "overlap"]

        similarity_results = {
            "text1": text1,
            "text2": text2,
            "analysis_types": similarity_types,
            "similarities": {},
            "timestamp": self._get_timestamp(),
        }

        if "cosine" in similarity_types:
            similarity_results["similarities"]["cosine"] = self._calculate_cosine_similarity(
                text1, text2
            )

        if "jaccard" in similarity_types:
            similarity_results["similarities"]["jaccard"] = self._calculate_jaccard_similarity(
                text1, text2
            )

        if "overlap" in similarity_types:
            similarity_results["similarities"]["overlap"] = self._calculate_overlap_similarity(
                text1, text2
            )

        if "semantic" in similarity_types and self.rag_pipeline:
            similarity_results["similarities"]["semantic"] = self._calculate_semantic_similarity(
                text1, text2
            )

        avg_similarity = sum(similarity_results["similarities"].values()) / len(
            similarity_results["similarities"]
        )
        similarity_results["average_similarity"] = avg_similarity
        similarity_results["interpretation"] = self._interpret_similarity(avg_similarity)

        return similarity_results

    def _analyze_text_basic(self, text: str) -> Dict:
        """Análisis básico de texto"""
        words = text.split()
        sentences = text.split(".")
        paragraphs = text.split("\n\n")

        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len([s for s in sentences if s.strip()])
            if sentences
            else 0,
            "unique_words": len(set(word.lower() for word in words)),
            "lexical_diversity": len(set(word.lower() for word in words)) / len(words)
            if words
            else 0,
        }

    def _analyze_text_linguistic(self, text: str) -> Dict:
        """Análisis lingüístico del texto"""
        words = text.split()

        word_freq = {}
        for word in words:
            clean_word = word.lower().strip('.,!?;:"()[]')
            word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

        most_frequent = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        uppercase_count = sum(1 for char in text if char.isupper())
        digit_count = sum(1 for char in text if char.isdigit())
        punctuation_count = sum(1 for char in text if char in ".,!?;:")

        return {
            "most_frequent_words": most_frequent,
            "uppercase_ratio": uppercase_count / len(text) if text else 0,
            "digit_ratio": digit_count / len(text) if text else 0,
            "punctuation_ratio": punctuation_count / len(text) if text else 0,
            "vocabulary_richness": len(word_freq) / len(words) if words else 0,
        }

    def _analyze_corpus_similarity(self, text: str) -> Dict:
        """Analizar similitud del texto con el corpus"""
        if not self.rag_pipeline or not self.rag_pipeline.documents:
            return {"status": "no_corpus_available"}

        search_results = self.rag_pipeline.search(text, top_k=10)
        top_similarities = search_results[:5]

        scores = [result.get("score", 0.0) for result in search_results]
        avg_similarity = sum(scores) / len(scores) if scores else 0

        return {
            "avg_corpus_similarity": avg_similarity,
            "top_similar_documents": [
                {
                    "document": result.get("content", "")[:100] + "..."
                    if len(result.get("content", "")) > 100
                    else result.get("content", ""),
                    "similarity": result.get("score", 0.0),
                    "document_id": result.get("id", ""),
                }
                for result in top_similarities
            ],
            "similarity_distribution": self._calculate_similarity_distribution(
                [(result.get("content", ""), result.get("score", 0.0)) for result in search_results]
            ),
        }

    def _analyze_text_topics(self, text: str) -> Dict:
        """Análisis básico de tópicos en el texto"""
        words = text.lower().split()

        topic_keywords = {
            "technology": [
                "python",
                "código",
                "programación",
                "software",
                "computadora",
                "algoritmo",
                "datos",
            ],
            "science": [
                "ciencia",
                "investigación",
                "experimento",
                "análisis",
                "método",
                "resultado",
                "hipótesis",
            ],
            "education": [
                "aprender",
                "enseñar",
                "educación",
                "estudiante",
                "curso",
                "clase",
                "conocimiento",
            ],
            "health": [
                "salud",
                "medicina",
                "médico",
                "tratamiento",
                "diagnóstico",
                "síntoma",
                "enfermedad",
            ],
            "business": [
                "negocio",
                "empresa",
                "mercado",
                "venta",
                "cliente",
                "proyecto",
                "estrategia",
            ],
        }

        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for word in words if word in keywords)
            topic_scores[topic] = score / len(words) if words else 0

        main_topic = (
            max(topic_scores.keys(), key=lambda k: topic_scores[k]) if topic_scores else "general"
        )

        return {
            "topic_scores": topic_scores,
            "main_topic": main_topic,
            "topic_confidence": topic_scores.get(main_topic, 0),
        }

    def _analyze_readability(self, text: str) -> Dict:
        """Métricas básicas de legibilidad"""
        words = text.split()
        sentences = [s for s in text.split(".") if s.strip()]

        if not words or not sentences:
            return {"status": "insufficient_text"}

        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)

        flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)

        if flesch_score >= 90:
            level = "muy_facil"
        elif flesch_score >= 80:
            level = "facil"
        elif flesch_score >= 70:
            level = "bastante_facil"
        elif flesch_score >= 60:
            level = "estandar"
        elif flesch_score >= 50:
            level = "bastante_dificil"
        elif flesch_score >= 30:
            level = "dificil"
        else:
            level = "muy_dificil"

        return {
            "avg_words_per_sentence": avg_words_per_sentence,
            "avg_syllables_per_word": avg_syllables_per_word,
            "flesch_score": flesch_score,
            "readability_level": level,
        }

    def _extract_entities(self, text: str) -> Dict:
        """Extracción básica de entidades"""
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        url_pattern = (
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        number_pattern = r"\b\d+(?:\.\d+)?\b"
        date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"

        entities = {
            "emails": re.findall(email_pattern, text),
            "urls": re.findall(url_pattern, text),
            "numbers": re.findall(number_pattern, text),
            "dates": re.findall(date_pattern, text),
            "capitalized_words": re.findall(r"\b[A-Z][a-z]+\b", text),
        }

        entity_counts = {k: len(v) for k, v in entities.items()}

        return {
            "entities": entities,
            "entity_counts": entity_counts,
            "total_entities": sum(entity_counts.values()),
        }

    def _analyze_sentiment(self, text: str) -> Dict:
        """Análisis básico de sentimiento"""
        positive_words = [
            "bueno",
            "excelente",
            "genial",
            "fantástico",
            "perfecto",
            "increíble",
            "maravilloso",
            "positivo",
            "útil",
            "efectivo",
        ]
        negative_words = [
            "malo",
            "terrible",
            "horrible",
            "pésimo",
            "negativo",
            "inútil",
            "problemático",
            "difícil",
            "complicado",
            "error",
        ]

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            sentiment = "neutral"
            confidence = 0.5
        else:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            if sentiment_score > 0.2:
                sentiment = "positive"
            elif sentiment_score < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            confidence = abs(sentiment_score)

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_words_found": positive_count,
            "negative_words_found": negative_count,
            "sentiment_ratio": (positive_count - negative_count) / len(words) if words else 0,
        }

    def _count_syllables(self, word: str) -> int:
        """Contar sílabas aproximadamente"""
        word = word.lower()
        vowels = "aeiouáéíóú"
        syllables = sum(1 for char in word if char in vowels)
        return max(1, syllables)

    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud coseno entre dos textos"""
        if self.rag_pipeline:
            tokens1 = self.rag_pipeline.embedder.tokenize(text1)
            tokens2 = self.rag_pipeline.embedder.tokenize(text2)

            tf1 = self.rag_pipeline.embedder.compute_tf(tokens1)
            tf2 = self.rag_pipeline.embedder.compute_tf(tokens2)

            return self.rag_pipeline.embedder.cosine_similarity(tf1, tf2)
        else:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0

    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calcular índice de Jaccard"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0

    def _calculate_overlap_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud de superposición"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1 & words2)
        min_size = min(len(words1), len(words2))
        return intersection / min_size if min_size > 0 else 0

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud semántica usando embeddings"""
        if not self.rag_pipeline:
            return 0.0
        return self._calculate_cosine_similarity(text1, text2)

    def _calculate_similarity_distribution(self, similarities: List[Tuple]) -> Dict:
        """Calcular distribución de similitudes"""
        if not similarities:
            return {"status": "no_data"}

        scores = [score for _, score in similarities]

        return {
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores),
            "high_similarity_count": sum(1 for score in scores if score > 0.7),
            "medium_similarity_count": sum(1 for score in scores if 0.3 <= score <= 0.7),
            "low_similarity_count": sum(1 for score in scores if score < 0.3),
        }

    def _interpret_similarity(self, similarity_score: float) -> str:
        """Interpretar score de similitud"""
        if similarity_score >= 0.9:
            return "muy_alta"
        elif similarity_score >= 0.7:
            return "alta"
        elif similarity_score >= 0.5:
            return "media"
        elif similarity_score >= 0.3:
            return "baja"
        else:
            return "muy_baja"

    def _get_timestamp(self) -> str:
        """Obtener timestamp actual"""
        import time

        return time.strftime("%Y-%m-%d %H:%M:%S")
