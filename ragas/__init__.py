"""
Ragas Evaluation System Package

A comprehensive evaluation framework for RAG applications using Ragas by Exploding Gradients.
"""

from .ragas_evaluator import (
    RagasEvaluator,
    RAGSystem,
    TestSetGenerator,
    EvaluationConfig
)

__version__ = "1.0.0"
__author__ = "AI Evaluation Framework"

__all__ = [
    "RagasEvaluator",
    "RAGSystem", 
    "TestSetGenerator",
    "EvaluationConfig"
] 