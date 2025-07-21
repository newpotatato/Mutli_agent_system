"""
Core components module

Основные компоненты системы: задачи, SPSA алгоритм, граф связности.
"""

from .task import Task
from .spsa import SPSA
from .graph import GraphService

__all__ = ['Task', 'SPSA', 'GraphService']
