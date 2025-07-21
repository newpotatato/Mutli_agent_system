"""
Models module

Модели машинного обучения для предсказания нагрузки и времени ожидания.
"""

from .models import predict_load, predict_waiting_time, LoadPredictor, WaitingTimePredictor

__all__ = ['predict_load', 'predict_waiting_time', 'LoadPredictor', 'WaitingTimePredictor']
