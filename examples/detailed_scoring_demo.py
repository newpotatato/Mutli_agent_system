#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстрация детального скоринга для анализа работы функции _determine_task_type
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.task import Task
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def detailed_task_analysis(prompt, priority, complexity):
    """
    Детальный анализ классификации задачи с выводом всех скоров
    """
    print(f"=== Анализ задачи: '{prompt}' ===\n")
    
    # Создаем объект задачи
    task = Task(prompt, priority, complexity)
    
    # Получаем доступ к типам задач
    TASK_TYPES = task.TASK_TYPES
    
    # Приводим промпт к нижнему регистру
    prompt_lower = prompt.lower()
    
    # Словарь для хранения скоров
    type_scores = {}
    
    # 1. Проверка ключевых слов
    print("1. Анализ ключевых слов:")
    for task_type, info in TASK_TYPES.items():
        keyword_score = 0
        found_keywords = []
        
        for keyword in info['keywords']:
            if keyword.lower() in prompt_lower:
                keyword_score += 1
                found_keywords.append(keyword)
        
        normalized_keyword_score = keyword_score / len(info['keywords'])
        type_scores[task_type] = {'keyword_score': normalized_keyword_score, 'found_keywords': found_keywords}
        
        print(f"   {task_type}: {normalized_keyword_score:.3f} (найдено: {found_keywords})")
    
    # 2. Косинусное сходство
    print("\n2. Косинусное сходство:")
    
    # Подготавливаем тексты для векторизации
    texts = [prompt]
    for task_type in TASK_TYPES.keys():
        texts.append(TASK_TYPES[task_type]['description'])
    
    # Создаем TF-IDF векторизатор
    vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Вычисляем косинусное сходство
    prompt_vector = tfidf_matrix.todense()[0:1]
    
    task_types_list = list(TASK_TYPES.keys())
    for i, task_type in enumerate(task_types_list):
        description_vector = tfidf_matrix.todense()[i+1:i+2]
        similarity = cosine_similarity(prompt_vector, description_vector)[0][0]
        type_scores[task_type]['cosine_score'] = similarity
        
        print(f"   {task_type}: {similarity:.3f}")
    
    # 3. Итоговые скоры
    print("\n3. Итоговые скоры:")
    final_scores = {}
    
    for task_type in type_scores:
        keyword_weight = 0.4
        cosine_weight = 0.6
        
        final_score = (type_scores[task_type]['keyword_score'] * keyword_weight + 
                      type_scores[task_type]['cosine_score'] * cosine_weight)
        final_scores[task_type] = final_score
        
        print(f"   {task_type}: {final_score:.3f} (ключевые: {type_scores[task_type]['keyword_score']:.3f}, косинус: {type_scores[task_type]['cosine_score']:.3f})")
    
    # Результат
    best_type = max(final_scores, key=lambda x: final_scores[x])
    print(f"\nРезультат: {best_type} (скор: {final_scores[best_type]:.3f})")
    print(f"Созданная задача: {task}")
    
    return task, final_scores

def demo_analysis():
    """Демонстрация детального анализа нескольких задач"""
    
    test_cases = [
        ("Решить квадратное уравнение 2x^2 + 3x - 5 = 0", 5, 6),
        ("Написать функцию на Python для сортировки списка", 6, 5),
        ("Создать креативную концепцию для рекламы", 4, 7),
        ("Проанализировать данные продаж и найти тренды", 7, 8),
        ("Объяснить принцип работы нейронных сетей", 5, 8)
    ]
    
    for prompt, priority, complexity in test_cases:
        detailed_task_analysis(prompt, priority, complexity)
        print("=" * 100)

if __name__ == "__main__":
    demo_analysis()
