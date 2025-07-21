import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, List, Tuple
import logging

class Task:
    # Определения типов задач для LLM с ключевыми словами, описаниями и весами
    TASK_TYPES = {
        'math': {
            'description': "Задачи, связанные с математическими вычислениями, включая формулы, уравнения, алгебру, геометрию, статистику, расчеты чисел и операции над ними.",
            'keywords': ['математика', 'вычисления', 'формула', 'уравнение', 'алгебра', 'геометрия', 
                        'статистика', 'расчет', 'число', 'сумма', 'произведение', 'интеграл', 
                        'дифференциал', 'решить', 'вычислить', 'найти значение', 'график', 'функция',
                        'калькулятор', 'арифметика', 'тригонометрия', 'логарифм', 'матрица',
                        'вектор', 'производная', 'предел', 'ряд', 'вероятность'],
            'weight': 1.2,  # Повышенный вес для точных наук
            'patterns': [r'\d+[\+\-\*/]\d+', r'\b\d+\s*[xх]\s*\d+', r'\d+\s*[=равно]']
        },
        'code': {
            'description': "Задачи, касающиеся программирования и разработки, включая написание кода, отладку, рефакторинг и реализацию алгоритмов.",
            'keywords': ['код', 'программирование', 'python', 'java', 'javascript', 'алгоритм', 
                        'функция', 'класс', 'метод', 'дебаг', 'отладка', 'рефакторинг', 
                        'написать код', 'исправить ошибку', 'оптимизировать', 'реализовать', 
                        'API', 'база данных', 'фреймворк', 'библиотека', 'модуль', 'переменная',
                        'цикл', 'условие', 'массив', 'объект', 'наследование', 'полиморфизм',
                        'git', 'github', 'docker', 'тестирование', 'unittest', 'pytest'],
            'weight': 1.1,
            'patterns': [r'def\s+\w+', r'class\s+\w+', r'import\s+\w+', r'[{}\[\]()]']
        },
        'text': {
            'description': "Задачи, связанные с текстом, включая написание статей, перевод, редактирование, проверку грамматики и стиля.",
            'keywords': ['текст', 'написать', 'статья', 'перевод', 'редактирование', 'грамматика', 
                        'стиль', 'контент', 'креативное письмо', 'эссе', 'резюме', 'письмо', 
                        'проверить текст', 'исправить ошибки', 'орфография', 'пунктуация',
                        'литература', 'рассказ', 'поэзия', 'сочинение', 'изложение',
                        'копирайтинг', 'контент-план', 'заголовок', 'аннотация'],
            'weight': 1.0,
            'patterns': [r'"[^"]+"', r'«[^»]+»']
        },
        'analysis': {
            'description': "Задачи, направленные на анализ данных, включая исследование, аналитику, сравнение данных и извлечение статистических выводов.",
            'keywords': ['анализ', 'данные', 'исследование', 'аналитика', 'сравнение', 'изучение', 
                        'обзор', 'статистика', 'проанализировать', 'сравнить', 'исследовать', 
                        'выводы', 'тренды', 'паттерны', 'метрики', 'показатели', 'диаграмма',
                        'график', 'таблица', 'отчет', 'dashboard', 'визуализация', 'корреляция',
                        'регрессия', 'кластеризация', 'машинное обучение', 'нейросеть'],
            'weight': 1.0,
            'patterns': [r'\d+%', r'\d+\s*(процент|percent)']
        },
        'creative': {
            'description': "Задачи, требующие творческого подхода, включая генерацию идей, создание концепций и разработку дизайна.",
            'keywords': ['творчество', 'креативность', 'идеи', 'концепция', 'дизайн', 'создание', 
                        'генерация', 'творческое решение', 'придумать', 'создать', 'сгенерировать', 
                        'вдохновение', 'креативный подход', 'иллюстрация', 'логотип', 'брендинг',
                        'фантазия', 'воображение', 'художественный', 'музыка', 'видео',
                        'реклама', 'маркетинг', 'слоган', 'название'],
            'weight': 0.9,
            'patterns': []
        },
        'explanation': {
            'description': "Задачи, требующие разъяснения и объяснения сложных понятий и механизмов.",
            'keywords': ['объяснение', 'разъяснение', 'понятие', 'определение', 'как работает', 
                        'принцип', 'механизм', 'теория', 'объяснить', 'разъяснить', 'что такое', 
                        'как это работает', 'принцип работы', 'почему', 'зачем', 'каким образом',
                        'суть', 'смысл', 'значение', 'расскажи', 'опиши', 'поясни', 'демонстрация',
                        'пример', 'иллюстрация', 'аналогия', 'детализация', 'описание процесса',
                        'шаги', 'пошаговый', 'обзор', 'разбор', 'анализ', 'инструкция'],
            'weight': 0.8,
            'patterns': [r'что\s+такое', r'как\s+работает', r'почему', r'зачем', r'объясните', r'разъясните']
        },
        'planning': {
            'description': "Задачи, связанные с планированием и организацией, включая разработку стратегий и составление планов.",
            'keywords': ['планирование', 'план', 'стратегия', 'организация', 'структура', 
                        'этапы', 'шаги', 'roadmap', 'составить план', 'спланировать', 'организовать', 
                        'структурировать', 'последовательность', 'график', 'расписание', 'таймлайн',
                        'проект', 'задачи', 'цели', 'приоритеты', 'ресурсы', 'бюджет'],
            'weight': 1.0,
            'patterns': [r'\d+\s*(этап|шаг|пункт)']
        },
        'research': {
            'description': "Задачи, связанные с поиском информации, исследованиями и изучением тем.",
            'keywords': ['исследование', 'поиск', 'найти информацию', 'изучить', 'узнать', 
                        'выяснить', 'разузнать', 'источники', 'литература', 'статьи',
                        'публикации', 'обзор литературы', 'библиография', 'реферат'],
            'weight': 0.9,
            'patterns': []
        },
        'optimization': {
            'description': "Задачи по оптимизации процессов, производительности и эффективности.",
            'keywords': ['оптимизация', 'улучшение', 'эффективность', 'производительность',
                        'скорость', 'оптимизировать', 'ускорить', 'улучшить', 'повысить',
                        'снизить затраты', 'экономия', 'автоматизация'],
            'weight': 1.1,
            'patterns': []
        }
    }
    
    def __init__(self, prompt: str, priority: int, complexity: int, debug: bool = False):
        """
        Инициализация задачи с описанием, приоритетом и сложностью.
        Автоматически определяет тип задачи при создании.
        
        :param prompt: Описание задачи
        :param priority: Приоритет (1-10)
        :param complexity: Сложность (1-10)
        :param debug: Включить отладочную информацию
        """
        self.prompt = prompt
        self.priority = priority
        self.complexity = complexity
        self.debug = debug
        self.classification_scores = {}  # Для хранения результатов классификации
        self.type = self._determine_task_type()  # Автоматическое определение типа

    def _preprocess_text(self, text: str) -> str:
        """
        Предобработка текста для лучшей классификации.
        
        :param text: Исходный текст
        :return: Обработанный текст
        """
        # Удаляем лишние пробелы и символы
        text = re.sub(r'\s+', ' ', text.strip())
        # Удаляем знаки препинания, кроме важных для контекста
        text = re.sub(r'[^\w\s\-+*/=()<>"«»]', ' ', text)
        return text.lower()
    
    def _calculate_keyword_score(self, prompt_lower: str, keywords: List[str]) -> Tuple[float, int]:
        """
        Вычисляет скор на основе ключевых слов с учетом важности совпадений.
        
        :param prompt_lower: Промпт в нижнем регистре
        :param keywords: Список ключевых слов
        :return: (нормализованный скор, количество совпадений)
        """
        matches = 0
        weighted_score = 0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in prompt_lower:
                matches += 1
                # Вес основан на длине ключевого слова и количестве слов в нем
                word_count = len(keyword_lower.split())
                # Более длинные и составные ключевые слова получают больший вес
                word_weight = word_count * 2 + len(keyword_lower) * 0.1
                weighted_score += word_weight
        
        # Простая нормализация: скор на совпадение
        if matches > 0:
            # Базовый скор + бонус за количество совпадений
            base_score = 0.3 + (matches * 0.1)
            # Дополнительный бонус за вес ключевых слов
            weight_bonus = min(weighted_score / 100, 0.6)  
            normalized_score = min(base_score + weight_bonus, 1.0)
        else:
            normalized_score = 0.0
        
        return normalized_score, matches
    
    def _calculate_pattern_score(self, prompt: str, patterns: List[str]) -> float:
        """
        Вычисляет скор на основе регулярных выражений.
        
        :param prompt: Исходный промпт
        :param patterns: Список паттернов для поиска
        :return: Скор паттернов
        """
        if not patterns:
            return 0.0
        
        pattern_matches = 0
        for pattern in patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                pattern_matches += 1
        
        return pattern_matches / len(patterns)
    
    def _calculate_enhanced_similarity(self, processed_prompt: str, detailed_scores: Dict) -> None:
        """
        Вычисляет улучшенное косинусное сходство с описаниями задач
        
        :param processed_prompt: Предобработанный промпт
        :param detailed_scores: Словарь с детальными скорами
        """
        # Создаем список текстов для векторизации
        texts = [processed_prompt]
        task_types_list = list(self.TASK_TYPES.keys())
        
        # Добавляем описания всех типов задач
        for task_type in task_types_list:
            description = self._preprocess_text(self.TASK_TYPES[task_type]['description'])
            # Обогащаем описание ключевыми словами
            keywords_text = ' '.join(self.TASK_TYPES[task_type]['keywords'][:10])  # Топ-10 ключевых слов
            enriched_description = f"{description} {keywords_text}"
            texts.append(self._preprocess_text(enriched_description))
        
        # Улучшенный TF-IDF векторизатор
        vectorizer = TfidfVectorizer(
            stop_words=None,
            lowercase=True,
            ngram_range=(1, 2),  # Униграммы и биграммы
            max_features=1500,
            min_df=1,  # Минимальная частота документа
            max_df=0.95  # Максимальная частота документа
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            prompt_vector = tfidf_matrix.todense()[0:1]
            
            for i, task_type in enumerate(task_types_list):
                description_vector = tfidf_matrix.todense()[i+1:i+2]
                similarity = cosine_similarity(prompt_vector, description_vector)[0][0]
                
                # Нормализация и усиление сходства
                if similarity > 0:
                    # Повышаем значимость высокого сходства
                    enhanced_similarity = min(similarity * 1.5, 1.0)
                else:
                    enhanced_similarity = max(0.3, similarity)  # Минимальный базовый скор
                
                detailed_scores[task_type]['cosine_score'] = enhanced_similarity
                
        except Exception as e:
            # Fallback в случае ошибки векторизации
            if self.debug:
                print(f"Ошибка векторизации: {e}")
            for task_type in task_types_list:
                detailed_scores[task_type]['cosine_score'] = 0.5  # Базовый скор
    
    def _determine_task_type(self) -> str:
        """
        Оптимизированное определение типа задачи с повышенной уверенностью:
        1. Усиленная проверка ключевых слов
        2. Улучшенные паттерны и их обработка
        3. Более точное косинусное сходство
        4. Динамические веса для максимизации различий
        """
        # Предобработка промпта
        processed_prompt = self._preprocess_text(self.prompt)
        prompt_lower = self.prompt.lower()
        
        # Словарь для хранения всех скоров
        detailed_scores = {}
        max_keyword_score = 0
        
        # 1. Вычисляем скоры для каждого типа задачи
        for task_type, info in self.TASK_TYPES.items():
            scores = {}
            
            # Скор ключевых слов с улучшенным алгоритмом
            keyword_score, keyword_matches = self._calculate_keyword_score(
                prompt_lower, info['keywords']
            )
            scores['keyword_score'] = keyword_score
            scores['keyword_matches'] = keyword_matches
            max_keyword_score = max(max_keyword_score, keyword_score)
            
            # Скор паттернов
            pattern_score = self._calculate_pattern_score(
                self.prompt, info.get('patterns', [])
            )
            scores['pattern_score'] = pattern_score
            
            detailed_scores[task_type] = scores
        
        # 2. Улучшенное косинусное сходство
        self._calculate_enhanced_similarity(processed_prompt, detailed_scores)
        
        # 3. Вычисляем итоговый скор с динамическими весами
        final_scores = {}
        best_keyword_matches = max([scores['keyword_matches'] for scores in detailed_scores.values()])
        
        for task_type in detailed_scores:
            scores = detailed_scores[task_type]
            
            # Динамические веса для повышения различий
            if scores['keyword_matches'] > 0:
                # Если есть совпадения ключевых слов - приоритет им
                keyword_weight = 0.7 if scores['keyword_matches'] >= best_keyword_matches else 0.6
                cosine_weight = 0.25
                pattern_weight = 0.05
                
                # Бонус за лидерство в ключевых словах
                if scores['keyword_score'] == max_keyword_score and max_keyword_score > 0:
                    leadership_bonus = 0.3
                else:
                    leadership_bonus = 0
            else:
                # Если нет совпадений - полагаемся на косинусное сходство
                keyword_weight = 0.1
                cosine_weight = 0.8
                pattern_weight = 0.1
                leadership_bonus = 0
            
            # Базовый скор
            base_score = (
                scores['keyword_score'] * keyword_weight +
                scores['cosine_score'] * cosine_weight +
                scores['pattern_score'] * pattern_weight
            )
            
            # Применяем тип-специфичный вес
            type_weight = self.TASK_TYPES[task_type].get('weight', 1.0)
            
            # Финальный скор с бонусами
            final_score = (base_score + leadership_bonus) * type_weight
            
            final_scores[task_type] = final_score
            
            # Сохраняем детальную информацию для отладки
            detailed_scores[task_type]['final_score'] = final_score
            detailed_scores[task_type]['type_weight'] = type_weight
            detailed_scores[task_type]['leadership_bonus'] = leadership_bonus
        
        # Сохраняем результаты для анализа
        self.classification_scores = detailed_scores
        
        # Находим лучший тип
        best_type = max(final_scores, key=lambda x: final_scores[x])
        best_score = final_scores[best_type]
        
        # Отладочная информация
        if self.debug:
            print(f"\nКлассификация для: '{self.prompt}'")
            print("Детальные скоры:")
            for task_type, scores in detailed_scores.items():
                print(f"  {task_type}: {scores['final_score']:.3f} "
                      f"(kw:{scores['keyword_score']:.3f}, "
                      f"cs:{scores['cosine_score']:.3f}, "
                      f"pt:{scores['pattern_score']:.3f}, "
                      f"bonus:{scores.get('leadership_bonus', 0):.2f}, "
                      f"matches:{scores['keyword_matches']})")
            print(f"Выбран тип: {best_type} (скор: {best_score:.3f})")
        
        return best_type

    def get_classification_details(self) -> Dict:
        """
        Возвращает детальную информацию о процессе классификации.
        
        :return: Словарь с результатами классификации
        """
        return self.classification_scores
    
    def get_confidence_score(self) -> float:
        """
        Возвращает улучшенный уровень уверенности в классификации (0-1).
        
        :return: Скор уверенности
        """
        if not self.classification_scores:
            return 0.0
        
        scores = [info['final_score'] for info in self.classification_scores.values()]
        if len(scores) < 2:
            return 0.0
        
        scores.sort(reverse=True)
        
        # Многофакторный расчет уверенности
        # 1. Разница между лучшим и вторым
        score_gap = scores[0] - scores[1] if scores[1] > 0 else scores[0] * 0.5
        
        # 2. Абсолютная высота лучшего скора
        absolute_score = min(scores[0] / 2.0, 0.5)  # Нормализуем
        
        # 3. Наличие ключевых слов у лидера
        best_type_info = max(self.classification_scores.items(), key=lambda x: x[1]['final_score'])[1]
        keyword_bonus = 0.3 if best_type_info.get('keyword_matches', 0) > 0 else 0
        
        # 4. Лидерский бонус
        leadership_bonus = best_type_info.get('leadership_bonus', 0) / 3  # Нормализуем
        
        # Общая уверенность
        total_confidence = score_gap + absolute_score + keyword_bonus + leadership_bonus
        
        return min(total_confidence, 1.0)
    
    def suggest_alternative_types(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Предлагает альтернативные типы задач с их скорами.
        
        :param top_n: Количество альтернатив для возврата
        :return: Список кортежей (тип, скор)
        """
        if not self.classification_scores:
            return []
        
        scored_types = [
            (task_type, info['final_score'])
            for task_type, info in self.classification_scores.items()
        ]
        
        scored_types.sort(key=lambda x: x[1], reverse=True)
        return scored_types[:top_n]
    
    def __repr__(self) -> str:
        confidence = self.get_confidence_score()
        return f"Task(prompt='{self.prompt[:50]}...', priority={self.priority}, " \
               f"complexity={self.complexity}, type='{self.type}', " \
               f"confidence={confidence:.2f})"
    
    def __str__(self) -> str:
        return f"Задача: {self.prompt}\nТип: {self.type}\nПриоритет: {self.priority}\nСложность: {self.complexity}"

    