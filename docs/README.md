# Multi-Agents System

This repository contains a multi-agent system using various algorithms such as SPSA, LVP, and machine learning models.

## New Project Structure

- `src/`
  - `core/`: Core components like tasks, SPSA algorithm, and connectivity graph.
  - `agents/`: Brokers and executors.
  - `models/`: Machine learning models for load and waiting time prediction.
  - `utils/`: Utilities for visualization and additional functionalities.

- `assets/`
  - `images/`: All PNG images moved here.

- `configs/`
  - Configuration files such as `config.py`.

- `tests/`
  - Test files.

- `examples/`
  - Example scripts demonstrating usage of system components.

- `docs/`
  - Documentation including README and system documentation.

## How to Run

Use `python` to execute and test different modules. Ensure all Python files in `src/` have proper relative imports.

To run tests, navigate to the `tests/` directory and run:

```bash
pytest
```

## Dependencies

All dependencies can be found in `docs/requirements.txt`. To install dependencies, navigate to the root project directory and run:

```bash
pip install -r docs/requirements.txt
```

## Author

Multi-Agent System Team

---

## Возможности

- **Автоматическая классификация задач** на основе текста промпта
- **Семь типов задач**: математика, программирование, текст, анализ, творчество, объяснение, планирование
- **Гибридный подход**: комбинация поиска ключевых слов и косинусного сходства
- **Настраиваемые веса** для балансировки различных методов классификации

## Установка

1. Убедитесь, что у вас есть Python 3.11 или выше
2. Клонируйте репозиторий или скачайте файлы
3. Установите зависимости:

```bash
pip install -r requirements.txt
```

## Типы задач

Система классифицирует задачи на следующие типы:

### 🔢 Math (Математика)
- Решение уравнений и систем
- Вычисления и расчеты
- Работа с формулами
- Статистика и алгебра

### 💻 Code (Программирование)
- Написание кода
- Отладка и исправление ошибок
- Создание алгоритмов
- Работа с API и базами данных

### 📝 Text (Текст)
- Написание статей и эссе
- Переводы
- Редактирование и проверка грамматики
- Создание контента

### 📊 Analysis (Анализ)
- Анализ данных
- Исследования и сравнения
- Поиск трендов и паттернов
- Аналитические отчеты

### 🎨 Creative (Творчество)
- Генерация идей
- Креативные концепции
- Дизайн и творческие решения
- Создание оригинального контента

### 💡 Explanation (Объяснение)
- Объяснение концепций и принципов
- Разъяснение сложных тем
- Обучающие материалы
- Ответы на вопросы "как это работает?"

### 📋 Planning (Планирование)
- Составление планов и стратегий
- Организация процессов
- Структурирование проектов
- Создание roadmap

## Использование

### Базовое использование

```python
from task import Task

# Создание задачи
task = Task(
    prompt="Решить уравнение x^2 + 5x + 6 = 0",
    priority=7,
    complexity=6
)

print(task.type)  # 'math'
print(task)       # [MATH] Решить уравнение x^2 + 5x + 6 = 0 (Приоритет: 7/10, Сложность: 6/10)
```

### Пакетная обработка

```python
from task import Task

prompts = [
    "Написать функцию сортировки на Python",
    "Объяснить принцип работы блокчейна",
    "Проанализировать данные продаж",
    "Создать креативную рекламную кампанию"
]

tasks = []
for prompt in prompts:
    task = Task(prompt, priority=5, complexity=6)
    tasks.append(task)
    print(f"{task.type}: {task.prompt}")
```

## Алгоритм классификации

Функция `_determine_task_type()` использует двухэтапный подход:

1. **Анализ ключевых слов** (40% веса):
   - Поиск прямых совпадений ключевых слов в тексте
   - Нормализация по количеству ключевых слов для каждого типа

2. **Косинусное сходство** (60% веса):
   - TF-IDF векторизация промпта и описаний типов задач
   - Вычисление косинусного сходства между векторами
   - Выбор наиболее похожего типа

3. **Итоговый скор**:
   - Взвешенная комбинация двух методов
   - Выбор типа с максимальным скором
   - Fallback на 'explanation' при низких скорах

## Примеры и тесты

### Запуск базового теста
```bash
python test_task.py
```

### Детальный анализ скоринга
```bash
python detailed_scoring_demo.py
```

### Финальная демонстрация
```bash
python demo_final.py
```

## Структура проекта

```
├── task.py                    # Основной класс Task
├── test_task.py              # Базовые тесты
├── detailed_scoring_demo.py  # Детальный анализ скоринга
├── demo_final.py             # Финальная демонстрация
├── requirements.txt          # Зависимости
└── README.md                 # Документация
```

## Зависимости

- `numpy` - для числовых операций
- `pandas` - для работы с данными
- `scikit-learn` - для TF-IDF векторизации и косинусного сходства

## Настройка весов

Вы можете изменить веса для различных методов классификации в функции `_determine_task_type()`:

```python
keyword_weight = 0.4  # 40% веса на ключевые слова
cosine_weight = 0.6   # 60% веса на косинусное сходство
```

## Расширение системы

Для добавления новых типов задач:

1. Добавьте новый тип в словарь `TASK_TYPES`
2. Укажите описание и ключевые слова
3. Система автоматически начнет классифицировать новый тип

```python
'new_type': {
    'description': 'описание нового типа задач',
    'keywords': ['ключевое1', 'ключевое2', 'ключевое3']
}
```

## Производительность

Система эффективно обрабатывает задачи любой длины и сложности. Время классификации составляет несколько миллисекунд на задачу.

## Лицензия

Этот проект распространяется под MIT лицензией.
