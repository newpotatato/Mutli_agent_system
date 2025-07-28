# Multi-Agent System with SPSA and LVP Algorithms

Многоагентная система с использованием алгоритмов SPSA (Simultaneous Perturbation Stochastic Approximation) и LVP (Load Vector Protocol) для оптимального распределения задач между исполнителями.

## 📁 Структура проекта

```
mutli_agents_2/
├── src/                    # Исходный код
│   ├── core/              # Основные компоненты
│   │   ├── task.py        # Классификация и обработка задач
│   │   ├── spsa.py        # SPSA алгоритм
│   │   └── graph.py       # Граф связности брокеров
│   ├── agents/            # Агенты системы
│   │   ├── controller.py  # Брокеры для распределения задач
│   │   ├── executor.py    # Исполнители задач
│   │   └── real_llm_executor.py  # Реальные LLM исполнители
│   ├── models/            # Модели машинного обучения
│   │   └── models.py      # Предсказание нагрузки и времени ожидания
│   ├── llm_providers/     # Провайдеры LLM
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── groq_provider.py
│   │   └── provider_manager.py
│   ├── visualization/     # Модули визуализации
│   │   ├── comprehensive_visualization.py
│   │   ├── real_llm_visualization.py
│   │   └── enhanced_comparison_visualization.py
│   └── utils/             # Утилиты
│       ├── multi_agent_visualization.py
│       └── multi_agent_visualization_eng.py
├── configs/               # Конфигурационные файлы
│   └── config.py         # Основная конфигурация
├── tests/                 # Тестовые файлы
│   ├── test_full_architecture.py
│   ├── test_pipeline.py
│   ├── test_pipeline_eng.py
│   ├── test_real_llm_pipeline.py
│   └── test_task.py
├── examples/              # Примеры использования
│   ├── demo_final.py
│   ├── demo_spsa_consensus.py
│   ├── demo_spsa_consensus_eng.py
│   └── detailed_scoring_demo.py
├── assets/               # Ресурсы
│   └── images/          # Изображения и графики
└── docs/                # Документация
    ├── README.md        # Детальная документация
    ├── requirements.txt # Зависимости
    └── SYSTEM_DOCUMENTATION.md
```

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install -r docs/requirements.txt
```

### Запуск демо

```bash
# Основная демонстрация системы
python examples/demo_final.py

# SPSA консенсус демо
python examples/demo_spsa_consensus.py

# Детальный анализ классификации
python examples/detailed_scoring_demo.py

# Полная система с реальными LLM
python run_complete_real_llm_analysis.py
```

### Запуск тестов

```bash
# Полное тестирование архитектуры
python tests/test_full_architecture.py

# Тестирование пайплайна
python tests/test_pipeline.py

# Тестирование классификации задач
python tests/test_task.py

# Тестирование с реальными LLM
python tests/test_real_llm_pipeline.py
```

## 🏗️ Архитектура системы

### Основные компоненты

1. **Task Classification System** (`src/core/task.py`)
   - Автоматическая классификация задач на 9 типов
   - Гибридный подход: ключевые слова + TF-IDF + косинусное сходство
   - Типы: math, code, text, analysis, creative, explanation, planning, research, optimization

2. **SPSA Optimizer** (`src/core/spsa.py`)
   - Стохастическая аппроксимация градиента
   - Оптимизация параметров распределения

3. **Graph Service** (`src/core/graph.py`)
   - Управление графом связности брокеров
   - Консенсусные алгоритмы обновления

4. **Broker System** (`src/agents/controller.py`)
   - Распределение задач между исполнителями
   - LVP алгоритм балансировки нагрузки
   - Round-robin контроллер для сравнения

5. **Executor Pool** (`src/agents/executor.py`, `src/agents/real_llm_executor.py`)
   - Выполнение задач через симуляцию или реальные LLM API
   - Мониторинг производительности
   - Поддержка множественных провайдеров

6. **LLM Providers** (`src/llm_providers/`)
   - OpenAI GPT модели
   - Anthropic Claude
   - Groq (быстрые модели)
   - Локальные модели
   - Hugging Face модели

7. **ML Models** (`src/models/models.py`)
   - Предсказание нагрузки и времени ожидания
   - Адаптивные модели обучения

8. **Visualization System** (`src/visualization/`)
   - Комплексная визуализация метрик
   - Сравнительный анализ алгоритмов
   - Визуализация работы с реальными LLM

## ⚙️ Конфигурация

Основные параметры системы находятся в `configs/config.py`:

```python
# LVP параметры
LVP_PARAMS = {
    'h': 0.1,           # коэффициент взаимодействия соседей
    'gamma': 0.05,      # коэффициент локального баланса
}

# SPSA параметры
SPSA_PARAMS = {
    'alpha': 0.01,      # скорость обучения
    'beta': 0.1,        # размер возмущения
    'gamma_consensus': 0.02,  # коэффициент консенсуса
}

# Настройки провайдеров LLM
LLM_PROVIDERS = {
    'openai': {
        'models': ['gpt-3.5-turbo', 'gpt-4'],
        'api_key': 'your_key_here'
    },
    'anthropic': {
        'models': ['claude-3-sonnet', 'claude-3-haiku'],
        'api_key': 'your_key_here'
    },
    'groq': {
        'models': ['llama-3.1-70b', 'mixtral-8x7b'],
        'api_key': 'your_key_here'
    }
}
```

## 📊 Визуализация

Система включает мощные инструменты визуализации:

- **Метрики производительности брокеров**
- **Графики распределения нагрузки**
- **Анализ консенсуса SPSA**
- **Статистика выполнения задач**
- **Сравнение алгоритмов (SPSA vs Round-Robin)**
- **Анализ работы с реальными LLM**
- **Временные характеристики и токены**

Все графики автоматически сохраняются в `assets/images/`.

## 🧪 Тестирование

Проект включает полное покрытие тестами:

- **Unit тесты**: Отдельные компоненты
- **Integration тесты**: Взаимодействие модулей
- **Performance тесты**: Анализ производительности
- **Architecture тесты**: Проверка системной архитектуры
- **Real LLM тесты**: Тестирование с реальными API

## 📈 Алгоритмы

### SPSA (Simultaneous Perturbation Stochastic Approximation)
- Оптимизация параметров θ брокеров
- Консенсусное обновление между соседними узлами
- Адаптивная настройка под нагрузку

### LVP (Load Vector Protocol)
- Балансировка нагрузки между брокерами
- Учет локальной и глобальной нагрузки
- Динамическое перераспределение задач

### Round-Robin (для сравнения)
- Простое циклическое распределение
- Базовый алгоритм для сравнения эффективности

## 🔧 Разработка

### Импорты модулей

```python
from src.core.task import Task
from src.core.spsa import SPSA
from src.core.graph import GraphService
from src.agents.controller import Broker
from src.agents.executor import Executor
from src.agents.real_llm_executor import RealLLMExecutor
from src.models.models import predict_load, predict_waiting_time
from src.llm_providers.provider_manager import ProviderManager
```

### Создание задачи

```python
task = Task(
    prompt="Solve quadratic equation x^2 + 5x + 6 = 0",
    priority=7,
    complexity=6,
    debug=True
)
print(f"Task type: {task.type}")
print(f"Confidence: {task.get_confidence_score():.2f}")
```

### Работа с реальными LLM

```python
from src.llm_providers.provider_manager import ProviderManager

# Инициализация менеджера провайдеров
provider_manager = ProviderManager()

# Выполнение задачи
result = provider_manager.execute_task("openai", "gpt-3.5-turbo", 
                                      "What is 2+2?")
```

## 🚀 Новые возможности

### Версия 1.0.0 включает:

- ✅ **Полная интеграция с реальными LLM**
- ✅ **Расширенная система визуализации**
- ✅ **Сравнительный анализ алгоритмов**
- ✅ **Мониторинг токенов и затрат**
- ✅ **Многопровайдерная архитектура**
- ✅ **Улучшенное тестирование**
- ✅ **Детальная документация**

## 📝 Документация

- Полная документация: `docs/README.md`
- Системная документация: `docs/SYSTEM_DOCUMENTATION.md`
- Требования: `docs/requirements.txt`
- Руководство по настройке: `SETUP_GUIDE_RU.md`

## 🌟 Основные скрипты

- `run_complete_real_llm_analysis.py` - Полный анализ с реальными LLM
- `run_enhanced_comparison.py` - Расширенное сравнение алгоритмов
- `model_performance_evaluator.py` - Оценка производительности моделей
- `performance_data_updater.py` - Обновление данных производительности

## 👥 Авторы

Multi-Agent System Development Team

## 📄 Лицензия

MIT License

---

**Версия:** 1.0.0  
**Последнее обновление:** Июль 2025
