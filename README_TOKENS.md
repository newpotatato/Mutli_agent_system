# 🚀 Быстрая настройка API токенов

## ⚡ Быстрый старт (2 минуты)

1. **Откройте** файл `api_keys.json`
2. **Заполните** хотя бы Groq ключ (самый быстрый бесплатный):
   ```json
   "groq": {
     "api_key": "ВАШ_КЛЮЧ_ЗДЕСЬ"
   }
   ```
3. **Получите ключ**: [console.groq.com](https://console.groq.com) → API Keys → Create
4. **Проверьте**: `python check_config.py`

## 🎯 Рекомендуемые бесплатные сервисы

| Сервис | Получение | Плюсы | Лимиты |
|--------|-----------|-------|---------|
| **Groq** | [console.groq.com](https://console.groq.com) | ⚡ Сверхбыстро | 30 запросов/мин |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | 🆓 Полностью бесплатно | ~100 запросов/мин |
| **Together AI** | [api.together.xyz](https://api.together.xyz) | 🎁 $25 бесплатно | 60 запросов/мин |
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com) | 🧠 Очень умный | $5 бесплатно |

## 📁 Что делать дальше

1. **Проверить настройки**: `python check_config.py`
2. **Запустить пример**: `python examples/demo_final.py`
3. **Читать документацию**: `SETUP_GUIDE_RU.md`

## ❗ Важно

- 🔒 **НЕ публикуйте** файл `api_keys.json` с заполненными ключами
- 📁 Добавьте `api_keys.json` в `.gitignore`
- 🧪 Система работает даже с одним ключом (HuggingFace даже без ключа)

---

🎉 **Все готово!** Теперь у вас есть мощная мульти-агентная система!
