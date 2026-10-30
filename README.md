# 🎬 AI Video Clipper (OpusClip Clone) Telegram Bot

Асинхронный высокопроизводительный Telegram-бот на Python, превращающий длинные видео (YouTube-ссылки или прямые MP4 файлы) в вирусные вертикальные Shorts, Reels и TikTok (формат 9:16, 1080x1920) со стильными динамическими субтитрами.

---

## ⚡ Преимущества новой архитектуры

- **Мгновенная транскрибация**: Полностью удалена тяжелая локальная зависимость PyTorch/Whisper. Теперь используется **Groq Cloud API (`whisper-large-v3`)**, расшифровывающий речь за считанные секунды.
- **Умный AI-анализ**: Работает на актуальной модели **Gemini 2.5 Flash** со строгой валидацией схемы JSON (Pydantic), выделяя ТОП-3 самых вирусных момента (15–75 секунд).
- **Качественное видео 9:16**: FFmpeg кадрирует видео по формуле `crop=ih*9/16:ih,scale=1080:1920` с выравниванием по центру и вшивает стильные желто-белые субтитры (ASS) с черной обводкой в нижней трети экрана.
- **Динамический прогресс-бар в Telegram**:
  - `📥 Скачивание видео с YouTube...`
  - `⚡ Мгновенная расшифровка речи (Groq API)...`
  - `🧠 Gemini ищет вирусные моменты...`
  - `✂️ Нарезка видео 9:16 и наложение субтитров (FFmpeg)...`
  - `🚀 Отправка готовых Shorts!`
- **Автоматическая очистка**: Временные файлы удаляются сразу после отправки результатов.

---

## 🛠 Стек технологий

- **Язык**: Python 3.10+
- **Telegram Bot Framework**: `aiogram` v3 (полностью асинхронный)
- **Транскрибация**: `groq` Cloud API (`whisper-large-v3`)
- **AI / Аналитика**: `google-generativeai` (`gemini-2.5-flash`)
- **Обработка видео**: `ffmpeg` и `yt-dlp`
- **Валидация схемы**: `pydantic` v2

---

## 📋 Предварительные требования

### 1. Установка FFmpeg
FFmpeg обязателен для конвертации, нарезки и наложения субтитров.

- **Windows**:
  ```powershell
  winget install "FFmpeg (Essentials Build)"
  ```
  Или скачайте с [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) и добавьте папку `bin` в системную переменную `PATH`.

- **Linux (Ubuntu / Debian)**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```

- **macOS**:
  ```bash
  brew install ffmpeg
  ```

---

## ⚙️ Установка и запуск

### 1. Виртуальное окружение
```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 2. Установка зависимостей
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Настройка переменных окружения (`.env`)
Создайте файл `.env` на основе `.env.example`:
```env
# Токен бота из @BotFather в Telegram
BOT_TOKEN=1234567890:ABCdefGhIJKlmNoPQRsTUVwxyZ_1234567

# Ключ Gemini API из https://aistudio.google.com/
GEMINI_API_KEY=AIzaSy...

# Ключ Groq Cloud API из https://console.groq.com/
GROQ_API_KEY=gsk_...

# Максимальная длительность видео в секундах (по умолчанию: 1800 = 30 минут)
MAX_VIDEO_DURATION_SECONDS=1800

# Временная папка
TEMP_DIR=./temp
```

### 4. Запуск бота
```bash
python main.py
```

---

## 📁 Структура проекта

```
.
├── config.py                  # Конфигурация, загрузка .env и логирование
├── main.py                    # Точка входа в Telegram-бот
├── requirements.txt           # Список зависимостей
├── .env.example               # Пример файла конфигурации
├── README.md                  # Документация проекта
├── handlers/
│   ├── __init__.py
│   └── clipper.py             # Обработчики команд, сообщений и отправка клипов
└── services/
    ├── __init__.py
    ├── downloader.py          # Загрузка видео через yt-dlp и извлечение звука
    ├── transcriber.py         # Быстрая транскрибация через Groq Whisper API
    ├── ai_analyzer.py         # Выбор вирусных фрагментов через Gemini 2.5 Flash
    ├── ffmpeg_editor.py       # Кадрирование 9:16 (crop=ih*9/16:ih) и вшивание субтитров
    ├── video_editor.py        # Обратная совместимость для ffmpeg_editor
    └── video_processor.py     # Монолитный оркестратор конвейера обработки
```
