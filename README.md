# Neuropsychologist Agent


## 1. Контекст

AI-ассистент-нейропсихолог на базе RAG с использованием книг **Михаила Хорса** как единственного источника знаний.

**Цель:** отвечать на вопросы пользователей в стиле и логике Хорса, опираясь на его книги.

**Интерфейс:** OpenWebUI (совместимость с OpenAI API)

---

## 2. Решение

Один сервис с:
- OpenAI-совместимым API
- Единой точкой входа в LLM (`LLMClient`)
- RAG-пайплайном по одной коллекции в Qdrant
- Логированием в Langfuse + стандартный logging

---

## 3. Архитектура

```
POST /v1/chat/completions
         ↓
OpenAICompatController
         ↓
    MainPipeline
         ↓
1. Переписывание запроса (LLM-вызов)
   → rewritten_query
         ↓
2. rag_service.retrieve(query) по коллекции khors_books
   Qdrant hybrid search (dense + sparse)
         ↓
3. reranker.rank(chunks)
         ↓
4. llm_client.generate(messages=[system_prompt, query, chunks])
   → final_answer
         ↓
ChatCompletionResponse
```

---

## 4. Компоненты

| Модуль | Описание |
|---|---|
| `llm_client.py` | Единая точка входа для LLM (retry, логи, Langfuse) |
| `rag_service.py` | Hybrid search по Qdrant |
| `reranker.py` | Переранжирование чанков |
| `main_pipeline.py` | Весь пайплайн |
| `openai_compat_controller.py` | API-контроллер (OpenAI-совместимый) |

---

## 5. LLMClient

```python
class LLMClient:
    async def generate(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> ChatCompletion:
        """
        Единая точка входа для всех LLM-вызовов.
        
        Features:
        - retry/fallback
        - логирование в Langfuse (@observe)
        - стандартный logging
        """
```

---

## 6. RAGService

```python
class RAGService:
    async def retrieve(
        self,
        query: str,
        collection: str = "khors_books",
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """
        Hybrid search по Qdrant.
        
        1. Embed query
        2. Dense search (vector similarity)
        3. Sparse search (BM25)
        4. Fusion (reciprocal rank)
        """
```

---

## 7. Reranker

```python
class Reranker:
    async def rank(
        self,
        chunks: list[Chunk],
        top_k: int = 3,
    ) -> list[RankedChunk]:
        """
        Переранжирование чанков после гибридного поиска.
        """
```

---

## 8. MainPipeline

```python
class MainPipeline:
    @staticmethod
    async def process(user_query: str) -> str:
        # 1. Переписывание запроса (уточнение интента)
        rewritten_query = await llm_client.generate(
            messages=[system_prompt_rewrite, user_query]
        )

        # 2. Поиск по книгам Хорса
        chunks = await rag_service.retrieve(
            query=rewritten_query,
            collection="khors_books"
        )

        # 3. Реранкинг
        ranked_chunks = await reranker.rank(chunks)

        # 4. Генерация ответа с контекстом
        answer = await llm_client.generate(
            messages=[
                system_prompt_khors,
                f"Запрос пользователя: {rewritten_query}",
                f"Контекст из книг Хорса:\n{ranked_chunks}"
            ]
        )
        return answer
```

---

## 9. Observability

### Langfuse
- Трейсинг каждого LLM-вызова через `@observe`

### Logging
- Стандартный `logging` с уровнями (INFO, ERROR)

---

## 10. Архитектурная схема

```
┌─────────────┐
│  OpenWebUI  │
│  (клиент)   │
└──────┬──────┘
       │ POST /v1/chat/completions
       ▼
┌─────────────────────────────┐
│ OpenAICompatController      │
│ (FastAPI роутер)            │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│      MainPipeline           │
├─────────────────────────────┤
│ 1. LLMClient.rewrite()      │
│ 2. RAGService.retrieve()    │
│    └── Qdrant (hybrid)      │
│ 3. Reranker.rank()          │
│ 4. LLMClient.generate()     │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│ ChatCompletionResponse      │
└─────────────────────────────┘
```

---

## 11. Структура проекта

```
src/
├── main.py                         # Точка входа (FastAPI приложение)
├── controllers/
│   └── openai_compat_controller.py # OpenAI-совместимый API
├── services/
│   ├── llm_client.py               # Единая точка входа в LLM
│   ├── rag_service.py              # Hybrid search по Qdrant
│   └── reranker.py                 # Переранжирование
├── pipelines/
│   └── main_pipeline.py            # Основной пайплайн
├── routers/
│   ├── openai_compat.py            # Роутер для /v1/chat/completions
│   └── health.py                   # Healthcheck
├── langfuse_client.py              # Langfuse инициализация
└── settings.py                     # Конфигурация
```

---

## 12. Системные промпты

### Промпт для переписывания запроса
```
Ты — ассистент. Перепиши запрос пользователя так, чтобы он стал более точным и полным для поиска в базе знаний по психологии. Сохрани суть вопроса.
```

### Промпт для генерации ответа
```
Ты — нейропсихолог, работающий в подходе Михаила Хорса.

Принципы:
1. Эмоции зависят не от обстоятельств, а от отношения к ним
2. Негативные установки можно менять
3. Реальность многовариантна
4. Ментальное здоровье — работа с мыслями

Отвечай:
- Чётко и по делу
- С опорой на книги Хорса (используй контекст из RAG)
- Если не знаешь — скажи честно
- Без воды и обещаний чуда
- В стиле Хорса — прямо, без эзотерики
```

---

## 13. Технологический стек

| Компонент | Технология |
|---|---|
| API | FastAPI |
| LLM | OpenAI API (или совместимые) |
| Vector DB | Qdrant |
| Embeddings | text-embedding-3-small / аналоги |
| Sparse search | Qdrant BM25 |
| Tracing | Langfuse |
| Logging | Python logging |


---

## 14. Запросы и ответы (примеры)

### Пример 1

**Запрос:**
```
что делать если я постоянно тревожусь
```

**Ответ (в стиле Хорса):**
```
Тревога — это не враг. Это сигнал, что твои мысли интерпретируют ситуацию как опасную. 

По Хорсу, эмоции зависят не от событий, а от того, как мы их трактуем. 
Задай себе вопрос: "Что именно я говорю себе об этой ситуации?" 
Скорее всего, ты используешь катастрофизацию ("а вдруг", "что если"). 

Попробуй технику "многовариантности": напиши 3-5 альтернативных сценариев развития событий, включая позитивные и нейтральные. 
Это снижает напряжение, потому что мозг перестаёт зацикливаться на одном страшном сценарии.
```

### Пример 2

**Запрос:**
```
как перестать зависеть от мнения других
```

**Ответ (в стиле Хорса):**
```
Зависимость от чужого мнения — это привычка ставить оценку других выше своей собственной.

Хорс говорит: "Ты не жертва обстоятельств". То же самое и здесь — ты не жертва чужого мнения. 
Начни с простого: раздели факты и интерпретации. 
"Они сказали Х" — это факт. "Они сказали Х, значит я плохой" — это твоя интерпретация.

Замени установку "мне нужно, чтобы меня одобряли" на "мне важно, чтобы я сам себя уважал". 
Звучит банально, но это работает через ежедневное проговаривание и отслеживание автоматических мыслей.
```

---

## 15. Индексация книг (подготовка данных)

Для наполнения Qdrant:

```python
# scripts/index_books.py
async def index_books():
    books = load_khors_books()  # Загрузка книг Хорса
    chunks = split_into_chunks(books, chunk_size=500)
    embeddings = await embed(chunks)
    await qdrant.upsert(
        collection="khors_books",
        points=chunks_with_embeddings
    )
```

**Формат чанка:**
```json
{
  "id": "uuid",
  "vector": [0.1, 0.2, ...],
  "payload": {
    "text": "текст чанка",
    "book": "Название книги",
    "chapter": "Глава",
    "page": 123
  }
}
```

---

## 16. Запуск

```bash
# Создать окружение под совместимую версию Python
uv venv --python 3.12

# Установить зависимости из pyproject.toml
uv sync

# Запуск сервиса
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

# Проверка health
curl http://localhost:8000/health
```

Если `Python 3.12` ещё не установлен локально, `uv` обычно может скачать его сам:

```bash
uv python install 3.12
uv venv --python 3.12
uv sync
```

---

## 17. Переменные окружения

```env
# LLM
LLM_API_KEY=your_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=openai/gpt-4o-mini

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=khors_books
QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=sparse
QDRANT_FUSION=rrf

# Embeddings
EMBEDDING_MODEL_NAME=BAAI/bge-m3
SPARSE_EMBEDDING_MODEL_NAME=Qdrant/bm25

# Reranker
RERANKER_API_BASE=http://localhost:8001
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

*Документация проекта Neuropsychologist Agent.*
```
