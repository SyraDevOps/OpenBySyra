"""
Luna / Gemma Local · V4
=======================
Chat local GGUF com memória híbrida persistente e recuperação adaptativa:

- Gemma via llama-cpp-python
- streaming de resposta
- Structured Outputs JSON Schema no ReAct
- cache de prefixo / prompt via LlamaCache
- histórico ajustado dinamicamente ao N_CTX
- prompt_toolkit opcional (histórico + autocomplete)
- sandbox com Popen e suporte a eventos/logs intermediários
- módulos + agentes
- roteamento explícito, semântico leve e fuzzy matching
- correção automática de nomes digitados incorretamente
- contador de tokens da conversa inteira
- métricas de geração
- interface de terminal organizada
- memória episódica SQLite + FAISS
- recuperação semântica de memórias antigas
- compactação automática do histórico
- tentativas adaptativas com mudança de estratégia e limite anti-loop

Dependências:
    py -m pip install llama-cpp-python colorama tqdm prompt_toolkit faiss-cpu sentence-transformers numpy

O prompt_toolkit é opcional; se não estiver instalado, há fallback para input().
"""

from __future__ import annotations

from pathlib import Path
from math import sqrt
from difflib import SequenceMatcher
from typing import Any, Iterable
import json
import os
import sqlite3
import hashlib
import queue
import re
import subprocess
import sys
import threading
import time
import unicodedata
import contextlib
import io
import logging

# O carregamento do modelo de memória ocorre em background lógico do runtime;
# não deve interromper a conversa com barras de download e avisos de cache.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from llama_cpp import Llama, LlamaCache
    from colorama import Fore, Style, init
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit(
        "Dependências obrigatórias ausentes.\n"
        "Instale com:\n"
        "  py -m pip install llama-cpp-python colorama tqdm prompt_toolkit faiss-cpu sentence-transformers numpy"
    ) from e

for logger_name in (
    "huggingface_hub",
    "transformers",
    "sentence_transformers",
):
    logging.getLogger(logger_name).setLevel(logging.ERROR)
try:
    from huggingface_hub import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
try:
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except Exception:
    pass

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.formatted_text import ANSI

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

init(autoreset=True)

# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT = Path(__file__).resolve().parent
SANDBOX = ROOT / "sandbox.py"

MODEL_ENV = os.getenv("GEMMA_MODEL", "").strip()
MODEL = ROOT / MODEL_ENV if MODEL_ENV else None

N_CTX = int(os.getenv("GEMMA_CTX", "8192"))
MAX_TOKENS = int(os.getenv("GEMMA_MAX_TOKENS", "1024"))
TOOL_PLAN_TOKENS = int(os.getenv("GEMMA_PLAN_TOKENS", "512"))
CONTEXT_MARGIN = int(os.getenv("GEMMA_CONTEXT_MARGIN", "128"))

THREADS = int(
    os.getenv(
        "GEMMA_THREADS",
        str(max(2, (os.cpu_count() or 4) // 2)),
    )
)
BATCH = int(os.getenv("GEMMA_BATCH", "512"))
GPU_LAYERS = int(os.getenv("GEMMA_GPU_LAYERS", "0"))

TIMEOUT = int(os.getenv("GEMMA_TOOL_TIMEOUT", "60"))
MAX_STEPS = int(os.getenv("GEMMA_REACT_STEPS", "5"))
AUTOWRITE = os.getenv("GEMMA_AUTONOMOUS_WRITE", "0") == "1"

# Memória híbrida
MEMORY_ENABLED = os.getenv("GEMMA_MEMORY", "1") != "0"
MEMORY_TRIGGER_PERCENT = float(os.getenv("GEMMA_MEMORY_TRIGGER", "72"))
MEMORY_RECALL_TOP_K = int(os.getenv("GEMMA_MEMORY_TOP_K", "3"))
MEMORY_MIN_SCORE = float(os.getenv("GEMMA_MEMORY_MIN_SCORE", "0.28"))
MEMORY_SUMMARY_TOKENS = int(os.getenv("GEMMA_MEMORY_SUMMARY_TOKENS", "420"))
MEMORY_CONTEXT_TOKENS = int(os.getenv("GEMMA_MEMORY_CONTEXT_TOKENS", "850"))
EMBED_MODEL = os.getenv(
    "GEMMA_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Recuperação / anti-loop de ferramentas
MAX_TOOL_ATTEMPTS = int(os.getenv("GEMMA_TOOL_ATTEMPTS", "4"))
MAX_REPEAT_ACTION = int(os.getenv("GEMMA_MAX_REPEAT_ACTION", "1"))

# Cache RAM para prefixos/prompt.
# 256 MiB por padrão, bem mais conservador que 2 GiB.
CACHE_MB = int(os.getenv("GEMMA_PROMPT_CACHE_MB", "256"))
CACHE_BYTES = max(16, CACHE_MB) * 1024 * 1024
ENABLE_PROMPT_CACHE = os.getenv("GEMMA_PROMPT_CACHE", "0") != "0"

# Correção de nomes.
AUTO_MATCH_THRESHOLD = float(os.getenv("GEMMA_MATCH_AUTO", "0.78"))
SUGGEST_MATCH_THRESHOLD = float(os.getenv("GEMMA_MATCH_SUGGEST", "0.55"))

# Máximo de texto bruto do resultado de ferramenta oferecido ao modelo.
MAX_TOOL_RESULT_CHARS = int(os.getenv("GEMMA_TOOL_RESULT_CHARS", "14000"))

# Quantos termos após "módulo/agente" usar na identificação inicial.
MAX_EXPLICIT_NAME_WORDS = int(os.getenv("GEMMA_NAME_WORDS", "6"))
PROMPTS_DIR = ROOT / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.json"

COMMANDS = [
    "/agentes",
    "/modulos",
    "/workspace",
    "/catalogo",
    "/tokens",
    "/status",
    "/cache",
    "/memoria",
    "/memorias",
    "/memoria_apagar",
    "/memoria_editar",
    "/limpar",
    "/ajuda",
    "/sair",
]

DEFAULT_SYSTEM_PROMPT = """
Você é Luna, uma assistente local executada por um modelo Gemma.

Idioma e comunicação:
- Responda em português do Brasil, salvo se o usuário pedir outro idioma.
- Seja natural, sociável e fluida.
- Não seja seca nem excessivamente curta.
- Perguntas simples podem ter respostas simples; perguntas técnicas devem receber
  explicação suficiente para o usuário entender o resultado, a causa e o próximo passo.
- Quando houver números, resultados ou execução de módulos, explique o que eles significam.
- Evite repetir a pergunta do usuário ou encher a resposta com introduções desnecessárias.
- Pode conversar casualmente, mantendo coerência com o histórico.

Confiabilidade:
- Não exponha cadeia de pensamento ou raciocínio interno.
- Não invente execução de ferramentas, módulos, agentes, rede ou arquivos.
- Quando receber um resultado real do sandbox, baseie a resposta exclusivamente nele.
- Se algo falhou, diga claramente que falhou e explique o erro real.
- Nunca transforme uma falha em sucesso.
""".strip()


def load_system_prompt() -> str:
    try:
        data = json.loads(SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"))
        sections = [
            f"Nome: {data.get('name', 'Luna')}",
            f"Função: {data.get('role', 'assistente local')}",
            f"Idioma: {data.get('language', 'pt-BR')}",
        ]
        for key, title in (
            ("style", "Estilo"),
            ("reliability", "Confiabilidade"),
            ("execution", "Execução"),
        ):
            values = data.get(key, [])
            if isinstance(values, list) and values:
                sections.append(title + ":\n- " + "\n- ".join(map(str, values)))
        return "\n\n".join(sections)
    except (OSError, json.JSONDecodeError, TypeError):
        return DEFAULT_SYSTEM_PROMPT


FAST_SYSTEM = load_system_prompt()

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["tool", "final"],
        },
        "name": {
            "type": "string",
            "enum": [
                "run_agent",
                "run_module",
                "http_request",
                "list_directory",
                "read_file",
                "write_file",
                "create_agent",
            ],
        },
        "arguments": {
            "type": "object",
        },
        "answer": {
            "type": "string",
        },
        "strategy": {
            "type": "string",
        },
        "can_retry": {
            "type": "boolean",
        },
        "silent": {
            "type": "boolean",
        },
    },
    "required": ["type"],
    "additionalProperties": False,
}

REGISTRY = {
    "fingerprint": None,
    "resources": [],
    "catalogue": {"ok": True, "agents": [], "modules": []},
}


# ============================================================
# VISUAL
# ============================================================

def hr(char: str = "─", width: int = 76) -> None:
    print(Fore.LIGHTBLACK_EX + char * width + Style.RESET_ALL)


def title(text: str) -> None:
    hr("═")
    print(Fore.MAGENTA + Style.BRIGHT + f"  {text}" + Style.RESET_ALL)
    hr("═")


def section(icon: str, text: str, color: str = Fore.CYAN) -> None:
    print()
    print(color + Style.BRIGHT + f"{icon} {text}" + Style.RESET_ALL)


def success(text: str) -> None:
    print(Fore.GREEN + "✓ " + text + Style.RESET_ALL)


def warning(text: str) -> None:
    print(Fore.YELLOW + "⚠ " + text + Style.RESET_ALL)


def error(text: str) -> None:
    print(Fore.RED + "✗ " + text + Style.RESET_ALL)


def transient(text: str) -> int:
    print(Fore.LIGHTBLACK_EX + text + Style.RESET_ALL, end="", flush=True)
    return len(text)


def erase(width: int) -> None:
    print("\r" + (" " * width) + "\r", end="", flush=True)


# ============================================================
# MODELO
# ============================================================

def find_model() -> Path:
    """
    Prioridade:
      1. GEMMA_MODEL
      2. nomes padrão
      3. *.gguf contendo 'gemma'
      4. único *.gguf
    """
    if MODEL is not None:
        if MODEL.is_file():
            return MODEL
        raise SystemExit(
            Fore.RED + f"GEMMA_MODEL aponta para um arquivo inexistente: {MODEL}"
        )

    preferred = (
        ROOT / "Gemma.gguf",
        ROOT / "gemma.gguf",
        ROOT / "GEMMA.gguf",
        ROOT / "model.gguf",
    )
    for p in preferred:
        if p.is_file():
            return p

    ggufs = sorted(ROOT.glob("*.gguf"))
    gemmas = [p for p in ggufs if "gemma" in p.name.casefold()]

    if len(gemmas) == 1:
        return gemmas[0]

    if len(gemmas) > 1:
        return max(gemmas, key=lambda p: p.stat().st_size)

    if len(ggufs) == 1:
        return ggufs[0]

    if not ggufs:
        raise SystemExit(
            Fore.RED
            + "Nenhum GGUF encontrado. Coloque o Gemma ao lado do script "
              "ou defina GEMMA_MODEL=arquivo.gguf."
        )

    names = "\n".join(f"  • {p.name}" for p in ggufs)
    raise SystemExit(
        Fore.YELLOW
        + "Há vários GGUF e não identifiquei o Gemma com segurança.\n"
          "Defina, por exemplo:\n"
          "  set GEMMA_MODEL=gemma-3-1b-it-Q8_0.gguf\n\n"
          "Arquivos encontrados:\n"
        + names
    )


def load_model():
    model_path = find_model()

    title("LUNA · GEMMA LOCAL · V3")
    print(Fore.LIGHTBLACK_EX + f"Modelo       : {model_path.name}")
    print(Fore.LIGHTBLACK_EX + f"Contexto     : {N_CTX:,} tokens")
    print(Fore.LIGHTBLACK_EX + f"Resposta máx.: {MAX_TOKENS:,} tokens")
    print(Fore.LIGHTBLACK_EX + f"Threads      : {THREADS}")
    print(Fore.LIGHTBLACK_EX + f"Batch        : {BATCH}")
    print(Fore.LIGHTBLACK_EX + f"GPU layers   : {GPU_LAYERS}")
    print(
        Fore.LIGHTBLACK_EX
        + f"Prompt cache : {'ON · ' + str(CACHE_MB) + ' MiB' if ENABLE_PROMPT_CACHE else 'OFF'}"
    )
    print()

    bar = tqdm(
        total=1,
        desc="Carregando Gemma",
        unit="modelo",
        dynamic_ncols=True,
    )
    started = time.perf_counter()

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=N_CTX,
            n_threads=THREADS,
            n_threads_batch=THREADS,
            n_batch=BATCH,
            n_gpu_layers=GPU_LAYERS,
            flash_attn=True,
            verbose=False,
        )

        if ENABLE_PROMPT_CACHE:
            try:
                llm.set_cache(LlamaCache(capacity_bytes=CACHE_BYTES))
            except Exception as cache_error:
                warning(
                    "Não foi possível ativar LlamaCache; continuando sem cache. "
                    f"Detalhe: {cache_error}"
                )

        bar.update(1)
    finally:
        bar.close()

    success(f"Gemma pronto em {time.perf_counter() - started:.1f}s")
    return llm, model_path


# ============================================================
# TOKENS / CONTEXTO
# ============================================================

def count_text_tokens(llm, text: str) -> int:
    if not text:
        return 0

    try:
        return len(
            llm.tokenize(
                text.encode("utf-8"),
                add_bos=False,
                special=True,
            )
        )
    except Exception:
        # Fallback apenas para a interface, nunca para o cálculo principal se
        # tokenize estiver funcionando.
        return max(1, len(text) // 4)


def count_message_tokens(llm, message: dict[str, Any]) -> int:
    # 4 tokens de overhead é uma estimativa conservadora do envelope/template.
    return count_text_tokens(llm, str(message.get("content", ""))) + 4


def dynamic_generation_budget(
    llm,
    messages: list[dict[str, str]],
    requested: int,
) -> int:
    used = sum(count_message_tokens(llm, message) for message in messages)
    available = N_CTX - used - CONTEXT_MARGIN
    return max(64, min(int(requested), max(64, available)))


def fit_history_to_context(
    llm,
    history: list[dict[str, str]],
    system_prompt: str,
    *,
    current_user: str = "",
    max_gen_tokens: int = MAX_TOKENS,
    extra_messages: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    Retém as mensagens mais recentes que cabem no N_CTX.

    Reserva:
      - system prompt
      - mensagem atual
      - mensagens extras
      - tokens da geração
      - margem de segurança

    Nunca corta o conteúdo de uma mensagem no meio. Se o texto atual sozinho
    for grande demais, compact_message_to_budget() deve ser usado antes.
    """
    extra_messages = extra_messages or []

    reserved = (
        count_text_tokens(llm, system_prompt)
        + count_text_tokens(llm, current_user)
        + sum(count_message_tokens(llm, m) for m in extra_messages)
        + max_gen_tokens
        + CONTEXT_MARGIN
        + 16
    )

    available = max(0, N_CTX - reserved)

    fitted: list[dict[str, str]] = []
    accumulated = 0

    for msg in reversed(history):
        tokens = count_message_tokens(llm, msg)

        if accumulated + tokens > available:
            break

        fitted.insert(0, msg)
        accumulated += tokens

    return fitted


def compact_text_to_budget(llm, text: str, token_budget: int) -> str:
    """
    Corta um texto enorme preservando começo e fim.
    Útil para resultados de módulos e entradas gigantes.
    """
    if token_budget <= 16:
        return ""

    if count_text_tokens(llm, text) <= token_budget:
        return text

    tokens = llm.tokenize(
        text.encode("utf-8"),
        add_bos=False,
        special=True,
    )

    keep = max(8, (token_budget - 12) // 2)
    head = tokens[:keep]
    tail = tokens[-keep:]

    head_text = llm.detokenize(head).decode("utf-8", errors="replace")
    tail_text = llm.detokenize(tail).decode("utf-8", errors="replace")

    return (
        head_text
        + "\n\n[… conteúdo intermediário removido para caber no contexto …]\n\n"
        + tail_text
    )


def conversation_stats(llm, history: list[dict[str, str]]) -> dict[str, Any]:
    user_tokens = 0
    assistant_tokens = 0

    for msg in history:
        tokens = count_text_tokens(llm, msg.get("content", ""))

        if msg.get("role") == "user":
            user_tokens += tokens
        elif msg.get("role") == "assistant":
            assistant_tokens += tokens

    system_tokens = count_text_tokens(llm, FAST_SYSTEM)
    overhead = len(history) * 4
    context_estimate = (
        user_tokens + assistant_tokens + system_tokens + overhead
    )

    return {
        "user": user_tokens,
        "assistant": assistant_tokens,
        "system": system_tokens,
        "conversation": user_tokens + assistant_tokens,
        "context_estimate": context_estimate,
        "remaining": max(0, N_CTX - context_estimate),
        "percent": min(
            100.0,
            context_estimate / max(1, N_CTX) * 100.0,
        ),
    }


def show_token_bar(
    llm,
    history: list[dict[str, str]],
    session_totals: dict[str, int] | None = None,
) -> None:
    st = conversation_stats(llm, history)

    width = 28
    filled = min(width, round(width * st["percent"] / 100))
    bar = ("█" * filled) + ("░" * (width - filled))

    print(
        Fore.LIGHTBLACK_EX
        + f"  Contexto ativo [{bar}] {st['percent']:.1f}% "
          f"· ~{st['context_estimate']:,}/{N_CTX:,}"
        + Style.RESET_ALL
    )
    if session_totals is None:
        total_user = st["user"]
        total_assistant = st["assistant"]
    else:
        total_user = int(session_totals.get("user", 0))
        total_assistant = int(session_totals.get("assistant", 0))
    print(
        Fore.LIGHTBLACK_EX
        + f"  Conversa inteira: {total_user + total_assistant:,} tokens "
          f"· você {total_user:,} · Luna {total_assistant:,} "
          f"· ativos agora {st['conversation']:,}"
        + Style.RESET_ALL
    )


def show_runtime_panel(
    llm,
    history: list[dict[str, str]],
    session_totals: dict[str, int] | None = None,
    memory: "LongTermMemory | None" = None,
    last_result: dict[str, Any] | None = None,
    prompt_text: str = FAST_SYSTEM,
) -> None:
    """Painel compacto para separar contexto, resposta e dados coletados."""
    st = conversation_stats(llm, history)
    reserved = min(MAX_TOKENS, max(0, N_CTX - st["context_estimate"] - CONTEXT_MARGIN))
    threshold = N_CTX * MEMORY_TRIGGER_PERCENT / 100.0
    print(Fore.LIGHTBLACK_EX + "  ┌─ runtime ───────────────────────────────────────────────┐" + Style.RESET_ALL)
    print(
        Fore.CYAN
        + f"  │ contexto {st['context_estimate']:,}/{N_CTX:,} · {st['percent']:.1f}%"
        + Fore.LIGHTBLACK_EX
        + f" · síntese em {threshold:,.0f} tokens ({MEMORY_TRIGGER_PERCENT:.0f}%)"
        + Style.RESET_ALL
    )
    print(
        Fore.MAGENTA
        + f"  │ resposta reservada: até {reserved:,} tokens"
        + Fore.LIGHTBLACK_EX
        + f" · margem {CONTEXT_MARGIN:,}"
        + Style.RESET_ALL
    )
    prompt_tokens = count_text_tokens(llm, prompt_text)
    print(
        Fore.YELLOW
        + f"  │ system prompt: ~{prompt_tokens:,} tokens"
        + Fore.LIGHTBLACK_EX
        + f" · orçamento restante ~{max(0, N_CTX - st['context_estimate'] - prompt_tokens):,}"
        + Style.RESET_ALL
    )
    memory_count = memory.count() if memory is not None else 0
    print(
        Fore.GREEN
        + f"  │ memória persistente: {'ON' if memory is not None else 'OFF'}"
        + Fore.LIGHTBLACK_EX
        + f" · {memory_count} episódios · contexto antigo vira resumo e sai daqui"
        + Style.RESET_ALL
    )
    if last_result is not None:
        status = "OK" if last_result.get("ok") else "FALHA"
        status_color = Fore.GREEN if last_result.get("ok") else Fore.RED
        detail = []
        if last_result.get("method") and last_result.get("status") is not None:
            detail.append(f"HTTP {last_result['method']} {last_result['status']}")
        if last_result.get("url"):
            detail.append(str(last_result["url"]))
        if last_result.get("path"):
            detail.append(f"arquivo: {last_result['path']}")
        if isinstance(last_result.get("files"), list):
            detail.append(f"itens coletados: {len(last_result['files'])}")
        if last_result.get("error"):
            detail.append(str(last_result["error"])[:180])
        print(
            status_color
            + f"  │ último resultado: {status}"
            + Fore.LIGHTBLACK_EX
            + (" · " + " · ".join(detail) if detail else "")
            + Style.RESET_ALL
        )
    print(Fore.LIGHTBLACK_EX + "  └──────────────────────────────────────────────────────────┘" + Style.RESET_ALL)


# ============================================================
# MEMÓRIA HÍBRIDA · SQLITE + FAISS
# ============================================================

class LongTermMemory:
    """Memória episódica persistente com texto no SQLite e vetores no FAISS."""

    def __init__(self, storage_dir: Path):
        self._lock = threading.RLock()
        self.dir = storage_dir / "memory"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "memory.db"
        self.index_path = self.dir / "memory.index"
        self.embedder = None
        self.dimension = None
        self.index = None
        self._init_sqlite()
        self._load_existing_index()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_sqlite(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    summary TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    source_turns INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT UNIQUE
                )
                """
            )
            # Migração defensiva para bancos criados por versões antigas.
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(memories)")
            }
            if "source_turns" not in cols:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN source_turns INTEGER NOT NULL DEFAULT 0"
                )
            if "content_hash" not in cols:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN content_hash TEXT"
                )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_hash "
                "ON memories(content_hash) WHERE content_hash IS NOT NULL"
            )
            conn.commit()

    def _load_existing_index(self):
        if not self.index_path.is_file():
            return
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.dimension = int(self.index.d)
        except Exception as exc:
            warning(f"Índice FAISS inválido; será reconstruído quando necessário: {exc}")
            self.index = None
            self.dimension = None

    def _ensure_embedder(self):
        if self.embedder is not None:
            return
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.embedder = SentenceTransformer(
                    EMBED_MODEL,
                    local_files_only=os.getenv("GEMMA_EMBED_LOCAL_ONLY", "0") == "1",
                    show_progress_bar=False,
                )
            self.dimension = int(self.embedder.get_sentence_embedding_dimension())
        except TypeError:
            # Compatibilidade com versões antigas de sentence-transformers.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.embedder = SentenceTransformer(EMBED_MODEL)
            self.dimension = int(self.embedder.get_sentence_embedding_dimension())

    def _ensure_index(self):
        self._ensure_embedder()
        if self.index is not None:
            compatible = isinstance(self.index, faiss.IndexIDMap2)
            if int(self.index.d) != int(self.dimension) or not compatible:
                warning(
                    "Índice FAISS antigo/incompatível detectado; reconstruindo com IDs reais do SQLite."
                )
                self.rebuild_index()
            return
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))

    def _encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_embedder()
        vectors = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _atomic_write_index(self):
        if self.index is None:
            return
        tmp = self.index_path.with_suffix(".index.tmp")
        faiss.write_index(self.index, str(tmp))
        os.replace(tmp, self.index_path)

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            return int(row[0] if row else 0)

    def ensure_consistency(self):
        db_count = self.count()
        index_count = int(self.index.ntotal) if self.index is not None else 0
        if db_count == 0:
            if index_count:
                self.index = None
                try:
                    self.index_path.unlink(missing_ok=True)
                except Exception:
                    pass
            return
        if self.index is None or index_count != db_count:
            self.rebuild_index()

    def rebuild_index(self):
        self._ensure_embedder()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, summary FROM memories ORDER BY id"
            ).fetchall()
        fresh = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))
        if rows:
            ids = np.asarray([int(r[0]) for r in rows], dtype=np.int64)
            vectors = self._encode([str(r[1]) for r in rows])
            fresh.add_with_ids(vectors, ids)
        self.index = fresh
        self._atomic_write_index()

    def save_memory(self, summary_text: str, tags: str = "", source_turns: int = 0):
        with self._lock:
            summary_text = summary_text.strip()
            if not summary_text:
                return None

            digest = hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT id FROM memories WHERE content_hash = ?",
                    (digest,),
                ).fetchone()
                if existing:
                    return int(existing[0])

            self._ensure_index()
            vector = self._encode([summary_text])

            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO memories
                    (created_at, summary, tags, source_turns, content_hash)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (time.time(), summary_text, tags, int(source_turns), digest),
                )
                doc_id = int(cur.lastrowid)
                conn.commit()

            try:
                ids = np.asarray([doc_id], dtype=np.int64)
                self.index.add_with_ids(vector, ids)
                self._atomic_write_index()
            except Exception:
                try:
                    self.index.remove_ids(np.asarray([doc_id], dtype=np.int64))
                except Exception:
                    pass
                with self._connect() as conn:
                    conn.execute("DELETE FROM memories WHERE id = ?", (doc_id,))
                    conn.commit()
                raise

            return doc_id

    def recall(self, query: str, top_k: int = MEMORY_RECALL_TOP_K) -> list[dict[str, Any]]:
        with self._lock:
            if not query.strip() or self.count() == 0:
                return []

            self._ensure_index()
            self.ensure_consistency()
            if self.index is None or self.index.ntotal == 0:
                return []

            query_vector = self._encode([query])
            search_k = min(max(top_k * 3, top_k), int(self.index.ntotal))
            scores, ids = self.index.search(query_vector, search_k)

            wanted = [
                (int(doc_id), float(score))
                for doc_id, score in zip(ids[0], scores[0])
                if int(doc_id) >= 0 and float(score) >= MEMORY_MIN_SCORE
            ][:top_k]
            if not wanted:
                return []

            out = []
            with self._connect() as conn:
                for doc_id, score in wanted:
                    row = conn.execute(
                        "SELECT summary, tags, created_at, source_turns "
                        "FROM memories WHERE id = ?",
                        (doc_id,),
                    ).fetchone()
                    if row:
                        out.append(
                            {
                                "id": doc_id,
                                "summary": str(row[0]),
                                "tags": str(row[1] or ""),
                                "created_at": float(row[2]),
                                "source_turns": int(row[3] or 0),
                                "score": score,
                            }
                        )
            return out

    def recent(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, summary, tags, source_turns "
                "FROM memories ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {
                "id": int(r[0]),
                "created_at": float(r[1]),
                "summary": str(r[2]),
                "tags": str(r[3] or ""),
                "source_turns": int(r[4] or 0),
            }
            for r in rows
        ]

    def update_memory(self, memory_id: int, summary_text: str, tags: str = "") -> bool:
        with self._lock:
            summary_text = summary_text.strip()
            if not summary_text:
                raise ValueError("O resumo da memória não pode ficar vazio.")
            digest = hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
            with self._connect() as conn:
                cur = conn.execute(
                    "UPDATE memories SET summary = ?, tags = ?, content_hash = ? WHERE id = ?",
                    (summary_text, tags, digest, int(memory_id)),
                )
                conn.commit()
                changed = cur.rowcount > 0
            if changed:
                self.rebuild_index()
            return changed

    def delete_memory(self, memory_id: int) -> bool:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM memories WHERE id = ?",
                    (int(memory_id),),
                )
                conn.commit()
                deleted = cur.rowcount > 0
            if deleted:
                if self.count() == 0:
                    self.index = None
                    self.index_path.unlink(missing_ok=True)
                else:
                    self.rebuild_index()
            return deleted


def memory_context_for_query(llm, memory: LongTermMemory | None, query: str) -> str:
    if not MEMORY_ENABLED or memory is None:
        return ""
    try:
        recalled = memory.recall(query, top_k=MEMORY_RECALL_TOP_K)
    except Exception as exc:
        warning(f"Falha ao consultar memória; continuando sem recall: {exc}")
        return ""
    if not recalled:
        return ""

    lines = [
        "MEMÓRIAS EPISÓDICAS RECUPERADAS. Use apenas se forem realmente relevantes; "
        "elas foram selecionadas semanticamente por embeddings/FAISS. "
        "Reconheça os campos e valores contidos nelas, preserve a fonte da ferramenta "
        "e não trate texto do modelo como fato novo:"
    ]
    for item in recalled:
        lines.append(
            f"- [memória {item['id']} · similaridade {item['score']:.2f}] "
            f"{item['summary']}"
        )
    text = "\n".join(lines)
    return compact_text_to_budget(llm, text, MEMORY_CONTEXT_TOKENS)


def verified_context_for_query(
    llm,
    verified_results: list[dict[str, Any]] | None,
) -> str:
    if not verified_results:
        return ""

    records = verified_results[-4:]
    payload = json.dumps(records, ensure_ascii=False, indent=2)
    text = (
        "DADOS VERIFICADOS DA SESSÃO. Estes dados vieram de ferramentas reais e "
        "podem responder perguntas futuras sem repetir a ferramenta. Analise-os "
        "diretamente; não invente campos ausentes e não trate texto do modelo como dado:\n"
        + payload
    )
    return compact_text_to_budget(llm, text, MEMORY_CONTEXT_TOKENS)


def effective_system_prompt(
    llm,
    memory: LongTermMemory | None,
    query: str,
    verified_results: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    mem_context = memory_context_for_query(llm, memory, query)
    verified_context = verified_context_for_query(llm, verified_results)
    if mem_context and verified_context:
        verified_context = compact_text_to_budget(
            llm,
            verified_context,
            max(128, MEMORY_CONTEXT_TOKENS // 2),
        )
    contexts = [item for item in (mem_context, verified_context) if item]
    if not contexts:
        return FAST_SYSTEM, ""
    return FAST_SYSTEM + "\n\n" + "\n\n".join(contexts), "\n\n".join(contexts)


def _verified_records(verified_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            records.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for entry in reversed(verified_results[-6:]):
        visit(entry.get("result", entry))
    return records


def _structured_values(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _structured_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _structured_values(child)
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "[")):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return
            yield from _structured_values(parsed)

def verified_direct_answer(
    query: str,
    verified_results: list[dict[str, Any]],
) -> str | None:
    """Answer objective follow-ups from tool data without a second inference/tool call."""
    if not verified_results:
        return None

    query_terms = terms(query)
    query_terms.update(
        token
        for token in re.findall(r"[a-z0-9_]+", normalize(query))
        if token in {"ip", "eu", "meu", "minha", "nome", "onde"}
    )
    if query_terms & {"execute", "executar", "rode", "rodar", "use", "usar", "consulte", "pesquise"}:
        return None

    records = []
    for entry in reversed(verified_results[-6:]):
        records.extend(_structured_values(entry.get("result", entry)))
    latest_module_data = next(
        (
            result.get("data")
            for entry in reversed(verified_results)
            for result in [entry.get("result", {})]
            if isinstance(result, dict) and isinstance(result.get("data"), dict)
        ),
        None,
    )

    if isinstance(latest_module_data, dict):
        if "public" in query_terms and "ip" in query_terms:
            value = latest_module_data.get("public_ip")
            if value:
                return f"Seu IP público verificado é {value}."

        if query_terms & {"cidade", "city"}:
            location = latest_module_data.get("location")
            value = location.get("city") if isinstance(location, dict) else None
            if value:
                return f"A cidade informada pelo último resultado é {value}."

        if query_terms & {"hostname", "host"} and latest_module_data.get("hostname"):
            return f"O hostname verificado é {latest_module_data['hostname']}."

    if query_terms & {"preco", "preço", "valor", "cotacao", "cotação"}:
        for record in records:
            for coin_id, coin_data in record.items():
                if isinstance(coin_data, dict) and any(
                    key in coin_data for key in ("brl", "usd")
                ):
                    values = ", ".join(
                        f"{key.upper()}: {coin_data[key]}"
                        for key in ("brl", "usd")
                        if key in coin_data
                    )
                    return f"Preço verificado de {coin_id}: {values}."
            for key, value in record.items():
                if key in {"brl", "usd", "price", "value"} and isinstance(value, (int, float, str)):
                    return f"O valor verificado é {key.upper()}: {value}."

    return None


def semantic_direct_answer(
    memory: LongTermMemory | None,
    query: str,
) -> str | None:
    """Read objective facts from semantically recalled memories without inference."""
    if memory is None or not MEMORY_ENABLED:
        return None
    query_terms = terms(query)
    query_terms.update(
        token
        for token in re.findall(r"[a-z0-9_]+", normalize(query))
        if token in {"ip", "eu", "meu", "minha", "nome", "onde"}
    )
    try:
        recalled = memory.recall(query, top_k=MEMORY_RECALL_TOP_K)
    except Exception:
        return None

    def walk(value: Any):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for item in recalled:
        summary = str(item.get("summary", ""))
        marker = "Resultado real:"
        if marker not in summary:
            continue
        raw = summary.split(marker, 1)[1].strip()
        try:
            parsed, _ = json.JSONDecoder().raw_decode(raw)
        except json.JSONDecodeError:
            continue
        for record in walk(parsed):
            if "public" in query_terms and "ip" in query_terms and record.get("public_ip"):
                return f"Seu IP público verificado é {record['public_ip']}."
            if query_terms & {"cidade", "city"}:
                location = record.get("location")
                if isinstance(location, dict) and location.get("city"):
                    return f"A cidade informada pelo último resultado é {location['city']}."
            if query_terms & {"hostname", "host"} and record.get("hostname"):
                return f"O hostname verificado é {record['hostname']}."
            if query_terms & {"preco", "preço", "valor", "cotacao", "cotação"}:
                for key in ("brl", "usd", "price", "value"):
                    if isinstance(record.get(key), (int, float, str)):
                        return f"O valor verificado é {key.upper()}: {record[key]}."
    return None

def remember_verified_result(
    memory: LongTermMemory | None,
    question: str,
    tool_name: Any,
    result: dict[str, Any],
) -> None:
    """Arquiva um resultado real em formato curto e semanticamente indexável."""
    if memory is None or not result.get("ok"):
        return

    def compact(value: Any, depth: int = 0) -> Any:
        if depth > 3:
            return "[...]"
        if isinstance(value, dict):
            important = {
                "ok", "status", "module", "agent", "public_ip", "hostname",
                "location", "city", "region", "country", "timezone",
                "brl", "usd", "price", "value", "url", "method", "http_status",
                "path", "items", "content", "body", "data", "crypto_id",
            }
            return {
                str(key): compact(item, depth + 1)
                for key, item in value.items()
                if key in important
            }
        if isinstance(value, list):
            return [compact(item, depth + 1) for item in value[:8]]
        if isinstance(value, str):
            return value[:600]
        return value

    payload = json.dumps(compact(result), ensure_ascii=False, separators=(",", ":"))
    payload = payload[:4000]
    summary = (
        f"Fato verificado pela ferramenta {tool_name}: pedido='{question}'. "
        f"Resultado real: {payload}"
    )
    try:
        memory.save_memory(
            summary,
            tags=f"verified,tool:{tool_name}",
            source_turns=2,
        )
    except Exception as exc:
        warning(f"Não foi possível indexar o resultado na memória semântica: {exc}")


def compress_context_if_needed(
    llm,
    history: list[dict[str, str]],
    memory: LongTermMemory | None,
    trigger_percent: float = MEMORY_TRIGGER_PERCENT,
) -> list[dict[str, str]]:
    if not MEMORY_ENABLED or memory is None:
        return history
    stats = conversation_stats(llm, history)
    if stats["percent"] < trigger_percent or len(history) < 6:
        return history

    # Mantém fronteira em pares user/assistant quando possível.
    split_idx = max(2, (len(history) // 2))
    if split_idx % 2:
        split_idx -= 1
    old_slice = history[:split_idx]
    active_slice = history[split_idx:]
    if not old_slice:
        return history

    warning(
        f"Contexto em {stats['percent']:.1f}%. Arquivando {len(old_slice)} mensagens antigas..."
    )

    chat_text = "\n".join(
        f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
        for msg in old_slice
    )
    summary_system = (
        "Você é o compressor de memória episódica da Luna. Extraia somente informações "
        "úteis para conversas futuras: fatos declarados pelo usuário, preferências, decisões, "
        "nomes de projetos, restrições, resultados técnicos importantes, pendências e relações "
        "entre esses itens. Não invente, não inclua conversa social descartável e não exponha "
        "cadeia de pensamento. Escreva tópicos densos e autoexplicativos em português."
    )
    budget = max(
        256,
        N_CTX
        - count_text_tokens(llm, summary_system)
        - MEMORY_SUMMARY_TOKENS
        - CONTEXT_MARGIN
        - 64,
    )
    safe_chat = compact_text_to_budget(llm, chat_text, budget)

    width = transient("◌ Gemma está consolidando memória antiga...")
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": summary_system},
                {
                    "role": "user",
                    "content": "Consolide esta conversa antiga em memória de longo prazo:\n\n" + safe_chat,
                },
            ],
            temperature=0.18,
            top_p=0.80,
            max_tokens=MEMORY_SUMMARY_TOKENS,
            stream=False,
        )
        summary = str(response["choices"][0]["message"]["content"] or "").strip()
    finally:
        erase(width)

    if not summary:
        warning("O resumo de memória veio vazio; o histórico não foi descartado.")
        return history

    try:
        doc_id = memory.save_memory(summary, source_turns=len(old_slice))
        success(f"Memória episódica {doc_id} salva; contexto ativo foi reduzido.")
        return active_slice
    except Exception as exc:
        error(f"Falha ao persistir memória; mantendo histórico intacto: {exc}")
        return history


# ============================================================
# SANDBOX · POPEN + EVENTOS
# ============================================================

def _stderr_reader(pipe, events: queue.Queue):
    try:
        for line in iter(pipe.readline, ""):
            if line:
                events.put(("stderr", line.rstrip("\r\n")))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _display_sandbox_event(line: str) -> bool:
    """
    Suporta eventos opcionais emitidos pelo sandbox, por exemplo:
      {"event":"log","message":"Baixando página..."}
      {"event":"step","message":"Etapa 1 concluída"}

    Retorna True se a linha foi tratada como evento intermediário.
    """
    stripped = line.strip()
    if not stripped:
        return True

    try:
        obj = json.loads(stripped)
    except Exception:
        return False

    if not isinstance(obj, dict) or "event" not in obj:
        return False

    event = str(obj.get("event", "")).casefold()
    msg = str(
        obj.get("message")
        or obj.get("text")
        or obj.get("detail")
        or ""
    )

    if event in {"log", "progress", "step", "info"}:
        print(Fore.CYAN + f"  [sandbox: {msg}]" + Style.RESET_ALL)
    elif event in {"warning", "warn"}:
        print(Fore.YELLOW + f"  ⚠ {msg}" + Style.RESET_ALL)
    elif event in {"error", "stderr"}:
        print(Fore.RED + f"  ✗ {msg}" + Style.RESET_ALL)
    else:
        print(Fore.LIGHTBLACK_EX + f"  · [{event}] {msg}" + Style.RESET_ALL)

    return True


def _parse_sandbox_stdout(lines: list[str]) -> dict[str, Any]:
    """
    Mantém compatibilidade com sandbox antigo que imprime apenas um JSON final
    e com sandbox novo que pode emitir eventos linha a linha.
    """
    payload_lines: list[str] = []

    for line in lines:
        if not _display_sandbox_event(line):
            payload_lines.append(line)

    raw = "\n".join(payload_lines).strip()

    if not raw:
        return {
            "ok": False,
            "error": "sandbox.py terminou sem retornar um resultado JSON final.",
        }

    # Caso comum: JSON final puro.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        return {"ok": True, "data": obj}
    except Exception:
        pass

    # Recupera o último objeto JSON completo por linha.
    for line in reversed(payload_lines):
        try:
            obj = json.loads(line.strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return {
        "ok": False,
        "error": "Não consegui interpretar a saída final do sandbox como JSON.",
        "stdout": raw[-4000:],
    }


def sandbox(*args, live: bool = True) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(SANDBOX),
        "--json",
        *map(str, args),
    ]

    try:
        p = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}

    assert p.stdout is not None
    assert p.stderr is not None

    stderr_events: queue.Queue = queue.Queue()
    stderr_thread = threading.Thread(
        target=_stderr_reader,
        args=(p.stderr, stderr_events),
        daemon=True,
    )
    stderr_thread.start()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    started = time.monotonic()

    try:
        while True:
            if time.monotonic() - started > TIMEOUT:
                p.kill()
                return {
                    "ok": False,
                    "error": f"Sandbox excedeu o timeout de {TIMEOUT}s.",
                    "stderr": "\n".join(stderr_lines)[-4000:],
                }

            line = p.stdout.readline()

            if line:
                clean = line.rstrip("\r\n")
                stdout_lines.append(clean)

                # Só exibe imediatamente se for evento explícito.
                if live:
                    _display_sandbox_event(clean)

            while True:
                try:
                    kind, msg = stderr_events.get_nowait()
                except queue.Empty:
                    break

                stderr_lines.append(msg)
                if live and msg.strip():
                    print(
                        Fore.LIGHTBLACK_EX
                        + f"  · sandbox: {msg}"
                        + Style.RESET_ALL
                    )

            if line == "" and p.poll() is not None:
                break

            if not line:
                time.sleep(0.02)

        p.wait(timeout=1)

    except Exception as e:
        try:
            p.kill()
        except Exception:
            pass
        return {
            "ok": False,
            "error": str(e),
            "stderr": "\n".join(stderr_lines)[-4000:],
        }

    stderr_thread.join(timeout=0.2)

    try:
        p.stdout.close()
    except Exception:
        pass

    # Drena stderr restante.
    while True:
        try:
            _, msg = stderr_events.get_nowait()
        except queue.Empty:
            break
        stderr_lines.append(msg)

    # Faz parse sem reimprimir eventos já mostrados.
    result = _parse_sandbox_stdout_no_display(stdout_lines)

    if stderr_lines and not result.get("stderr"):
        result["stderr"] = "\n".join(stderr_lines)[-4000:]

    if p.returncode not in (0, None) and result.get("ok", False):
        result["ok"] = False
        result["error"] = (
            result.get("error")
            or f"sandbox.py encerrou com código {p.returncode}."
        )

    return result


def _parse_sandbox_stdout_no_display(lines: list[str]) -> dict[str, Any]:
    payload_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        try:
            obj = json.loads(stripped)
        except Exception:
            payload_lines.append(line)
            continue

        if isinstance(obj, dict) and "event" in obj:
            continue

        # Linha JSON não-evento pode já ser o resultado final.
        payload_lines.append(line)

    raw = "\n".join(payload_lines).strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        return {"ok": True, "data": obj}
    except Exception:
        pass

    for line in reversed(payload_lines):
        try:
            obj = json.loads(line.strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return {
        "ok": False,
        "error": "sandbox.py não retornou JSON final válido.",
        "stdout": raw[-4000:],
    }


def catalogue() -> dict[str, Any]:
    return sandbox("--catalog", live=False)


# ============================================================
# CATÁLOGO / REGISTRO
# ============================================================

def normalize(text: Any) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", str(text).casefold())
        if not unicodedata.combining(c)
    )


def terms(value: Any) -> set[str]:
    plain = normalize(value)

    return {
        x
        for x in re.findall(r"[a-z0-9_]{3,}", plain)
        if x
        not in {
            "para",
            "com",
            "que",
            "uma",
            "por",
            "dos",
            "das",
            "como",
            "sobre",
            "modulo",
            "module",
            "agente",
            "agent",
            "execute",
            "executar",
            "usar",
            "use",
        }
    }


def resource_fingerprint():
    files = (
        list((ROOT / "core").glob("*.py"))
        + list((ROOT / "agents").glob("*.json"))
    )

    return tuple(
        sorted(
            (
                str(x),
                x.stat().st_mtime_ns,
                x.stat().st_size,
            )
            for x in files
            if not x.name.startswith("_")
        )
    )


def registry(force: bool = False) -> dict[str, Any]:
    fp = resource_fingerprint()

    if not force and fp == REGISTRY["fingerprint"]:
        return REGISTRY

    data = catalogue()
    items: list[dict[str, Any]] = []

    for kind, key in (
        ("module", "modules"),
        ("agent", "agents"),
    ):
        for item in data.get(key, []):
            name = str(item.get("name", ""))

            doc = " ".join(
                (
                    name,
                    str(item.get("description", "")),
                    json.dumps(
                        item.get("steps", []),
                        ensure_ascii=False,
                    ),
                )
            )

            items.append(
                {
                    "kind": kind,
                    "name": name,
                    "normalized": normalize(name),
                    "tokens": terms(doc),
                    "description": str(item.get("description", "")),
                }
            )

    REGISTRY.update(
        fingerprint=fp,
        resources=items,
        catalogue=data,
    )

    return REGISTRY


def discovery() -> dict[str, Any]:
    data = registry()

    return {
        "modules": [
            {
                "name": x["name"],
                "description": x["description"],
            }
            for x in data["resources"]
            if x["kind"] == "module"
        ],
        "agents": [
            {
                "name": x["name"],
                "description": x["description"],
            }
            for x in data["resources"]
            if x["kind"] == "agent"
        ],
    }


# ============================================================
# FUZZY MATCHING
# ============================================================

def name_similarity(a: str, b: str) -> float:
    a = normalize(a).replace("-", "_").strip("_ ")
    b = normalize(b).replace("-", "_").strip("_ ")

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    containment = 0.08 if a in b or b in a else 0.0

    ratio = SequenceMatcher(None, a, b).ratio()
    compact = SequenceMatcher(
        None,
        a.replace("_", ""),
        b.replace("_", ""),
    ).ratio()

    return min(1.0, max(ratio, compact) + containment)


def resolve_resource_name(kind: str, typed_name: str) -> dict[str, Any]:
    data = registry(force=True)

    candidates = [
        x for x in data["resources"]
        if x["kind"] == kind
    ]

    if not candidates:
        return {
            "ok": False,
            "resolved": None,
            "exact": False,
            "corrected": False,
            "score": 0.0,
            "suggestions": [],
            "error": (
                "Nenhum módulo disponível."
                if kind == "module"
                else "Nenhum agente disponível."
            ),
        }

    typed_norm = normalize(typed_name)

    for item in candidates:
        if item["normalized"] == typed_norm:
            return {
                "ok": True,
                "resolved": item["name"],
                "exact": True,
                "corrected": False,
                "score": 1.0,
                "suggestions": [],
            }

    ranked = sorted(
        (
            {
                "name": item["name"],
                "score": name_similarity(
                    typed_name,
                    item["name"],
                ),
                "description": item.get("description", ""),
            }
            for item in candidates
        ),
        key=lambda x: x["score"],
        reverse=True,
    )

    best = ranked[0]

    suggestions = [
        x for x in ranked[:5]
        if x["score"] >= SUGGEST_MATCH_THRESHOLD
    ]

    if best["score"] >= AUTO_MATCH_THRESHOLD:
        # Se dois recursos são praticamente empatados, não executa no escuro.
        if (
            len(ranked) > 1
            and ranked[1]["score"] >= best["score"] - 0.04
        ):
            return {
                "ok": False,
                "resolved": None,
                "exact": False,
                "corrected": False,
                "score": best["score"],
                "suggestions": suggestions,
                "ambiguous": True,
            }

        return {
            "ok": True,
            "resolved": best["name"],
            "exact": False,
            "corrected": True,
            "score": best["score"],
            "suggestions": suggestions,
        }

    return {
        "ok": False,
        "resolved": None,
        "exact": False,
        "corrected": False,
        "score": best["score"],
        "suggestions": suggestions,
        "ambiguous": bool(suggestions),
    }


def resolve_action_target(action: dict[str, Any]):
    tool = action.get("name")
    args = action.get("arguments", {})

    if tool not in {"run_module", "run_agent"}:
        return action, None

    if not isinstance(args, dict):
        return action, {
            "ok": False,
            "error": "arguments inválido.",
        }

    kind = "module" if tool == "run_module" else "agent"

    target = (
        args.get("name")
        or args.get("module")
        or args.get("module_name")
        or args.get("agent")
        or args.get("agent_name")
    )

    if not target:
        return action, {
            "ok": False,
            "kind": kind,
            "error": "Nenhum nome foi informado.",
        }

    match = resolve_resource_name(kind, str(target))

    if match["ok"]:
        fixed = dict(action)
        fixed_args = dict(args)
        fixed_args["name"] = match["resolved"]
        fixed["arguments"] = fixed_args

        return fixed, {
            "kind": kind,
            "original": target,
            **match,
        }

    return action, {
        "kind": kind,
        "original": target,
        **match,
    }


def show_match_feedback(match: dict[str, Any] | None) -> None:
    if not match:
        return

    kind = match.get("kind")
    noun = "módulo" if kind == "module" else "agente"

    if match.get("ok") and match.get("corrected"):
        print(
            Fore.YELLOW
            + f"↪ Corrigi o nome: “{match['original']}” → "
              f"“{match['resolved']}” "
              f"({match['score'] * 100:.0f}% de similaridade)"
            + Style.RESET_ALL
        )
        return

    if not match.get("ok") and match.get("suggestions"):
        warning(
            f"Não encontrei exatamente o {noun} “{match.get('original', '')}”."
        )
        print(Fore.LIGHTBLACK_EX + "  Candidatos encontrados:")
        for item in match["suggestions"]:
            desc = (
                f" — {item['description']}"
                if item.get("description")
                else ""
            )
            print(
                Fore.LIGHTBLACK_EX
                + f"  • {item['name']} "
                  f"({item['score'] * 100:.0f}%){desc}"
                + Style.RESET_ALL
            )


# ============================================================
# EXECUÇÃO DE TOOLS
# ============================================================

def execute(call: dict[str, Any], context: str) -> dict[str, Any]:
    name = call.get("name")
    args = call.get("arguments", {})

    if not isinstance(args, dict):
        return {
            "ok": False,
            "error": "arguments deve ser um objeto JSON.",
        }

    

    target = (
        args.get("name")
        or args.get("module")
        or args.get("module_name")
        or args.get("agent")
        or args.get("agent_name")
    )

    if name == "run_agent":
        if not target:
            return {
                "ok": False,
                "error": "run_agent não informou o nome do agente.",
            }

        return sandbox(
            "--agent",
            target,
            "--context",
            args.get("context", context),
            live=not bool(call.get("silent")),
        )

    if name == "run_module":
        if not target:
            return {
                "ok": False,
                "error": "run_module não informou o nome do módulo.",
            }

        return sandbox(
            "--module",
            target,
            "--module-args",
            json.dumps(
                args.get("args", args.get("input", {})),
                ensure_ascii=False,
            ),
            live=not bool(call.get("silent")),
        )

    if name in {"http_request", "http", "curl"}:
        request = {
            "url": str(args.get("url", "")).strip(),
            "method": str(args.get("method", "GET")).upper(),
        }

        for key in ("headers", "body", "data", "json"):
            if key in args:
                request[key] = args[key]

        if not request["url"]:
            return {"ok": False, "error": "http_request precisa de um url."}

        preflight = http_preflight_steps({"name": name, "arguments": request})
        for item in preflight:
            if item.get("catalog"):
                pre_result = sandbox("--list", item["catalog"], live=False)
            else:
                pre_result = sandbox(
                    "--tool",
                    item["name"],
                    "--tool-args",
                    json.dumps(item["arguments"], ensure_ascii=False),
                    live=False,
                )
            if not pre_result.get("ok"):
                break

        return sandbox(
            "--tool",
            "http",
            "--tool-args",
            json.dumps(request, ensure_ascii=False),
            live=not bool(call.get("silent")),
        )

    if name == "list_directory":
        requested_path = str(args.get("path", ".")).strip()
        normalized_path = requested_path.replace("\\", "/").casefold().rstrip("/")
        virtual_workspace_paths = {
            "",
            ".",
            "./",
            "workspace",
            "/workspace",
            "home/luna/workspace",
            "/home/luna/workspace",
        }
        if normalized_path in virtual_workspace_paths:
            requested_path = "."
        payload = {
            "path": requested_path,
            "recursive": bool(args.get("recursive", False)),
            "limit": int(args.get("limit", 500)),
        }
        return sandbox(
            "--tool",
            "list",
            "--tool-args",
            json.dumps(payload, ensure_ascii=False),
            live=not bool(call.get("silent")),
        )

    if name == "read_file":
        payload = {"path": str(args.get("path", "."))}
        return sandbox(
            "--tool",
            "read",
            "--tool-args",
            json.dumps(payload, ensure_ascii=False),
            live=not bool(call.get("silent")),
        )

    if name == "write_file":
        payload = {
            "path": str(args.get("path", "")),
            "content": args.get("content", ""),
        }
        if not payload["path"]:
            return {"ok": False, "error": "write_file precisa de um path."}
        return sandbox(
            "--tool",
            "write",
            "--tool-args",
            json.dumps(payload, ensure_ascii=False),
            live=True,
        )

    if name == "create_agent":
        if not AUTOWRITE:
            return {
                "ok": False,
                "needs_confirmation": True,
                "error": (
                    "Criação autônoma bloqueada. "
                    "Defina GEMMA_AUTONOMOUS_WRITE=1 para habilitar."
                ),
                "proposal": args,
            }

        command = [
            "--create-agent",
            str(args.get("name", "")),
            "--definition",
            json.dumps(
                args.get("definition", {}),
                ensure_ascii=False,
            ),
        ]

        if args.get("overwrite"):
            command.append("--overwrite")

        return sandbox(*command, live=True)

    return {
        "ok": False,
        "error": f"Ferramenta não permitida: {name}",
    }


# ============================================================
# ROTEAMENTO
# ============================================================

def react_prompt(base_system: str = FAST_SYSTEM) -> str:
    available = discovery()

    return (
        base_system
        + """

Você possui acesso controlado a recursos locais.
A saída desta etapa é validada por JSON Schema; não use Markdown.

Política de execução adaptativa:
- Você pode definir `silent:true` quando uma etapa deve apenas trabalhar no sandbox,
  atualizar arquivos/memória ou coletar dados sem mostrar progresso intermediário ao usuário.
  Use `silent:false` quando o usuário precisa acompanhar uma operação longa ou importante.
- Decida cedo entre ação direta, inspeção antes da ação e planejamento:
    - ação direta: tarefa simples e específica, sem ambiguidade nem múltiplas dependências;
    - inspeção antes da ação: antes de qualquer GET/POST/PUT/PATCH, liste o workspace e confirme o contexto local relevante;
    - planejamento: tarefas com múltiplos passos, ambiguidade, investigação profunda ou muitas variáveis.
- Se a tarefa exigir investigação, planeje primeiro: inspecione o escopo, liste diretórios, leia arquivos ou verifique contexto antes de executar operações pesadas.
- Se uma tentativa falhar, NÃO repita mecanicamente a mesma ação com os mesmos argumentos.
- Antes de tentar novamente, mude algo relevante: ferramenta, módulo/agente, parâmetros,
  consulta, URL, escopo, payload, autenticação ou decomposição da tarefa.
- Se o resultado for parcial ou "quase certo", preserve o que funcionou e tente completar
  apenas a parte que falta enquanto ainda houver tentativas disponíveis.
- Quando houver ambiguidade de nome, descubra primeiro e só então execute.
- Não entre em loop. Se não existir estratégia materialmente diferente, finalize explicando
  o melhor resultado obtido e a limitação real.
- Nunca invente sucesso.
- Em tarefas de rede, prefira HTTP de consulta simples e confirme se o endpoint o que você está acessando faz sentido antes de enviar POST/PUT/PATCH.

Tipos:
1. Módulo:
{"type":"tool","name":"run_module","arguments":{"name":"NOME","args":{}},"silent":false,"strategy":"descrição curta"}
2. Agente:
{"type":"tool","name":"run_agent","arguments":{"name":"NOME","context":"pedido"},"strategy":"descrição curta"}
3. HTTP simples:
{"type":"tool","name":"http_request","arguments":{"url":"https://...","method":"GET"},"strategy":"descrição curta"}
4. Inspecionar diretório:
{"type":"tool","name":"list_directory","arguments":{"path":".","recursive":false,"limit":200},"strategy":"descrição curta"}
5. Ler arquivo:
{"type":"tool","name":"read_file","arguments":{"path":"caminho/arquivo.txt"},"strategy":"descrição curta"}
6. Criar agente SOMENTE se solicitado:
{"type":"tool","name":"create_agent","arguments":{"name":"NOME","definition":{}},"strategy":"descrição curta"}
7. Finalizar:
{"type":"final","answer":"resposta","can_retry":false}

Nunca invente recurso fora da descoberta atual. O Python valida nomes e impede repetição cega.
Recursos atuais:
"""
        + json.dumps(
            available,
            ensure_ascii=False,
            separators=(",", ":"),
        )[:12000]
    )


def classify_tool_result(result: dict[str, Any]) -> str:
    """Classifica execução como success, partial ou failure sem depender do modelo."""
    if result.get("ok"):
        if result.get("partial") is True or result.get("complete") is False:
            return "partial"
        return "success"

    useful_payload = False
    for key in ("data", "stdout", "result", "items", "results"):
        value = result.get(key)
        if value not in (None, "", [], {}, ()):
            useful_payload = True
            break

    text = json.dumps(result, ensure_ascii=False).casefold()
    partial_words = (
        "partial", "parcial", "quase", "incompleto", "incomplete",
        "alguns resultados", "parte conclu", "faltou", "missing",
    )
    if useful_payload or any(word in text for word in partial_words):
        return "partial"
    return "failure"


def decision_policy(text: str, action: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decide se a tarefa deve ir direto, inspecionar antes ou planejar antes."""
    low = text.casefold()
    action_name = str((action or {}).get("name", "")).strip()
    method = str(((action or {}).get("arguments", {}) or {}).get("method", "GET")).upper()

    if needs_planning(text):
        return {
            "mode": "plan",
            "reason": "a tarefa requer exploração, múltiplos passos ou contexto ambíguo antes da ação.",
        }

    if should_auto_inspect(text):
        return {
            "mode": "inspect_then_action",
            "reason": "a ação depende de contexto local ou de uma investigação prévia do workspace.",
        }

    if action_name in {"http_request", "http", "curl"}:
        if method in {"POST", "PUT", "PATCH"}:
            return {
                "mode": "inspect_then_action",
                "reason": "requisições mutáveis exigem confirmação do contexto antes de enviar payloads.",
            }
        if any(token in low for token in ("http", "curl", "api", "url", "endpoint")):
            return {
                "mode": "inspect_then_action",
                "reason": "consultas de rede devem começar com uma inspeção mínima do contexto local.",
            }

    if any(token in low for token in ("pesquise", "consulte", "verifique", "analise", "investigue")):
        return {
            "mode": "inspect_then_action",
            "reason": "a tarefa pede investigação antes de concluir algo.",
        }

    return {
        "mode": "direct",
        "reason": "a tarefa parece simples, específica e executável sem planejamento excessivo.",
    }


def http_preflight_steps(action: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not action or str(action.get("name", "")).strip() not in {"http_request", "http", "curl"}:
        return []

    args = action.get("arguments", {}) or {}
    method = str(args.get("method", "GET")).upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
        return []

    steps: list[dict[str, Any]] = [
        {
            "type": "tool",
            "name": "list_directory",
            "arguments": {"path": ".", "recursive": False, "limit": 50},
            "strategy": "inspecionar o contexto local antes da consulta da rede",
        }
    ]

    if method in {"POST", "PUT", "PATCH"}:
        steps.extend([
            {
                "type": "tool",
                "name": "list_directory",
                "arguments": {"path": "agents", "recursive": False, "limit": 50},
                "catalog": "agents",
                "strategy": "confirmar agentes e payloads relevantes antes do POST",
            },
            {
                "type": "tool",
                "name": "list_directory",
                "arguments": {"path": "core", "recursive": False, "limit": 50},
                "catalog": "core",
                "strategy": "confirmar módulos disponíveis antes de persistir a ação de rede",
            },
        ])
    return steps


def action_signature(action: dict[str, Any]) -> str:
    payload = {
        "name": action.get("name"),
        "arguments": action.get("arguments", {}),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def recovery_observation(
    result: dict[str, Any],
    outcome: str,
    attempt: int,
    max_attempts: int,
    *,
    repeated: bool = False,
) -> str:
    raw = json.dumps(result, ensure_ascii=False)[:MAX_TOOL_RESULT_CHARS]
    if outcome == "success":
        instruction = "A execução foi concluída. Finalize com base no resultado real, salvo se faltar uma etapa explicitamente pedida."
    elif outcome == "partial":
        instruction = (
            "O resultado é PARCIAL. Não abandone o que já funcionou. Preserve os dados úteis e, "
            "se ainda houver tentativas, escolha uma estratégia diferente para completar somente o que falta."
        )
    else:
        instruction = (
            "A tentativa FALHOU. Se houver alternativa viável, mude de estratégia de forma material. "
            "Não repita a mesma chamada com os mesmos argumentos."
        )
    if repeated:
        instruction += " Esta ação já foi tentada; repetir exatamente a mesma assinatura está bloqueado."
    return (
        f"RESULTADO REAL DA TENTATIVA {attempt}/{max_attempts} · estado={outcome}:\n"
        f"{raw}\n\n{instruction}"
    )


def may_need_tools(text: str) -> bool:
    low = text.casefold()

    action_words = (
        "execute",
        "executa",
        "rode",
        "roda",
        "use o modulo",
        "use o módulo",
        "módulo",
        "modulo",
        "agente",
        "agent",
        "pesquise",
        "pesquisa",
        "consulte",
        "acesse",
        "baixe",
        "salve",
        "crie um agente",
        "criar agente",
        "api ",
        "http",
        "curl",
        "url",
        "lista",
        "liste",
        "listar",
        "list",
        "dir",
        "ls ",
        "inspecione",
        "investigue",
        "verifique",
        "analise",
        "analisa",
        "diretório",
        "diretorio",
        "workspace",
        "pasta",
        "arquivo",
        "arquivo(s)",
        "equipe",
        "projeto",
        "json",
        "post",
        "payload",
    )

    return any(word in low for word in action_words)


def should_auto_inspect(text: str) -> bool:
    low = text.casefold()

    if not may_need_tools(text):
        return False

    if any(token in low for token in (
        "liste o diretório",
        "lista o diretório",
        "lista os arquivos",
        "mostre a estrutura",
        "veja a pasta",
        "ver o workspace",
        "explore o workspace",
        "investigue o diretório",
        "inspecione os arquivos",
        "descubra quais arquivos",
        "procure primeiro",
        "confira primeiro",
    )):
        return True

    if re.search(r"\b(?:list|lista|listar|dir|ls|mostre|veja|ver|explore|investigue|inspecione)\b", low) and re.search(r"\b(?:arquivo|arquivos|pasta|pastas|workspace|diretório|diretorio|projeto|conteúdo|conteudo)\b", low):
        return True

    if any(term in low for term in ("api", "http", "curl", "url", "post", "json", "payload")) and any(term in low for term in ("arquivo", "pasta", "workspace", "projeto", "diretório", "diretorio")):
        return True

    if len(re.findall(r"\b(?:pesquise|consulte|verifique|liste|listar|analise|acesse|baixe|rode|use|execute|crie|salve|inspecione|investigue)\b", low)) >= 2:
        return True

    if len(re.findall(r"\b\w+\b", low)) >= 16:
        return True

    return False


def needs_planning(text: str) -> bool:
    low = text.casefold()

    if not may_need_tools(text):
        return False

    triggers = (
        "investigue",
        "inspecione",
        "liste o diretório",
        "liste o diretorio",
        "verifique antes",
        "analise antes",
        "confira antes",
        "faça um plano",
        "crie um plano",
        "planeje",
        "descubra",
        "explore",
        "navegue",
        "compare",
        "reconheça",
        "reconheca",
        "pesquise e",
        "consulte e",
        "verifique e",
        "descubra quais arquivos",
        "procure primeiro",
        "confira primeiro",
        "lista os arquivos",
        "veja o diretório",
        "faça uma inspeção",
        "faca uma inspecao",
        "tenho que descobrir",
    )

    if any(trigger in low for trigger in triggers):
        return True

    if should_auto_inspect(text):
        return True

    if len(re.findall(r"\b(?:pesquise|consulte|verifique|liste|listar|analise|acesse|baixe|rode|use|execute|crie|salve|inspecione|investigue)\b", low)) > 1:
        return True

    if len(re.findall(r"\b\w+\b", low)) >= 22:
        return True

    if re.search(r"\b(?:dir|ls|list|lista|listar)\b", low) and re.search(r"\b(?:arquivo|pasta|workspace|diretório|diretorio|projeto|conteúdo|conteudo)\b", low):
        return True

    return False


def _best_explicit_candidate(
    kind: str,
    tail: list[str],
) -> str:
    """
    Em vez de tail[0], testa combinações progressivas:
      gerar
      gerar_relatorio
      gerar_relatorio_mensal
      ...

    Escolhe a combinação que mais se parece com algum recurso real.
    """
    if not tail:
        return ""

    candidates = [
        "_".join(tail[:n])
        for n in range(
            1,
            min(len(tail), MAX_EXPLICIT_NAME_WORDS) + 1,
        )
    ]

    resources = [
        x for x in registry()["resources"]
        if x["kind"] == kind
    ]

    if not resources:
        return candidates[-1]

    best_candidate = candidates[0]
    best_score = -1.0

    for candidate in candidates:
        score = max(
            name_similarity(candidate, item["name"])
            for item in resources
        )

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate



def extract_json_payload(text: str) -> Any | None:
    for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        candidate = match.group(0)
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def explicit_http_route(text: str) -> dict[str, Any] | None:
    """
    Roteia pedidos explícitos de curl/HTTP sem gastar uma chamada de planner.

    Exemplos:
      "faça um curl para g1.com"
      "curl https://api.github.com"
      "faça um POST para https://... com json {...}"
      "acesse example.com/teste"
    """
    cleaned = re.sub(
        r"(?<=[A-Za-z0-9]),(?=[A-Za-z]{2,6}(?:\b|/))",
        ".",
        text.strip(),
    )

    method = "GET"
    method_match = re.search(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD)\b", cleaned, re.I)
    if method_match:
        method = method_match.group(1).upper()

    if re.search(r"\b(post|poster|envie|enviar|submit|payload|json)\b", cleaned, re.I):
        method = "POST"

    payload = extract_json_payload(cleaned)
    json_payload = payload if isinstance(payload, (dict, list)) else None

    patterns = (
        r"\bcurl\s+(?:-X\s*(?P<method>[A-Z]+)\s+)?(?:para\s+|em\s+|no\s+|na\s+)?(?P<url>https?://[^\s\"']+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s\"']*)?)",
        r"\b(?:faça\s+um|faca\s+um|faça\s+o|faca\s+o|curl|acesse|consulte|envie|poste|post|envia|enviar)\s+(?:-X\s*(?P<method2>[A-Z]+)\s+)?(?:para\s+|em\s+|no\s+|na\s+)?(?P<url2>https?://[^\s\"']+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s\"']*)?)",
    )

    for pattern in patterns:
        match = re.search(pattern, cleaned, re.I)
        if not match:
            continue

        target = (match.groupdict().get("url") or match.groupdict().get("url2") or "").strip().rstrip(".,;!?")
        detected_method = (match.groupdict().get("method") or match.groupdict().get("method2") or method).upper()

        if not target:
            continue

        if not target.startswith(("http://", "https://")):
            if "." in target and "//" not in target:
                target = "https://" + target

        if (
            "://" in target
            or "." in target
            or target.casefold() == "localhost"
        ):
            arguments = {"url": target, "method": detected_method}
            if json_payload is not None and detected_method.upper() in {"POST", "PUT", "PATCH"}:
                arguments["json"] = json_payload
            return {
                "type": "tool",
                "name": "http_request",
                "arguments": arguments,
                "_explicit": True,
                "_deterministic": True,
            }

    return None


def explicit_route(text: str) -> dict[str, Any] | None:
    http_action = explicit_http_route(text)
    if http_action:
        return http_action

    low = text.casefold()
    if (
        re.search(r"\b(?:ip|endereço|endereco)\b", low)
        and re.search(r"\b(?:público|publico|privado|local|cidade|localização|localizacao|onde)\b", low)
    ) or re.search(r"\b(?:qual|onde).{0,30}\b(?:minha|meu)\b.{0,20}\b(?:cidade|localização|localizacao)\b", low):
        return {
            "type": "tool",
            "name": "run_module",
            "arguments": {
                "name": "infoself",
                "args": {"geolocation": True},
                "context": text,
            },
            "_explicit": True,
            "_deterministic": True,
        }

    if re.search(r"\b(?:lista|liste|listar|mostre|ver|veja|dir|ls)\b", low) and re.search(r"\b(?:arquivo|arquivos|pasta|pastas|workspace|diretório|diretorio|conteúdo|conteudo)\b", low):
        return {
            "type": "tool",
            "name": "list_directory",
            "arguments": {"path": ".", "recursive": False, "limit": 500},
            "_explicit": True,
            "_deterministic": True,
        }

    if re.search(r"\b(?:leia|lê|le o|abra|read)\b", low) and re.search(r"\b(?:arquivo|file|txt|json|md|py)\b", low):
        path_match = re.search(r"(?:arquivo|file|\bpath\b|\bpath do arquivo\b)\s+[\"']?([A-Za-z0-9_./\\-]+)[\"']?", text, re.I)
        return {
            "type": "tool",
            "name": "read_file",
            "arguments": {"path": path_match.group(1) if path_match else "."},
            "_explicit": True,
            "_deterministic": True,
        }

    plain = normalize(text)
    words = re.findall(r"[a-z0-9_]+", plain)

    mappings = (
        ("modulo", "run_module", "module"),
        ("module", "run_module", "module"),
        ("agente", "run_agent", "agent"),
        ("agent", "run_agent", "agent"),
    )

    for label, tool, kind in mappings:
        if label not in words:
            continue

        index = words.index(label)
        tail = words[index + 1 :]

        while tail and tail[0] in {
            "o",
            "a",
            "um",
            "uma",
            "chamado",
            "chamada",
            "nome",
            "de",
            "por",
            "favor",
        }:
            tail.pop(0)

        if not tail:
            continue

        stop_words = {
            "e", "me", "diga", "fale", "mostre", "depois",
            "entao", "então", "para", "quero", "preciso",
            "sobre", "com", "que",
        }
        resource_tail = []
        for token in tail:
            if resource_tail and token in stop_words:
                break
            resource_tail.append(token)

        candidate_name = _best_explicit_candidate(
            kind,
            resource_tail or tail[:1],
        )

        return {
            "type": "tool",
            "name": tool,
            "arguments": {
                "name": candidate_name,
                "context": text,
            },
            "_explicit": True,
        }

    return None


def direct_route(text: str) -> dict[str, Any] | None:
    if not may_need_tools(text):
        return None

    low = text.casefold()

    wanted = (
        "module"
        if ("modulo" in low or "módulo" in low)
        else "agent"
        if ("agente" in low or "agent" in low)
        else None
    )

    query = terms(text)
    best = None
    best_score = 0.0

    for item in registry()["resources"]:
        if wanted and item["kind"] != wanted:
            continue

        name_terms = terms(item["name"])

        overlap = len(query & item["tokens"]) / max(
            0.1,
            sqrt(
                max(1, len(query))
                * max(1, len(item["tokens"]))
            ),
        )

        name_bonus = (
            2.0
            if name_terms and name_terms <= query
            else 0.0
        )

        score = overlap + name_bonus

        if score > best_score:
            best = item
            best_score = score

    if best and (
        best_score >= 2.0
        or (wanted and best_score >= 0.30)
    ):
        tool = (
            "run_module"
            if best["kind"] == "module"
            else "run_agent"
        )

        return {
            "type": "tool",
            "name": tool,
            "arguments": {
                "name": best["name"],
                "context": text,
            },
            "_score": best_score,
        }

    return None


# ============================================================
# STRUCTURED OUTPUT / REACT
# ============================================================

def decode_plan_response(response: dict[str, Any]) -> dict[str, Any]:
    try:
        content = response["choices"][0]["message"]["content"]
    except Exception:
        return {
            "type": "final",
            "answer": "A etapa de planejamento não retornou conteúdo.",
        }

    if not content:
        return {
            "type": "final",
            "answer": "A etapa de planejamento retornou conteúdo vazio.",
        }

    try:
        data = json.loads(content)
    except Exception:
        # Normalmente não ocorrerá por causa de response_format/schema.
        return {
            "type": "final",
            "answer": content,
        }

    if not isinstance(data, dict):
        return {
            "type": "final",
            "answer": str(data),
        }

    return data



def normalize_chat_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Normaliza mensagens para templates rígidos como Gemma:
      system -> user -> assistant -> user -> ...

    - junta mensagens consecutivas do mesmo papel;
    - remove assistant órfão no início;
    - mantém no máximo um system no começo.

    Isso evita:
      Conversation roles must alternate user/assistant/user/assistant/...
    """
    system_parts: list[str] = []
    body: list[dict[str, str]] = []

    for raw in messages:
        if not isinstance(raw, dict):
            continue

        role = str(raw.get("role", "")).strip()
        content = str(raw.get("content", "") or "").strip()

        if not content:
            continue

        if role == "system":
            system_parts.append(content)
            continue

        if role not in {"user", "assistant"}:
            continue

        # Gemma não aceita assistant como primeira mensagem depois do system.
        if not body and role == "assistant":
            continue

        if body and body[-1]["role"] == role:
            body[-1]["content"] += "\n\n" + content
        else:
            body.append({
                "role": role,
                "content": content,
            })

    out: list[dict[str, str]] = []

    if system_parts:
        out.append({
            "role": "system",
            "content": "\n\n".join(system_parts),
        })

    out.extend(body)
    return out


def ask_plan(llm, messages: list[dict[str, str]]):
    width = transient("◌ Analisando ação...")
    started = time.perf_counter()
    messages = normalize_chat_messages(messages)

    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.10,
            top_p=0.75,
            max_tokens=dynamic_generation_budget(llm, messages, TOOL_PLAN_TOKENS),
            stream=False,
            response_format={
                "type": "json_object",
                "schema": PLAN_SCHEMA,
            },
        )
    finally:
        erase(width)

    return decode_plan_response(response), time.perf_counter() - started


# ============================================================
# GERAÇÃO
# ============================================================

import itertools

class ThinkingAnimator:
    """Anima frases dinâmicas enquanto o modelo processa o prompt."""

    DEFAULT_PHRASES = [
        "Um minuto...",
        "Pensando um pouco...",
        "Analisando o contexto...",
        "Organizando as ideias...",
        "Formulando a resposta...",
        "Consultando a memória...",
        "Quase pronto...",
    ]

    def __init__(self, phrases: list[str] | None = None, interval: float = 1.6):
        self.phrases = phrases or self.DEFAULT_PHRASES
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_len = 0

    def _animate(self) -> None:
        dots_cycle = itertools.cycle([".  ", ".. ", "..."])
        phrase_cycle = itertools.cycle(self.phrases)
        
        while not self._stop_event.is_set():
            phrase = next(phrase_cycle)
            # Troca os pontinhos dentro da mesma frase antes de mudar de frase
            for _ in range(int(self.interval / 0.35)):
                if self._stop_event.is_set():
                    break
                dot = next(dots_cycle)
                text = Fore.LIGHTBLACK_EX + f"  ◌ {phrase.rstrip('.')} {dot}" + Style.RESET_ALL
                
                # Apaga a linha anterior e escreve a nova
                print(f"\r{' ' * max(self._last_len + 4, len(text))}\r{text}", end="", flush=True)
                self._last_len = len(text)
                time.sleep(0.35)

    def start(self) -> "ThinkingAnimator":
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.3)
        # Limpa completamente a linha antes de Luna começar a falar
        print(f"\r{' ' * (self._last_len + 10)}\r", end="", flush=True)


def stream_final(
    llm,
    messages: list[dict[str, str]],
    max_tokens: int = MAX_TOKENS,
):
    # Inicia a animação em background
    animator = ThinkingAnimator().start()
    started = time.perf_counter()
    first = None
    answer = ""
    messages = normalize_chat_messages(messages)
    generation_budget = dynamic_generation_budget(llm, messages, max_tokens)

    try:
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=0.68,
            top_p=0.90,
            top_k=40,
            min_p=0.05,
            repeat_penalty=1.08,
            max_tokens=generation_budget,
            stream=True,
        )

        for chunk in stream:
            text = (
                chunk.get("choices", [{}])[0]
                .get("delta", {})
                .get("content", "")
            )

            if not text:
                continue

            if first is None:
                # O primeiro token chegou: para a animação e apaga a frase
                animator.stop()
                first = time.perf_counter()
                print(
                    Fore.MAGENTA
                    + Style.BRIGHT
                    + "Luna > "
                    + Style.RESET_ALL,
                    end="",
                    flush=True,
                )

            answer += text
            print(
                Fore.WHITE + text + Style.RESET_ALL,
                end="",
                flush=True,
            )

    except MemoryError:
        animator.stop()
        try:
            llm.set_cache(None)
        except Exception:
            pass
        warning("Memória insuficiente para manter o cache do modelo; resposta reduzida.")
        answer = "Não consegui gerar uma explicação longa agora, mas a ferramenta foi executada."
        print(Fore.MAGENTA + Style.BRIGHT + "Luna > " + Style.RESET_ALL + answer)
        return answer, (0.0, 0.0, 0, time.perf_counter() - started)
    except ValueError as e:
        animator.stop()
        error(f"Falha de inferência: {e}")
        return (
            "Não consegui gerar a resposta porque o contexto excedeu o limite "
            "ou a configuração do modelo recusou a entrada.",
            (0.0, 0.0, 0, time.perf_counter() - started),
        )
    finally:
        # Garante que a animação seja parada em qualquer cenário
        animator.stop()

    end = time.perf_counter()

    if first is None:
        print(
            Fore.MAGENTA
            + Style.BRIGHT
            + "Luna > "
            + Style.RESET_ALL
            + "(sem conteúdo)"
        )
    else:
        print()

    tokens = count_text_tokens(llm, answer) if answer else 0
    generation_time = max(0.001, end - (first or started))
    tok_s = tokens / generation_time if tokens else 0.0
    first_token = (first or end) - started

    return (
        answer.strip(),
        (
            tok_s,
            first_token,
            tokens,
            end - started,
        ),
    )

def build_chat_messages(
    llm,
    history: list[dict[str, str]],
    question: str,
    system_prompt: str = FAST_SYSTEM,
) -> list[dict[str, str]]:
    question_budget = max(
        128,
        N_CTX
        - count_text_tokens(llm, system_prompt)
        - MAX_TOKENS
        - CONTEXT_MARGIN
        - 32,
    )
    safe_question = compact_text_to_budget(llm, question, question_budget)
    fitted = fit_history_to_context(
        llm,
        history,
        system_prompt,
        current_user=safe_question,
        max_gen_tokens=MAX_TOKENS,
    )
    return [
        {"role": "system", "content": system_prompt},
        *fitted,
        {"role": "user", "content": safe_question},
    ]


def build_react_messages(
    llm,
    history: list[dict[str, str]],
    question: str,
    base_system: str = FAST_SYSTEM,
) -> list[dict[str, str]]:
    sys_prompt = react_prompt(base_system)
    question_budget = max(
        128,
        N_CTX
        - count_text_tokens(llm, sys_prompt)
        - TOOL_PLAN_TOKENS
        - CONTEXT_MARGIN
        - 32,
    )
    safe_question = compact_text_to_budget(llm, question, question_budget)
    fitted = fit_history_to_context(
        llm,
        history,
        sys_prompt,
        current_user=safe_question,
        max_gen_tokens=TOOL_PLAN_TOKENS,
    )
    return [
        {"role": "system", "content": sys_prompt},
        *fitted,
        {"role": "user", "content": safe_question},
    ]


def _final_from_trace(
    llm,
    question: str,
    history: list[dict[str, str]],
    base_system: str,
    trace: list[dict[str, Any]],
    answer_hint: str = "",
):
    trace_text = json.dumps(trace[-MAX_TOOL_ATTEMPTS:], ensure_ascii=False)[:MAX_TOOL_RESULT_CHARS]
    instruction = (
        "Responda ao pedido com base apenas nos resultados reais das tentativas. "
        "Explique o que foi concluído, o que ficou parcial e qualquer limitação. "
        "Não mencione cadeia de pensamento nem invente sucesso."
    )
    if answer_hint:
        instruction += "\nResposta-base estruturada: " + answer_hint
    payload = f"Pedido: {question}\n\nHistórico real de tentativas:\n{trace_text}\n\n{instruction}"
    fitted = fit_history_to_context(
        llm,
        history,
        base_system,
        current_user=payload,
        max_gen_tokens=MAX_TOKENS,
    )
    return stream_final(
        llm,
        [
            {"role": "system", "content": base_system},
            *fitted,
            {"role": "user", "content": payload},
        ],
    )


def react(
    llm,
    question: str,
    history: list[dict[str, str]],
    base_system: str = FAST_SYSTEM,
    *,
    seed_action: dict[str, Any] | None = None,
    seed_result: dict[str, Any] | None = None,
):
    messages = build_react_messages(llm, history, question, base_system)
    seen: dict[str, int] = {}
    trace: list[dict[str, Any]] = []
    attempts = 0
    last_outcome = ""

    if seed_action is not None and seed_result is not None:
        sig = action_signature(seed_action)
        seen[sig] = 1
        attempts = 1
        last_outcome = classify_tool_result(seed_result)
        trace.append({
            "attempt": attempts,
            "action": seed_action,
            "outcome": last_outcome,
            "result": seed_result,
        })
        observation = recovery_observation(
            seed_result,
            last_outcome,
            attempts,
            MAX_TOOL_ATTEMPTS,
        )
        sys_prompt = react_prompt(base_system)
        messages = [
            {"role": "system", "content": sys_prompt},
            *fit_history_to_context(
                llm,
                history,
                sys_prompt,
                current_user=question + observation,
                max_gen_tokens=TOOL_PLAN_TOKENS,
            ),
            {"role": "user", "content": question},
            {"role": "assistant", "content": json.dumps(seed_action, ensure_ascii=False)},
            {"role": "user", "content": observation},
        ]
        if last_outcome == "success":
            return _final_from_trace(llm, question, history, base_system, trace)

    plan_cycles = 0
    max_plan_cycles = max(MAX_STEPS, MAX_TOOL_ATTEMPTS + 2)

    while plan_cycles < max_plan_cycles:
        plan_cycles += 1
        action, _ = ask_plan(llm, messages)

        if action.get("type") != "tool" and not trace and may_need_tools(question):
            fallback = explicit_route(question) or direct_route(question)
            if fallback is not None:
                action = fallback
            else:
                messages.extend([
                    {
                        "role": "assistant",
                        "content": json.dumps(action, ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": (
                            "O pedido exige uma ferramenta. Não invente dados nem finalize; "
                            "retorne uma ação JSON executável ou explique que não há ferramenta adequada."
                        ),
                    },
                ])
                continue

        if action.get("type") != "tool":
            answer_hint = str(action.get("answer", "")).strip()
            can_retry = action.get("can_retry")

            # Um resultado parcial não pode encerrar cedo enquanto ainda existe orçamento.
            if last_outcome == "partial" and attempts < MAX_TOOL_ATTEMPTS:
                push = (
                    "O último resultado ainda é parcial e há tentativas disponíveis. "
                    "Não finalize agora: proponha uma ação materialmente diferente para completar o que falta."
                )
                messages.extend([
                    {
                        "role": "assistant",
                        "content": json.dumps(action, ensure_ascii=False),
                    },
                    {"role": "user", "content": push},
                ])
                continue

            if trace:
                return _final_from_trace(
                    llm,
                    question,
                    history,
                    base_system,
                    trace,
                    answer_hint,
                )

            final_instruction = (
                "Responda ao usuário em linguagem natural, sem JSON e sem mencionar planejamento interno."
            )
            if answer_hint:
                final_instruction += "\nUse como base: " + answer_hint
            final_messages = [
                {"role": "system", "content": base_system},
                *fit_history_to_context(
                    llm,
                    history,
                    base_system,
                    current_user=question + final_instruction,
                    max_gen_tokens=MAX_TOKENS,
                ),
                {
                    "role": "user",
                    "content": question + "\n\n" + final_instruction,
                },
            ]
            return stream_final(llm, final_messages)

        action, match = resolve_action_target(action)
        show_match_feedback(match)
        if match and not match.get("ok"):
            synthetic = {
                "ok": False,
                "error": "Nome de recurso ambíguo ou inexistente.",
                "suggestions": [x["name"] for x in match.get("suggestions", [])],
            }
            last_outcome = "failure"
            observation = recovery_observation(
                synthetic,
                last_outcome,
                max(1, attempts),
                MAX_TOOL_ATTEMPTS,
            )
            messages.extend([
                {
                    "role": "assistant",
                    "content": json.dumps(action, ensure_ascii=False),
                },
                {"role": "user", "content": observation},
            ])
            continue

        sig = action_signature(action)
        repeat_count = seen.get(sig, 0)
        if repeat_count >= MAX_REPEAT_ACTION:
            blocked = {
                "ok": False,
                "error": "Ação idêntica bloqueada pelo controlador anti-loop.",
                "action": action,
            }
            last_outcome = (
                "partial" if any(t.get("outcome") == "partial" for t in trace) else "failure"
            )
            messages.extend([
                {
                    "role": "assistant",
                    "content": json.dumps(action, ensure_ascii=False),
                },
                {
                    "role": "user",
                    "content": recovery_observation(
                        blocked,
                        last_outcome,
                        max(1, attempts),
                        MAX_TOOL_ATTEMPTS,
                        repeated=True,
                    ),
                },
            ])
            continue

        if attempts >= MAX_TOOL_ATTEMPTS:
            break

        attempts += 1
        seen[sig] = repeat_count + 1
        strategy = str(action.get("strategy", "")).strip()
        label = f"Tentativa {attempts}/{MAX_TOOL_ATTEMPTS}"
        if strategy:
            label += f" · {strategy[:90]}"
        section("⚙", label, Fore.YELLOW)

        result = execute(action, question)
        outcome = classify_tool_result(result)
        last_outcome = outcome
        trace.append({
            "attempt": attempts,
            "action": action,
            "outcome": outcome,
            "result": result,
        })

        if outcome == "success":
            success(f"Tentativa {attempts} concluída")
        elif outcome == "partial":
            warning(f"Tentativa {attempts} trouxe resultado parcial; vou tentar completar.")
        else:
            error(f"Tentativa {attempts} falhou; a próxima ação deve mudar de estratégia.")

        observation = recovery_observation(
            result,
            outcome,
            attempts,
            MAX_TOOL_ATTEMPTS,
        )
        sys_prompt = react_prompt(base_system)
        operational = messages[1:] + [
            {
                "role": "assistant",
                "content": json.dumps(action, ensure_ascii=False),
            },
        ]
        messages = normalize_chat_messages([
            {"role": "system", "content": sys_prompt},
            *fit_history_to_context(
                llm,
                operational,
                sys_prompt,
                current_user=observation,
                max_gen_tokens=TOOL_PLAN_TOKENS,
            ),
            {"role": "user", "content": observation},
        ])

        if outcome == "success":
            # Ainda dá uma chance ao planner de perceber se a tarefa tinha mais de uma etapa.
            continue

    if trace:
        return _final_from_trace(llm, question, history, base_system, trace)

    answer = "Não encontrei uma estratégia executável dentro do limite de tentativas."
    print(Fore.MAGENTA + Style.BRIGHT + "Luna > " + Style.RESET_ALL + answer)
    return answer, (0.0, 0.0, 0, 0.0)

def explain_direct(
    llm,
    question: str,
    result: dict[str, Any],
    history: list[dict[str, str]],
    base_system: str = FAST_SYSTEM,
):
    # Extrai o texto limpo do arquivo lido ou da resposta da ferramenta
    conteudo_real = ""
    if isinstance(result, dict) and "steps" in result:
        for s in reversed(result.get("steps", [])):
            res = s.get("result", {})
            if isinstance(res, dict) and res.get("content"):
                conteudo_real = res.get("content")
                break
            elif isinstance(res, dict) and res.get("body"):
                conteudo_real = res.get("body")
                break
            elif isinstance(res, dict) and res.get("data"):
                conteudo_real = json.dumps(res.get("data"), ensure_ascii=False)
                break
    
    if not conteudo_real:
        conteudo_real = json.dumps(result, ensure_ascii=False)

    safe_result = compact_text_to_budget(llm, str(conteudo_real), 3500)

    system_prompt = (
        FAST_SYSTEM
        + "\n\nVocê acabou de receber os dados reais coletados pela ferramenta. "
          "Responda à pergunta do usuário EXCLUSIVAMENTE com base nesses dados. "
          "NÃO retorne código JSON e NÃO mencione etapas internas do sandbox. "
          "Responda em português natural, direto e correto."
    )

    user_payload = (
        f"Pergunta do usuário:\n{question}\n\n"
        f"Dados coletados pela ferramenta:\n{safe_result}\n\n"
        "Com base nos dados coletados acima, responda à pergunta do usuário com precisão."
    )

    short_history = []

    fitted = fit_history_to_context(
        llm,
        short_history,
        system_prompt,
        current_user=user_payload,
        max_gen_tokens=MAX_TOKENS,
    )

    return stream_final(
        llm,
        [
            {"role": "system", "content": system_prompt},
            *fitted,
            {"role": "user", "content": user_payload},
        ],
        max_tokens=min(MAX_TOKENS, 512),
    )

# ============================================================
# RESULTADOS / STATUS
# ============================================================

def show_actual_result(result: dict[str, Any]) -> None:
    if result.get("error"):
        print(
            Fore.RED
            + "  erro real: "
            + str(result["error"])
            + Style.RESET_ALL
        )

    if result.get("stderr"):
        print(
            Fore.RED
            + "  stderr: "
            + str(result["stderr"])[:1800]
            + Style.RESET_ALL
        )

    if result.get("stdout"):
        print(
            Fore.LIGHTBLACK_EX
            + "  saída: "
            + str(result["stdout"])[:700]
            + Style.RESET_ALL
        )

    if result.get("data") is not None:
        data = result["data"]
        if isinstance(data, dict):
            compact: dict[str, Any] = {}
            for key in ("status", "hostname", "public_ip", "private_ips", "module", "agent"):
                if key in data:
                    compact[key] = data[key]
            location = data.get("location")
            if isinstance(location, dict):
                compact["location"] = {
                    key: location[key]
                    for key in ("city", "region", "country", "timezone")
                    if key in location
                }
            data = compact or {"campos": sorted(data)[:12]}
        print(
            Fore.LIGHTBLACK_EX
            + "  dados: "
            + json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            )[:1200]
            + Style.RESET_ALL
        )


def print_help() -> None:
    section("⌘", "Comandos")
    print("  /agentes      lista agentes")
    print("  /modulos      lista módulos")
    print("  /workspace    lista workspace")
    print("  /catalogo     mostra catálogo completo")
    print("  /tokens       mostra uso acumulado de tokens")
    print("  /status       mostra configuração do runtime")
    print("  /cache        mostra configuração do prompt cache")
    print("  /memoria      mostra status da memória de longo prazo")
    print("  /memorias     lista memórias episódicas recentes")
    print("  /memoria_editar ID TEXTO  altera uma memória persistida")
    print("  /memoria_apagar ID        remove uma memória persistida")
    print("  /limpar       limpa apenas o contexto ativo da conversa")
    print("  /ajuda        mostra esta ajuda")
    print("  /sair         encerra")


def print_status(
    model_path: Path,
    llm,
    history: list[dict[str, str]],
    session_totals: dict[str, int] | None = None,
    memory: LongTermMemory | None = None,
) -> None:
    resources = registry()["resources"]

    section("●", "Status")
    print(f"  Modelo       : {model_path.name}")
    print(f"  Contexto     : {N_CTX:,}")
    print(f"  Resposta máx.: {MAX_TOKENS:,}")
    print(f"  Plan máx.    : {TOOL_PLAN_TOKENS:,}")
    print(f"  Threads      : {THREADS}")
    print(f"  Batch        : {BATCH}")
    print(f"  GPU layers   : {GPU_LAYERS}")
    print(
        f"  Prompt cache : "
        f"{'ON · ' + str(CACHE_MB) + ' MiB' if ENABLE_PROMPT_CACHE else 'OFF'}"
    )
    print(
        f"  CLI avançada : "
        f"{'prompt_toolkit' if HAS_PROMPT_TOOLKIT else 'input() fallback'}"
    )
    print(
        "  Agentes      : "
        + str(len([x for x in resources if x["kind"] == "agent"]))
    )
    print(
        "  Módulos      : "
        + str(len([x for x in resources if x["kind"] == "module"]))
    )
    print(
        f"  Memória LTP  : "
        f"{'ON · ' + str(memory.count()) + ' episódios' if memory is not None and MEMORY_ENABLED else 'OFF'}"
    )
    print(f"  Tool attempts: {MAX_TOOL_ATTEMPTS} · repetição idêntica máx. {MAX_REPEAT_ACTION}")
    show_runtime_panel(llm, history, session_totals, memory)


# ============================================================
# PROMPT TOOLKIT
# ============================================================

def make_prompt_session():
    if not HAS_PROMPT_TOOLKIT:
        return None

    completer = WordCompleter(
        COMMANDS,
        ignore_case=True,
        sentence=True,
    )

    return PromptSession(
        history=InMemoryHistory(),
        completer=completer,
        complete_while_typing=False,
    )


def read_user_input(prompt_session) -> str:
    if prompt_session is None:
        return input(
            Fore.CYAN
            + Style.BRIGHT
            + "Você > "
            + Style.RESET_ALL
        ).strip()

    # ANSI preserva cor sem contaminar o texto capturado.
    return prompt_session.prompt(
        ANSI(
            Fore.CYAN
            + Style.BRIGHT
            + "Você > "
            + Style.RESET_ALL
        )
    ).strip()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    for p in (
        ROOT / "agents",
        ROOT / "core",
        ROOT / "sandbox_data",
    ):
        p.mkdir(exist_ok=True)

    if not SANDBOX.is_file():
        raise SystemExit(
            Fore.RED + "sandbox.py não encontrado ao lado deste arquivo."
        )

    llm, model_path = load_model()
    history: list[dict[str, str]] = []
    session_totals = {"user": 0, "assistant": 0}
    last_result: dict[str, Any] | None = None
    verified_results: list[dict[str, Any]] = []
    base_system = FAST_SYSTEM

    memory = LongTermMemory(ROOT / "sandbox_data") if MEMORY_ENABLED else None

    # Pré-aquece catálogo de agentes/módulos, não o embedder.
    registry(force=True)
    prompt_session = make_prompt_session()

    print()
    print(
        Fore.WHITE
        + "Luna está pronta. Converse normalmente ou peça para executar módulos e agentes."
        + Style.RESET_ALL
    )
    print(
        Fore.LIGHTBLACK_EX
        + "Nomes incorretos são comparados com core/ e agents/. Falhas de ferramenta podem "
          "ser retentadas com estratégia diferente, com limite anti-loop."
        + Style.RESET_ALL
    )
    if memory is not None:
        print(
            Fore.LIGHTBLACK_EX
            + f"Memória episódica: ON · gatilho {MEMORY_TRIGGER_PERCENT:.0f}% · "
              f"{memory.count()} episódios persistidos."
            + Style.RESET_ALL
        )
    if HAS_PROMPT_TOOLKIT:
        print(
            Fore.LIGHTBLACK_EX
            + "Use ↑/↓ para histórico e Tab para completar comandos."
            + Style.RESET_ALL
        )

    print_help()
    print()

    while True:
        try:
            q = read_user_input(prompt_session)
        except (EOFError, KeyboardInterrupt):
            print("\nAté mais.")
            break

        if not q:
            continue

        command = q.casefold()

        if command in {"/sair", "/exit", "/quit"}:
            print(
                Fore.MAGENTA + Style.BRIGHT + "Luna > " + Style.RESET_ALL + "Até mais."
            )
            break

        if command == "/limpar":
            history.clear()
            verified_results.clear()
            success(
                "Contexto ativo limpo. As memórias de longo prazo e os totais da sessão foram preservados."
            )
            continue

        if command == "/tokens":
            show_runtime_panel(llm, history, session_totals, memory, last_result, base_system)
            continue

        if command == "/ajuda":
            print_help()
            continue

        if command == "/status":
            print_status(
                model_path,
                llm,
                history,
                session_totals,
                memory,
            )
            continue

        if command == "/cache":
            section("◆", "Prompt cache")
            print(f"  Estado     : {'ativado' if ENABLE_PROMPT_CACHE else 'desativado'}")
            print(f"  Capacidade : {CACHE_MB} MiB")
            print("  Estratégia : LlamaCache associado ao modelo via set_cache().")
            continue

        if command == "/memoria":
            section("◆", "Memória de longo prazo")
            if memory is None:
                print("  Estado      : desativada")
            else:
                print("  Estado      : ativada")
                print(f"  Episódios   : {memory.count()}")
                print(f"  Diretório   : {memory.dir}")
                print(f"  Gatilho     : {MEMORY_TRIGGER_PERCENT:.0f}% do contexto")
                print(f"  Recall      : top {MEMORY_RECALL_TOP_K} · score mínimo {MEMORY_MIN_SCORE:.2f}")
                print(f"  Embedder    : {EMBED_MODEL}")
                print(
                    f"  Índice      : "
                    f"{int(memory.index.ntotal) if memory.index is not None else 'não carregado'}"
                )
            continue

        if command == "/memorias":
            section("◆", "Memórias episódicas recentes")
            if memory is None:
                print("  Memória desativada.")
            else:
                rows = memory.recent(8)
                if not rows:
                    print("  (nenhuma memória arquivada ainda)")
                for item in rows:
                    stamp = time.strftime(
                        "%Y-%m-%d %H:%M",
                        time.localtime(item["created_at"]),
                    )
                    preview = item["summary"].replace("\n", " ").strip()
                    if len(preview) > 260:
                        preview = preview[:257] + "..."
                    print(
                        f"  • #{item['id']} · {stamp} · {item['source_turns']} msgs\n"
                        f"    {preview}"
                    )
            continue

        if command.startswith("/memoria_apagar "):
            if memory is None:
                error("Memória desativada.")
            else:
                try:
                    memory_id = int(command.split(maxsplit=1)[1])
                    if memory.delete_memory(memory_id):
                        success(f"Memória #{memory_id} apagada.")
                    else:
                        warning(f"Memória #{memory_id} não encontrada.")
                except (ValueError, IndexError) as exc:
                    error(f"Uso: /memoria_apagar ID ({exc})")
            continue

        if command.startswith("/memoria_editar "):
            if memory is None:
                error("Memória desativada.")
            else:
                parts = q.split(maxsplit=2)
                if len(parts) < 3:
                    error("Uso: /memoria_editar ID TEXTO")
                else:
                    try:
                        memory_id = int(parts[1])
                        if memory.update_memory(memory_id, parts[2]):
                            success(f"Memória #{memory_id} atualizada.")
                        else:
                            warning(f"Memória #{memory_id} não encontrada.")
                    except (ValueError, sqlite3.Error) as exc:
                        error(f"Não foi possível editar a memória: {exc}")
            continue

        if command in {"/agentes", "/modulos", "/workspace"}:
            result = sandbox("--list", command[1:], live=False)
            files = result.get("files", [])
            section("▸", command[1:].capitalize())
            if files:
                print("\n".join("  • " + str(x) for x in files))
            else:
                print(Fore.LIGHTBLACK_EX + "(vazio)" + Style.RESET_ALL)
            continue

        if command == "/catalogo":
            print(json.dumps(catalogue(), ensure_ascii=False, indent=2))
            continue

        # ----------------------------------------------------
        # RECALL SEMÂNTICO ANTES DA RESPOSTA
        # ----------------------------------------------------
        base_system, recalled_memory = effective_system_prompt(
            llm,
            memory,
            q,
            verified_results,
        )
        if recalled_memory:
            print(
                Fore.LIGHTBLACK_EX
                + "↺ Memória relevante recuperada para esta pergunta."
                + Style.RESET_ALL
            )

        direct_verified = semantic_direct_answer(memory, q)
        if direct_verified is None:
            direct_verified = verified_direct_answer(q, verified_results)
        if direct_verified is not None:
            answer = direct_verified
            stats = (0.0, 0.0, 0, 0.0)
            print(Fore.MAGENTA + Style.BRIGHT + "Luna > " + Style.RESET_ALL + answer)
            session_totals["user"] += count_text_tokens(llm, q)
            session_totals["assistant"] += count_text_tokens(llm, answer)
            history.extend([
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer},
            ])
            history = compress_context_if_needed(
                llm,
                history,
                memory,
                trigger_percent=MEMORY_TRIGGER_PERCENT,
            )
            show_runtime_panel(llm, history, session_totals, memory, last_result, base_system)
            print()
            continue

        # ----------------------------------------------------
        # ROTEAMENTO
        # ----------------------------------------------------
        # ----------------------------------------------------
        # ROTEAMENTO
        # ----------------------------------------------------
        action = explicit_route(q) or direct_route(q)

        # Se o usuário pediu explicitamente um agente/módulo, executa direto sem inspecionar
        if not action and should_auto_inspect(q) and may_need_tools(q):
            section("◌", "Inspecionando estrutura antes da ação", Fore.CYAN)
            preflight_result = execute(
                {
                    "type": "tool",
                    "name": "list_directory",
                    "arguments": {"path": ".", "recursive": False, "limit": 50},
                },
                q,
            )
            last_result = preflight_result
            if preflight_result.get("ok"):
                verified_results.append({
                    "request": q,
                    "tool": "list_directory",
                    "result": preflight_result,
                })
            show_actual_result(preflight_result)

        # Rota explícita tem prioridade máxima
        if action :
            action, match = resolve_action_target(action)
            show_match_feedback(match)

            if match and not match.get("ok"):
                suggestions = [
                    x["name"] for x in match.get("suggestions", [])
                ]
                answer = (
                    f"Não encontrei com segurança o recurso “{match.get('original', '')}”."
                )
                if suggestions:
                    answer += " Os nomes mais próximos são: " + ", ".join(suggestions) + "."
                else:
                    answer += (
                        " Verifiquei core/ e agents/, mas não apareceu uma correspondência segura."
                    )
                print(
                    Fore.MAGENTA + Style.BRIGHT + "Luna > " + Style.RESET_ALL + answer
                )
                stats = (0.0, 0.0, 0, 0.0)

            else:
                target = str(action["arguments"].get("name", action.get("name", "ação")))
                section("⚙", f"Executando {target}", Fore.YELLOW)
                result = execute(action, q)
                last_result = result
                outcome = classify_tool_result(result)
                if outcome == "success":
                    verified_results.append({
                        "request": q,
                        "tool": action.get("name"),
                        "result": result,
                    })
                    remember_verified_result(memory, q, action.get("name"), result)
                show_actual_result(result)

                if outcome == "success":
                    success(f"{target} concluído")
                    answer, stats = explain_direct(
                        llm,
                        q,
                        result,
                        history,
                        base_system,
                    )
                else:
                    if outcome == "partial":
                        warning(
                            f"{target} retornou algo útil, mas incompleto. A Luna vai tentar completar sem perder o resultado."
                        )
                    else:
                        error(
                            f"{target} falhou. A Luna vai procurar uma estratégia diferente antes de desistir."
                        )

                    answer, stats = react(
                        llm,
                        q,
                        history,
                        base_system,
                        seed_action=action,
                        seed_result=result,
                    )

        elif may_need_tools(q):
            if needs_planning(q):
                answer, stats = react(
                    llm,
                    q,
                    history,
                    base_system,
                )
            else:
                answer, stats = stream_final(
                    llm,
                    build_chat_messages(
                        llm,
                        history,
                        q,
                        base_system,
                    ),
                )

        else:
            answer, stats = stream_final(
                llm,
                build_chat_messages(
                    llm,
                    history,
                    q,
                    base_system,
                ),
            )

        # ----------------------------------------------------
        # HISTÓRICO + CONTAGEM TOTAL DA SESSÃO
        # ----------------------------------------------------
        session_totals["user"] += count_text_tokens(llm, q)
        session_totals["assistant"] += count_text_tokens(llm, answer)

        history.extend(
            [
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer},
            ]
        )

        # ----------------------------------------------------
        # CONSOLIDAÇÃO AUTOMÁTICA DA MEMÓRIA
        # ----------------------------------------------------
        history = compress_context_if_needed(
            llm,
            history,
            memory,
            trigger_percent=MEMORY_TRIGGER_PERCENT,
        )

        if stats[2]:
            print(
                Fore.LIGHTBLACK_EX
                + f"  ⚡ {stats[0]:.1f} tok/s"
                  f" · 1º token {stats[1]:.2f}s"
                  f" · resposta {stats[2]} tokens"
                  f" · total {stats[3]:.2f}s"
                + Style.RESET_ALL
            )

        show_token_bar(llm, history, session_totals)
        show_runtime_panel(llm, history, session_totals, memory, last_result, base_system)
        print()


if __name__ == "__main__":
    main()
