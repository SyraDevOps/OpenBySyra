"""
Astra · Assistente (Groq ou local) · V4
=======================
Chat com memória híbrida persistente e recuperação adaptativa, falando com
a API em nuvem da Groq por padrão (ou com um GGUF local, se preferir):

- backend Groq (padrão) ou GGUF local via llama-cpp-python — veja
  "BACKEND GROQ" abaixo para colar sua chave
- streaming de resposta
- Structured Outputs JSON Schema no ReAct
- cache de prefixo / prompt via LlamaCache (só no backend local)
- histórico ajustado dinamicamente ao N_CTX
- prompt_toolkit opcional (histórico + autocomplete)
- sandbox com asyncio.subprocess e suporte a eventos/logs intermediários
- módulos + agentes
- roteamento explícito, semântico (embeddings) e fuzzy matching
- correção automática de nomes digitados incorretamente
- contador de tokens da conversa inteira
- métricas de geração
- interface de terminal organizada
- memória episódica SQLite + FAISS
- recuperação semântica de memórias antigas
- compactação automática do histórico
- tentativas adaptativas com mudança de estratégia e limite anti-loop
- entrada por voz opcional: microfone sempre ligado, wake word "astra",
  VAD e transcrição local com Whisper (veja voice_io.py)

>>> PARA USAR A GROQ: procure "GROQ_API_KEY = " logo abaixo das outras
>>> importações e cole sua chave de https://console.groq.com/keys no lugar
>>> de "COLE_SUA_CHAVE_GROQ_AQUI" (ou defina a variável de ambiente
>>> ASTRA_GROQ_API_KEY, que tem prioridade). Isso já é suficiente — o
>>> BACKENDS DE IA (escolha na linha de comando; o padrão é "auto"):
>>>     python main.py                       # auto: Groq → Gemini → GGUF local
>>>     python main.py --groq                # só Groq
>>>     python main.py --groq openai/gpt-oss-20b
>>>     python main.py --google              # só Google Gemini (gemini-2.5-flash-lite)
>>>     python main.py --google gemini-2.5-flash
>>>     python main.py --groq --key gsk_...          # chave só desta execução
>>>     python main.py --google gemini-2.5-flash --key AIza...
>>>     python main.py --gguf                # só local/offline (model.gguf)
>>>     python main.py --gguf --model outro.gguf
>>>     python main.py --backend auto|groq|google|gguf
>>> ou pela variável ASTRA_BACKEND. A linha de comando tem prioridade.
>>> No modo auto, se o backend em uso cair no meio da conversa, a Astra
>>> troca sozinha para o próximo da lista (ASTRA_AUTO_ORDER=groq,google,gguf).
>>> Durante a conversa: /backend mostra o atual; /backend google troca.
>>>
>>> CHAVES: use --key, ou cole nas constantes GROQ_API_KEY / GOOGLE_API_KEY logo abaixo das
>>> importações (ou use as variáveis ASTRA_GROQ_API_KEY, ASTRA_GOOGLE_API_KEY,
>>> GEMINI_API_KEY, que têm prioridade). Sem chave, a Astra pergunta na hora
>>> (a digitação fica oculta). Chave do Google: aistudio.google.com/apikey

Dependências (obrigatórias — servem para memória/roteamento semântico e a
interface, independente do backend de chat escolhido):
    py -m pip install colorama tqdm prompt_toolkit faiss-cpu sentence-transformers numpy

Dependências do backend (instale UM dos dois):
    py -m pip install groq tiktoken        # backend Groq (nuvem)
    (Google Gemini não precisa de pacote extra: usa só a biblioteca padrão)
    py -m pip install llama-cpp-python     # backend local/offline (--gguf)

Dependências opcionais:
    py -m pip install rapidfuzz                       # fuzzy matching melhor
    py -m pip install sounddevice faster-whisper piper-tts   # entrada/saída por voz
    py -m pip install silero-vad-notorch               # VAD neural (ONNX, sem torch), opcional
    No Linux, o modo de voz também precisa da lib nativa do PortAudio:
        sudo apt install libportaudio2

O prompt_toolkit é opcional; se não estiver instalado, há fallback para input().
Sem sounddevice/faster-whisper, o programa cai automaticamente para digitação
(ASTRA_VOICE_MODE=0 desativa o modo de voz manualmente). Rode
`python voice_io.py --selftest` para diagnosticar microfone e Whisper.
"""

from __future__ import annotations

from pathlib import Path
from math import sqrt
from difflib import SequenceMatcher
from typing import Any, Iterable
import json
import os
import queue
import sqlite3
import hashlib
import asyncio
import re
import subprocess
import sys
import threading
import time
import unicodedata
import getpass
import urllib.error
import urllib.request
import http.client
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
    from colorama import Fore, Style, init
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit(
        "Dependências obrigatórias ausentes.\n"
        "Instale com:\n"
        "  py -m pip install colorama tqdm prompt_toolkit faiss-cpu sentence-transformers numpy"
    ) from e

# llama-cpp-python (backend local) e groq+tiktoken (backend em nuvem) são
# opcionais — só o backend escolhido em ASTRA_BACKEND precisa estar
# instalado de verdade. Ver load_model() e a classe GroqLLM mais abaixo.
try:
    from llama_cpp import Llama, LlamaCache

    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

try:
    import groq as groq_sdk
    from groq import Groq

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

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
    from prompt_toolkit.patch_stdout import patch_stdout

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    import voice_io

    HAS_VOICE_IO = True
except ImportError:
    HAS_VOICE_IO = False

init(autoreset=True)


def fuzzy_ratio(a: str, b: str) -> float:
    """
    Similaridade 0..1 entre dois textos já normalizados.

    Usa RapidFuzz quando disponível: é bem mais tolerante a palavras fora de ordem
    ("pesquisa-web" vs "web-pesquisa") e a pequenos erros de digitação do que o
    difflib puro. Se a dependência não estiver instalada, cai para
    SequenceMatcher (mais fraco, mas sem dependência externa).
    """
    if not a or not b:
        return 0.0

    if HAS_RAPIDFUZZ:
        return max(
            _rapidfuzz_fuzz.WRatio(a, b),
            _rapidfuzz_fuzz.token_sort_ratio(a, b),
        ) / 100.0

    return SequenceMatcher(None, a, b).ratio()

# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT = Path(__file__).resolve().parent
SANDBOX = ROOT / "sandbox.py"

MODEL_ENV = os.getenv("ASTRA_MODEL", "").strip()
MODEL = ROOT / MODEL_ENV if MODEL_ENV else None

# ------------------------------------------------------------
# Backend Groq (API em nuvem) — substitui o modelo GGUF local
# ------------------------------------------------------------
# Cole sua chave aqui (console.groq.com/keys) para usar a Groq em vez de um
# modelo local. ASTRA_GROQ_API_KEY (variável de ambiente) tem prioridade
# sobre o valor colado aqui, caso prefira não deixar a chave no arquivo.
GROQ_API_KEY = "COLE_SUA_CHAVE_GROQ_AQUI"

GROQ_API_KEY = os.getenv("ASTRA_GROQ_API_KEY", "").strip() or GROQ_API_KEY

# openai/gpt-oss-120b é o modelo de produção recomendado pela Groq para uso
# geral (substituiu o llama-3.3-70b-versatile, aposentado). Para respostas
# mais rápidas/baratas trocando um pouco de qualidade, use
# ASTRA_GROQ_MODEL=openai/gpt-oss-20b.
GROQ_MODEL = os.getenv("ASTRA_GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_REASONING_EFFORT = os.getenv("ASTRA_GROQ_REASONING_EFFORT", "low")  # low/medium/high
# "hidden" mantém a saída limpa (sem o raciocínio interno do modelo
# aparecendo na resposta) — é o que faz o GPT-OSS se comportar como os
# modelos instruct pequenos que este app usava localmente.
GROQ_REASONING_FORMAT = os.getenv("ASTRA_GROQ_REASONING_FORMAT", "hidden")
GROQ_MAX_RETRIES = int(os.getenv("ASTRA_GROQ_MAX_RETRIES", "4"))
GROQ_RETRY_BASE_DELAY = float(os.getenv("ASTRA_GROQ_RETRY_DELAY", "1.5"))
GROQ_CONTEXT_WINDOW = int(os.getenv("ASTRA_GROQ_CTX", "131072"))

# ------------------------------------------------------------
# GOOGLE GEMINI (API em nuvem, endpoint compatível com OpenAI)
# ------------------------------------------------------------
# Chave grátis em https://aistudio.google.com/apikey. Variáveis de ambiente
# (ASTRA_GOOGLE_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY) têm prioridade.
GOOGLE_API_KEY = "COLE_SUA_CHAVE_GOOGLE_AQUI"

GOOGLE_API_KEY = (
    os.getenv("ASTRA_GOOGLE_API_KEY", "").strip()
    or os.getenv("GEMINI_API_KEY", "").strip()
    or os.getenv("GOOGLE_API_KEY", "").strip()
    or GOOGLE_API_KEY
)
DEFAULT_GOOGLE_MODEL = "gemini-2.5-flash-lite"
GOOGLE_MODEL = os.getenv("ASTRA_GOOGLE_MODEL", DEFAULT_GOOGLE_MODEL).strip() or DEFAULT_GOOGLE_MODEL
GOOGLE_BASE_URL = os.getenv(
    "ASTRA_GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
).rstrip("/")
GOOGLE_MAX_RETRIES = int(os.getenv("ASTRA_GOOGLE_MAX_RETRIES", "4"))
GOOGLE_RETRY_BASE_DELAY = float(os.getenv("ASTRA_GOOGLE_RETRY_DELAY", "1.5"))
GOOGLE_TIMEOUT = float(os.getenv("ASTRA_GOOGLE_TIMEOUT", "120"))
# O Gemini 2.5 aceita ~1M tokens, mas uma janela de trabalho menor mantém
# custo e latência sob controle e a memória compacta o histórico antes.
# Suba com ASTRA_GOOGLE_CTX=1000000 se quiser usar tudo.
GOOGLE_CONTEXT_WINDOW = int(os.getenv("ASTRA_GOOGLE_CTX", "131072"))


def _key_ok(key: str) -> bool:
    key = (key or "").strip()
    return bool(key) and not key.startswith("COLE_")

_BACKEND_ALIASES = {
    "auto": "auto", "": "auto",
    "groq": "groq", "cloud": "groq", "nuvem": "groq", "online": "groq",
    "google": "google", "gemini": "google", "gai": "google",
    "gguf": "gguf", "local": "gguf", "offline": "gguf", "llama": "gguf",
}


def _normalize_argv(argv: list[str]) -> list[str]:
    """Aceita --Google, --GROQ etc.: só o nome da flag é minúsculo; valores
    (nomes de modelo, caminhos) ficam como o usuário digitou."""
    out = []
    for a in argv:
        if a.startswith("--"):
            name, eq, val = a.partition("=")
            a = name.lower() + eq + val
        out.append(a)
    return out


def _parse_backend_cli(argv: list[str]) -> tuple[str, str | None, str | None, str | None]:
    """Devolve (modo, modelo_da_nuvem|None, caminho_gguf|None, chave|None).
    CLI > ASTRA_BACKEND > auto."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="main.py",
        description="Astra — assistente de terminal. Backends: Groq, Google Gemini (nuvem) ou GGUF (offline).",
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--groq", nargs="?", const="", default=None, metavar="MODELO",
                       help="usa só a Groq; modelo opcional (ex.: openai/gpt-oss-20b)")
    group.add_argument("--google", nargs="?", const="", default=None, metavar="MODELO",
                       help=f"usa só o Google Gemini; modelo opcional (padrão {DEFAULT_GOOGLE_MODEL})")
    group.add_argument("--gguf", "--local", "--offline", dest="gguf", action="store_true",
                       help="usa só o modelo local model.gguf (offline)")
    group.add_argument("--auto", action="store_true", help="Groq → Gemini → GGUF, com troca automática (padrão)")
    group.add_argument("--backend", default=None, metavar="MODO", help="auto | groq | google | gguf")
    ap.add_argument("--key", "--api-key", dest="key", default=None, metavar="CHAVE",
                    help="chave de API do backend escolhido (só nesta execução)")
    ap.add_argument("--model", default=None, metavar="MODELO_OU_ARQUIVO.gguf",
                    help="modelo do backend escolhido, ou caminho de um .gguf")
    args, _unknown = ap.parse_known_args(_normalize_argv(argv))

    cloud_model: str | None = None
    if args.groq is not None:
        mode, cloud_model = "groq", args.groq
    elif args.google is not None:
        mode, cloud_model = "google", args.google
    elif args.gguf:
        mode = "gguf"
    elif args.auto:
        mode = "auto"
    elif args.backend is not None:
        mode = _BACKEND_ALIASES.get(args.backend.strip().casefold())
        if mode is None:
            ap.error(f"--backend inválido: {args.backend!r} (use auto, groq, google ou gguf)")
    else:
        mode = _BACKEND_ALIASES.get(os.getenv("ASTRA_BACKEND", "auto").strip().casefold(), "auto")

    gguf_path: str | None = None
    if args.model:
        if args.model.casefold().endswith(".gguf"):
            gguf_path = args.model
        elif mode in ("groq", "google") and not cloud_model:
            cloud_model = args.model
    return mode, (cloud_model or None), gguf_path, (args.key.strip() if args.key else None)


BACKEND_MODE, _CLI_CLOUD_MODEL, _CLI_GGUF, _CLI_KEY = _parse_backend_cli(sys.argv[1:])
if _CLI_GGUF:
    MODEL = Path(_CLI_GGUF).expanduser()
if _CLI_CLOUD_MODEL:
    if BACKEND_MODE == "groq":
        GROQ_MODEL = _CLI_CLOUD_MODEL
    elif BACKEND_MODE == "google":
        GOOGLE_MODEL = _CLI_CLOUD_MODEL

if _CLI_KEY:
    # Em modo explícito a chave é do backend escolhido; no auto, adivinha
    # pelo prefixo (Groq: gsk_..., Google: AIza...).
    if BACKEND_MODE == "groq" or (BACKEND_MODE == "auto" and _CLI_KEY.startswith("gsk_")):
        GROQ_API_KEY = _CLI_KEY
    elif BACKEND_MODE == "google" or (BACKEND_MODE == "auto" and _CLI_KEY.startswith("AIza")):
        GOOGLE_API_KEY = _CLI_KEY
    else:
        print("⚠ --key ignorada: use junto de --groq ou --google (ou uma chave gsk_.../AIza...).")

N_CTX_LOCAL = int(os.getenv("ASTRA_CTX", "8192"))

BACKEND_LABELS = {
    "groq": "Groq (nuvem)",
    "google": "Google Gemini (nuvem)",
    "local": "local (GGUF, offline)",
}


def _ctx_for(backend: str) -> int:
    return {
        "groq": GROQ_CONTEXT_WINDOW,
        "google": GOOGLE_CONTEXT_WINDOW,
    }.get(backend, N_CTX_LOCAL)


# Backend REALMENTE em uso ("groq", "google" ou "local"); load_model() e o
# failover o atualizam. N_CTX (usado por todo o orçamento de tokens do app)
# e ACTIVE_MODEL acompanham.
BACKEND = {"gguf": "local", "google": "google"}.get(BACKEND_MODE, "groq")
N_CTX = _ctx_for(BACKEND)
ACTIVE_MODEL = {"groq": GROQ_MODEL, "google": GOOGLE_MODEL}.get(BACKEND, "model.gguf")
MAX_TOKENS = int(os.getenv("ASTRA_MAX_TOKENS", "1024"))
TOOL_PLAN_TOKENS = int(os.getenv("ASTRA_PLAN_TOKENS", "512"))
CONTEXT_MARGIN = int(os.getenv("ASTRA_CONTEXT_MARGIN", "128"))

THREADS = int(
    os.getenv(
        "ASTRA_THREADS",
        str(max(2, (os.cpu_count() or 4) // 2)),
    )
)
BATCH = int(os.getenv("ASTRA_BATCH", "512"))
GPU_LAYERS = int(os.getenv("ASTRA_GPU_LAYERS", "0"))

TIMEOUT = int(os.getenv("ASTRA_TOOL_TIMEOUT", "60"))
MAX_STEPS = int(os.getenv("ASTRA_REACT_STEPS", "5"))
AUTOWRITE = os.getenv("ASTRA_AUTONOMOUS_WRITE", "0") == "1"

# Memória híbrida
MEMORY_ENABLED = os.getenv("ASTRA_MEMORY", "1") != "0"
MEMORY_TRIGGER_PERCENT = float(os.getenv("ASTRA_MEMORY_TRIGGER", "72"))
MEMORY_RECALL_TOP_K = int(os.getenv("ASTRA_MEMORY_TOP_K", "3"))
MEMORY_MIN_SCORE = float(os.getenv("ASTRA_MEMORY_MIN_SCORE", "0.28"))
MEMORY_SUMMARY_TOKENS = int(os.getenv("ASTRA_MEMORY_SUMMARY_TOKENS", "420"))
MEMORY_CONTEXT_TOKENS = int(os.getenv("ASTRA_MEMORY_CONTEXT_TOKENS", "850"))
EMBED_MODEL = os.getenv(
    "ASTRA_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Recuperação / anti-loop de ferramentas
MAX_TOOL_ATTEMPTS = int(os.getenv("ASTRA_TOOL_ATTEMPTS", "4"))
MAX_REPEAT_ACTION = int(os.getenv("ASTRA_MAX_REPEAT_ACTION", "1"))

# Cache RAM para prefixos/prompt.
# 256 MiB por padrão, bem mais conservador que 2 GiB.
CACHE_MB = int(os.getenv("ASTRA_PROMPT_CACHE_MB", "256"))
CACHE_BYTES = max(16, CACHE_MB) * 1024 * 1024
ENABLE_PROMPT_CACHE = os.getenv("ASTRA_PROMPT_CACHE", "0") != "0"

# Correção de nomes.
AUTO_MATCH_THRESHOLD = float(os.getenv("ASTRA_MATCH_AUTO", "0.78"))
SUGGEST_MATCH_THRESHOLD = float(os.getenv("ASTRA_MATCH_SUGGEST", "0.55"))

# Máximo de texto bruto do resultado de ferramenta oferecido ao modelo.
MAX_TOOL_RESULT_CHARS = int(os.getenv("ASTRA_TOOL_RESULT_CHARS", "14000"))

# Quantos termos após "módulo/agente" usar na identificação inicial.
MAX_EXPLICIT_NAME_WORDS = int(os.getenv("ASTRA_NAME_WORDS", "6"))
PROMPTS_DIR = ROOT / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.json"

COMMANDS = [
    "/agentes",
    "/modulos",
    "/workspace",
    "/catalogo",
    "/tokens",
    "/status",
    "/backend",
    "/backend auto",
    "/backend groq",
    "/backend google",
    "/backend gguf",
    "/cache",
    "/memoria",
    "/memorias",
    "/memoria_apagar",
    "/memoria_editar",
    "/voice",
    "/voice on",
    "/voice off",
    "/voz",
    "/vozes",
    "/idioma",
    "/limpar",
    "/ajuda",
    "/sair",
]

DEFAULT_SYSTEM_PROMPT = """
Você é Astra, uma assistente local.

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
            f"Nome: {data.get('name', 'Astra')}",
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


# "Roxo choque" pedido explicitamente: colorama não tem esse tom pronto, então
# usamos ANSI truecolor (24 bits) direto — funciona no Windows Terminal e no
# cmd.exe moderno (Windows 10+), que é o que os logs anteriores mostraram.
SHOCK_PURPLE = "\033[38;2;177;0;255m"
SHOCK_PURPLE_BOLD = Style.BRIGHT + SHOCK_PURPLE

# Banner grande "ASTRA" (estilo blocos), montado letra a letra a partir de
# glifos com largura fixa — evita desalinhamento ao juntar strings na mão.
_BANNER_A = [
    " █████╗ ",
    "██╔══██╗",
    "███████║",
    "██╔══██║",
    "██║  ██║",
    "╚═╝  ╚═╝",
]
_BANNER_S = [
    "███████╗",
    "██╔════╝",
    "███████╗",
    "╚════██║",
    "███████║",
    "╚══════╝",
]
_BANNER_T = [
    "████████╗",
    "╚══██╔══╝",
    "   ██║   ",
    "   ██║   ",
    "   ██║   ",
    "   ╚═╝   ",
]
_BANNER_R = [
    "██████╗ ",
    "██╔══██╗",
    "██████╔╝",
    "██╔══██╗",
    "██║  ██║",
    "╚═╝  ╚═╝",
]
ASTRA_BANNER = "\n".join(
    " ".join(letter[row] for letter in (_BANNER_A, _BANNER_S, _BANNER_T, _BANNER_R, _BANNER_A))
    for row in range(6)
)


def print_astra_banner() -> None:
    print()
    for line in ASTRA_BANNER.splitlines():
        print(SHOCK_PURPLE_BOLD + line + Style.RESET_ALL)
    print()


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
      1. ASTRA_MODEL
      2. nomes padrão
      3. *.gguf contendo 'astra'
      4. único *.gguf
    """
    if MODEL is not None:
        if MODEL.is_file():
            return MODEL
        raise SystemExit(
            Fore.RED + f"ASTRA_MODEL aponta para um arquivo inexistente: {MODEL}"
        )

    preferred = (
        ROOT / "Astra.gguf",
        ROOT / "astra.gguf",
        ROOT / "ASTRA.gguf",
        ROOT / "model.gguf",
    )
    for p in preferred:
        if p.is_file():
            return p

    ggufs = sorted(ROOT.glob("*.gguf"))
    astras = [p for p in ggufs if "astra" in p.name.casefold()]

    if len(astras) == 1:
        return astras[0]

    if len(astras) > 1:
        return max(astras, key=lambda p: p.stat().st_size)

    if len(ggufs) == 1:
        return ggufs[0]

    if not ggufs:
        raise SystemExit(
            Fore.RED
            + "Nenhum GGUF encontrado. Coloque o Astra ao lado do script "
              "ou defina ASTRA_MODEL=arquivo.gguf."
        )

    names = "\n".join(f"  • {p.name}" for p in ggufs)
    raise SystemExit(
        Fore.YELLOW
        + "Há vários GGUF e não identifiquei o Astra com segurança.\n"
          "Defina, por exemplo:\n"
          "  set ASTRA_MODEL=astra-3-1b-it-Q8_0.gguf\n\n"
          "Arquivos encontrados:\n"
        + names
    )


# ============================================================
# BACKENDS EM NUVEM (GROQ, GOOGLE GEMINI) E FAILOVER
# ============================================================
# GroqLLM expõe exatamente os 4 métodos que o resto do arquivo chama num
# objeto `llm` (create_chat_completion, tokenize, detokenize, set_cache) —
# por isso stream_final, ask_plan, compact_text_to_budget, classify_intent
# etc. funcionam sem nenhuma mudança, sem saber se estão falando com um
# .gguf local, com a Groq ou com o Google Gemini.

def _groq_translate_response_format(
    response_format: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    O resto do código pede `{"type": "json_object", "schema": {...}}`
    (convenção do llama-cpp-python). A Groq (API OpenAI-compatible) quer
    `{"type": "json_schema", "json_schema": {"name":..., "schema":..., "strict": false}}`.
    Isolar essa tradução aqui é o que permite manter ask_plan(),
    classify_intent(), extract_http_intent() etc. inalterados.
    """
    if not response_format:
        return None

    schema = response_format.get("schema")
    if not schema:
        return {"type": "json_object"}

    name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(schema.get("title") or "response")).strip("_")[:60] or "response"

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            # strict=true exige listar TODO campo em "required"; nossos
            # schemas têm campos opcionais (ex.: start_with_inspection), e
            # o código já valida/decodifica com tolerância a JSON
            # imperfeito, então best-effort é a opção mais robusta aqui.
            "strict": False,
        },
    }


def _groq_response_to_dict(response: Any) -> dict[str, Any]:
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    return {
        "choices": [
            {
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content or "",
                },
                "finish_reason": choice.finish_reason,
            }
        ],
        "usage": (
            {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage
            else {}
        ),
    }


def _groq_chunk_to_dict(chunk: Any) -> dict[str, Any]:
    delta = chunk.choices[0].delta if chunk.choices else None
    return {"choices": [{"delta": {"content": (delta.content if delta else None) or ""}}]}


class CloudUnavailable(RuntimeError):
    """O provedor em nuvem não respondeu após todas as tentativas (rede,
    limite de taxa, erro 5xx). É o sinal que dispara o failover."""


class CloudAPIError(RuntimeError):
    """Erro HTTP da API (4xx/5xx) com status e mensagem limpos."""

    def __init__(self, status: int, message: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.retry_after = retry_after


class _CloudLLMBase:
    """Parte comum dos backends em nuvem: tokenização aproximada e no-op de cache."""

    _enc = None

    def _init_tokenizer(self) -> None:
        self._enc = None
        if HAS_TIKTOKEN:
            try:
                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._enc = None

    # ---------------- tokenização aproximada ----------------

    _FALLBACK_CHUNK = 4  # bytes por "token" falso quando tiktoken não está disponível

    def tokenize(self, text_bytes: bytes, add_bos: bool = False, special: bool = True) -> list[int]:
        if self._enc is not None:
            try:
                text = text_bytes.decode("utf-8", errors="ignore")
                return self._enc.encode(text, disallowed_special=())
            except Exception:
                pass
        # Fallback sem tiktoken (pacote ausente, ou sem rede para baixar o
        # vocabulário cl100k_base na primeira vez): cada "token" empacota
        # até 4 bytes UTF-8 num inteiro. Não é um tokenizador de verdade
        # (a contagem fica perto de len(texto)//4, igual ao fallback de
        # count_text_tokens), mas — ao contrário de só contar e descartar —
        # detokenize() reconstrói o texto ORIGINAL exatamente a partir
        # desses inteiros, então compact_text_to_budget continua cortando
        # cabeça/cauda de verdade em vez de devolver string vazia.
        n = self._FALLBACK_CHUNK
        chunks = [text_bytes[i:i + n] for i in range(0, len(text_bytes), n)] or [b""]
        return [int.from_bytes(c.ljust(n, b"\x00"), "big") for c in chunks]

    def detokenize(self, tokens: list[int]) -> bytes:
        if self._enc is not None:
            try:
                return self._enc.decode(tokens).encode("utf-8", errors="ignore")
            except Exception:
                pass
        n = self._FALLBACK_CHUNK
        raw = b"".join(t.to_bytes(n, "big") for t in tokens)
        return raw.rstrip(b"\x00")

    def set_cache(self, cache: Any) -> None:
        # A Groq não expõe cache de prefixo local; no-op só para
        # llm.set_cache(None) (chamado condicionalmente) não quebrar.
        pass


class GroqLLM(_CloudLLMBase):
    """
    Adaptador: dá ao resto do app a mesma interface que llama_cpp.Llama,
    só que por trás fala com a API em nuvem da Groq em vez de um GGUF local.

    Tokenização aproximada (tiktoken, cl100k_base): a Groq não expõe
    tokenização local, e cada modelo hospedado lá usa um vocabulário
    próprio diferente. Para orçamento de contexto (cortar histórico,
    truncar JSON grande) uma contagem aproximada é suficiente — o pior
    caso é reservar um pouco mais ou menos margem que o exato, nunca
    estourar o contexto real (que na Groq é bem maior que os 8k locais).
    """

    def __init__(self, api_key: str, model: str = GROQ_MODEL):
        if not HAS_GROQ:
            raise SystemExit(
                Fore.RED + "Pacote 'groq' não instalado. Rode: pip install groq"
            )
        if not api_key or api_key.strip() in ("", "COLE_SUA_CHAVE_GROQ_AQUI"):
            raise SystemExit(
                Fore.RED
                + "Defina GROQ_API_KEY no topo do main.py (ou a variável de "
                  "ambiente ASTRA_GROQ_API_KEY) com sua chave de "
                  "https://console.groq.com/keys antes de rodar."
            )

        self.model = model
        self._client = Groq(api_key=api_key)
        self._init_tokenizer()

    # ---------------- chat completion ----------------

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        stop: Any = None,
        **_ignored: Any,  # top_k, min_p, repeat_penalty... específicos do llama.cpp
    ):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_format": GROQ_REASONING_FORMAT,
        }
        if "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = GROQ_REASONING_EFFORT
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop is not None:
            kwargs["stop"] = stop

        groq_response_format = _groq_translate_response_format(response_format)
        if groq_response_format is not None:
            # A Groq recusa json_schema junto com stream=True ("Streaming
            # and tool use are not currently supported with Structured
            # Outputs"). Nada neste app pede os dois juntos hoje, mas se
            # pedir, o schema ganha prioridade sobre o streaming.
            stream = False
            kwargs["response_format"] = groq_response_format

        if stream:
            return self._stream(kwargs)
        return self._call_with_retry(kwargs)

    def _is_retryable(self, exc: Exception) -> bool:
        return isinstance(
            exc,
            (
                groq_sdk.RateLimitError,
                groq_sdk.APIConnectionError,
                groq_sdk.APITimeoutError,
                groq_sdk.InternalServerError,
            ),
        )

    def _call_with_retry(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(GROQ_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(**kwargs, stream=False)
                return _groq_response_to_dict(response)
            except Exception as exc:
                if not self._is_retryable(exc):
                    raise  # chave inválida, requisição malformada etc.: tentar de novo não ajuda
                last_exc = exc
                if attempt < GROQ_MAX_RETRIES:
                    delay = GROQ_RETRY_BASE_DELAY * (2**attempt)
                    warning(
                        f"Groq: {type(exc).__name__}, tentando de novo em "
                        f"{delay:.1f}s ({attempt + 1}/{GROQ_MAX_RETRIES})..."
                    )
                    time.sleep(delay)
                # Na última tentativa, cai pro raise RuntimeError depois do
                # loop em vez de deixar a exceção crua escapar sem contexto.
        raise CloudUnavailable(f"Groq API falhou após {GROQ_MAX_RETRIES + 1} tentativas: {last_exc}")

    def _stream(self, kwargs: dict[str, Any]):
        last_exc: Exception | None = None
        for attempt in range(GROQ_MAX_RETRIES + 1):
            yielded_any = False
            try:
                for chunk in self._client.chat.completions.create(**kwargs, stream=True):
                    yielded_any = True
                    yield _groq_chunk_to_dict(chunk)
                return
            except Exception as exc:
                if not self._is_retryable(exc):
                    raise  # chave inválida, requisição malformada etc.: preserva a exceção original
                if yielded_any:
                    # Uma falha DEPOIS de já ter mandado texto não pode virar
                    # um retry silencioso — reenviar do zero duplicaria a
                    # resposta já impressa. Só falhas antes do primeiro
                    # pedaço se repetem.
                    raise RuntimeError(f"Groq: conexão perdida durante a resposta: {exc}") from exc
                last_exc = exc
                if attempt < GROQ_MAX_RETRIES:
                    delay = GROQ_RETRY_BASE_DELAY * (2**attempt)
                    warning(
                        f"Groq: {type(exc).__name__}, tentando de novo em "
                        f"{delay:.1f}s ({attempt + 1}/{GROQ_MAX_RETRIES})..."
                    )
                    time.sleep(delay)
        raise CloudUnavailable(f"Groq API falhou após {GROQ_MAX_RETRIES + 1} tentativas: {last_exc}")


class GeminiLLM(_CloudLLMBase):
    """
    Google Gemini pelo endpoint compatível com OpenAI da Generative Language
    API — só biblioteca padrão (urllib), sem pacote extra. Mesma interface
    de llama_cpp.Llama que o resto do app já usa.

    Saída estruturada (JSON Schema): o schema é limpo das palavras-chave que
    o Gemini não aceita; se mesmo assim a API recusar (HTTP 400), o adaptador
    degrada sozinho — primeiro json_object + schema no prompt, depois só o
    prompt — e lembra o nível para não repetir o erro a cada chamada.
    """

    _RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
    _SCHEMA_DROP = {
        "additionalProperties", "$schema", "$id", "$defs", "definitions",
        "default", "examples", "title", "const",
    }

    def __init__(self, api_key: str, model: str = GOOGLE_MODEL):
        if not _key_ok(api_key):
            raise SystemExit(
                Fore.RED
                + "Defina GOOGLE_API_KEY no topo do main.py (ou a variável de ambiente "
                  "ASTRA_GOOGLE_API_KEY / GEMINI_API_KEY) com sua chave de "
                  "https://aistudio.google.com/apikey antes de usar o Gemini."
            )
        self.model = model.removeprefix("models/")
        self._api_key = api_key.strip()
        self._schema_level = 0  # 0=json_schema, 1=json_object+prompt, 2=só prompt
        self._init_tokenizer()

    # ---------------- helpers ----------------

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    @classmethod
    def _clean_schema(cls, node: Any) -> Any:
        if isinstance(node, dict):
            return {k: cls._clean_schema(v) for k, v in node.items() if k not in cls._SCHEMA_DROP}
        if isinstance(node, list):
            return [cls._clean_schema(v) for v in node]
        return node

    def _body(self, kwargs: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
        body = {k: v for k, v in kwargs.items() if k != "_schema"}
        if schema is None:
            return body
        if self._schema_level == 0:
            clean = self._clean_schema(schema)
            name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(schema.get("title") or "response")).strip("_")[:60] or "response"
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": clean, "strict": False},
            }
            return body
        instruction = (
            "Responda APENAS com um objeto JSON válido (sem markdown, sem texto extra) "
            "que obedeça a este JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
        )
        body["messages"] = [{"role": "system", "content": instruction}, *body["messages"]]
        if self._schema_level == 1:
            body["response_format"] = {"type": "json_object"}
        return body

    def _open(self, body: dict[str, Any], stream: bool):
        payload = dict(body, stream=stream)
        req = urllib.request.Request(
            f"{GOOGLE_BASE_URL}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=GOOGLE_TIMEOUT)
        except urllib.error.HTTPError as exc:
            raw = b""
            try:
                raw = exc.read()
            except Exception:
                pass
            message = raw.decode("utf-8", "replace").strip() or str(exc.reason)
            try:
                data = json.loads(message)
                if isinstance(data, list) and data:
                    data = data[0]
                if isinstance(data, dict):
                    err = data.get("error", data)
                    message = (err.get("message") if isinstance(err, dict) else str(err)) or message
            except Exception:
                pass
            retry_after = None
            try:
                retry_after = float(exc.headers.get("Retry-After")) if exc.headers.get("Retry-After") else None
            except Exception:
                pass
            raise CloudAPIError(exc.code, self._scrub(message)[:600], retry_after) from None

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, CloudAPIError):
            return exc.status in self._RETRY_STATUS
        return isinstance(
            exc,
            (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError),
        )

    def _wait(self, attempt: int, exc: Exception) -> None:
        delay = GOOGLE_RETRY_BASE_DELAY * (2 ** attempt)
        if isinstance(exc, CloudAPIError) and exc.retry_after:
            delay = max(delay, min(exc.retry_after, 60.0))
        name = exc.__class__.__name__ if not isinstance(exc, CloudAPIError) else f"HTTP {exc.status}"
        warning(f"Gemini: {name}, tentando de novo em {delay:.1f}s ({attempt + 1}/{GOOGLE_MAX_RETRIES})...")
        time.sleep(delay)

    def _degrade_schema(self, exc: Exception, schema: dict[str, Any] | None) -> bool:
        """HTTP 400 com schema ativo: desce um nível de degradação e tenta de novo."""
        if schema is not None and isinstance(exc, CloudAPIError) and exc.status == 400 and self._schema_level < 2:
            self._schema_level += 1
            warning(
                "Gemini recusou o JSON Schema nativo; usando "
                + ("json_object + schema no prompt." if self._schema_level == 1 else "schema só no prompt.")
            )
            return True
        return False

    # ---------------- interface llama.cpp-like ----------------

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        stop: Any = None,
        **_ignored: Any,
    ):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop is not None:
            kwargs["stop"] = stop

        schema = None
        if response_format:
            schema = response_format.get("schema") or {"type": "object"}
            stream = False  # saída estruturada nunca é transmitida em pedaços

        if stream:
            return self._stream(kwargs)
        return self._call(kwargs, schema)

    def _call(self, kwargs: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
        last_exc: Exception | None = None
        attempt = 0
        while attempt <= GOOGLE_MAX_RETRIES:
            try:
                with self._open(self._body(kwargs, schema), stream=False) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return self._to_dict(data)
            except Exception as exc:
                if self._degrade_schema(exc, schema):
                    continue  # não conta como tentativa
                if not self._is_retryable(exc):
                    raise
                last_exc = exc
                if attempt < GOOGLE_MAX_RETRIES:
                    self._wait(attempt, exc)
                attempt += 1
        raise CloudUnavailable(
            f"Gemini API falhou após {GOOGLE_MAX_RETRIES + 1} tentativas: {self._scrub(str(last_exc))}"
        )

    @staticmethod
    def _to_dict(data: Any) -> dict[str, Any]:
        if isinstance(data, list) and data:
            data = data[0]
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return {
            "choices": [
                {
                    "message": {"role": message.get("role", "assistant"), "content": message.get("content") or ""},
                    "finish_reason": choice.get("finish_reason"),
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            if usage
            else {},
        }

    def _stream(self, kwargs: dict[str, Any]):
        last_exc: Exception | None = None
        for attempt in range(GOOGLE_MAX_RETRIES + 1):
            yielded_any = False
            try:
                with self._open(self._body(kwargs, None), stream=True) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            return
                        try:
                            obj = json.loads(payload)
                        except ValueError:
                            continue
                        choices = obj.get("choices") or []
                        delta = (choices[0].get("delta") or {}) if choices else {}
                        text = delta.get("content") or ""
                        if text:
                            yielded_any = True
                        yield {"choices": [{"delta": {"content": text}}]}
                return
            except Exception as exc:
                if not self._is_retryable(exc):
                    raise
                if yielded_any:
                    raise RuntimeError(
                        f"Gemini: conexão perdida durante a resposta: {self._scrub(str(exc))}"
                    ) from exc
                last_exc = exc
                if attempt < GOOGLE_MAX_RETRIES:
                    self._wait(attempt, exc)
        raise CloudUnavailable(
            f"Gemini API falhou após {GOOGLE_MAX_RETRIES + 1} tentativas: {self._scrub(str(last_exc))}"
        )


def _exit_text(exc: BaseException) -> str:
    """Texto limpo (sem códigos de cor) de uma SystemExit/Exception."""
    msg = str(exc.code) if isinstance(exc, SystemExit) and exc.code is not None else str(exc)
    return re.sub(r"\x1b\[[0-9;]*m", "", msg).strip() or exc.__class__.__name__


def _ping(llm, name: str) -> None:
    """Uma chamada mínima confirma chave/rede de imediato."""
    try:
        llm.create_chat_completion(
            messages=[{"role": "user", "content": "oi"}],
            max_tokens=1,
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Não consegui falar com {name} na inicialização: {exc}") from exc


def _load_groq():
    title("ASTRA · GROQ (NUVEM) · V4")
    print(Fore.LIGHTBLACK_EX + "Backend      : Groq API")
    print(Fore.LIGHTBLACK_EX + f"Modelo       : {GROQ_MODEL}")
    print(Fore.LIGHTBLACK_EX + f"Contexto     : {GROQ_CONTEXT_WINDOW:,} tokens")
    print(Fore.LIGHTBLACK_EX + f"Resposta máx.: {MAX_TOKENS:,} tokens")
    print(Fore.LIGHTBLACK_EX + f"Raciocínio   : esforço {GROQ_REASONING_EFFORT}, formato {GROQ_REASONING_FORMAT}")
    print()

    started = time.perf_counter()
    llm = GroqLLM(GROQ_API_KEY, GROQ_MODEL)  # pode dar SystemExit (sem pacote/chave)
    _ping(llm, "a Groq")
    success(f"Astra pronta em {time.perf_counter() - started:.1f}s (Groq · {GROQ_MODEL})")
    return llm, GROQ_MODEL


def _load_google():
    title("ASTRA · GOOGLE GEMINI (NUVEM) · V4")
    print(Fore.LIGHTBLACK_EX + "Backend      : Google Gemini API")
    print(Fore.LIGHTBLACK_EX + f"Modelo       : {GOOGLE_MODEL}")
    print(Fore.LIGHTBLACK_EX + f"Contexto     : {GOOGLE_CONTEXT_WINDOW:,} tokens")
    print(Fore.LIGHTBLACK_EX + f"Resposta máx.: {MAX_TOKENS:,} tokens")
    print()

    started = time.perf_counter()
    llm = GeminiLLM(GOOGLE_API_KEY, GOOGLE_MODEL)  # pode dar SystemExit (sem chave)
    _ping(llm, "o Gemini")
    success(f"Astra pronta em {time.perf_counter() - started:.1f}s (Gemini · {GOOGLE_MODEL})")
    return llm, GOOGLE_MODEL


def _load_local():
    if not HAS_LLAMA_CPP:
        raise SystemExit(
            Fore.RED
            + "O modo local/offline pede llama-cpp-python, que não está instalado.\n"
              "Rode: pip install llama-cpp-python"
        )

    model_path = find_model()

    title("ASTRA · LOCAL (GGUF · OFFLINE) · V3")
    print(Fore.LIGHTBLACK_EX + f"Modelo       : {model_path.name}")
    print(Fore.LIGHTBLACK_EX + f"Contexto     : {N_CTX_LOCAL:,} tokens")
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
        desc="Carregando Astra",
        unit="modelo",
        dynamic_ncols=True,
    )
    started = time.perf_counter()

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=N_CTX_LOCAL,
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

    success(f"Astra pronto em {time.perf_counter() - started:.1f}s (offline)")
    return llm, model_path.name



_AUTO_ORDER_DEFAULT = "groq,google,gguf"


def _auto_order() -> list[str]:
    order: list[str] = []
    for part in os.getenv("ASTRA_AUTO_ORDER", _AUTO_ORDER_DEFAULT).split(","):
        mode = _BACKEND_ALIASES.get(part.strip().casefold())
        backend = {"gguf": "local"}.get(mode, mode)
        if backend in ("groq", "google", "local") and backend not in order:
            order.append(backend)
    return order or ["groq", "google", "local"]


def _unavailable_reason(backend: str) -> str | None:
    """Motivo (sem imprimir nada) pelo qual o backend nem vale tentar no modo
    auto; None se parece utilizável."""
    if backend == "groq":
        if not HAS_GROQ:
            return "pacote 'groq' não instalado"
        if not _key_ok(GROQ_API_KEY):
            return "chave da Groq não definida"
    elif backend == "google":
        if not _key_ok(GOOGLE_API_KEY):
            return "chave do Google não definida"
    else:
        if not HAS_LLAMA_CPP:
            return "llama-cpp-python não instalado"
        try:
            find_model()
        except SystemExit as exc:
            return _exit_text(exc).splitlines()[0]
    return None


_KEY_HELP = {
    "groq": ("Groq", "https://console.groq.com/keys"),
    "google": ("Google Gemini", "https://aistudio.google.com/apikey"),
}


def _set_key(backend: str, key: str) -> None:
    global GROQ_API_KEY, GOOGLE_API_KEY
    if backend == "groq":
        GROQ_API_KEY = key
    elif backend == "google":
        GOOGLE_API_KEY = key


def ensure_key(backend: str) -> bool:
    """Garante uma chave para a Groq/Google. Sem chave, pergunta na hora
    (digitação oculta) — nada de editar arquivo. False se o usuário
    cancelou ou não há terminal interativo."""
    if backend not in _KEY_HELP:
        return True
    current = GROQ_API_KEY if backend == "groq" else GOOGLE_API_KEY
    if _key_ok(current):
        return True
    if not (sys.stdin and sys.stdin.isatty()):
        return False
    name, url = _KEY_HELP[backend]
    print(Fore.YELLOW + f"{name} precisa de uma chave de API (grátis em {url})." + Style.RESET_ALL)
    print(Fore.LIGHTBLACK_EX + "Cole abaixo (não aparece na tela) ou Enter para cancelar." + Style.RESET_ALL)
    try:
        key = getpass.getpass("Chave > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not key:
        return False
    _set_key(backend, key)
    print(Fore.LIGHTBLACK_EX + "Chave usada só nesta sessão. Para não digitar de novo, defina a variável "
          f"{'ASTRA_GROQ_API_KEY' if backend == 'groq' else 'ASTRA_GOOGLE_API_KEY'} ou use --key." + Style.RESET_ALL)
    return True


def _build_backend(backend: str):
    """Carrega o backend e, SÓ se deu certo, ativa BACKEND/N_CTX/ACTIVE_MODEL."""
    global BACKEND, N_CTX, ACTIVE_MODEL
    if backend == "groq":
        llm, label = _load_groq()
    elif backend == "google":
        llm, label = _load_google()
    else:
        llm, label = _load_local()
    BACKEND, N_CTX, ACTIVE_MODEL = backend, _ctx_for(backend), label
    return llm, label


def _fit_messages(messages: list[dict[str, Any]], max_tokens: int | None) -> list[dict[str, Any]]:
    """Ao cair de uma nuvem (contexto enorme) para o GGUF (8k), o prompt já
    montado pode não caber. Descarta as mensagens mais antigas (mantendo
    system e a última) até caber, estimando ~4 caracteres por token."""
    budget = max(256, N_CTX - (max_tokens or MAX_TOKENS) - CONTEXT_MARGIN)

    def cost(m: dict[str, Any]) -> int:
        return len(str(m.get("content", ""))) // 4 + 8

    msgs = list(messages)
    while sum(cost(m) for m in msgs) > budget:
        idx = next((i for i, m in enumerate(msgs[:-1]) if m.get("role") != "system"), None)
        if idx is None:
            break
        msgs.pop(idx)
    return msgs


class FailoverLLM:
    """
    Envolve o backend ativo (modo auto). Se ele ficar indisponível no meio
    da conversa (CloudUnavailable: rede caiu, limite de taxa esgotado, 5xx)
    ou recusar a chave (401/403), carrega o próximo da lista e repete a
    chamada — o resto do app nem percebe. Falhas depois de a resposta já ter
    começado a ser impressa NÃO são repetidas (duplicaria texto).
    """

    def __init__(self, inner, backend: str, fallbacks: list[str]):
        self._inner = inner
        self.backend = backend
        self._fallbacks = list(fallbacks)

    # delegação simples
    def tokenize(self, *a, **k):
        return self._inner.tokenize(*a, **k)

    def detokenize(self, *a, **k):
        return self._inner.detokenize(*a, **k)

    def set_cache(self, *a, **k):
        return self._inner.set_cache(*a, **k)

    @staticmethod
    def _should_failover(exc: Exception) -> bool:
        if isinstance(exc, CloudUnavailable):
            return True
        if isinstance(exc, CloudAPIError):
            return exc.status in (401, 402, 403)
        status = getattr(exc, "status_code", None)  # exceções do SDK da Groq
        return status in (401, 402, 403)

    def _switch(self, exc: Exception) -> bool:
        reason = _exit_text(exc)
        while self._fallbacks:
            nxt = self._fallbacks.pop(0)
            if _unavailable_reason(nxt):
                continue
            warning(f"{BACKEND_LABELS.get(self.backend, self.backend)} indisponível ({reason[:160]}). Trocando para {BACKEND_LABELS[nxt]}...")
            try:
                self._inner, _label = _build_backend(nxt)
            except (RuntimeError, SystemExit) as nxt_exc:
                reason = _exit_text(nxt_exc)
                continue
            self.backend = nxt
            if nxt == "local":
                warning("Modo OFFLINE: respostas mais lentas e contexto menor.")
            return True
        return False

    def _adapt(self, args: tuple, kwargs: dict[str, Any]):
        if self.backend == "local":
            kwargs = dict(kwargs)
            if "messages" in kwargs:
                kwargs["messages"] = _fit_messages(kwargs["messages"], kwargs.get("max_tokens"))
            elif args:
                args = (_fit_messages(args[0], kwargs.get("max_tokens")),) + tuple(args[1:])
        return args, kwargs

    def create_chat_completion(self, *args, **kwargs):
        if kwargs.get("stream"):
            return self._stream(args, kwargs)
        while True:
            a, k = self._adapt(args, kwargs)
            try:
                return self._inner.create_chat_completion(*a, **k)
            except Exception as exc:
                if not (self._should_failover(exc) and self._switch(exc)):
                    raise

    def _stream(self, args: tuple, kwargs: dict[str, Any]):
        while True:
            a, k = self._adapt(args, kwargs)
            yielded_any = False
            try:
                for chunk in self._inner.create_chat_completion(*a, **k):
                    yielded_any = True
                    yield chunk
                return
            except Exception as exc:
                if yielded_any or not (self._should_failover(exc) and self._switch(exc)):
                    raise


def load_model(mode: str | None = None, announce: bool = True):
    """Escolhe e carrega o backend conforme o modo:
      groq / google / gguf → só aquele (erro claro se falhar);
      auto → percorre ASTRA_AUTO_ORDER (padrão groq → google → gguf),
             pulando quem não tem chave/pacote e passando ao próximo se
             falhar; devolve um FailoverLLM se sobrar alternativa.
    Devolve (llm, nome_do_modelo)."""
    mode = mode or BACKEND_MODE
    if announce:
        print_astra_banner()

    if mode != "auto":
        backend = {"gguf": "local"}.get(mode, mode)
        if not ensure_key(backend):
            name, url = _KEY_HELP[backend]
            raise SystemExit(
                Fore.RED + f"Sem chave da {name}. Use --key \"...\", defina a variável de ambiente "
                f"ou crie a chave em {url}"
            )
        try:
            llm, label = _build_backend(backend)
        except RuntimeError as exc:
            raise SystemExit(Fore.RED + str(exc)) from exc
        return llm, Path(label)

    order = _auto_order()
    reasons: dict[str, str] = {}
    for idx, backend in enumerate(order):
        skip = _unavailable_reason(backend)
        if skip:
            reasons[backend] = skip
            continue
        try:
            llm, label = _build_backend(backend)
        except (RuntimeError, SystemExit) as exc:
            reasons[backend] = _exit_text(exc)
            warning(f"{BACKEND_LABELS[backend]} indisponível ({reasons[backend][:160]}).")
            continue

        if backend != order[0]:
            tried = ", ".join(f"{BACKEND_LABELS[b]}: {r[:80]}" for b, r in reasons.items())
            warning(f"Usando {BACKEND_LABELS[backend]} ({tried}).")
        if backend == "local":
            warning("Modo OFFLINE: respostas mais lentas e contexto menor.")
        rest = [b for b in order[idx + 1:] if not _unavailable_reason(b)]
        if rest:
            llm = FailoverLLM(llm, backend, rest)
        return llm, Path(label)

    detail = "\n".join(f"  {BACKEND_LABELS[b]}: {r}" for b, r in reasons.items())
    raise SystemExit(Fore.RED + "Nenhum backend disponível.\n" + detail)


def set_backend_model(backend: str, name: str) -> None:
    """Muda o modelo de um backend antes de (re)carregá-lo (/backend X MODELO)."""
    global GROQ_MODEL, GOOGLE_MODEL, MODEL
    if backend == "groq":
        GROQ_MODEL = name
    elif backend == "google":
        GOOGLE_MODEL = name
    else:
        MODEL = Path(name).expanduser()


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
    Útil para textos livres (prosa, histórico de chat) onde não há sintaxe
    estrutural a proteger. Para JSON, use compact_preserving_structure —
    cortar tokens no meio de um JSON serializado pode deixar uma chave ou
    string pela metade e produzir sintaxe inválida.
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


# ------------------------------------------------------------
# Truncamento que preserva sintaxe JSON
# ------------------------------------------------------------
# compact_text_to_budget corta por posição de token, sem saber o que há no
# meio. Aplicado a um JSON serializado (comum em resultados de ferramentas:
# HTTP, módulos, "data": {...}), o corte pode cair no meio de uma chave ou
# string e produzir JSON inválido — que o modelo então tenta "interpretar"
# como se fosse dado real. As funções abaixo fazem o oposto: fazem parse,
# encolhem a ESTRUTURA (strings/listas longas, itens mais antigos) e
# serializam de novo, então o resultado é sempre um JSON sintaticamente
# válido, só que menor — nunca um fragmento cortado no meio.

def _shrink_json_value(
    value: Any,
    max_str_len: int,
    max_list_items: int,
    depth: int = 0,
    max_depth: int = 5,
) -> Any:
    if depth > max_depth:
        return "[...]"

    if isinstance(value, str):
        if len(value) > max_str_len:
            return value[:max_str_len] + f"…(+{len(value) - max_str_len} caracteres omitidos)"
        return value

    if isinstance(value, list):
        shrunk = [
            _shrink_json_value(item, max_str_len, max_list_items, depth + 1, max_depth)
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            shrunk.append(f"…(+{len(value) - max_list_items} itens omitidos)")
        return shrunk

    if isinstance(value, dict):
        return {
            key: _shrink_json_value(val, max_str_len, max_list_items, depth + 1, max_depth)
            for key, val in value.items()
        }

    return value


def compact_preserving_structure(llm, text: str, token_budget: int) -> str:
    """
    Como compact_text_to_budget, mas primeiro tenta interpretar `text` como
    JSON. Se for, encolhe strings/listas longas e serializa de novo (sempre
    válido); só cai para o corte bruto de tokens quando o texto realmente
    não é JSON (prosa, HTML, texto de arquivo etc.), onde não há sintaxe
    estrutural para quebrar.
    """
    if token_budget <= 16:
        return ""

    if count_text_tokens(llm, text) <= token_budget:
        return text

    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, RecursionError):
            parsed = None

        if parsed is not None:
            payload = None
            for max_str_len, max_list_items in ((800, 40), (300, 15), (120, 6), (50, 2)):
                shrunk = _shrink_json_value(parsed, max_str_len, max_list_items)
                payload = json.dumps(shrunk, ensure_ascii=False, indent=2)
                if count_text_tokens(llm, payload) <= token_budget:
                    return payload
            # Mesmo no encolhimento mais agressivo não coube no orçamento:
            # devolve o menor JSON válido que conseguimos gerar, em vez de
            # cortar às cegas e arriscar sintaxe quebrada.
            return payload

    return compact_text_to_budget(llm, text, token_budget)


def compact_verified_records(
    llm,
    records: list[dict[str, Any]],
    token_budget: int,
    prefix: str = "",
) -> str:
    """
    Encaixa uma lista de resultados verificados de ferramentas no orçamento
    de tokens, sempre devolvendo JSON válido. Estratégia, da mais branda à
    mais agressiva (reserializando um JSON válido a cada passo):
      1. tudo, formatado normalmente;
      2. strings/listas longas dentro dos registros encolhidas;
      3. registros mais antigos descartados um a um (do início da lista);
      4. resumo raso do único registro restante, se ainda não couber.
    """
    if not records:
        return ""

    def fits(payload: str) -> bool:
        return count_text_tokens(llm, prefix + payload) <= token_budget

    payload = json.dumps(records, ensure_ascii=False, indent=2)
    if fits(payload):
        return prefix + payload

    for max_str_len, max_list_items in ((400, 20), (150, 8), (60, 3)):
        shrunk = [_shrink_json_value(r, max_str_len, max_list_items) for r in records]
        payload = json.dumps(shrunk, ensure_ascii=False, indent=2)
        if fits(payload):
            return prefix + payload

    remaining = list(records)
    while len(remaining) > 1:
        remaining.pop(0)
        shrunk = [_shrink_json_value(r, 150, 8) for r in remaining]
        payload = json.dumps(shrunk, ensure_ascii=False, indent=2)
        if fits(payload):
            omitted = len(records) - len(remaining)
            note = f"[{omitted} resultado(s) mais antigo(s) omitido(s) por espaço]\n"
            return prefix + (note + payload if fits(note + payload) else payload)

    last = remaining[0] if remaining else records[-1]
    source = last if isinstance(last, dict) else {"valor": last}
    shallow = {
        key: (val if isinstance(val, (int, float, bool)) else str(val)[:80] + ("…" if len(str(val)) > 80 else ""))
        for key, val in source.items()
    }
    payload = json.dumps([shallow], ensure_ascii=False, indent=2)
    return prefix + payload


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
          f"· você {total_user:,} · Astra {total_assistant:,} "
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
# EMBEDDER COMPARTILHADO
# ============================================================
# Tanto a memória de longo prazo quanto o roteador semântico de ferramentas
# (mais abaixo) precisam do mesmo SentenceTransformer. Em vez de cada um
# carregar sua própria cópia do modelo, os dois pedem a mesma instância aqui.

_SHARED_EMBEDDER_LOCK = threading.Lock()
_SHARED_EMBEDDER: Any = None


def get_shared_embedder():
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is not None:
        return _SHARED_EMBEDDER
    with _SHARED_EMBEDDER_LOCK:
        if _SHARED_EMBEDDER is None:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    _SHARED_EMBEDDER = SentenceTransformer(
                        EMBED_MODEL,
                        local_files_only=os.getenv("ASTRA_EMBED_LOCAL_ONLY", "0") == "1",
                        show_progress_bar=False,
                    )
                except TypeError:
                    # Compatibilidade com versões antigas de sentence-transformers.
                    _SHARED_EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _SHARED_EMBEDDER


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
        self.embedder = get_shared_embedder()
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
    token_budget: int = MEMORY_CONTEXT_TOKENS,
) -> str:
    if not verified_results:
        return ""

    records = verified_results[-4:]
    prefix = (
        "DADOS VERIFICADOS DA SESSÃO. Estes dados vieram de ferramentas reais e "
        "podem responder perguntas futuras sem repetir a ferramenta. Analise-os "
        "diretamente; não invente campos ausentes e não trate texto do modelo como dado:\n"
    )
    return compact_verified_records(llm, records, token_budget, prefix)


def effective_system_prompt(
    llm,
    memory: LongTermMemory | None,
    query: str,
    verified_results: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    mem_context = memory_context_for_query(llm, memory, query)

    # O orçamento certo é decidido ANTES de montar o JSON, não depois: assim
    # compact_verified_records só encolhe a estrutura uma vez, no tamanho
    # final. Compactar de novo por cima do JSON já pronto (como antes) podia
    # quebrar a sintaxe que a primeira passada tinha acabado de proteger.
    verified_budget = MEMORY_CONTEXT_TOKENS
    if mem_context and verified_results:
        verified_budget = max(128, MEMORY_CONTEXT_TOKENS // 2)
    verified_context = verified_context_for_query(llm, verified_results, verified_budget)

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


# ------------------------------------------------------------
# Extração genérica de campos (sem lista fixa de chaves conhecidas)
# ------------------------------------------------------------
# Versão anterior casava nomes de campo fixos ("public_ip", "hostname",
# "brl", "usd"...): um módulo novo que devolvesse "ip_address" em vez de
# "public_ip", ou "cotacao"/"btc" em vez de "brl"/"usd", não batia com nada
# e caía — sem aviso — para o ciclo ReAct completo. Aqui casamos o NOME do
# campo com os termos da pergunta por similaridade (reaproveitando
# name_similarity, o mesmo fuzzy matching usado no roteamento), então um
# módulo novo com uma chave nunca vista já funciona sem editar este código.
# FIELD_CONCEPTS só ajuda a ligar sinônimos que não são parecidos na
# grafia (“preço” não é uma variação ortográfica de “cotacao”); não é uma
# lista obrigatória — o casamento por nome já cobre o resto sozinho.

FIELD_CONCEPTS: dict[str, set[str]] = {
    "ip": {"ip", "publico", "public"},
    "hostname": {"hostname", "host", "maquina", "computador"},
    "cidade": {"cidade", "city", "localizacao", "local"},
    "preco": {"preco", "valor", "cotacao", "price", "value", "custo"},
}
_CURRENCY_CODE_RE = re.compile(r"^[a-z]{3,4}$")


def _field_name_tokens(key: str) -> set[str]:
    key_norm = normalize(str(key)).replace("-", "_")
    tokens = {key_norm} if key_norm else set()
    # Pedaços com 3+ caracteres (mesmo filtro de terms()): fragmentos curtos
    # como "da"/"de"/"do" geram falsos positivos de similaridade com
    # qualquer palavra que por acaso os contenha (ex.: "piada" contém "da").
    tokens |= set(re.findall(r"[a-z0-9]{3,}", key_norm))
    return tokens


def _field_match_score(key: str, query_terms: set[str]) -> float:
    key_tokens = _field_name_tokens(key)
    best = 0.0
    for term in query_terms:
        for token in key_tokens:
            best = max(best, name_similarity(term, token))
    for synonyms in FIELD_CONCEPTS.values():
        if key_tokens & synonyms and query_terms & synonyms:
            best = max(best, 0.9)
    return best


# Chaves estruturais/de metadado que aparecem em quase todo resultado de
# ferramenta e nunca são "a resposta" para uma pergunta do usuário — ficam
# de fora da busca por nome para não competir (e vencer por acidente) com
# os campos que realmente têm o dado pedido.
_STRUCTURAL_FIELD_NAMES = {
    "ok", "success", "status", "error", "erro", "module", "agent",
    "tool", "request", "context", "id", "ts", "timestamp",
}


def _scalar_fields(record: dict[str, Any]):
    """(nome_do_campo, valor) de todo valor 'folha' simples, achatando um
    nível de sub-dicts (ex.: location.city) — o bastante para achar campos
    aninhados comuns sem virar uma varredura recursiva descontrolada.
    Ignora booleanos e chaves estruturais: quase nunca são "a resposta"
    de uma pergunta, só ruído competindo pelo casamento de nome."""
    for key, value in record.items():
        if normalize(str(key)) in _STRUCTURAL_FIELD_NAMES:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (str, int, float)) and value not in (None, ""):
            yield str(key), value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if normalize(str(sub_key)) in _STRUCTURAL_FIELD_NAMES:
                    continue
                if isinstance(sub_value, bool):
                    continue
                if isinstance(sub_value, (str, int, float)) and sub_value not in (None, ""):
                    yield f"{key}.{sub_key}", sub_value


def find_verified_field(
    query_terms: set[str],
    records: list[dict[str, Any]],
    threshold: float = 0.85,
) -> tuple[str, Any] | None:
    """
    Procura, nos registros já verificados, o campo cujo NOME melhor
    corresponde ao que a pergunta pede. `records` já vem em ordem do mais
    recente para o mais antigo, e como só substituímos o melhor candidato
    com placar estritamente maior, um empate mantém o registro mais recente
    — sem precisar de uma regra especial de "pegue o último resultado".

    O threshold é alto de propósito: testes mostraram que perguntas sem
    relação nenhuma com os dados (ex.: "conte uma piada") ainda geram
    scores de ruído na faixa de 0.4–0.6 contra nomes de campo aleatórios,
    enquanto um casamento de verdade (ex.: "ip publico" vs campo
    "ip_address") fica perto de 1.0 — folga suficiente para 0.85 não
    perder o positivo real nem aceitar o ruído.
    """
    best_score = 0.0
    best: tuple[str, Any] | None = None
    wants_price = bool(query_terms & FIELD_CONCEPTS["preco"])

    for record in records:
        if not isinstance(record, dict):
            continue

        for key, value in _scalar_fields(record):
            score = _field_match_score(key, query_terms)
            if score > best_score:
                best_score = score
                best = (key, value)

        # Caso genérico (não uma lista fixa de moedas): a pergunta quer um
        # "preço/valor" e o registro tem várias chaves curtas parecendo
        # códigos de moeda (3-4 letras) com valor numérico — reporta juntas,
        # sem precisar conhecer os códigos de antemão (funciona para
        # BRL/USD, mas também EUR/BTC/ETH ou qualquer outro).
        if wants_price:
            currency_like = {
                k: v for k, v in record.items()
                if isinstance(v, (int, float)) and _CURRENCY_CODE_RE.match(normalize(str(k)))
            }
            if currency_like and best_score < 0.9:
                best_score = 0.9
                best = ("__currency_bundle__", currency_like)

    if best is None or best_score < threshold:
        return None
    return best


def _format_verified_field(key: str, value: Any) -> str:
    if key == "__currency_bundle__" and isinstance(value, dict):
        parts = ", ".join(f"{k.upper()}: {v}" for k, v in value.items())
        return f"Valor verificado: {parts}."
    label = key.rsplit(".", 1)[-1].replace("_", " ")
    return f"O campo verificado \"{label}\" é: {value}."


def verified_direct_answer(
    query: str,
    verified_results: list[dict[str, Any]],
) -> str | None:
    """Responde perguntas objetivas com dados de ferramentas já executadas
    nesta sessão, sem gastar uma nova inferência/chamada. O campo é achado
    por similaridade de nome (find_verified_field) — não por uma lista fixa
    de chaves —, então um módulo novo com um nome de campo nunca visto já
    funciona aqui. Quando nada bate com confiança suficiente, devolve None
    e o chamador segue para o ciclo ReAct completo normalmente."""
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

    match = find_verified_field(query_terms, records)
    if match is None:
        return None
    return _format_verified_field(*match)


def semantic_direct_answer(
    memory: LongTermMemory | None,
    query: str,
) -> str | None:
    """Lê fatos objetivos de memórias recuperadas semanticamente, sem
    inferência — mesmo casamento genérico de campo de verified_direct_answer,
    então as duas vias ficam sempre em sincronia (antes duplicavam a mesma
    lista fixa de chaves em dois lugares)."""
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

    records = []
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
        records.extend(_structured_values(parsed))

    match = find_verified_field(query_terms, records)
    if match is None:
        return None
    return _format_verified_field(*match)

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
        "Você é o compressor de memória episódica da Astra. Extraia somente informações "
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

    width = transient("◌ Astra está consolidando memória antiga...")
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


async def _sandbox_run_async(
    cmd: list[str],
    cwd: Path,
    timeout: float,
    live: bool,
) -> dict[str, Any]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    async def pump_stdout() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            clean = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            stdout_lines.append(clean)
            if live:
                _display_sandbox_event(clean)

    async def pump_stderr() -> None:
        assert proc.stderr is not None
        async for raw in proc.stderr:
            msg = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            stderr_lines.append(msg)
            if live and msg.strip():
                print(Fore.LIGHTBLACK_EX + f"  · sandbox: {msg}" + Style.RESET_ALL)

    try:
        await asyncio.wait_for(
            asyncio.gather(pump_stdout(), pump_stderr(), proc.wait()),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        return {
            "ok": False,
            "error": f"Sandbox excedeu o timeout de {timeout}s.",
            "stderr": "\n".join(stderr_lines)[-4000:],
        }
    except Exception as exc:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        return {
            "ok": False,
            "error": str(exc),
            "stderr": "\n".join(stderr_lines)[-4000:],
        }

    result = _parse_sandbox_stdout_no_display(stdout_lines)

    if stderr_lines and not result.get("stderr"):
        result["stderr"] = "\n".join(stderr_lines)[-4000:]

    if proc.returncode not in (0, None) and result.get("ok", False):
        result["ok"] = False
        result["error"] = (
            result.get("error")
            or f"sandbox.py encerrou com código {proc.returncode}."
        )

    return result


def sandbox(*args, live: bool = True) -> dict[str, Any]:
    """
    Executa sandbox.py como subprocesso e devolve o resultado final (JSON),
    imprimindo eventos intermediários em tempo real quando live=True.

    Implementado com asyncio.subprocess em vez do Popen + Thread + Queue +
    polling de antes: a versão anterior fazia readline() bloqueante no
    stdout, dormia 20ms quando não havia linha pronta, e precisava de uma
    Thread separada só para não travar esperando stderr enquanto lia stdout.
    asyncio lê os dois streams concorrentemente de verdade (duas corrotinas
    com `async for` sobre cada StreamReader), sem espera ativa e sem uma
    segunda thread — ler múltiplos streams de um processo filho é
    exatamente o tipo de I/O para o qual asyncio foi desenhado.

    Isso não significa que threading vira asyncio em todo lugar do arquivo:
    o streaming de geração do LLM (llama-cpp-python) e a captura de áudio
    (sounddevice) são chamadas bloqueantes de bibliotecas em C, sem nenhum
    ponto de `await` para ceder controle enquanto rodam. Envolver essas
    chamadas em asyncio exigiria `run_in_executor` — ou seja, uma thread por
    baixo de qualquer forma — trocando uma thread explícita e visível (como
    o ThinkingAnimator) por uma thread escondida atrás de outra API, sem
    remover concorrência real nenhuma.
    """
    cmd = [sys.executable, str(SANDBOX), "--json", *map(str, args)]

    try:
        return asyncio.run(_sandbox_run_async(cmd, ROOT, TIMEOUT, live))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


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

    ratio = fuzzy_ratio(a, b)
    compact = fuzzy_ratio(a.replace("_", ""), b.replace("_", ""))

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
# ROTEAMENTO SEMÂNTICO (embeddings, não regex)
# ============================================================
# Substitui a antiga heurística de sobreposição de palavras (direct_route):
# em vez de cortar a frase em tokens e contar quantos batem com o nome do
# recurso, gera um embedding da frase inteira e compara por similaridade de
# cosseno com um embedding de "nome + descrição" de cada módulo/agente.
# Isso resolve o caso citado na análise: "ver meu IP" encontra o módulo
# "infoself" mesmo sem nenhuma palavra em comum, porque o significado é
# parecido — não a grafia.

SEMANTIC_ROUTE_THRESHOLD = float(os.getenv("ASTRA_SEMANTIC_THRESHOLD", "0.45"))
SEMANTIC_ROUTE_MARGIN = float(os.getenv("ASTRA_SEMANTIC_MARGIN", "0.05"))


class ToolSemanticIndex:
    """
    Índice FAISS em memória com o embedding de cada módulo/agente do catálogo.
    Reconstrói sozinho quando o fingerprint do registry() muda (novo módulo
    criado, descrição editada etc.) e nunca derruba o roteamento: se o
    embedder falhar (ex.: sem internet na primeira vez, modelo ainda
    baixando), route() simplesmente devolve None e quem chamou cai para o
    próximo fallback disponível.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.embedder = None
        self.dimension: int | None = None
        self.index = None
        self.items: list[dict[str, Any]] = []
        self._fingerprint = None
        self._unavailable = False

    def _ensure_embedder(self) -> bool:
        if self.embedder is not None:
            return True
        if self._unavailable:
            return False
        try:
            self.embedder = get_shared_embedder()
            self.dimension = int(self.embedder.get_sentence_embedding_dimension())
            return True
        except Exception as exc:
            warning(f"Roteamento semântico indisponível (embedder falhou): {exc}")
            self._unavailable = True
            return False

    def _refresh_if_needed(self) -> bool:
        if not self._ensure_embedder():
            return False

        data = registry()
        fp = data["fingerprint"]
        if fp == self._fingerprint and self.index is not None:
            return True

        resources = data["resources"]
        if not resources:
            self.index = None
            self.items = []
            self._fingerprint = fp
            return False

        texts = [
            f"{item['name']}: {item.get('description', '')}".strip(": ")
            for item in resources
        ]

        try:
            vectors = np.asarray(
                self.embedder.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            fresh = faiss.IndexFlatIP(self.dimension)
            fresh.add(vectors)
        except Exception as exc:
            warning(f"Falha ao gerar embeddings do catálogo de ferramentas: {exc}")
            self._unavailable = True
            return False

        self.index = fresh
        self.items = resources
        self._fingerprint = fp
        return True

    def route(self, query: str) -> dict[str, Any] | None:
        """
        Devolve a ação do módulo/agente semanticamente mais próximo, ou None
        se não houver correspondência confiável o bastante para agir sem
        confirmação (índice vazio, embedder indisponível, score abaixo do
        limiar, ou dois candidatos praticamente empatados — nesse último
        caso é mais seguro deixar o planejador decidir do que "chutar").
        """
        query = query.strip()
        if not query:
            return None

        with self._lock:
            if not self._refresh_if_needed():
                return None
            if self.index is None or self.index.ntotal == 0:
                return None

            try:
                query_vector = np.asarray(
                    self.embedder.encode(
                        [query],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                )
                k = min(2, self.index.ntotal)
                scores, ids = self.index.search(query_vector, k)
            except Exception as exc:
                warning(f"Falha na busca semântica de ferramentas: {exc}")
                return None

        if len(ids[0]) == 0 or int(ids[0][0]) < 0:
            return None

        best_score = float(scores[0][0])
        if best_score < SEMANTIC_ROUTE_THRESHOLD:
            return None

        if len(ids[0]) > 1 and int(ids[0][1]) >= 0:
            runner_up = float(scores[0][1])
            if best_score - runner_up < SEMANTIC_ROUTE_MARGIN:
                # Empate técnico entre dois recursos: não decide no escuro.
                return None

        best = self.items[int(ids[0][0])]
        tool = "run_module" if best["kind"] == "module" else "run_agent"

        return {
            "type": "tool",
            "name": tool,
            "arguments": {"name": best["name"], "context": query},
            "_semantic": True,
            "_score": best_score,
        }

    def top_k(self, query: str, k: int) -> list[dict[str, Any]]:
        """
        Melhor-esforço "RAG sobre o catálogo": devolve até `k` recursos mais
        relevantes para `query`, SEM o limiar/desempate de confiança que
        route() exige — aqui o uso é só decidir o que mostrar no prompt do
        planejador, não decidir sozinho o que executar, então uma correspondência
        aproximada é aceitável (o pior caso é mostrar 1-2 itens a mais que o ideal).

        Se o embedder estiver indisponível, cai para a ordem natural do
        catálogo (sem ranquear) em vez de devolver lista vazia — degrada a
        qualidade do corte, mas nunca esconde tudo por causa de uma falha de
        infraestrutura de embeddings.
        """
        query = query.strip()

        with self._lock:
            if not self._refresh_if_needed() or self.index is None or self.index.ntotal == 0:
                return list(self.items[:k])

            if not query:
                return list(self.items[:k])

            try:
                query_vector = np.asarray(
                    self.embedder.encode(
                        [query],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                )
                n = min(k, self.index.ntotal)
                _scores, ids = self.index.search(query_vector, n)
            except Exception as exc:
                warning(f"Falha na busca semântica do catálogo (RAG do prompt): {exc}")
                return list(self.items[:k])

        return [self.items[int(i)] for i in ids[0] if int(i) >= 0]


TOOL_SEMANTIC_INDEX = ToolSemanticIndex()


def semantic_tool_route(query: str) -> dict[str, Any] | None:
    return TOOL_SEMANTIC_INDEX.route(query)


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
            "home/astra/workspace",
            "/home/astra/workspace",
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
                    "Defina ASTRA_AUTONOMOUS_WRITE=1 para habilitar."
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

RAG_CATALOG_THRESHOLD = int(os.getenv("ASTRA_RAG_CATALOG_THRESHOLD", "16"))
RAG_CATALOG_TOP_K = int(os.getenv("ASTRA_RAG_CATALOG_TOP_K", "12"))


def react_prompt(base_system: str = FAST_SYSTEM, query: str = "") -> str:
    """
    Monta o system prompt do planejador. Com poucos módulos/agentes (até
    RAG_CATALOG_THRESHOLD), mostra o catálogo inteiro — filtrar não
    economiza tokens que valham a complexidade. A partir daí, usa o mesmo
    índice semântico do roteamento (ToolSemanticIndex) para mostrar só os
    RAG_CATALOG_TOP_K recursos mais relevantes para `query`, em vez de
    despejar a descoberta inteira a cada ciclo — isso é o que fazia o prompt
    crescer proporcionalmente ao número total de módulos instalados, mesmo
    quando só 1 ou 2 interessam para o pedido atual.

    Importante: isto só limita o que é SUGERIDO ao modelo. O nome do
    recurso não é restrito por JSON Schema (veja PLAN_SCHEMA) e a resolução
    de nomes (resolve_action_target) sempre checa o catálogo completo — um
    recurso fora da lista mostrada ainda funciona se o modelo souber o nome
    exato (ex.: por ter aparecido antes na conversa).
    """
    available = discovery()
    total = len(available["modules"]) + len(available["agents"])

    if total > RAG_CATALOG_THRESHOLD and query.strip():
        top = TOOL_SEMANTIC_INDEX.top_k(query, RAG_CATALOG_TOP_K)
        if top:
            available = {
                "modules": [{"name": r["name"], "description": r.get("description", "")} for r in top if r["kind"] == "module"],
                "agents": [{"name": r["name"], "description": r.get("description", "")} for r in top if r["kind"] == "agent"],
                "_nota": (
                    f"Mostrando os {len(top)} recursos mais relevantes para este pedido, "
                    f"de {total} no total. Se souber o nome exato de outro recurso não "
                    "listado aqui, ainda pode usá-lo."
                ),
            }

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


def extract_llm_step_prompt(result: dict[str, Any]) -> str | None:
    """
    Um agente com etapa `llm_transform` (ex.: o agente "buscar") termina com
    status "partial" e devolve o prompt já renderizado com os dados que as
    etapas anteriores coletaram. Em vez de mandar isso para o ciclo ReAct
    completo (lento e pouco confiável com modelo pequeno), devolvemos o
    prompt para ser respondido direto, numa única geração.
    """
    if not isinstance(result, dict) or not result.get("needs_llm"):
        return None
    for item in reversed(result.get("steps") or []):
        step_result = item.get("result") if isinstance(item, dict) else None
        if isinstance(step_result, dict) and step_result.get("needs_llm") and step_result.get("prompt"):
            return str(step_result["prompt"])
    return None


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


# ------------------------------------------------------------
# Rede de segurança (fallback), NÃO a lógica principal.
# ------------------------------------------------------------
# classify_intent() (abaixo) é quem decide, em condições normais, se um
# pedido precisa de ferramenta. Isso substitui as antigas may_need_tools /
# should_auto_inspect / needs_planning, que eram dezenas de regex e listas
# de palavras que quebravam com qualquer sinônimo fora da lista.
#
# TRIGGERS só entra em ação se a chamada ao modelo falhar (ex.: o processo
# do llama.cpp travou, JSON Schema não suportado nesta build) — nesse caso,
# em vez de travar o roteamento inteiro, caímos num critério simples e
# conservador baseado em conjuntos de palavras.
TRIGGERS: dict[str, set[str]] = {
    "acao": {
        "execute", "executa", "rode", "roda", "use", "usar",
        "pesquise", "pesquisa", "consulte", "acesse", "baixe",
        "salve", "crie", "criar",
    },
    "recurso": {"modulo", "módulo", "module", "agente", "agent"},
    "rede": {"http", "curl", "api", "url", "post", "payload", "json", "endpoint", "site", "pagina", "website"},
    "arquivo": {
        "arquivo", "arquivos", "pasta", "pastas", "workspace",
        "diretorio", "diretório", "projeto", "ls", "dir",
        "liste", "listar", "lista", "leia", "abra",
    },
}


def _fallback_needs_tool(text: str) -> bool:
    words = set(re.findall(r"[a-z0-9_]+", normalize(text)))
    return any(words & group for group in TRIGGERS.values())


INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_tool": {"type": "boolean"},
        "start_with_inspection": {"type": "boolean"},
    },
    "required": ["needs_tool"],
    "additionalProperties": False,
}


def classify_intent(llm, question: str) -> dict[str, Any]:
    """
    Pergunta ao próprio modelo se o pedido precisa de ferramenta, numa
    chamada curta e barata (poucos tokens, JSON Schema), em vez de tentar
    adivinhar via regex/palavras-chave. É exatamente o "LLM Router rápido"
    da análise: o modelo entende sinônimos e contexto; uma lista de
    palavras não entende.

    start_with_inspection permite pedir, na mesma chamada, se vale a pena
    listar o workspace antes de agir — substituindo o antigo
    should_auto_inspect sem precisar de uma segunda bateria de regex.
    """
    prompt = (
        "Classifique o pedido do usuário quanto ao uso de recursos locais "
        "(módulos, agentes, HTTP, arquivos, diretórios).\n\n"
        f"Pedido: {question}\n\n"
        "needs_tool = true somente se responder bem exigir executar algo real "
        "(rodar um módulo/agente, chamar HTTP, ler ou listar arquivos/diretórios). "
        "needs_tool = false para conversa comum, opinião, explicação de conceito "
        "ou qualquer coisa que já pode ser respondida sem consultar nada.\n"
        "start_with_inspection = true apenas se, antes de agir, valer a pena listar "
        "o workspace para entender o contexto local primeiro."
    )

    messages = normalize_chat_messages([
        {
            "role": "system",
            "content": (
                "Você é um classificador rápido de intenção. "
                "Responda apenas com o JSON pedido, sem explicações."
            ),
        },
        {"role": "user", "content": prompt},
    ])

    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=24,
            stream=False,
            response_format={"type": "json_object", "schema": INTENT_SCHEMA},
        )
        data = json.loads(response["choices"][0]["message"]["content"])
        if not isinstance(data, dict):
            raise ValueError("resposta de classificação não é um objeto JSON.")
    except Exception as exc:
        warning(f"Classificador de intenção indisponível; usando fallback conservador: {exc}")
        return {
            "needs_tool": _fallback_needs_tool(question),
            "start_with_inspection": False,
            "_fallback": True,
        }

    return {
        "needs_tool": bool(data.get("needs_tool", False)),
        "start_with_inspection": bool(data.get("start_with_inspection", False)),
    }


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


HTTP_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "method": {
            "type": "string",
            "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
        },
        "json": {"type": "object"},
    },
    "required": ["url"],
    "additionalProperties": False,
}


def extract_http_intent(llm, text: str) -> dict[str, Any] | None:
    """
    Fallback via LLM para quando o regex de explicit_http_route não acha nada,
    mas a frase claramente fala de rede ("faz um get na url google.com com
    json tal"). Regex trabalha com padrões fixos; o modelo entende a frase
    misturada em linguagem natural — é o ponto 4 da análise (NER/extração
    dinâmica em vez de regex cada vez mais complexo).
    """
    if llm is None:
        return None

    messages = normalize_chat_messages([
        {
            "role": "system",
            "content": (
                "Extraia a URL, o método HTTP e o payload JSON (se houver) do "
                "pedido do usuário. Responda apenas o JSON do schema. Se não "
                "houver nenhuma URL real no pedido, responda com url vazio."
            ),
        },
        {"role": "user", "content": text},
    ])

    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=200,
            stream=False,
            response_format={"type": "json_object", "schema": HTTP_EXTRACT_SCHEMA},
        )
        data = json.loads(response["choices"][0]["message"]["content"])
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    url = str(data.get("url", "")).strip()
    if not url:
        return None
    if not url.startswith(("http://", "https://")):
        if "." not in url and url.casefold() != "localhost":
            return None
        url = "https://" + url

    method = str(data.get("method", "GET") or "GET").upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
        method = "GET"

    arguments: dict[str, Any] = {"url": url, "method": method}
    payload = data.get("json")
    if isinstance(payload, dict) and method in {"POST", "PUT", "PATCH"}:
        arguments["json"] = payload

    return {
        "type": "tool",
        "name": "http_request",
        "arguments": arguments,
        "_explicit": True,
        "_llm_extracted": True,
    }


_CURL_METHOD_RE = re.compile(r"\bcurl\s+-X\s*(?P<method>[A-Z]+)\b", re.I)
_URL_OR_DOMAIN_RE = re.compile(
    r"https?://[^\s\"'<>]+"
    r"|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/[^\s\"'<>]*)?"
    r"|\blocalhost(?::\d+)?(?:/[^\s\"'<>]*)?\b",
    re.I,
)


def explicit_http_route(text: str, llm=None) -> dict[str, Any] | None:
    """
    Roteia pedidos de HTTP sem gastar uma chamada de planner, mas só para o
    caso realmente inequívoco: uma URL ou domínio (ex.: "https://...",
    "api.github.com/repos", "g1.com") em algum lugar da frase, com ou sem
    "curl" na frente.

    A versão anterior tentava também casar o VERBO da frase inteira numa
    alternância de regex ("faça um", "curl", "acesse", "consulte", "envie",
    "poste"...) — cada nova forma de pedir a mesma coisa ("dá uma olhada
    em...", "abre o site...") exigia editar essa alternância, e mesmo assim
    sempre faltaria alguma. Como o alvo (a URL/domínio) é o único dado que
    realmente precisa ser extraído com precisão, e tem uma FORMA bem
    definida (não depende de sinônimo nenhum), procurar só por essa forma —
    em qualquer frase, com qualquer verbo — cobre tanto ou mais casos com
    bem menos regex. Frases sem nenhuma URL/domínio literal (ex.: "faça uma
    consulta na API do clima") não têm o que uma regex de forma possa achar
    de qualquer jeito; essas sempre dependeram do extrator via LLM
    (extract_http_intent), que continua sendo o fallback.
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

    curl_method = _CURL_METHOD_RE.search(cleaned)
    if curl_method:
        method = curl_method.group("method").upper()

    payload = extract_json_payload(cleaned)
    json_payload = payload if isinstance(payload, (dict, list)) else None

    target_match = _URL_OR_DOMAIN_RE.search(cleaned)
    if target_match:
        target = target_match.group(0).strip().rstrip(".,;!?")
        if not target.startswith(("http://", "https://")):
            target = "https://" + target

        arguments = {"url": target, "method": method}
        if json_payload is not None and method in {"POST", "PUT", "PATCH"}:
            arguments["json"] = json_payload
        return {
            "type": "tool",
            "name": "http_request",
            "arguments": arguments,
            "_explicit": True,
            "_deterministic": True,
        }

    # Nenhuma URL/domínio literal na frase, mas ela claramente fala de rede
    # ("acesse o site", "faça uma chamada pra API do clima"...) — delega pro
    # extrator via LLM em vez de tentar prever toda frase possível em regex.
    if llm is not None:
        rede_words = set(re.findall(r"[a-z0-9_]+", normalize(cleaned)))
        if rede_words & TRIGGERS["rede"]:
            return extract_http_intent(llm, text)

    return None


def explicit_route(text: str, llm=None) -> dict[str, Any] | None:
    http_action = explicit_http_route(text, llm=llm)
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
    """
    Roteador por sobreposição literal de palavras — a heurística original
    apontada na análise como frágil ("ver meu IP" não bate com "infoself").

    Não é mais a primeira linha de roteamento: isso agora é
    semantic_tool_route(), que entende sinônimos via embeddings. Esta função
    fica só como último fallback determinístico, para o caso raro de o
    embedder estar indisponível (ex.: primeira execução sem internet para
    baixar o modelo de embeddings) — melhor ter uma tentativa fraca do que
    nenhuma.
    """
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
    Normaliza mensagens para templates rígidos como Astra:
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

        # Astra não aceita assistant como primeira mensagem depois do system.
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
        # Limpa completamente a linha antes de Astra começar a falar
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
                    + "Astra > "
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
        print(Fore.MAGENTA + Style.BRIGHT + "Astra > " + Style.RESET_ALL + answer)
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
            + "Astra > "
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
    sys_prompt = react_prompt(base_system, question)
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
        sys_prompt = react_prompt(base_system, question)
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

        # react() só é chamado quando já sabemos que uma ferramenta é
        # relevante (classify_intent disse needs_tool=True, ou uma ação
        # anterior falhou/veio parcial). Então, se a primeira resposta do
        # planejador não foi uma ação de ferramenta, vale tentar as rotas
        # determinística e semântica antes de cobrar do modelo de novo.
        if action.get("type") != "tool" and not trace:
            fallback = (
                explicit_route(question, llm=llm)
                or semantic_tool_route(question)
                or direct_route(question)
            )
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
        sys_prompt = react_prompt(base_system, question)
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
    print(Fore.MAGENTA + Style.BRIGHT + "Astra > " + Style.RESET_ALL + answer)
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

    safe_result = compact_preserving_structure(llm, str(conteudo_real), 3500)

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
    print("  /backend      mostra o backend de IA; /backend groq|google|gguf|auto [MODELO] troca")
    print("  /cache        mostra configuração do prompt cache")
    print("  /memoria      mostra status da memória de longo prazo")
    print("  /memorias     lista memórias episódicas recentes")
    print("  /memoria_editar ID TEXTO  altera uma memória persistida")
    print("  /memoria_apagar ID        remove uma memória persistida")
    print("  /voice on|off liga/desliga a voz inteira: escuta (Whisper) e fala (Piper)")
    print("  /voz          mostra status do modo de voz (wake word, microfone, voz ativa)")
    print("  /vozes        lista as vozes Piper disponíveis para resposta falada")
    print("  /voz N        troca a voz falada (N = número de /vozes, ou o nome dela)")
    print("  /idioma CODIGO  trava o Whisper num idioma (ex.: pt, en) ou 'auto' p/ geral")
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
    print(f"  Backend      : {BACKEND_LABELS.get(BACKEND, BACKEND)}  [modo: {BACKEND_MODE}]")
    print(f"  Modelo       : {ACTIVE_MODEL}")
    print(f"  Contexto     : {N_CTX:,}")
    print(f"  Resposta máx.: {MAX_TOKENS:,}")
    print(f"  Plan máx.    : {TOOL_PLAN_TOKENS:,}")
    if BACKEND == "groq":
        print(f"  Raciocínio   : esforço {GROQ_REASONING_EFFORT}, formato {GROQ_REASONING_FORMAT}")
    elif BACKEND == "google":
        print(f"  Endpoint     : {GOOGLE_BASE_URL}")
    else:
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


# ------------------------------------------------------------
# Entrada por voz
# ------------------------------------------------------------
# Comandos de barra (/sair, /ajuda, ...) não fazem sentido ditados por voz
# ("barra sair"); esses apelidos deixam o modo de voz utilizável sozinho,
# sem que o usuário precise saber a sintaxe de texto.
VOICE_COMMAND_ALIASES: dict[str, str] = {
    "sair": "/sair",
    "encerrar": "/sair",
    "tchau": "/sair",
    "até mais": "/sair",
    "ajuda": "/ajuda",
    "socorro": "/ajuda",
    "limpar": "/limpar",
    "limpar conversa": "/limpar",
    "status": "/status",
    "memória": "/memoria",
    "memorias": "/memorias",
    "agentes": "/agentes",
    "módulos": "/modulos",
    "catálogo": "/catalogo",
    # Desliga a entrada por voz (Whisper) e a fala (Piper) de uma vez.
    "voice off": "/voice off",
    "voice of": "/voice off",
    "voz off": "/voice off",
    "desligar voz": "/voice off",
    "desligar a voz": "/voice off",
    "desativar voz": "/voice off",
    "desativar a voz": "/voice off",
    "modo texto": "/voice off",
    "parar de ouvir": "/voice off",
}


def _voice_key(text: str) -> str:
    """Casefold + sem acentos + sem pontuação: o Whisper devolve 'Sair.' ou
    'Voice, off!', que nunca casariam com uma chave exata do dicionário."""
    key = normalize(text)
    key = re.sub(r"[^\w\s]", " ", key)
    return re.sub(r"\s+", " ", key).strip()


def resolve_voice_command(text: str) -> str:
    """Mapeia uma frase falada para um comando de barra quando aplicável."""
    key = _voice_key(text)
    for alias, command in VOICE_COMMAND_ALIASES.items():
        if _voice_key(alias) == key:
            return command
    return text


_KB_EOF = object()  # sentinela: o teclado fechou (Ctrl-D / Ctrl-C)


class KeyboardChannel:
    """
    Mantém o teclado SEMPRE utilizável, inclusive com o modo de voz ligado.

    Uma thread daemon fica com o prompt aberto e joga cada linha digitada
    numa fila de eventos compartilhada (`events`) — a mesma que a thread de
    escuta de voz usa. Quem espera entrada (main thread) só lê dessa fila,
    então "o que chegar primeiro" ganha: digitou + Enter, ou falou "Astra".
    O prompt só reabre depois que a linha anterior foi consumida (arm()),
    para não ficar empilhando "Você >" durante a geração da resposta.
    """

    def __init__(self, prompt_session) -> None:
        self.prompt_session = prompt_session
        self.events: queue.Queue = queue.Queue()
        self._want = threading.Event()
        self._lock = threading.Lock()
        self._prompting = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="astra-keyboard", daemon=True)
            self._thread.start()

    def arm(self) -> None:
        """Garante que há um prompt aberto esperando o usuário digitar."""
        self.start()
        with self._lock:
            if not self._prompting:
                self._want.set()

    def _run(self) -> None:
        while True:
            self._want.wait()
            with self._lock:
                self._want.clear()
                self._prompting = True
            try:
                line = read_user_input(self.prompt_session)
            except (EOFError, KeyboardInterrupt):
                self.events.put(("eof", _KB_EOF))
                return
            except Exception:
                with self._lock:
                    self._prompting = False
                time.sleep(0.2)
                continue
            with self._lock:
                self._prompting = False
            if line:
                self.events.put(("kb", line))
            else:
                self._want.set()  # linha vazia: reabre o prompt

    def drain(self, kinds: tuple[str, ...] = ("voice",)) -> list[tuple[str, Any]]:
        """Tira da fila eventos velhos dos tipos dados; devolve os demais."""
        keep: list[tuple[str, Any]] = []
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            if ev[0] not in kinds:
                keep.append(ev)
        for ev in keep:
            self.events.put(ev)
        return keep


def _get_event(kb_channel: KeyboardChannel, timeout: float = 0.2):
    """queue.get com timeout curto: mantém Ctrl-C funcionando na main thread."""
    try:
        return kb_channel.events.get(timeout=timeout)
    except queue.Empty:
        return None


def read_user_input_voice(voice_loop, kb_channel: KeyboardChannel) -> tuple[str, bool]:
    """
    Espera o usuário por VOZ ("Astra" + pedido) OU por TECLADO, o que vier
    primeiro. Devolve (texto, veio_por_voz). Digitar + Enter cancela a
    escuta de voz na hora. Falas sem a wake word são ignoradas.
    """
    kb_channel.arm()
    while True:
        cancel = threading.Event()

        def worker() -> None:
            while not cancel.is_set():
                try:
                    command = voice_loop.listen_for_command(cancel)
                except Exception as exc:
                    warning(f"Falha na captura/transcrição de voz: {exc}")
                    time.sleep(0.5)
                    continue
                if command is None:
                    continue  # sem wake word (ou cancelado): volta a escutar
                kb_channel.events.put(("voice", command))
                return

        thread = threading.Thread(target=worker, name="astra-voice", daemon=True)
        thread.start()

        event = None
        try:
            while event is None:
                event = _get_event(kb_channel)
        except BaseException:
            cancel.set()
            raise

        cancel.set()
        thread.join(timeout=8.0)
        kind, payload = event

        if kind == "eof":
            raise EOFError
        if kind == "kb":
            kb_channel.drain(("voice",))
            return payload, False

        # kind == "voice"
        command = (payload or "").strip()
        if not command:
            print(
                Fore.LIGHTBLACK_EX
                + "(ouvi a wake word, mas não entendi o comando — pode repetir)"
                + Style.RESET_ALL
            )
            continue
        print(Fore.CYAN + Style.BRIGHT + "Você (voz) > " + Style.RESET_ALL + command)
        return resolve_voice_command(command), True


def read_user_input_auto(prompt_session, voice_loop, kb_channel: KeyboardChannel) -> tuple[str, bool]:
    """Devolve (texto, veio_por_voz). Texto digitado nunca é 'por voz'."""
    if voice_loop is not None:
        return read_user_input_voice(voice_loop, kb_channel)
    kb_channel.arm()
    while True:
        event = _get_event(kb_channel)
        if event is None:
            continue
        kind, payload = event
        if kind == "eof":
            raise EOFError
        if kind == "kb":
            return payload, False


BARGE_IN_VOICE = os.getenv("ASTRA_BARGE_IN_VOICE", "1") == "1"


def speak_with_interrupt(tts, voice_loop, kb_channel: KeyboardChannel, text: str) -> tuple[str, str | None]:
    """
    Fala `text` e se deixa interromper. Devolve (desfecho, texto):
      ("done", None)      terminou de falar
      ("keyboard", texto) o usuário digitou algo + Enter (áudio cortado)
      ("voice", None)     o usuário disse a wake word (áudio cortado)
      ("eof", None)       teclado fechado
    Teclado tem prioridade sobre voz se ambos ocorrerem juntos.
    """
    cancel = threading.Event()
    done = threading.Event()
    voice_seen = threading.Event()

    def speaker() -> None:
        try:
            tts.speak(text, cancel_event=cancel)
        finally:
            done.set()

    def listener() -> None:
        while not cancel.is_set() and not done.is_set():
            try:
                hit = voice_loop.detect_wake_word_once(cancel)
            except Exception:
                time.sleep(0.5)
                continue
            if hit:
                voice_seen.set()
                cancel.set()
                return

    kb_channel.arm()
    t_speak = threading.Thread(target=speaker, name="astra-tts", daemon=True)
    t_listen = None
    if voice_loop is not None and BARGE_IN_VOICE:
        t_listen = threading.Thread(target=listener, name="astra-barge-in", daemon=True)
    t_speak.start()
    if t_listen is not None:
        t_listen.start()

    kb_text: str | None = None
    kb_eof = False
    try:
        while not done.is_set() and not cancel.is_set():
            event = _get_event(kb_channel, timeout=0.05)
            if event is None:
                continue
            kind, payload = event
            if kind == "kb":
                kb_text = payload
                cancel.set()
            elif kind == "eof":
                kb_eof = True
                cancel.set()
    except BaseException:
        cancel.set()
        raise
    finally:
        cancel.set()  # encerra o que ainda estiver rodando
        t_speak.join(timeout=3.0)
        if t_listen is not None:
            t_listen.join(timeout=8.0)

    if kb_eof:
        return "eof", None
    if kb_text is None:
        # o teclado pode ter chegado junto com o fim da fala
        for kind, payload in kb_channel.drain(("kb", "eof")):
            pass
    if kb_text is not None:
        return "keyboard", kb_text
    if voice_seen.is_set():
        return "voice", None
    return "done", None


def setup_voice_loop(force: bool = False):
    """
    Prepara o modo de voz: verifica microfone, calibra o VAD e (na primeira
    vez) baixa/carrega o Whisper local. Se qualquer verificação crítica
    falhar, cai para digitação em vez de travar o programa — "sempre
    escutando" só faz sentido se o microfone realmente está disponível.
    """
    # force=True (usado por "/voice on") ignora ASTRA_VOICE_MODE=0 — quem liga
    # à mão em runtime quer ligar, independente do padrão da inicialização.
    if not (HAS_VOICE_IO and (force or voice_io.VOICE_MODE)):
        return None

    section("🎙", "Modo de voz")

    if not voice_io.HAS_SOUNDDEVICE:
        warning("sounddevice não instalado — caindo para digitação. `pip install sounddevice`.")
        return None
    if not voice_io.HAS_FASTER_WHISPER:
        warning("faster-whisper não instalado — caindo para digitação. `pip install faster-whisper`.")
        return None

    mic = check_microphone_safe()
    if mic is None or mic.hard_fail:
        warning(
            (mic.error if mic else "Não foi possível acessar o microfone.")
            + " Caindo para digitação. Rode `python voice_io.py --selftest` para diagnosticar."
        )
        return None

    if mic.quiet:
        warning(
            f"{mic.error} (nível medido: {mic.dbfs:.0f} dBFS). Continuando em modo de "
            "voz mesmo assim — se a Astra não reagir quando você falar, isso é o "
            "primeiro lugar pra olhar."
        )
    else:
        print(
            Fore.LIGHTBLACK_EX
            + f"Microfone OK ({mic.device_name or 'dispositivo padrão'}, {mic.dbfs:.0f} dBFS de ruído ambiente)."
            + Style.RESET_ALL
        )

    voice_loop = voice_io.VoiceLoop()
    voice_loop.vad.calibrate(mic.rms)

    # "Pode transcrever tudo, numa cor mais tranquila": mostra toda fala
    # ouvida (ligado por padrão — não é só um modo de debug), numa cor
    # discreta, para o usuário ver que o microfone está de fato captando.
    # Só o texto pós-wake-word (is_command=True) é o que realmente vira
    # pergunta para o modelo; o resto é só transparência do que foi ouvido.
    if os.getenv("ASTRA_VOICE_SHOW_TRANSCRIPT", "1") == "1":
        def _show_transcript(text: str, is_command: bool) -> None:
            label = "comando" if is_command else "ouvido"
            print(Fore.LIGHTBLACK_EX + f"({label}: {text!r})" + Style.RESET_ALL)

        voice_loop.on_transcript = _show_transcript

    # Confirmação visível assim que a wake word é reconhecida — não depende
    # do dispositivo de áudio de saída (o plim pode falhar silenciosamente
    # em alguns setups do Windows); isto aqui garante que o usuário sempre
    # vê que foi ouvido, antes mesmo do texto do comando aparecer.
    voice_loop.on_wake_detected = lambda: print(
        Fore.GREEN + Style.BRIGHT + f"🔔 \"{voice_io.WAKE_WORD}\" reconhecida — pode falar." + Style.RESET_ALL
    )

    try:
        voice_loop.transcriber.ensure_loaded()
        voice_loop.wake_transcriber.ensure_loaded()
    except Exception as exc:
        warning(f"{exc}\nCaindo para digitação.")
        return None

    print(
        Fore.LIGHTBLACK_EX
        + f"Wake word: \"{voice_io.WAKE_WORD}\" · detecção: modelo "
          f"{voice_io.WHISPER_WAKE_MODEL_SIZE} (rápido) · comando: modelo "
          f"{voice_io.WHISPER_MODEL_SIZE} (preciso) · pasta: {voice_io.MODELS_DIR}"
        + Style.RESET_ALL
    )
    print(
        Fore.LIGHTBLACK_EX
        + f"VAD: {type(voice_loop.vad).__name__}"
        + Style.RESET_ALL
    )
    print(
        Fore.LIGHTBLACK_EX
        + f"Diga \"{voice_io.WAKE_WORD}\" seguido do seu pedido a qualquer momento. "
          "(defina ASTRA_VOICE_MODE=0 para voltar a digitar)"
        + Style.RESET_ALL
    )
    return voice_loop


def check_microphone_safe():
    try:
        return voice_io.check_microphone()
    except Exception as exc:
        warning(f"Verificação de microfone falhou: {exc}")
        return None


def setup_tts(force: bool = False):
    """
    Prepara a saída de voz (Piper): baixa a voz padrão na primeira vez e
    deixa pronta para falar. Independente do modo de entrada — dá pra digitar
    e ainda assim ouvir a resposta, ou escutar sem falar de volta
    (ASTRA_TTS_MODE=0).
    """
    if not (HAS_VOICE_IO and (force or voice_io.TTS_MODE)):
        return None
    if not voice_io.HAS_PIPER:
        warning("piper-tts não instalado — respostas ficam só em texto. `pip install piper-tts`.")
        return None

    tts = voice_io.PiperTTS()
    try:
        tts.ensure_voice_downloaded(tts.voice_name)
    except Exception as exc:
        warning(f"{exc}\nRespostas ficam só em texto por enquanto.")
        return None

    label = voice_io.PIPER_VOICES[tts.voice_name]["label"]
    print(
        Fore.LIGHTBLACK_EX
        + f"Voz ativa: {label} · troque com /voz (veja /vozes)"
        + Style.RESET_ALL
    )
    return tts


# ============================================================
# MAIN
# ============================================================

def _main() -> None:
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
    voice_loop = setup_voice_loop()
    tts = setup_tts()
    # "/voice off" solta as referências ativas mas guarda os objetos aqui,
    # então "/voice on" religa na hora, sem recarregar Whisper/Piper.
    voice_cache: dict[str, Any] = {"loop": voice_loop, "tts": tts}

    print()
    print(
        Fore.WHITE
        + "Astra está pronta. Converse normalmente ou peça para executar módulos e agentes."
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

    show_banner_every_turn = os.getenv("ASTRA_BANNER_EVERY_TURN", "1") == "1"

    kb_channel = KeyboardChannel(prompt_session)
    # Entrada já obtida durante a fala anterior (digitada) e/ou pedido de
    # tratar uma interrupção por voz ("Astra" dito enquanto ela falava).
    pending_input: tuple[str, bool] | None = None
    pending_voice_followup = False
    pending_eof = False

    while True:
        if pending_eof:
            print("\nAté mais.")
            break

        if show_banner_every_turn:
            print_astra_banner()

        try:
            if pending_input is not None:
                q, from_voice = pending_input
                pending_input = None
                if from_voice:
                    print(Fore.CYAN + Style.BRIGHT + "Você (voz) > " + Style.RESET_ALL + q)
            elif pending_voice_followup and voice_loop is not None:
                pending_voice_followup = False
                if voice_loop.on_wake_detected:
                    voice_loop.on_wake_detected()
                voice_io.play_chime()
                spoken = voice_loop.record_command().strip()
                if not spoken:
                    print(
                        Fore.LIGHTBLACK_EX
                        + "(ouvi a wake word, mas não entendi o comando — pode repetir)"
                        + Style.RESET_ALL
                    )
                    continue
                print(Fore.CYAN + Style.BRIGHT + "Você (voz) > " + Style.RESET_ALL + spoken)
                q, from_voice = resolve_voice_command(spoken), True
            else:
                pending_voice_followup = False
                q, from_voice = read_user_input_auto(prompt_session, voice_loop, kb_channel)
        except (EOFError, KeyboardInterrupt):
            print("\nAté mais.")
            break

        if not q:
            continue

        command = q.casefold()

        if command in {"/sair", "/exit", "/quit"}:
            print(
                Fore.MAGENTA + Style.BRIGHT + "Astra > " + Style.RESET_ALL + "Até mais."
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

        if command == "/backend" or command.startswith("/backend "):
            parts = q.split(None, 2)
            if len(parts) == 1:
                section("◆", "Backend de IA")
                print(f"  Em uso   : {BACKEND_LABELS.get(BACKEND, BACKEND)} · {ACTIVE_MODEL}")
                print(f"  Modo     : {BACKEND_MODE}" + (" (troca automática se cair)" if isinstance(llm, FailoverLLM) else ""))
                print(f"  Contexto : {N_CTX:,} tokens")
                for b in ("groq", "google", "local"):
                    why = _unavailable_reason(b)
                    mark = "indisponível: " + why if why else "pronto"
                    print(f"  {BACKEND_LABELS[b]:<24}: {mark}")
                print("  Trocar   : /backend groq|google|gguf|auto [MODELO]")
                continue
            target = _BACKEND_ALIASES.get(parts[1].strip().casefold())
            if target is None:
                error("Backend inválido. Use: auto, groq, google ou gguf.")
                continue
            if len(parts) == 3 and target != "auto":
                set_backend_model({"gguf": "local"}.get(target, target), parts[2].strip())
            switch_to = {"gguf": "local"}.get(target, target)
            old_key = {"groq": GROQ_API_KEY, "google": GOOGLE_API_KEY}.get(switch_to)
            if switch_to in _KEY_HELP and not ensure_key(switch_to):
                error(f"Troca cancelada: falta a chave da {_KEY_HELP[switch_to][0]}. "
                      f"Tente de novo ou crie uma em {_KEY_HELP[switch_to][1]}")
                continue
            try:
                new_llm, _ = load_model(mode=target, announce=False)
            except SystemExit as exc:
                text = _exit_text(exc)
                error(f"Não consegui trocar de backend: {text}")
                if switch_to in _KEY_HELP and re.search(r"\b(401|403)\b|api key|invalid", text, re.I):
                    _set_key(switch_to, old_key or "")  # chave recusada: esquece, pergunta de novo na próxima
                    warning("A chave parece inválida; digite /backend " + target + " para informar outra.")
                continue
            llm = new_llm
            success(f"Agora usando {BACKEND_LABELS.get(BACKEND, BACKEND)} · {ACTIVE_MODEL}. A conversa continua.")
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

        if command == "/voice" or command.startswith("/voice "):
            arg = command.split(maxsplit=1)[1].strip() if " " in command else ""

            if arg in {"off", "desligar", "desativar", "0"}:
                if voice_loop is None and tts is None:
                    warning("A voz já está desligada. Use /voice on para ligar.")
                else:
                    voice_cache["loop"] = voice_loop or voice_cache.get("loop")
                    voice_cache["tts"] = tts or voice_cache.get("tts")
                    voice_loop, tts = None, None
                    success("Voz desligada: sem escuta (Whisper) e sem fala (Piper). Volto a ler o teclado.")

            elif arg in {"on", "ligar", "ativar", "1"}:
                if voice_loop is not None and tts is not None:
                    warning("A voz já está ligada. Use /voice off para desligar.")
                else:
                    # Religa só o que estiver desligado: primeiro reaproveita os
                    # objetos guardados (instantâneo); se não houver, inicializa
                    # de novo (ex.: o microfone falhou na abertura do programa).
                    if voice_loop is None:
                        voice_loop = voice_cache.get("loop") or setup_voice_loop(force=True)
                    if tts is None:
                        tts = voice_cache.get("tts") or setup_tts(force=True)
                    voice_cache["loop"], voice_cache["tts"] = voice_loop, tts

                    if voice_loop is None and tts is None:
                        error("Não consegui ligar a voz (veja os avisos acima). Rode `python voice_io.py --selftest`.")
                    else:
                        parts = []
                        if voice_loop is not None:
                            parts.append(f"escuta por \"{voice_io.WAKE_WORD}\"")
                        if tts is not None:
                            parts.append("fala em voz alta")
                        success("Voz ligada: " + " e ".join(parts) + ".")

            elif arg in {"", "status"}:
                listening = "ligada" if voice_loop is not None else "desligada"
                speaking = "ligada" if tts is not None else "desligada"
                print(f"  Escuta (Whisper): {listening} · Fala (Piper): {speaking}")
                print("  Use: /voice on  ·  /voice off")

            else:
                error("Uso: /voice on  ou  /voice off")
            continue

        if command == "/voz":
            section("🎙", "Status do modo de voz")
            if voice_loop is None:
                if voice_cache.get("loop") is not None:
                    reason = "desligada com /voice off (use /voice on)"
                elif HAS_VOICE_IO and not voice_io.VOICE_MODE:
                    reason = "desativada (ASTRA_VOICE_MODE=0; /voice on liga)"
                else:
                    reason = "indisponível (veja avisos na inicialização)"
                print(f"  Entrada por voz : {reason}")
                print("  Diagnóstico     : rode `python voice_io.py --selftest`")
            else:
                mic = check_microphone_safe()
                print(f"  Entrada por voz : ON")
                print(f"  Wake word       : \"{voice_io.WAKE_WORD}\"")
                lang_desc = "geral (auto-detecção)" if voice_loop.transcriber.language in ("", "auto") else voice_loop.transcriber.language
                print(f"  Idioma Whisper  : {lang_desc} · comando: {voice_io.WHISPER_MODEL_SIZE} · detecção: {voice_io.WHISPER_WAKE_MODEL_SIZE} ({voice_io.MODELS_DIR})")
                if mic is not None:
                    mic_label = "PROBLEMA" if mic.hard_fail else ("baixo, mas OK" if mic.quiet else "OK")
                    print(f"  Microfone       : {mic_label} · {mic.dbfs:.0f} dBFS · {mic.device_name or 'padrão'}")
            if tts is None:
                if voice_cache.get("tts") is not None:
                    reason = "desligada com /voice off (use /voice on)"
                elif HAS_VOICE_IO and not voice_io.TTS_MODE:
                    reason = "desativada (ASTRA_TTS_MODE=0; /voice on liga)"
                else:
                    reason = "indisponível (veja avisos na inicialização)"
                print(f"  Saída falada    : {reason}")
            else:
                print(f"  Saída falada    : ON · {voice_io.PIPER_VOICES[tts.voice_name]['label']}")
                print("  Trocar voz      : /voz <número ou nome> (veja /vozes)")
            continue

        if command == "/vozes":
            section("🗣", "Vozes Piper disponíveis")
            if not HAS_VOICE_IO:
                print("  voice_io indisponível.")
            else:
                for i, (name, info) in enumerate(voice_io.PIPER_VOICES.items(), start=1):
                    active = tts is not None and tts.voice_name == name
                    marker = "→" if active else " "
                    print(f"  {marker} {i}. {info['label']}  [{name}]")
                if tts is None:
                    print("  (saída falada está desativada — veja /voz)")
                else:
                    print("  Troque com: /voz <número ou nome>")
            continue

        if command.startswith("/voz "):
            arg = q.split(maxsplit=1)[1].strip()
            if not HAS_VOICE_IO:
                error("voice_io indisponível.")
            elif tts is None:
                error("Saída falada está desativada (ASTRA_TTS_MODE=0 ou piper-tts não instalado).")
            else:
                names = list(voice_io.PIPER_VOICES)
                target = None
                if arg.isdigit() and 1 <= int(arg) <= len(names):
                    target = names[int(arg) - 1]
                elif arg in voice_io.PIPER_VOICES:
                    target = arg
                else:
                    # aceita nome parcial/case-insensitive, ex.: "amy", "faber"
                    matches = [n for n in names if arg.casefold() in n.casefold()]
                    if len(matches) == 1:
                        target = matches[0]

                if target is None:
                    error(f"Voz '{arg}' não reconhecida. Use /vozes para ver as opções.")
                else:
                    try:
                        tts.set_voice(target)
                        success(f"Voz trocada para {voice_io.PIPER_VOICES[target]['label']}.")
                    except Exception as exc:
                        error(f"Não consegui trocar de voz: {exc}")
            continue

        if command.startswith("/idioma"):
            arg = q.split(maxsplit=1)[1].strip() if len(q.split(maxsplit=1)) > 1 else ""
            if voice_loop is None:
                error("Entrada por voz está desativada — não há Whisper para configurar.")
            elif not arg:
                error("Uso: /idioma CODIGO (ex.: pt, en) ou /idioma auto para geral/multilíngue.")
            else:
                voice_loop.transcriber.language = arg.casefold()
                voice_loop.wake_transcriber.language = arg.casefold()
                success(f"Idioma do Whisper definido para: {voice_loop.transcriber.language}")
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
            print(Fore.MAGENTA + Style.BRIGHT + "Astra > " + Style.RESET_ALL + answer)
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
        # 1) rotas deterministicas: só para o que é inequívoco por construção
        #    (URL explícita, "módulo/agente NOME", listar/ler arquivo);
        # 2) roteamento semântico: embeddings contra nome+descrição de cada
        #    módulo/agente — entende sinônimos que a rota determinística não
        #    tem como prever ("ver meu IP" → módulo infoself);
        # 3) se nada bateu com confiança, o próprio modelo classifica se a
        #    tarefa precisa de ferramenta (classify_intent), em vez de uma
        #    bateria de regex tentando adivinhar.
        action = explicit_route(q, llm=llm) or semantic_tool_route(q) or direct_route(q)

        intent: dict[str, Any] | None = None
        if not action:
            intent = classify_intent(llm, q)
            if intent.get("start_with_inspection"):
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

        # Rota explícita ou semântica tem prioridade máxima
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
                    Fore.MAGENTA + Style.BRIGHT + "Astra > " + Style.RESET_ALL + answer
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

                llm_step_prompt = (
                    extract_llm_step_prompt(result) if outcome != "success" else None
                )

                if llm_step_prompt:
                    success(f"{target}: dados coletados, resumindo")
                    answer, stats = stream_final(
                        llm,
                        build_chat_messages(
                            llm,
                            history,
                            llm_step_prompt,
                            base_system,
                        ),
                    )
                elif outcome == "success":
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
                            f"{target} retornou algo útil, mas incompleto. A Astra vai tentar completar sem perder o resultado."
                        )
                    else:
                        error(
                            f"{target} falhou. A Astra vai procurar uma estratégia diferente antes de desistir."
                        )

                    answer, stats = react(
                        llm,
                        q,
                        history,
                        base_system,
                        seed_action=action,
                        seed_result=result,
                    )

        elif intent and intent.get("needs_tool"):
            # react() decide sozinho, via seu próprio prompt, entre ação
            # direta, inspecionar antes ou planejar em múltiplos passos —
            # não precisamos de uma segunda camada de heurística aqui.
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

        # ----------------------------------------------------
        # RESPOSTA FALADA — só quando a conversa começou por VOZ.
        # Mensagem digitada recebe só texto. Enquanto ela fala, digitar
        # algo + Enter ou dizer "Astra" corta o áudio na hora.
        # ----------------------------------------------------
        if from_voice and tts is not None and answer.strip():
            outcome, typed = speak_with_interrupt(tts, voice_loop, kb_channel, answer)
            if outcome == "keyboard" and typed:
                pending_input = (typed, False)
            elif outcome == "voice" and voice_loop is not None:
                pending_voice_followup = True
            elif outcome == "eof":
                pending_eof = True


def main() -> None:
    # Com o prompt vivo numa thread de fundo, qualquer print() normal
    # bagunçaria a linha de digitação; patch_stdout imprime ACIMA dela.
    use_patch = HAS_PROMPT_TOOLKIT and os.getenv("ASTRA_PATCH_STDOUT", "1") == "1"
    ctx = patch_stdout(raw=True) if use_patch else contextlib.nullcontext()
    with ctx:
        _main()


if __name__ == "__main__":
    main()
