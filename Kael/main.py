"""
LUNA AI CODER v10 — State of the Art 2026/2
============================================

Arquitetura multi-agente com verificação semi-formal:

  USER
    │
    ├── IntentClassifier (LLM)
    ├── ApiDiscoveryAgent (RAG + OpenAPI parsing + robust HTTP)
    ├── DependencyManager (auto-install uv/pip)
    ├── ContextManager (PACE-inspired)
    │
    └── DELIVERY PIPELINE
          ├── Architect      (plano estruturado)
          ├── Coder          (geração streaming)
          ├── Reviewer       (semi-formal reasoning certificate)
          ├── Tester         (execução + validação)
          ├── Debugger       (TrajAudit-style diagnosis + repair)
          └── Verifier       (evidência externa estruturada)
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from colorama import Fore, Style, init
from llama_cpp import Llama


init(autoreset=True)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.gguf"
WORKSPACE = BASE_DIR / "workspace"
LOGS_DIR = BASE_DIR / "logs"
MOLS_DIR = BASE_DIR / "mols"

for d in (WORKSPACE, LOGS_DIR, MOLS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# Modelo
N_CTX_BASE = 8192
N_CTX_MAX = 32768
N_THREADS = max(1, (os.cpu_count() or 4) - 1)
N_BATCH = 512
N_GPU_LAYERS = 0
USE_FLASH_ATTN = False


# Orçamentos dinâmicos de tokens
TOKEN_BUDGETS = {
    "router": (180, 400),
    "architect": (600, 1200),
    "coder": (1200, 3500),
    "quick": (150, 400),
    "reviewer": (400, 900),
    "debugger": (350, 800),
    "repair_code": (1200, 3000),
    "repair_surgical": (800, 1800),
    "verifier": (300, 600),
    "interpreter": (600, 1800),
    "dependency_resolver": (150, 400),
    "api_discovery": (500, 1000),
    "doc_reader": (400, 1000),
}


def token_budget_for(stage: str, complexity: int = 3) -> int:
    lo, hi = TOKEN_BUDGETS.get(stage, (300, 1000))
    if complexity <= 2:
        return lo
    if complexity >= 5:
        return hi
    return lo + (hi - lo) * (complexity - 2) // 3


# Amostragem
TEMPERATURE_ROUTER = 0.03
TEMPERATURE_ARCHITECT = 0.05
TEMPERATURE_CODER = 0.10
TEMPERATURE_QUICK = 0.04
TEMPERATURE_REVIEWER = 0.02
TEMPERATURE_DEBUGGER = 0.04
TEMPERATURE_REPAIR = 0.08
TEMPERATURE_SURGICAL = 0.04
TEMPERATURE_VERIFIER = 0.02
TEMPERATURE_INTERPRETER = 0.18
TEMPERATURE_DEP_RESOLVER = 0.02
TEMPERATURE_API_DISCOVERY = 0.05
TEMPERATURE_DOC_READER = 0.05

TOP_P = 0.92
TOP_K = 40
MIN_P = 0.05
REPEAT_PENALTY = 1.06
SEED = 42

STREAM = True
SHOW_MODEL_STREAM = True


# Limites operacionais
MAX_ATTEMPTS = 15
MAX_HYPOTHESIS_REPEATS = 2
MAX_LOOP_SIMILARITY = 0.92
LOOP_WINDOW = 5
EXECUTION_TIMEOUT = 120
INTERACTIVE_TIMEOUT = 3600
PROBE_TIMEOUT = 25
SHELL_TIMEOUT = 30
PIP_INSTALL_TIMEOUT = 180
API_TEST_TIMEOUT = 15

MAX_CODE_CHARS = 120_000
MAX_TRACEBACK_CHARS = 24_000
MAX_PREVIOUS_CODE_CHARS = 80_000
MAX_MODULE_BYTES = 600_000
MAX_CONTEXT_TOKENS = N_CTX_MAX - 2048

# HTTP robustness
HTTP_MAX_RETRIES = 3
HTTP_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8",
}


# Comportamento
ENABLE_PLANNER = True
ENABLE_REVIEWER = True
ENABLE_VERIFIER = True
ENABLE_DEBUGGER = True
ENABLE_LOOP_DETECTOR = True
ENABLE_CONTEXT_MANAGER = True
ENABLE_DEPENDENCY_MANAGER = True
ENABLE_API_DISCOVERY = True
PLANNER_COMPLEXITY_THRESHOLD = 3
REVIEWER_COMPLEXITY_THRESHOLD = 4

AUTO_FILL_INPUT = True
OFFER_INTERACTIVE = True
AUTO_INSTALL_DEPS = True
PREFER_UV_OVER_PIP = True


# ============================================================
# TIPOS
# ============================================================

@dataclass
class GenerationResult:
    text: str
    elapsed: float
    completion_tokens: int
    prompt_tokens: int
    tokens_per_second: float
    truncated: bool = False


@dataclass
class ExecutionResult:
    success: bool
    returncode: int
    stdout: str
    stderr: str
    elapsed: float
    timed_out: bool = False
    stdin_used: str = ""
    mode: str = "headless"

    @property
    def combined_output(self) -> str:
        parts = []
        if self.stdout:
            parts.append("STDOUT:\n" + self.stdout)
        if self.stderr:
            parts.append("STDERR:\n" + self.stderr)
        return "\n\n".join(parts).strip()


@dataclass
class AnalysisResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    uses_input: bool = False
    uses_network: bool = False
    uses_subprocess: bool = False
    uses_matplotlib_show: bool = False
    uses_savefig: bool = False
    saves_figures: list[str] = field(default_factory=list)
    opens_gui: bool = False
    antipatterns: list[tuple[str, str, str]] = field(default_factory=list)
    missing_packages: list[str] = field(default_factory=list)
    api_endpoints: list[str] = field(default_factory=list)


@dataclass
class ReasoningCertificate:
    premises: list[str]
    execution_path: list[str]
    conclusion: str
    confidence: str
    evidence: list[str] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)


@dataclass
class TaskPlan:
    goal: str
    requirements: list[str]
    checks: list[str]
    test_inputs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    interactive: bool = False
    probe_strategy: str = ""


@dataclass
class StackFrame:
    file: str
    line: int
    func: str
    code_text: str = ""
    is_user: bool = False


@dataclass
class TracebackInfo:
    exception_type: str = ""
    exception_message: str = ""
    user_frame: Optional[StackFrame] = None
    library_frame: Optional[StackFrame] = None
    all_frames: list[StackFrame] = field(default_factory=list)
    user_context: list[tuple[int, str]] = field(default_factory=list)
    raw: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.exception_type or self.user_frame or self.library_frame)


@dataclass
class AttemptResult:
    code: str
    execution: ExecutionResult
    analysis: AnalysisResult
    traceback_info: Optional[TracebackInfo] = None
    fingerprint: str = ""
    certificate: Optional[ReasoningCertificate] = None


@dataclass
class Intent:
    wants_delivery: bool
    wants_interpretation: bool
    read_requested: bool
    module_names: list[str]
    forbids_files: bool = False
    wants_visual: bool = False
    module_contents: list[tuple[str, str]] = field(default_factory=list)
    involves_api: bool = False
    api_urls: list[str] = field(default_factory=list)


@dataclass
class QuickCommand:
    kind: str
    command: str
    explanation: str = ""


@dataclass
class LoopSignal:
    kind: str
    similarity: float
    evidence: str
    severity: str


@dataclass
class FailureDiagnosis:
    root_cause_type: str
    root_cause: str
    location: str
    observed_value: str = ""
    admissible_alternatives: list[str] = field(default_factory=list)
    fix_strategy: str = ""
    must_avoid: list[str] = field(default_factory=list)
    hypothesis_key: str = ""
    suggested_dependencies: list[str] = field(default_factory=list)
    reasoning_certificate: Optional[ReasoningCertificate] = None


@dataclass
class DependencyInfo:
    name: str
    import_name: str
    is_installed: bool
    version: Optional[str] = None
    install_spec: str = ""


@dataclass
class ApiEndpoint:
    url: str
    method: str = "GET"
    parameters: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    sample_response: str = ""
    discovered_via: str = ""


@dataclass
class ApiTestResult:
    url: str
    method: str
    success: bool
    status_code: Optional[int] = None
    response_body: str = ""
    error: str = ""
    elapsed: float = 0.0
    retries: int = 0


@dataclass
class OpenApiSpec:
    url: str
    version: str
    base_url: str
    endpoints: list[ApiEndpoint] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    attempt_count: int = 0
    hypothesis_counts: dict[str, int] = field(default_factory=dict)
    fingerprints: list[str] = field(default_factory=list)
    recent_signatures: deque = field(default_factory=lambda: deque(maxlen=LOOP_WINDOW))
    context_used_tokens: int = 0
    context_summaries: list[str] = field(default_factory=list)
    total_tokens_generated: int = 0
    installed_packages: list[str] = field(default_factory=list)
    api_cache: dict[str, ApiTestResult] = field(default_factory=dict)
    spec_cache: dict[str, OpenApiSpec] = field(default_factory=dict)
    review_history: list[ReasoningCertificate] = field(default_factory=list)


# ============================================================
# LOGGER
# ============================================================

class LunaLogger:

    def __init__(self, base_dir: Path) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = base_dir / stamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.session_dir / "events.jsonl"

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize(v) for v in value]
        if hasattr(value, "__dict__"):
            return self._serialize(vars(value))
        return repr(value)

    def log(self, kind: str, data: Any = None) -> None:
        entry = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "kind": kind,
            "data": self._serialize(data),
        }
        try:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except (OSError, KeyboardInterrupt):
            pass

    def write_text(self, name: str, content: str) -> Path:
        path = self.session_dir / name
        try:
            path.write_text(content, encoding="utf-8")
        except OSError:
            pass
        return path


# ============================================================
# TERMINAL
# ============================================================

def log(m: str) -> None:
    print(f"{Fore.CYAN}[LUNA]{Style.RESET_ALL} {m}")

def success(m: str) -> None:
    print(f"{Fore.GREEN}[LUNA] ✓ {m}{Style.RESET_ALL}")

def warning(m: str) -> None:
    print(f"{Fore.YELLOW}[LUNA] ! {m}{Style.RESET_ALL}")

def error(m: str) -> None:
    print(f"{Fore.RED}[LUNA] ✗ {m}{Style.RESET_ALL}")

def stage(m: str) -> None:
    print(f"{Fore.MAGENTA}[LUNA] ◆ {m}{Style.RESET_ALL}")

def dim(m: str) -> None:
    print(f"{Style.DIM}{m}{Style.RESET_ALL}")

def stream_open(label: str) -> None:
    pad = max(0, 62 - len(label))
    print(f"{Fore.MAGENTA}┌── {label} " + "─" * pad + f"{Style.RESET_ALL}")

def stream_close(label: str) -> None:
    pad = max(0, 58 - len(label))
    print(f"{Fore.MAGENTA}└── fim {label} " + "─" * pad + f"{Style.RESET_ALL}")


# ============================================================
# UTILITÁRIOS
# ============================================================

def clean_text(text: str) -> str:
    return text.replace("\x00", "").strip()


def truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[... truncado pela Luna ...]"


def truncate_smart(text: str, limit: int, head_ratio: float = 0.6) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head = int(limit * head_ratio)
    tail = max(0, limit - head - 40)
    return text[:head] + "\n\n[... ...]\n\n" + text[-tail:]


def strip_code_fences(text: str) -> str:
    text = text.strip()
    fence = re.search(
        r"```(?:python|py|bash|sh|json|yaml|toml|text|)\s*\n(.*?)```",
        text, re.DOTALL | re.IGNORECASE,
    )
    if fence:
        return fence.group(1).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def normalize_code(code: str) -> str:
    code = strip_code_fences(code)
    lines = code.splitlines()
    while lines and re.match(
        r"^\s*(aqui|here|segue|below|abaixo)\b.*:?\s*$",
        lines[0], re.IGNORECASE,
    ):
        lines = lines[1:]
    code = "\n".join(lines)
    if code.lower().startswith("python\n"):
        code = code[7:]
    return code.strip()


def code_fingerprint(code: str) -> str:
    normalized = re.sub(r"\s+", "", code)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def make_diff(old: str, new: str) -> str:
    return "\n".join(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile="anterior.py", tofile="atual.py", lineterm="", n=2,
    ))


def safe_json(text: str) -> Optional[dict[str, Any]]:
    text = clean_text(text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start:end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None


def read_text_safely(path: Path, limit: int = MAX_MODULE_BYTES) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return f"[LUNA] Falha ao ler {path}: {exc}"
    if len(raw) > limit:
        raw = raw[:limit]
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ============================================================
# DEPENDENCY MANAGER
# ============================================================

class DependencyManager:

    KNOWN_MAPPINGS: dict[str, Optional[str]] = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "skimage": "scikit-image",
        "yaml": "PyYAML",
        "bs4": "beautifulsoup4",
        "dateutil": "python-dateutil",
        "dotenv": "python-dotenv",
        "OpenSSL": "pyOpenSSL",
        "Crypto": "pycryptodome",
        "serial": "pyserial",
        "usb": "pyusb",
        "git": "GitPython",
        "jwt": "PyJWT",
        "multipart": "python-multipart",
        "mplfinance": "mplfinance",
        "plotly": "plotly",
        "seaborn": "seaborn",
        "pandas": "pandas",
        "numpy": "numpy",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "requests": "requests",
        "aiohttp": "aiohttp",
        "httpx": "httpx",
        "flask": "Flask",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "sqlalchemy": "SQLAlchemy",
        "pydantic": "pydantic",
        "tqdm": "tqdm",
        "rich": "rich",
        "click": "click",
        "typer": "typer",
        "loguru": "loguru",
        "toml": "toml",
        "orjson": "orjson",
        "msgpack": "msgpack",
        "zmq": "pyzmq",
        "redis": "redis",
        "pymongo": "pymongo",
        "psycopg2": "psycopg2-binary",
        "MySQLdb": "mysqlclient",
        # stdlib (None = não instalar)
        "sqlite3": None, "tkinter": None, "collections": None,
        "itertools": None, "functools": None, "pathlib": None,
        "typing": None, "dataclasses": None, "enum": None,
        "abc": None, "io": None, "os": None, "sys": None,
        "re": None, "json": None, "math": None, "time": None,
        "datetime": None, "random": None, "hashlib": None,
        "base64": None, "urllib": None, "http": None,
        "socket": None, "threading": None, "multiprocessing": None,
        "asyncio": None, "subprocess": None, "shutil": None,
        "tempfile": None, "traceback": None, "logging": None,
        "warnings": None, "contextlib": None, "copy": None,
        "pprint": None, "textwrap": None, "string": None,
        "csv": None, "configparser": None, "argparse": None,
        "platform": None, "stat": None, "glob": None,
        "fnmatch": None, "struct": None, "array": None,
        "queue": None, "heapq": None, "bisect": None,
        "operator": None, "weakref": None, "gc": None,
        "inspect": None, "importlib": None, "pkgutil": None,
        "ast": None, "dis": None, "keyword": None,
        "unittest": None, "email": None, "smtplib": None,
        "html": None, "xml": None, "webbrowser": None,
        "ssl": None, "select": None, "signal": None,
        "mmap": None, "ctypes": None,
    }

    def __init__(self, model: "LunaModel", logger: LunaLogger) -> None:
        self.model = model
        self.logger = logger
        self._uv_available = shutil.which("uv")
        self._python_exe = sys.executable

    def _is_installed(self, import_name: str) -> tuple[bool, Optional[str]]:
        top = import_name.split(".")[0]
        if top in self.KNOWN_MAPPINGS and self.KNOWN_MAPPINGS[top] is None:
            return True, "stdlib"
        try:
            spec = importlib.util.find_spec(top)
            if spec is None:
                return False, None
            mod = importlib.import_module(top)
            version = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
            return True, str(version) if version else "installed"
        except Exception:
            return False, None

    def _extract_imports(self, code: str) -> list[str]:
        imports: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        return sorted(imports)

    def analyze(self, code: str) -> list[DependencyInfo]:
        imports = self._extract_imports(code)
        result: list[DependencyInfo] = []
        for imp in imports:
            installed, version = self._is_installed(imp)
            mapping = self.KNOWN_MAPPINGS.get(imp)
            install_spec = mapping if mapping else imp
            result.append(DependencyInfo(
                name=imp, import_name=imp,
                is_installed=installed, version=version,
                install_spec=install_spec or imp,
            ))
        return result

    def resolve_install_spec(self, import_name: str) -> Optional[str]:
        known = self.KNOWN_MAPPINGS.get(import_name)
        if known is not None:
            return known
        if import_name in self.KNOWN_MAPPINGS:
            return None

        prompt = (
            f'Qual é o nome do pacote no PyPI para importar `{import_name}` '
            f'em Python? Se for stdlib, responda null. '
            f'Responda SOMENTE JSON: {{"package": "nome"}} ou {{"package": null}}.'
        )

        try:
            result = self.model.generate(
                system="Você responde com JSON estrito.",
                user=prompt,
                max_tokens=120,
                temperature=TEMPERATURE_DEP_RESOLVER,
                response_format={"type": "json_object"},
                stream_output=False,
                label="dep-resolver",
            )
            data = safe_json(result.text)
            if data:
                pkg = data.get("package")
                if pkg:
                    return str(pkg)
        except Exception:
            pass
        return import_name

    def install(self, packages: list[str], timeout: int = PIP_INSTALL_TIMEOUT) -> tuple[bool, str]:
        if not packages:
            return True, "(nada)"

        cmd: list[str]
        use_uv = PREFER_UV_OVER_PIP and self._uv_available
        if use_uv:
            cmd = [self._uv_available, "pip", "install", "--quiet"] + packages
        else:
            cmd = [self._python_exe, "-m", "pip", "install", "--quiet"] + packages

        dim(f"   instalando: {' '.join(cmd)}")

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=timeout, cwd=str(WORKSPACE),
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return (proc.returncode == 0), output
        except subprocess.TimeoutExpired:
            return False, f"timeout ({timeout}s)"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def ensure(self, code: str, state: SessionState) -> tuple[bool, list[str], list[DependencyInfo]]:
        if not ENABLE_DEPENDENCY_MANAGER:
            return True, [], []

        deps = self.analyze(code)
        missing = [d for d in deps if not d.is_installed]
        if not missing:
            return True, [], deps

        warning(f"Pacotes ausentes: {[d.import_name for d in missing]}")
        if not AUTO_INSTALL_DEPS:
            return False, [], deps

        specs: list[str] = []
        for dep in missing:
            spec = dep.install_spec or dep.import_name
            if spec and spec not in specs:
                specs.append(spec)

        stage(f"Instalando {len(specs)} pacote(s)...")
        ok, log_str = self.install(specs)

        self.logger.log("dependency_install", {
            "packages": specs, "success": ok,
            "log": truncate(log_str, 4000),
        })

        if ok:
            success(f"Instalados: {specs}")
            state.installed_packages.extend(specs)
            return True, specs, deps
        error(f"Falha: {specs}")
        dim(truncate(log_str, 800))
        return False, [], deps


# ============================================================
# ROBUST HTTP CLIENT
# ============================================================

class RobustHTTPClient:
    """
    Cliente HTTP robusto com retry exponencial, headers realistas,
    tratamento de 429/5xx, session reuse.
    """

    def __init__(self, logger: LunaLogger) -> None:
        self.logger = logger

    def _build_request_code(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        timeout: int = 15,
        max_retries: int = HTTP_MAX_RETRIES,
    ) -> str:
        """Gera código Python one-liner para requisição robusta."""

        merged_headers = {**HTTP_DEFAULT_HEADERS, **(headers or {})}
        headers_str = repr(merged_headers)
        params_str = repr(params or {})
        json_str = repr(json_body) if json_body else "None"
        method_upper = method.upper()

        code = (
            "import requests, json, time, random\n"
            "session = requests.Session()\n"
            f"headers = {headers_str}\n"
            f"params = {params_str}\n"
            f"json_body = {json_str}\n"
            f"url = {url!r}\n"
            f"method = {method_upper!r}\n"
            f"max_retries = {max_retries}\n"
            f"timeout = {timeout}\n"
            "last_error = None\n"
            "result = None\n"
            "for attempt in range(max_retries + 1):\n"
            "    try:\n"
            "        kwargs = {'headers': headers, 'params': params, 'timeout': timeout, 'allow_redirects': True}\n"
            "        if method == 'POST' or method == 'PUT' or method == 'PATCH':\n"
            "            kwargs['json'] = json_body\n"
            "        r = session.request(method, url, **kwargs)\n"
            "        status = r.status_code\n"
            "        body = r.text[:8000]\n"
            "        if status == 429 or (500 <= status < 600):\n"
            "            if attempt < max_retries:\n"
            "                wait = (1.5 ** attempt) + random.uniform(0, 0.3)\n"
            "                time.sleep(wait)\n"
            "                last_error = 'HTTP ' + str(status)\n"
            "                continue\n"
            "        result = {'status': status, 'body': body, 'attempts': attempt + 1}\n"
            "        break\n"
            "    except requests.exceptions.Timeout:\n"
            "        last_error = 'timeout'\n"
            "        if attempt < max_retries:\n"
            "            time.sleep((1.5 ** attempt) + random.uniform(0, 0.3))\n"
            "            continue\n"
            "        result = {'status': -1, 'error': 'timeout after retries'}\n"
            "        break\n"
            "    except requests.exceptions.ConnectionError as e:\n"
            "        last_error = 'connection: ' + str(e)[:200]\n"
            "        if attempt < max_retries:\n"
            "            time.sleep((1.5 ** attempt) + random.uniform(0, 0.3))\n"
            "            continue\n"
            "        result = {'status': -2, 'error': last_error}\n"
            "        break\n"
            "    except Exception as e:\n"
            "        result = {'status': -3, 'error': type(e).__name__ + ': ' + str(e)[:200]}\n"
            "        break\n"
            "if result is None:\n"
            "    result = {'status': -4, 'error': last_error or 'unknown'}\n"
            "print(json.dumps(result))\n"
        )
        return code

    def request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        timeout: int = API_TEST_TIMEOUT,
        cache: Optional[dict[str, ApiTestResult]] = None,
    ) -> ApiTestResult:
        """Executa requisição HTTP robusta via python -c."""

        cache_key = f"{method}:{url}:{hash(str(params))}:{hash(str(json_body))}"
        if cache and cache_key in cache:
            return cache[cache_key]

        started = time.perf_counter()
        code = self._build_request_code(url, method, headers, params, json_body, timeout)

        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=timeout * (HTTP_MAX_RETRIES + 2),
                cwd=str(WORKSPACE),
            )

            if proc.returncode != 0:
                result = ApiTestResult(
                    url=url, method=method, success=False,
                    error=truncate(proc.stderr or "", 500),
                    elapsed=time.perf_counter() - started,
                )
            else:
                try:
                    data = json.loads(proc.stdout.strip())
                except json.JSONDecodeError:
                    result = ApiTestResult(
                        url=url, method=method, success=False,
                        error=f"resposta inválida: {truncate(proc.stdout, 300)}",
                        elapsed=time.perf_counter() - started,
                    )
                else:
                    status = data.get("status")
                    if status is None or status < 0:
                        result = ApiTestResult(
                            url=url, method=method, success=False,
                            status_code=status,
                            error=data.get("error", "erro desconhecido"),
                            elapsed=time.perf_counter() - started,
                            retries=data.get("attempts", 0),
                        )
                    else:
                        success_flag = 200 <= status < 400
                        result = ApiTestResult(
                            url=url, method=method,
                            success=success_flag,
                            status_code=status,
                            response_body=truncate(data.get("body", ""), 8000),
                            elapsed=time.perf_counter() - started,
                            retries=data.get("attempts", 1),
                        )
        except subprocess.TimeoutExpired:
            result = ApiTestResult(
                url=url, method=method, success=False,
                error=f"timeout global ({timeout * (HTTP_MAX_RETRIES + 2)}s)",
                elapsed=time.perf_counter() - started,
            )
        except Exception as exc:
            result = ApiTestResult(
                url=url, method=method, success=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed=time.perf_counter() - started,
            )

        if cache is not None:
            cache[cache_key] = result

        return result


# ============================================================
# API DISCOVERY AGENT
# ============================================================

class ApiDiscoveryAgent:
    """
    Descoberta de API: RAG + OpenAPI parsing + robust HTTP.
    """

    COMMON_SPEC_PATHS = [
        "/openapi.json", "/swagger.json", "/api-docs",
        "/v3/api-docs", "/docs/openapi.json",
        "/api/openapi.json", "/.well-known/openapi.json",
        "/swagger/v1/swagger.json", "/api/v1/openapi.json",
    ]

    def __init__(
        self,
        http: RobustHTTPClient,
        logger: LunaLogger,
        model: "LunaModel",
    ) -> None:
        self.http = http
        self.logger = logger
        self.model = model

    def discover(self, base_url: str, state: SessionState) -> Optional[OpenApiSpec]:
        base = base_url.rstrip("/")

        if base.endswith(".json") or base.endswith(".yaml"):
            spec = self._fetch_spec(base, state)
            if spec:
                return spec

        for path in self.COMMON_SPEC_PATHS:
            spec_url = base + path
            if spec_url in state.spec_cache:
                return state.spec_cache[spec_url]
            spec = self._fetch_spec(spec_url, state)
            if spec:
                return spec

        return None

    def _fetch_spec(self, url: str, state: SessionState) -> Optional[OpenApiSpec]:
        if url in state.spec_cache:
            return state.spec_cache[url]

        result = self.http.request(url, cache=state.api_cache)
        if not result.success or not result.response_body:
            return None

        try:
            data = json.loads(result.response_body)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        if "openapi" not in data and "swagger" not in data:
            return None

        version = str(data.get("openapi", data.get("swagger", "unknown")))
        base_url = ""

        servers = data.get("servers", [])
        if isinstance(servers, list) and len(servers) > 0:
            first = servers[0]
            if isinstance(first, dict):
                base_url = first.get("url", "")

        spec = OpenApiSpec(
            url=url, version=version, base_url=base_url,
            endpoints=self._parse_endpoints(data), raw=data,
        )

        state.spec_cache[url] = spec
        return spec

    def _parse_endpoints(self, spec: dict[str, Any]) -> list[ApiEndpoint]:
        endpoints: list[ApiEndpoint] = []
        paths = spec.get("paths", {})

        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue

            for method, info in methods.items():
                method_upper = method.upper()
                if method_upper not in {
                    "GET", "POST", "PUT", "DELETE",
                    "PATCH", "HEAD", "OPTIONS",
                }:
                    continue

                params: list[str] = []
                if isinstance(info, dict):
                    for p in info.get("parameters", []):
                        if isinstance(p, dict) and "name" in p:
                            params.append(p["name"])

                endpoints.append(ApiEndpoint(
                    url=path,
                    method=method_upper,
                    parameters=params,
                    discovered_via="openapi_spec",
                ))

        return endpoints

    def test_endpoints(
        self,
        base_url: str,
        endpoints: list[ApiEndpoint],
        state: SessionState,
        max_tests: int = 5,
    ) -> list[ApiTestResult]:
        results: list[ApiTestResult] = []
        base = base_url.rstrip("/")

        for ep in endpoints[:max_tests]:
            full_url = ep.url if ep.url.startswith("http") else base + ep.url
            result = self.http.request(
                full_url, method=ep.method, cache=state.api_cache,
            )
            results.append(result)

        return results


# ============================================================
# SEMI-FORMAL REASONER
# ============================================================

class SemiFormalReasoner:
    """
    Verificação semi-formal de código sem execução.
    Estrutura: premissas → caminho de execução → conclusão.
    """

    SYSTEM = """
Você é um revisor de código que usa RACIOCÍNIO SEMI-FORMAL.

Dado um código Python e a tarefa, produza um certificado de raciocínio.

Retorne SOMENTE JSON:

{
  "premises": ["premissa 1", "premissa 2"],
  "execution_path": ["passo 1", "passo 2"],
  "conclusion": "o código está correto / incorreto porque...",
  "confidence": "high",
  "evidence": ["evidência 1"],
  "counter_evidence": ["contra-evidência 1"]
}

REGRAS:
1. Premises: o que você assume como verdadeiro sobre o código.
2. Execution path: trace o fluxo do código passo a passo.
3. Conclusion: julgamento final (correto/incorreto e por quê).
4. Confidence: high se traçou todo o caminho, medium se assumiu partes,
   low se não conseguiu traçar.
5. Evidence: linhas de código que suportam a conclusão.
6. counter_evidence: casos onde o código poderia falhar.

NÃO execute o código. Apenas raciocine.
NÃO invente comportamentos. Trace o que está escrito.
""".strip()

    def __init__(self, model: "LunaModel", logger: LunaLogger) -> None:
        self.model = model
        self.logger = logger

    def review(self, task: str, code: str, complexity: int = 3) -> ReasoningCertificate:
        user = f"""
TAREFA:
{task}

CÓDIGO:
{truncate(code, 8000)}

Produza o certificado de raciocínio semi-formal.
""".strip()

        result = self.model.generate(
            system=self.SYSTEM,
            user=user,
            max_tokens=token_budget_for("reviewer", complexity),
            temperature=TEMPERATURE_REVIEWER,
            response_format={"type": "json_object"},
            stream_output=True,
            label="reviewer",
        )

        self.logger.log("reasoning_certificate", {"raw": result.text})

        data = safe_json(result.text)
        if not data:
            return ReasoningCertificate(
                premises=[], execution_path=[],
                conclusion="(revisão indisponível)",
                confidence="low",
            )

        def _as_list(key: str) -> list[str]:
            value = data.get(key, [])
            if not isinstance(value, list):
                return []
            return [str(item) for item in value if item]

        return ReasoningCertificate(
            premises=_as_list("premises"),
            execution_path=_as_list("execution_path"),
            conclusion=str(data.get("conclusion", "")),
            confidence=str(data.get("confidence", "low")).lower(),
            evidence=_as_list("evidence"),
            counter_evidence=_as_list("counter_evidence"),
        )


# ============================================================
# LOOP DETECTOR
# ============================================================

def _normalize_for_similarity(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
    return set(w for w in words if len(w) > 2)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def structural_signature(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    parts: list[str] = []

    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            parts.append(f"def:{node.name}")
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            parts.append(f"async:{node.name}")
            self.generic_visit(node)

        def visit_Call(self, node):
            func = _dotted_name(node.func)
            if func:
                parts.append(f"call:{func}")
            self.generic_visit(node)

        def visit_For(self, node):
            parts.append("for")
            self.generic_visit(node)

        def visit_While(self, node):
            parts.append("while")
            self.generic_visit(node)

        def visit_If(self, node):
            parts.append("if")
            self.generic_visit(node)

    Collector().visit(tree)
    return "|".join(sorted(set(parts)))[:200]


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    cur: Any = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def detect_loop(
    code: str,
    state: SessionState,
    last_error: str = "",
) -> Optional[LoopSignal]:
    if not ENABLE_LOOP_DETECTOR:
        return None

    fp = code_fingerprint(code)
    if fp in state.fingerprints:
        return LoopSignal(
            kind="exact", similarity=1.0,
            evidence=f"fingerprint {fp} já visto",
            severity="high",
        )

    sig = structural_signature(code)
    if sig and sig in state.recent_signatures:
        return LoopSignal(
            kind="structural", similarity=0.95,
            evidence=f"assinatura {sig[:8]} repetida",
            severity="high",
        )

    tokens_new = _normalize_for_similarity(code)
    max_sim = 0.0
    best_evidence = ""
    for prev_sig in state.recent_signatures:
        sim = jaccard_similarity(tokens_new, set(prev_sig.split()))
        if sim > max_sim:
            max_sim = sim
            best_evidence = f"similaridade {sim:.2f}"

    if max_sim >= MAX_LOOP_SIMILARITY:
        return LoopSignal(
            kind="semantic", similarity=max_sim,
            evidence=best_evidence, severity="medium",
        )

    return None


def update_loop_state(code: str, state: SessionState) -> None:
    fp = code_fingerprint(code)
    state.fingerprints.append(fp)
    sig = structural_signature(code)
    if sig:
        state.recent_signatures.append(sig)


# ============================================================
# TRACEBACK PARSER
# ============================================================

_FRAME_RE = re.compile(
    r'File "(?P<path>[^"]+)",\s+line\s+(?P<lineno>\d+),\s+in\s+(?P<func>.+)'
)

_EXCEPTION_RE = re.compile(
    r"^(?P<etype>[A-Za-z_][A-Za-z0-9_.]*"
    r"(?:Error|Exception|Warning|Interrupt|Exit|Timeout|Fault))"
    r"(?::\s*(?P<emsg>.*))?$"
)

_LIBRARY_HINTS = (
    "site-packages", "lib/python", "lib\\python",
    "matplotlib", "numpy", "scipy", "pandas", "PIL", "cv2",
)


def _is_library_path(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    if lowered.startswith("<") and lowered.endswith(">"):
        return False
    for hint in _LIBRARY_HINTS:
        if hint.lower() in lowered:
            return True
    return False


def parse_traceback(code: str, stderr: str) -> TracebackInfo:
    info = TracebackInfo(raw=stderr or "")
    if not stderr:
        return info

    lines = stderr.splitlines()
    frames: list[StackFrame] = []
    i = 0
    while i < len(lines):
        m = _FRAME_RE.search(lines[i])
        if m:
            file_path = m.group("path")
            try:
                lineno = int(m.group("lineno"))
            except (ValueError, TypeError):
                lineno = 0
            func = (m.group("func") or "").strip()
            code_text = ""
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and not nxt.startswith("File \"") and not nxt.startswith("^"):
                    code_text = nxt
            frames.append(StackFrame(
                file=file_path, line=lineno, func=func,
                code_text=code_text, is_user=not _is_library_path(file_path),
            ))
            i += 2
            continue
        i += 1

    info.all_frames = frames
    for frame in reversed(frames):
        if frame.is_user and info.user_frame is None:
            info.user_frame = frame
        if not frame.is_user and info.library_frame is None:
            info.library_frame = frame
        if info.user_frame and info.library_frame:
            break

    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        m = _EXCEPTION_RE.match(stripped)
        if m:
            info.exception_type = m.group("etype")
            info.exception_message = (m.group("emsg") or "").strip()
            break

    if info.user_frame and code:
        code_lines = code.splitlines()
        n = info.user_frame.line
        start = max(0, n - 4)
        end = min(len(code_lines), n + 3)
        for idx in range(start, end):
            info.user_context.append((idx + 1, code_lines[idx]))

    return info


def describe_traceback(info: TracebackInfo) -> str:
    if info.is_empty:
        return "(sem traceback estruturado)"
    parts: list[str] = []
    if info.exception_type:
        parts.append(f"EXCEÇÃO: {info.exception_type}")
    if info.exception_message:
        parts.append(f"MENSAGEM: {info.exception_message}")
    if info.user_frame:
        parts.append("")
        parts.append(">>> FALHA NO SEU CÓDIGO <<<")
        parts.append(
            f"  linha {info.user_frame.line} "
            f"(função {info.user_frame.func or '<module>'})"
        )
        if info.user_frame.code_text:
            parts.append(f"  código: {info.user_frame.code_text}")
    if info.library_frame:
        parts.append("")
        parts.append("(exceção levantada dentro da biblioteca)")
        parts.append(f"  {info.library_frame.file}")
        parts.append(f"  linha {info.library_frame.line} em {info.library_frame.func}")
    if info.user_context:
        parts.append("")
        parts.append("CONTEXTO (suas linhas):")
        for lineno, text in info.user_context:
            marker = " >>> " if lineno == info.user_frame.line else "     "
            parts.append(f"{marker}{lineno:>4} | {text}")
    return "\n".join(parts)


# ============================================================
# ANTI-PADRÃO
# ============================================================

ANTIPATTERN_RULES: list[tuple[str, str, str]] = [
    (
        r"plot_surface\s*\(\s*[^,]+,\s*[^,]+,\s*[0-9]+(?:\.[0-9]+)?\s*[,)]",
        "plot_surface com Z escalar",
        "use np.zeros_like(X) / np.full_like(X, valor)",
    ),
    (
        r"if\s+(?:abs\s*\(\s*)?[A-Za-z_]\w*\s*[<>=!]+\s*[0-9]",
        "comparação de array com escalar via if",
        "use np.where(...) ou np.errstate(...)",
    ),
    (
        r"np\.nan\s*==\s*|==\s*np\.nan|!=\s*np\.nan",
        "comparação direta com np.nan",
        "use np.isnan(x)",
    ),
]


def detect_antipatterns(code: str) -> list[tuple[str, str, str]]:
    hits: list[tuple[str, str, str]] = []
    for idx, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for pattern, reason, suggestion in ANTIPATTERN_RULES:
            if re.search(pattern, line):
                hits.append((f"linha {idx}: {stripped}", reason, suggestion))
    return hits


# ============================================================
# INPUT AUTO-FILL
# ============================================================

def extract_input_prompts(code: str) -> list[str]:
    prompts: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return prompts
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "input":
                if node.args and isinstance(node.args[0], ast.Constant):
                    prompts.append(str(node.args[0].value))
                else:
                    prompts.append("")
    return prompts


def synthesize_input_value(prompt: str) -> str:
    p = (prompt or "").lower()
    if any(k in p for k in ("número", "numero", "number", "inteiro", "int")):
        return "7"
    if any(k in p for k in ("decimal", "float", "real", "frac")):
        return "3.14"
    if any(k in p for k in ("nome", "name")):
        return "Luna"
    if "email" in p or "e-mail" in p:
        return "luna@example.com"
    if "senha" in p or "password" in p:
        return "senha123"
    if "idade" in p or "age" in p:
        return "25"
    if "s/n" in p or "y/n" in p or "sim" in p:
        return "s"
    if "arquivo" in p or "file" in p or "caminho" in p or "path" in p:
        return "dados.txt"
    if "url" in p or "endereço" in p or "endereco" in p:
        return "https://example.com"
    return "1"


def build_stdin_payload(values: list[str]) -> Optional[str]:
    if not values:
        return None
    return "\n".join(values) + "\n"


# ============================================================
# CONTEXT MANAGER
# ============================================================

class ContextManager:
    def __init__(self, model: "LunaModel", logger: LunaLogger) -> None:
        self.model = model
        self.logger = logger
        self.max_tokens = MAX_CONTEXT_TOKENS

    def would_overflow(self, *texts: str) -> bool:
        total = sum(estimate_tokens(t) for t in texts if t)
        return total > self.max_tokens

    def fit(self, texts: list[str], priorities: Optional[list[int]] = None) -> list[str]:
        if priorities is None:
            priorities = [1] * len(texts)

        total = sum(estimate_tokens(t) for t in texts if t)
        if total <= self.max_tokens:
            return texts

        order = sorted(range(len(texts)), key=lambda i: priorities[i])
        result = list(texts)
        current = total
        for idx in order:
            if current <= self.max_tokens:
                break
            excess = current - self.max_tokens
            text = result[idx] or ""
            t_tokens = estimate_tokens(text)
            if t_tokens <= excess + 100:
                result[idx] = truncate(text, max(200, len(text) - excess * 4))
                current -= t_tokens
            else:
                max_chars = max(500, len(text) - excess * 4)
                result[idx] = truncate_smart(text, max_chars)
                new_tokens = estimate_tokens(result[idx])
                current -= (t_tokens - new_tokens)
        return result


# ============================================================
# INTENT CLASSIFIER
# ============================================================

INTENT_SYSTEM = """
Você é o classificador de intenção do agente Luna.

Dada uma mensagem do usuário, classifique. Retorne SOMENTE JSON:

{
  "wants_delivery": true,
  "wants_interpretation": false,
  "read_requested": false,
  "module_names": [],
  "forbids_files": false,
  "wants_visual": false,
  "involves_api": false,
  "api_urls": []
}

Definições:
- wants_delivery: o usuário quer CRIAR/EXECUTAR/GERAR algo.
- wants_interpretation: o usuário quer EXPLICAR/ANALISAR/COMENTAR.
- read_requested: usuário pediu para LER/CARREGAR/ABRIR módulo/arquivo.
- module_names: nomes dos módulos/arquivos se read_requested=true.
- forbids_files: usuário disse "sem salvar", "sem arquivo", etc.
- wants_visual: usuário quer VER o resultado.
- involves_api: tarefa envolve consumir API HTTP/REST.
- api_urls: URLs completas mencionadas.

IMPORTANTE: Se o usuário faz uma PERGUNTA sobre um fato atual
(preço, clima, cotação, notícia), NÃO é delivery puro — o usuário
quer uma resposta. Trate como wants_delivery=true com is_api_task
se aplicável. Sempre prefira entregar um RESULTADO real.

Responda SOMENTE com JSON.
""".strip()


def classify_intent_llm(task: str, model: "LunaModel", logger: LunaLogger) -> Intent:
    result = model.generate(
        system=INTENT_SYSTEM,
        user=task,
        max_tokens=400,
        temperature=0.02,
        response_format={"type": "json_object"},
        stream_output=False,
        label="intent",
    )
    logger.log("intent_raw", {"raw": result.text})

    data = safe_json(result.text)
    if not data:
        data = {
            "wants_delivery": True,
            "wants_interpretation": False,
            "read_requested": False,
            "module_names": [],
            "forbids_files": False,
            "wants_visual": False,
            "involves_api": False,
            "api_urls": [],
        }

    module_names = data.get("module_names", [])
    if not isinstance(module_names, list):
        module_names = []

    api_urls = data.get("api_urls", [])
    if not isinstance(api_urls, list):
        api_urls = []

    return Intent(
        wants_delivery=bool(data.get("wants_delivery", True)),
        wants_interpretation=bool(data.get("wants_interpretation", False)),
        read_requested=bool(data.get("read_requested", False)),
        module_names=[str(m).strip() for m in module_names if m],
        forbids_files=bool(data.get("forbids_files", False)),
        wants_visual=bool(data.get("wants_visual", False)),
        involves_api=bool(data.get("involves_api", False)),
        api_urls=[str(u).strip() for u in api_urls if u],
    )


# ============================================================
# MODULE LOADER
# ============================================================

def find_module_files(name: str) -> list[Path]:
    target_full = name.lower()
    target_stem = Path(name).stem.lower()
    all_files = [p for p in MOLS_DIR.rglob("*") if p.is_file()]

    hits: list[Path] = []
    for path in all_files:
        if path.name.lower() == target_full:
            hits.append(path)
    if hits:
        return hits
    for path in all_files:
        if path.stem.lower() == target_stem:
            hits.append(path)
    if hits:
        return hits
    if target_stem and len(target_stem) >= 3:
        for path in all_files:
            if target_stem in path.stem.lower():
                hits.append(path)
    return hits


def load_modules(names: list[str], logger: LunaLogger) -> list[tuple[str, str]]:
    loaded: list[tuple[str, str]] = []
    if not names:
        entries = sorted(
            p.relative_to(MOLS_DIR) for p in MOLS_DIR.rglob("*") if p.is_file()
        )
        if entries:
            dim(f"   módulos disponíveis em ./mols: {len(entries)} arquivo(s)")
            for entry in entries[:20]:
                dim(f"     • {entry}")
        else:
            dim("   ./mols está vazio.")
        return loaded

    for name in names:
        hits = find_module_files(name)
        if not hits:
            warning(f"Módulo não encontrado em ./mols: {name}")
            logger.log("module_missing", {"name": name})
            continue
        for path in hits:
            content = read_text_safely(path)
            label = str(path.relative_to(MOLS_DIR))
            loaded.append((label, content))
            success(f"Módulo carregado: {label} ({len(content)} chars)")
    return loaded


def build_augmented_task(
    task: str,
    module_contents: list[tuple[str, str]],
    api_endpoints: list[ApiEndpoint],
    api_test_hints: str = "",
    doc_hints: str = "",
) -> str:
    parts: list[str] = [task]

    if module_contents:
        parts.append("")
        parts.append("=== MÓDULOS CARREGADOS (./mols) ===")
        for label, content in module_contents:
            parts.append(f"--- {label} ---")
            parts.append(content)
            parts.append(f"--- fim {label} ---")
            parts.append("")

    if api_endpoints:
        parts.append("")
        parts.append("=== ENDPOINTS DE API DESCOBERTOS ===")
        for ep in api_endpoints:
            line = f"[{ep.method}] {ep.url}"
            if ep.parameters:
                line += f"  params={ep.parameters}"
            parts.append(line)
            if ep.sample_response:
                parts.append(f"  amostra: {truncate(ep.sample_response, 300)}")
        parts.append("")

    if api_test_hints:
        parts.append("")
        parts.append("=== TESTES DE API REALIZADOS ===")
        parts.append(api_test_hints)
        parts.append("")

    if doc_hints:
        parts.append("")
        parts.append("=== DOCUMENTAÇÃO LIDA ===")
        parts.append(doc_hints)
        parts.append("")

    return "\n".join(parts)


# ============================================================
# MODELO
# ============================================================

class LunaModel:

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo não encontrado:\n{MODEL_PATH}")

        log("Carregando modelo...")
        self.n_ctx = N_CTX_BASE
        self.llm = self._load(N_CTX_BASE)
        success(f"Modelo carregado. Contexto: {self.n_ctx}")

    def _load(self, n_ctx: int) -> Llama:
        kwargs: dict[str, Any] = {
            "model_path": str(MODEL_PATH),
            "n_ctx": n_ctx,
            "n_threads": N_THREADS,
            "n_batch": N_BATCH,
            "n_gpu_layers": N_GPU_LAYERS,
            "verbose": False,
            "use_mmap": True,
            "seed": SEED,
        }
        if USE_FLASH_ATTN:
            kwargs["flash_attn"] = True
        return Llama(**kwargs)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:
            return estimate_tokens(text)

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.12,
        response_format: Optional[dict[str, Any]] = None,
        stream_output: Optional[bool] = None,
        label: str = "modelo",
    ) -> GenerationResult:

        if stream_output is None:
            stream_output = SHOW_MODEL_STREAM

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt_tokens = self.count_tokens(system + "\n" + user)
        started = time.perf_counter()

        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=TOP_P,
            top_k=TOP_K,
            min_p=MIN_P,
            repeat_penalty=REPEAT_PENALTY,
            max_tokens=max_tokens,
            seed=SEED,
            stream=STREAM,
            response_format=response_format,
        )

        parts: list[str] = []
        if stream_output:
            stream_open(label)

        try:
            for chunk in stream:
                try:
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        parts.append(content)
                        if stream_output:
                            print(content, end="", flush=True)
                except (AttributeError, KeyError, TypeError):
                    continue
        except KeyboardInterrupt:
            if stream_output:
                print()
                stream_close(f"{label} (interrompido)")
            raise

        if stream_output:
            print()
            stream_close(label)

        text = "".join(parts)
        elapsed = max(time.perf_counter() - started, 1e-6)
        completion_tokens = self.count_tokens(text)
        tps = completion_tokens / elapsed if completion_tokens else 0.0
        truncated = completion_tokens >= max_tokens - 5

        return GenerationResult(
            text=text, elapsed=elapsed,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens, tokens_per_second=tps,
            truncated=truncated,
        )


# ============================================================
# PROMPTS
# ============================================================

ROUTER_SYSTEM = """
Você é o router do agente Luna. Classifique a tarefa. Retorne SOMENTE JSON:

{
  "complexity": 3,
  "needs_planning": false,
  "needs_execution": true,
  "needs_verification": true,
  "task_type": "python",
  "expects_files": false,
  "is_quick": false,
  "quick_kind": "python",
  "needs_network": false,
  "wants_visual": false,
  "self_questioning": false,
  "is_api_task": false
}

complexity: 1 (trivial) a 5 (muito complexa).
task_type: python | coding | file | math | network | visualization | general.
expects_files: true SOMENTE se o usuário deu NOME EXPLÍCITO de arquivo.
is_quick: true se resolve com UM comando.
quick_kind: SEMPRE "python".
needs_network: true se acessa rede.
wants_visual: true se quer ver resultado.
self_questioning: true se a tarefa tem ambiguidades.
is_api_task: true se consome API HTTP/REST.

Se o usuário pergunta um FATO ATUAL (preço, clima, cotação), é
is_api_task=true. A Luna vai buscar via requests.

Responda SOMENTE com JSON.
""".strip()


ARCHITECT_SYSTEM = """
Você é o arquiteto técnico da Luna.

Você recebe uma tarefa e precisa planejar a solução ANTES de codar.

Retorne SOMENTE JSON:

{
  "goal": "objetivo em uma linha",
  "requirements": ["requisito 1", "requisito 2"],
  "checks": ["verificação 1", "verificação 2"],
  "test_inputs": ["valor 1"],
  "expected_outputs": ["saída esperada 1"],
  "interactive": false,
  "approach": "descrição curta da abordagem técnica",
  "risks": ["risco 1", "risco 2"]
}

REGRAS:
- requirements: condições funcionais que o programa DEVE cumprir.
- checks: verificações objetivas após execução.
- test_inputs: valores de stdin se houver input().
- expected_outputs: strings que devem aparecer no stdout.
- approach: como você vai resolver.
- risks: o que pode dar errado.
- NÃO invente requisitos que o usuário não pediu.
""".strip()


CODER_SYSTEM = """
Você é o engenheiro principal da Luna.

Escreva código Python 3 COMPLETO e EXECUTÁVEL.

═══════════════════════════════════════════════════════════════
REGRA CRÍTICA — REDE / API
═══════════════════════════════════════════════════════════════
- NUNCA use curl, wget, subprocess para HTTP.
- SEMPRE use `import requests` para HTTP/REST.
- Toda requisição DEVE ter timeout: requests.get(url, timeout=10).
- Trate erros:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        print(f"Erro: {e}")
- Para APIs que exigem User-Agent, use:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
- Se o usuário forneceu uma URL, use-a exatamente.

═══════════════════════════════════════════════════════════════
REGRA CRÍTICA — NÃO SALVAR ARQUIVOS SEM PEDIDO
═══════════════════════════════════════════════════════════════
- Se o usuário pedir um gráfico SEM nome de arquivo, use SOMENTE plt.show().
- Só use plt.savefig('nome.png') se o usuário deu nome explícito.
- Se o usuário disse "sem salvar"/"sem png", proibido savefig.
- Não crie arquivos .py/.txt/.json extras.

═══════════════════════════════════════════════════════════════
REGRAS GERAIS
═══════════════════════════════════════════════════════════════
1. Retorne SOMENTE código Python puro. Sem markdown.
2. Código completo (imports incluídos).
3. Nunca peça confirmação. Execute o que foi pedido.
4. Executável do início ao fim sem intervenção.
5. Não explique. Apenas código.

═══════════════════════════════════════════════════════════════
REGRAS CRÍTICAS — MATPLOTLIB 3D
═══════════════════════════════════════════════════════════════
- ax.plot_surface(X, Y, Z) exige X, Y, Z arrays 2D mesmo shape.

  ❌ ERRADO:   ax.plot_surface(X, Y, 0)
  ✅ CORRETO:  ax.plot_surface(X, Y, Z)
  ✅ CORRETO:  ax.plot_surface(X, Y, np.zeros_like(X))

- NUNCA use `if <array> ...:` — use np.where / np.errstate.

═══════════════════════════════════════════════════════════════
REGRAS CRÍTICAS — NUMPY
═══════════════════════════════════════════════════════════════
- NaN: use np.isnan(x), NUNCA x == np.nan.

═══════════════════════════════════════════════════════════════
QUALIDADE
═══════════════════════════════════════════════════════════════
- Prefira soluções curtas e corretas.
""".strip()


QUICK_SYSTEM = """
Você é o executor rápido da Luna. Tarefa TRIVIAL — UM comando.
Retorne SOMENTE JSON:

{
  "kind": "python",
  "command": "...",
  "explanation": "..."
}

REGRA ABSOLUTA: kind DEVE ser "python". O campo command é código Python
one-liner (uma linha). NUNCA use curl/wget/shell.

Exemplos:
- HTTP: "import requests; r=requests.get('https://...', timeout=10); print(r.json())"
- Versão: "import numpy; print(numpy.__version__)"
- Cálculo: "import math; print(math.sqrt(144))"

Responda SOMENTE com JSON.
""".strip()


REVIEWER_SYSTEM = """
Você é o revisor de código da Luna usando RACIOCÍNIO SEMI-FORMAL.

Retorne SOMENTE JSON:

{
  "premises": ["premissa sobre o código"],
  "execution_path": ["passo 1", "passo 2"],
  "conclusion": "correto / incorreto porque...",
  "confidence": "high|medium|low",
  "evidence": ["linha X faz Y"],
  "counter_evidence": ["falha se Z"]
}

NÃO execute. Apenas raciocine sobre o código.
""".strip()


INVESTIGATOR_SYSTEM = """
Você é o investigador da Luna.

A execução falhou. Se o traceback aponta CLARAMENTE o código do usuário,
NÃO peça probe. Se é ambíguo (ImportError, rede, permissão, API),
PEÇA UM probe.

Retorne SOMENTE JSON:

{
  "need_probe": false,
  "reasoning": "...",
  "kind": "python",
  "probe": ""
}

Erros autoexplicativos: AttributeError, TypeError, ValueError,
KeyError, IndexError, NameError, ZeroDivisionError, SyntaxError.

Erros ambíguos: ImportError, ModuleNotFoundError, ConnectionError,
FileNotFoundError, PermissionError, timeout, HTTPError.

kind = "python" SEMPRE.

UM probe. Rápido. Determinístico.
""".strip()


DEBUGGER_SYSTEM = """
Você é o analista de causa raiz da Luna.

Retorne SOMENTE JSON:

{
  "root_cause_type": "syntax|runtime|logic|env|loop|dependency|api|unknown",
  "root_cause": "explicação técnica precisa",
  "location": "linha N do SEU código",
  "observed_value": "valor que causou o erro",
  "admissible_alternatives": ["alternativa 1"],
  "fix_strategy": "como consertar",
  "must_avoid": ["padrão concreto 1"],
  "hypothesis_key": "chave curta normalizada",
  "suggested_dependencies": ["pacote1"]
}

REGRA CRÍTICA:
- Causa raiz está no SEU código, NÃO na biblioteca.
- must_avoid: padrões CONCRETOS que NÃO devem reaparecer.
- admissible_alternatives: o que PODERIA substituir.
- hypothesis_key: identificador estável.
- Se erro é de API, indique "api" e sugira ajustes.
""".strip()


REPAIR_CODE_SYSTEM = """
Você é o engenheiro de reparo da Luna.

Recebe análise de causa raiz e código que falhou.
Gere o código Python COMPLETO e CORRIGIDO.

REGRAS ABSOLUTAS:
1. Retorne SOMENTE código Python.
2. Corrija EXATAMENTE a causa raiz.
3. NÃO use construções em must_avoid.
4. Use preferencialmente admissible_alternatives.
5. NUNCA use curl/wget/shell para HTTP. SEMPRE requests.
6. Toda requisição com timeout.
7. Nunca use `if <array> ...:`. Nunca compare com np.nan usando ==.
8. Código executável do início ao fim.
9. Mantenha o mesmo comportamento — só corrija o que está errado.
""".strip()


REPAIR_SURGICAL_SYSTEM = """
Você é o reparador cirúrgico da Luna.

Linhas culpadas foram comentadas com `# LUNA-REMOVED`.
Substitua essas linhas pelas versões CORRETAS.

REGRAS:
1. Retorne SOMENTE código Python.
2. Cada linha `# LUNA-REMOVED` DEVE ser substituída.
   Não pode sobrar NENHUM `# LUNA-REMOVED` no output.
3. Mantenha o resto intacto.
""".strip()


VERIFIER_SYSTEM = """
Você é o verificador externo da Luna.

Retorne SOMENTE JSON:
{
  "status": "ok",
  "reason": "...",
  "correction": ""
}

status: "ok" ou "retry".

Use "retry" SOMENTE com evidência externa clara:
- stdout vazio quando deveria ter saída
- exceção visível
- arquivo esperado ausente
- saída contradiz o esperado

NUNCA exija melhorias que o usuário não pediu.
""".strip()


INTERPRETER_SYSTEM = """
Você é a Luna em modo interpretação. O usuário quer ENTENDER algo.

IMPORTANTE: Se o usuário pergunta um FATO ATUAL (preço, clima, cotação),
você NÃO deve responder "não posso". Você deve RECONHECER que precisa
buscar. Responda algo como: "Vou buscar essa informação via API." e
o pipeline mudará para delivery automaticamente.

Para perguntas conceituais, responda com clareza técnica.
Não gere arquivos. Não execute código diretamente.
""".strip()


# ============================================================
# AGENTE
# ============================================================

class LunaAgent:

    def __init__(self, model: LunaModel, logger: LunaLogger) -> None:
        self.model = model
        self.logger = logger
        self.ctx = ContextManager(model, logger) if ENABLE_CONTEXT_MANAGER else None
        self.deps = DependencyManager(model, logger) if ENABLE_DEPENDENCY_MANAGER else None
        self.http = RobustHTTPClient(logger)
        self.api_discovery = (
            ApiDiscoveryAgent(self.http, logger, model)
            if ENABLE_API_DISCOVERY else None
        )
        self.reviewer = (
            SemiFormalReasoner(model, logger)
            if ENABLE_REVIEWER else None
        )

    def route(self, task: str) -> dict[str, Any]:
        max_tok = token_budget_for("router", 3)
        result = self.model.generate(
            system=ROUTER_SYSTEM, user=task,
            max_tokens=max_tok, temperature=TEMPERATURE_ROUTER,
            response_format={"type": "json_object"},
            stream_output=True, label="router",
        )
        self.logger.log("router_response", {"raw": result.text})
        data = safe_json(result.text)
        if not data:
            warning("Router devolveu JSON inválido. Fallback.")
            data = {
                "complexity": 3, "needs_planning": False,
                "needs_execution": True, "needs_verification": True,
                "task_type": "python", "expects_files": False,
                "is_quick": False, "quick_kind": "python",
                "needs_network": False, "wants_visual": False,
                "self_questioning": False, "is_api_task": False,
            }
        complexity = int(data.get("complexity", 3))
        data["complexity"] = max(1, min(5, complexity))
        return data

    def architect(self, task: str, complexity: int) -> TaskPlan:
        max_tok = token_budget_for("architect", complexity)
        result = self.model.generate(
            system=ARCHITECT_SYSTEM, user=task,
            max_tokens=max_tok, temperature=TEMPERATURE_ARCHITECT,
            response_format={"type": "json_object"},
            stream_output=True, label="architect",
        )
        self.logger.log("architect_response", {"raw": result.text})
        data = safe_json(result.text)
        if not data:
            return TaskPlan(goal=task, requirements=[], checks=[])

        def _as_list(key: str) -> list[str]:
            value = data.get(key, [])
            if not isinstance(value, list):
                return []
            return [str(item) for item in value if item]

        return TaskPlan(
            goal=str(data.get("goal", task)),
            requirements=_as_list("requirements"),
            checks=_as_list("checks"),
            test_inputs=_as_list("test_inputs"),
            expected_outputs=_as_list("expected_outputs"),
            interactive=bool(data.get("interactive", False)),
            probe_strategy=str(data.get("approach", "") or ""),
        )

    def generate_code(
        self, task: str, plan: Optional[TaskPlan],
        complexity: int, hints: str = "",
    ) -> GenerationResult:
        if plan:
            requirements = "\n".join(f"- {x}" for x in plan.requirements) or "- nenhum"
            checks = "\n".join(f"- {x}" for x in plan.checks) or "- executar sem erro"
            user = f"""
OBJETIVO:
{plan.goal}

REQUISITOS:
{requirements}

CHECKS:
{checks}

SOLICITAÇÃO ORIGINAL:
{task}
""".strip()
        else:
            user = f"SOLICITAÇÃO ORIGINAL:\n\n{task}"

        if hints:
            user += f"\n\nDICAS:\n{hints}"

        max_tok = token_budget_for("coder", complexity)
        return self.model.generate(
            system=CODER_SYSTEM, user=user,
            max_tokens=max_tok, temperature=TEMPERATURE_CODER,
            stream_output=True, label="coder",
        )

    def generate_quick(self, task: str) -> Optional[QuickCommand]:
        max_tok = token_budget_for("quick", 1)
        result = self.model.generate(
            system=QUICK_SYSTEM, user=task,
            max_tokens=max_tok, temperature=TEMPERATURE_QUICK,
            response_format={"type": "json_object"},
            stream_output=True, label="quick",
        )
        self.logger.log("quick_response", {"raw": result.text})
        data = safe_json(result.text)
        if not data:
            return None
        kind = str(data.get("kind", "python")).lower()
        if kind != "python":
            kind = "python"
        command = str(data.get("command", "")).strip()
        if not command:
            return None
        return QuickCommand(
            kind=kind, command=command,
            explanation=str(data.get("explanation", "")).strip(),
        )

    def should_investigate(self, info: Optional[TracebackInfo]) -> bool:
        if not info or info.is_empty:
            return True
        etype = info.exception_type
        obvious = {
            "AttributeError", "TypeError", "ValueError",
            "ZeroDivisionError", "IndexError", "KeyError",
            "NameError", "UnboundLocalError", "IndentationError",
            "SyntaxError",
        }
        if etype in obvious and info.user_frame:
            return False
        return True

    def investigate(
        self, task: str, plan: Optional[TaskPlan], code: str,
        execution: ExecutionResult, analysis: AnalysisResult,
        traceback_info: Optional[TracebackInfo],
        history: list[str],
    ) -> str:

        if not ENABLE_DEBUGGER:
            return ""
        if not self.should_investigate(traceback_info):
            dim("   investigador: traceback autoexplicativo, pulando probe.")
            return ""

        history_snippet = "\n".join(history[-3:]) if history else "(vazio)"
        tb_desc = describe_traceback(traceback_info) if traceback_info else "(sem traceback)"

        user = f"""
OBJETIVO:
{task}

TRACEBACK ESTRUTURADO:
{tb_desc}

STDERR:
{truncate(execution.stderr, 3000)}

RETURN CODE: {execution.returncode}
TIMED OUT:   {execution.timed_out}

ANÁLISE ESTÁTICA:
missing_packages: {json.dumps(analysis.missing_packages)}
api_endpoints: {json.dumps(analysis.api_endpoints)}

HISTÓRICO:
{history_snippet}

Decida se precisa de probe.
""".strip()

        max_tok = token_budget_for("debugger", 3)
        result = self.model.generate(
            system=INVESTIGATOR_SYSTEM, user=user,
            max_tokens=max_tok, temperature=TEMPERATURE_DEBUGGER,
            response_format={"type": "json_object"},
            stream_output=True, label="investigator",
        )
        self.logger.log("investigator_response", {"raw": result.text})
        data = safe_json(result.text)
        if not data:
            return ""
        if not bool(data.get("need_probe", False)):
            dim("   investigador: sem probe necessário.")
            return ""

        probe = str(data.get("probe", "")).strip()
        reasoning = str(data.get("reasoning", "")).strip()
        if not probe:
            return ""

        stage(f"Investigação — {reasoning or 'probe'}")
        dim(f"   [python] {probe[:260]}")

        probe_exec = execute_probe(probe, kind="python", timeout=PROBE_TIMEOUT)
        self.logger.log("investigation_probe", {
            "probe": probe, "reasoning": reasoning,
        })
        self.logger.log("investigation_result", {
            "returncode": probe_exec.returncode,
            "stdout": truncate(probe_exec.stdout, 4000),
            "stderr": truncate(probe_exec.stderr, 4000),
        })

        if probe_exec.stdout.strip():
            print(Fore.WHITE + "   " + probe_exec.stdout.strip().replace("\n", "\n   ") + Style.RESET_ALL)
        if probe_exec.stderr.strip():
            print(Fore.YELLOW + "   " + probe_exec.stderr.strip().replace("\n", "\n   ") + Style.RESET_ALL)

        return (
            f"PROBE (python): {probe}\n"
            f"RAZÃO: {reasoning}\n"
            f"RETURN: {probe_exec.returncode}\n"
            f"STDOUT:\n{truncate(probe_exec.stdout, 4000)}\n"
            f"STDERR:\n{truncate(probe_exec.stderr, 4000)}\n"
        )

    def debug_analyze(
        self, task: str, plan: Optional[TaskPlan], code: str,
        execution: ExecutionResult, analysis: AnalysisResult,
        traceback_info: Optional[TracebackInfo],
        history: list[dict[str, Any]],
        diff_from_previous: str, investigation: str,
        loop_signal: Optional[LoopSignal] = None,
        api_test_hints: str = "",
        doc_hints: str = "",
        reviewer_certificate: Optional[ReasoningCertificate] = None,
    ) -> FailureDiagnosis:

        history_text: list[str] = []
        for item in history[-4:]:
            history_text.append(
                f"TENTATIVA {item.get('attempt')}:\n"
                f"  resultado: {item.get('result')}\n"
                f"  erro: {truncate(str(item.get('error', '')), 700)}\n"
            )
        history_block = "\n".join(history_text)

        tb_desc = describe_traceback(traceback_info) if traceback_info else "(sem traceback)"
        investigation_block = investigation.strip() or "(sem investigação)"
        diff_block = diff_from_previous.strip() or "(sem diff)"

        annotated_code = annotate_code_with_marker(
            code,
            traceback_info.user_frame.line if (traceback_info and traceback_info.user_frame) else None,
        )

        loop_block = ""
        if loop_signal:
            loop_block = (
                f"\n>>> LOOP DETECTADO ({loop_signal.kind}, "
                f"severidade={loop_signal.severity}): {loop_signal.evidence} <<<\n"
                "Você DEVE mudar a estratégia radicalmente.\n"
            )

        missing_hint = ""
        if "ModuleNotFoundError" in (execution.stderr or ""):
            m = re.search(r"No module named '([^']+)'", execution.stderr)
            if m:
                missing_hint = (
                    f"\n>>> MÓDULO AUSENTE: {m.group(1)}. "
                    f"Adicione em suggested_dependencies <<<\n"
                )

        api_hint = ""
        if api_test_hints:
            api_hint = f"\n>>> TESTES DE API JÁ REALIZADOS:\n{api_test_hints}\n"
        if doc_hints:
            api_hint += f"\n>>> DOCS LIDAS:\n{doc_hints}\n"

        reviewer_block = ""
        if reviewer_certificate:
            reviewer_block = (
                "\n>>> ANÁLISE SEMI-FORMAL DO CÓDIGO:\n"
                f"  premissas: {reviewer_certificate.premises}\n"
                f"  caminho: {reviewer_certificate.execution_path}\n"
                f"  conclusão: {reviewer_certificate.conclusion}\n"
                f"  confiança: {reviewer_certificate.confidence}\n"
                f"  contra-evidência: {reviewer_certificate.counter_evidence}\n"
            )

        user = f"""
OBJETIVO:
{task}
{loop_block}
{missing_hint}
{api_hint}
{reviewer_block}
==================================================
CÓDIGO (linha culpada com >>>)
==================================================
{truncate(annotated_code, MAX_PREVIOUS_CODE_CHARS)}

==================================================
TRACEBACK ESTRUTURADO
==================================================
{tb_desc}

==================================================
STDOUT
==================================================
{truncate(execution.stdout, 3000)}

==================================================
STDERR
==================================================
{truncate(execution.stderr, MAX_TRACEBACK_CHARS)}

==================================================
ANÁLISE ESTÁTICA
==================================================
erros: {json.dumps(analysis.errors, ensure_ascii=False)}
warnings: {json.dumps(analysis.warnings, ensure_ascii=False)}
missing_packages: {json.dumps(analysis.missing_packages, ensure_ascii=False)}
api_endpoints: {json.dumps(analysis.api_endpoints, ensure_ascii=False)}

==================================================
INVESTIGAÇÃO
==================================================
{investigation_block}

==================================================
DIFF vs. TENTATIVA ANTERIOR
==================================================
{diff_block}

==================================================
HISTÓRICO
==================================================
{history_block}

==================================================

Produza a análise de causa raiz estruturada.
""".strip()

        if self.ctx and self.ctx.would_overflow(user):
            user = self.ctx.fit([user], priorities=[1])[0]

        max_tok = token_budget_for("debugger", 3)
        result = self.model.generate(
            system=DEBUGGER_SYSTEM, user=user,
            max_tokens=max_tok, temperature=TEMPERATURE_DEBUGGER,
            response_format={"type": "json_object"},
            stream_output=True, label="debugger",
        )
        self.logger.log("debug_analysis", {"raw": result.text})
        data = safe_json(result.text)

        if not data:
            return FailureDiagnosis(
                root_cause_type="unknown",
                root_cause="(análise indisponível)",
                location="",
                fix_strategy="reescrever sem repetir o erro",
                hypothesis_key="unknown",
            )

        must_avoid = data.get("must_avoid") or []
        if not isinstance(must_avoid, list):
            must_avoid = []
        alternatives = data.get("admissible_alternatives") or []
        if not isinstance(alternatives, list):
            alternatives = []
        suggested_deps = data.get("suggested_dependencies") or []
        if not isinstance(suggested_deps, list):
            suggested_deps = []

        return FailureDiagnosis(
            root_cause_type=str(data.get("root_cause_type", "unknown")),
            root_cause=str(data.get("root_cause", "")),
            location=str(data.get("location", "")),
            observed_value=str(data.get("observed_value", "")),
            admissible_alternatives=[str(a) for a in alternatives],
            fix_strategy=str(data.get("fix_strategy", "")),
            must_avoid=[str(m) for m in must_avoid],
            hypothesis_key=str(data.get("hypothesis_key", "unknown")).strip().lower()[:60],
            suggested_dependencies=[str(d) for d in suggested_deps],
        )

    def repair_code(
        self, task: str, plan: Optional[TaskPlan],
        diagnosis: FailureDiagnosis, previous_code: str,
        execution: ExecutionResult,
        traceback_info: Optional[TracebackInfo],
        investigation: str, diff_from_previous: str,
        complexity: int = 3,
    ) -> GenerationResult:

        must_avoid_block = (
            "\n".join(f"  ❌ {x}" for x in diagnosis.must_avoid)
            if diagnosis.must_avoid else "  (nada)"
        )
        alternatives_block = (
            "\n".join(f"  ✅ {x}" for x in diagnosis.admissible_alternatives)
            if diagnosis.admissible_alternatives else "  (nenhuma sugerida)"
        )

        tb_desc = describe_traceback(traceback_info) if traceback_info else "(sem traceback)"
        investigation_block = investigation.strip() or "(sem investigação)"
        diff_block = diff_from_previous.strip() or "(sem diff)"

        annotated_code = annotate_code_with_marker(
            previous_code,
            traceback_info.user_frame.line if (traceback_info and traceback_info.user_frame) else None,
        )

        deps_block = ""
        if diagnosis.suggested_dependencies:
            deps_block = (
                "\nDEPENDÊNCIAS SUGERIDAS:\n"
                + "\n".join(f"  • {d}" for d in diagnosis.suggested_dependencies)
            )

        user = f"""
OBJETIVO:
{task}

==================================================
DIAGNÓSTICO ESTRUTURADO
==================================================
tipo:        {diagnosis.root_cause_type}
causa:       {diagnosis.root_cause}
localização: {diagnosis.location}
valor visto: {diagnosis.observed_value}
estratégia:  {diagnosis.fix_strategy}
{deps_block}

MUST_AVOID (NÃO use NENHUMA):
{must_avoid_block}

ALTERNATIVAS ADMISSÍVEIS:
{alternatives_block}

==================================================
TRACEBACK
==================================================
{tb_desc}

==================================================
CÓDIGO QUE FALHOU (>>> na linha culpada)
==================================================
{truncate(annotated_code, MAX_PREVIOUS_CODE_CHARS)}

==================================================
INVESTIGAÇÃO
==================================================
{investigation_block}

==================================================
DIFF vs. TENTATIVA ANTERIOR
==================================================
{diff_block}

==================================================

Gere o código Python COMPLETO e CORRIGIDO.
- Sem markdown.
- Corrija a linha marcada com >>>.
- Não use must_avoid.
- Prefira alternativas admissíveis.
- NUNCA use curl/shell para HTTP. SEMPRE requests.
""".strip()

        if self.ctx and self.ctx.would_overflow(user):
            user = self.ctx.fit([user], priorities=[1])[0]

        max_tok = token_budget_for("repair_code", complexity)
        return self.model.generate(
            system=REPAIR_CODE_SYSTEM, user=user,
            max_tokens=max_tok, temperature=TEMPERATURE_REPAIR,
            stream_output=True, label="repair·code",
        )

    def repair_surgical(
        self, task: str, diagnosis: FailureDiagnosis,
        code_with_markers: str,
        traceback_info: Optional[TracebackInfo],
    ) -> GenerationResult:

        tb_desc = describe_traceback(traceback_info) if traceback_info else "(sem traceback)"
        must_avoid_block = "\n".join(f"  ❌ {x}" for x in diagnosis.must_avoid) or "  (nada)"

        user = f"""
OBJETIVO:
{task}

CAUSA RAIZ:
{diagnosis.root_cause}

LOCALIZAÇÃO:
{diagnosis.location}

MUST_AVOID:
{must_avoid_block}

TRACEBACK:
{tb_desc}

==================================================
CÓDIGO COM LINHAS REMOVIDAS (marcadas com # LUNA-REMOVED)
==================================================
{truncate(code_with_markers, MAX_PREVIOUS_CODE_CHARS)}

==================================================

Substitua cada linha `# LUNA-REMOVED` por código correto.
Retorne o código Python completo.
""".strip()

        max_tok = token_budget_for("repair_surgical", 3)
        return self.model.generate(
            system=REPAIR_SURGICAL_SYSTEM, user=user,
            max_tokens=max_tok, temperature=TEMPERATURE_SURGICAL,
            stream_output=True, label="repair·surgical",
        )

    def verify(
        self, task: str, code: str,
        execution: ExecutionResult,
        plan: Optional[TaskPlan],
        expected_files: list[str],
    ) -> dict[str, Any]:

        checks: list[str] = []
        expected: list[str] = []
        if plan:
            checks = plan.checks
            expected = plan.expected_outputs

        files_status = []
        for fname in expected_files:
            path = WORKSPACE / fname
            if path.exists():
                try:
                    size = path.stat().st_size
                    files_status.append(f"{fname}: existe ({size} bytes)")
                except OSError:
                    files_status.append(f"{fname}: erro ao ler")
            else:
                files_status.append(f"{fname}: AUSENTE")

        stdout_len = len(execution.stdout.strip())
        stderr_len = len(execution.stderr.strip())

        user = f"""
TAREFA:
{task}

CHECKS:
{json.dumps(checks, ensure_ascii=False)}

SAÍDAS ESPERADAS:
{json.dumps(expected, ensure_ascii=False)}

EVIDÊNCIA EXTERNA:
- modo: {execution.mode}
- returncode: {execution.returncode}
- timed_out: {execution.timed_out}
- stdout: {stdout_len} chars
- stderr: {stderr_len} chars
- arquivos esperados:
{chr(10).join('  ' + s for s in files_status) if files_status else '  (nenhum)'}

STDOUT:
{truncate(execution.stdout, 6000)}

STDERR:
{truncate(execution.stderr, 4000)}
""".strip()

        max_tok = token_budget_for("verifier", 3)
        result = self.model.generate(
            system=VERIFIER_SYSTEM, user=user,
            max_tokens=max_tok, temperature=TEMPERATURE_VERIFIER,
            response_format={"type": "json_object"},
            stream_output=True, label="verifier",
        )
        data = safe_json(result.text)
        if not data:
            return {"status": "ok", "reason": "", "correction": ""}
        return data

    def interpret(self, task: str, module_contents: list[tuple[str, str]]) -> GenerationResult:
        user = task
        if module_contents:
            blocks: list[str] = [
                "O usuário carregou os módulos abaixo. Baseie-se neles.", "",
            ]
            for label, content in module_contents:
                blocks.append(f"=== MÓDULO: {label} ===")
                blocks.append(content)
                blocks.append(f"=== FIM DO MÓDULO: {label} ===")
                blocks.append("")
            blocks.append("=== PERGUNTA DO USUÁRIO ===")
            blocks.append(task)
            user = "\n".join(blocks)

        if self.ctx and self.ctx.would_overflow(user):
            user = self.ctx.fit([user], priorities=[1])[0]

        max_tok = token_budget_for("interpreter", 3)
        return self.model.generate(
            system=INTERPRETER_SYSTEM, user=user,
            max_tokens=max_tok, temperature=TEMPERATURE_INTERPRETER,
            stream_output=True, label="interpreter",
        )


# ============================================================
# HELPERS
# ============================================================

def annotate_code_with_marker(code: str, failing_line: Optional[int]) -> str:
    if not code:
        return code
    lines = code.splitlines()
    out: list[str] = []
    for idx, line in enumerate(lines, start=1):
        marker = ">>>" if idx == failing_line else "   "
        out.append(f"{marker} {idx:>4} | {line}")
    return "\n".join(out)


_LUNA_REMOVED_PREFIX = "# LUNA-REMOVED"


def comment_out_lines(code: str, line_numbers: set[int]) -> str:
    if not line_numbers:
        return code
    lines = code.splitlines()
    out: list[str] = []
    for idx, line in enumerate(lines, start=1):
        if idx in line_numbers:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}{_LUNA_REMOVED_PREFIX} {line.strip()}")
        else:
            out.append(line)
    return "\n".join(out)


def find_line_numbers_of_text(code: str, needle: str) -> set[int]:
    hits: set[int] = set()
    if not needle:
        return hits
    needle_clean = needle.strip()
    if not needle_clean:
        return hits
    for idx, line in enumerate(code.splitlines(), start=1):
        if needle_clean in line.strip():
            hits.add(idx)
    return hits


def strip_luna_markers(code: str) -> str:
    lines = [
        line for line in code.splitlines()
        if _LUNA_REMOVED_PREFIX not in line
    ]
    return "\n".join(lines)


def strip_savefig_lines(code: str) -> str:
    out: list[str] = []
    for line in code.splitlines():
        if "savefig" in line.lower() and not line.strip().startswith("#"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}# LUNA: savefig removido")
        else:
            out.append(line)
    return "\n".join(out)


# ============================================================
# ANÁLISE ESTÁTICA
# ============================================================

def _call_dotted(node: ast.AST) -> str:
    parts: list[str] = []
    cur: Any = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def static_analyze(code: str, deps: Optional[DependencyManager] = None) -> AnalysisResult:
    result = AnalysisResult(ok=True)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result.ok = False
        result.errors.append(f"SyntaxError linha {exc.lineno}: {exc.msg}")
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            result.imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "input":
                result.uses_input = True
            full = _call_dotted(node.func)
            if full:
                lowered = full.lower()
                if any(k in lowered for k in (
                    "requests.", "urllib.", "httpx.", "aiohttp.", "socket.",
                )):
                    result.uses_network = True
                if any(k in lowered for k in (
                    "subprocess.", "os.system", "os.popen",
                )):
                    result.uses_subprocess = True
                if lowered.endswith("savefig"):
                    result.uses_savefig = True
                    if node.args and isinstance(node.args[0], ast.Constant):
                        result.saves_figures.append(str(node.args[0].value))

    if "plt.show(" in code or ".show()" in code:
        result.uses_matplotlib_show = True
    if result.uses_matplotlib_show:
        result.opens_gui = True

    for m in re.finditer(r'https?://[^\s"\'<>)]+', code):
        result.api_endpoints.append(m.group(0))

    if result.uses_input and AUTO_FILL_INPUT:
        result.warnings.append("input() detectado — stdin será preenchido.")

    anti = detect_antipatterns(code)
    if anti:
        result.antipatterns = anti
        for where, reason, suggestion in anti:
            result.warnings.append(f"ANTI-PADRÃO [{where}]: {reason} → {suggestion}")

    if deps is not None:
        try:
            dep_infos = deps.analyze(code)
            result.missing_packages = [
                d.import_name for d in dep_infos if not d.is_installed
            ]
        except Exception:
            pass

    return result


# ============================================================
# EXECUÇÃO
# ============================================================

def _build_env(headless: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["LUNA_WORKSPACE"] = str(WORKSPACE)
    env["LUNA_MOLS"] = str(MOLS_DIR)

    parts = [str(BASE_DIR), str(WORKSPACE)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    if headless:
        env["MPLBACKEND"] = "Agg"
    else:
        env.pop("MPLBACKEND", None)
    return env


def _popen_kwargs(headless: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if headless and sys.platform.startswith("win"):
        try:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    return kwargs


def execute_python(
    code: str,
    stdin_data: Optional[str] = None,
    timeout: Optional[int] = EXECUTION_TIMEOUT,
    headless: bool = True,
) -> ExecutionResult:

    started = time.perf_counter()
    env = _build_env(headless=headless)

    if not headless and timeout == EXECUTION_TIMEOUT:
        timeout = INTERACTIVE_TIMEOUT

    use_pipe = stdin_data is not None
    stdin_arg = subprocess.PIPE if use_pipe else subprocess.DEVNULL
    mode = "headless" if headless else "interactive"

    try:
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=str(WORKSPACE),
            stdin=stdin_arg,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            env=env, **_popen_kwargs(headless),
        )
    except Exception as exc:
        return ExecutionResult(
            success=False, returncode=-1, stdout="",
            stderr=f"Falha ao iniciar processo:\n{type(exc).__name__}: {exc}",
            elapsed=time.perf_counter() - started,
            stdin_used=stdin_data or "",
            mode=mode,
        )

    try:
        try:
            stdout, stderr = process.communicate(
                input=stdin_data if use_pipe else None,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                stdout, stderr = process.communicate()
            except Exception:
                stdout, stderr = "", ""
            return ExecutionResult(
                success=False, returncode=-1,
                stdout=stdout or "",
                stderr=(stderr or "") + f"\nLUNA: timeout ({timeout}s).",
                elapsed=time.perf_counter() - started,
                timed_out=True,
                stdin_used=stdin_data or "",
                mode=mode,
            )
        return ExecutionResult(
            success=process.returncode == 0,
            returncode=process.returncode,
            stdout=stdout or "", stderr=stderr or "",
            elapsed=time.perf_counter() - started,
            stdin_used=stdin_data or "",
            mode=mode,
        )
    except KeyboardInterrupt:
        try:
            process.terminate()
        except Exception:
            pass
        try:
            process.wait(timeout=3)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        raise


def execute_probe(probe: str, kind: str, timeout: int = PROBE_TIMEOUT) -> ExecutionResult:
    """Sempre via python -c, independente de kind."""
    started = time.perf_counter()
    env = _build_env(headless=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(WORKSPACE), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout, env=env, **_popen_kwargs(headless=True),
        )
        return ExecutionResult(
            success=proc.returncode == 0, returncode=proc.returncode,
            stdout=proc.stdout or "", stderr=proc.stderr or "",
            elapsed=time.perf_counter() - started,
            mode="headless",
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False, returncode=-1, stdout="",
            stderr=f"probe timeout ({timeout}s)",
            elapsed=time.perf_counter() - started, timed_out=True,
            mode="headless",
        )
    except Exception as exc:
        return ExecutionResult(
            success=False, returncode=-1, stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
            elapsed=time.perf_counter() - started,
            mode="headless",
        )


def should_run_interactive(
    code: str,
    intent: Intent,
    analysis: AnalysisResult,
) -> bool:
    if analysis.uses_matplotlib_show and not analysis.uses_savefig:
        return True
    if intent.wants_visual and analysis.opens_gui and not analysis.uses_savefig:
        return True
    if analysis.uses_input and not AUTO_FILL_INPUT:
        return True
    return False


# ============================================================
# WORKSPACE
# ============================================================

def snapshot_workspace() -> dict[str, float]:
    snap: dict[str, float] = {}
    for path in WORKSPACE.rglob("*"):
        if path.is_file():
            try:
                snap[str(path.relative_to(WORKSPACE))] = path.stat().st_mtime
            except OSError:
                pass
    return snap


def changed_files(before: dict[str, float]) -> list[str]:
    changed: list[str] = []
    for path in WORKSPACE.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative = str(path.relative_to(WORKSPACE))
            mtime = path.stat().st_mtime
            if relative not in before or mtime != before[relative]:
                changed.append(relative)
        except OSError:
            continue
    return sorted(changed)


def extract_expected_files(task: str) -> list[str]:
    found: set[str] = set()
    patterns = [
        r"(?:salve|salvar|grave|gravar|exporte|exportar|escreva|escrever)"
        r"\s+(?:como\s+|em\s+)?[\"']?([A-Za-z0-9_.-]+\.[A-Za-z0-9]+)",
        r"(?:crie|criar|gere|gerar)"
        r"\s+(?:o\s+arquivo\s+|um\s+arquivo\s+|a\s+imagem\s+|o\s+gr[aá]fico\s+)"
        r"[\"']?([A-Za-z0-9_.-]+\.[A-Za-z0-9]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, task, flags=re.IGNORECASE):
            found.add(match.group(1))
    return sorted(found)


def verify_expected_files(task: str) -> list[str]:
    missing: list[str] = []
    for filename in extract_expected_files(task):
        if not (WORKSPACE / filename).exists():
            missing.append(filename)
    return missing


def verify_figure_files(saved_figures: list[str]) -> list[str]:
    problems: list[str] = []
    for name in saved_figures:
        path = WORKSPACE / name
        if not path.exists():
            problems.append(f"{name} não foi salvo")
            continue
        try:
            size = path.stat().st_size
        except OSError:
            problems.append(f"{name} erro ao ler")
            continue
        if size < 512:
            problems.append(f"{name} tem {size} bytes (suspeito)")
    return problems


def choose_output_path(task: str) -> Optional[Path]:
    expected = extract_expected_files(task)
    if expected:
        return WORKSPACE / expected[0]
    return None


def save_program(code: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code, encoding="utf-8")


# ============================================================
# UMA TENTATIVA
# ============================================================

def run_attempt(
    code: str,
    stdin_data: Optional[str],
    headless: bool = True,
    deps: Optional[DependencyManager] = None,
) -> AttemptResult:

    code = normalize_code(code)
    fp = code_fingerprint(code)

    if not code:
        analysis = AnalysisResult(ok=False, errors=["Código vazio."])
        execution = ExecutionResult(
            success=False, returncode=-1, stdout="",
            stderr="Código vazio.", elapsed=0.0,
            mode="headless" if headless else "interactive",
        )
        return AttemptResult(
            code=code, execution=execution, analysis=analysis, fingerprint=fp,
        )

    if len(code) > MAX_CODE_CHARS:
        analysis = AnalysisResult(ok=False, errors=["Código excedeu o limite."])
        execution = ExecutionResult(
            success=False, returncode=-1, stdout="",
            stderr="Código grande demais.", elapsed=0.0,
            mode="headless" if headless else "interactive",
        )
        return AttemptResult(
            code=code, execution=execution, analysis=analysis, fingerprint=fp,
        )

    analysis = static_analyze(code, deps)

    if not analysis.ok:
        execution = ExecutionResult(
            success=False, returncode=-2, stdout="",
            stderr="\n".join(analysis.errors), elapsed=0.0,
            stdin_used=stdin_data or "",
            mode="headless" if headless else "interactive",
        )
        tb = parse_traceback(code, execution.stderr)
        return AttemptResult(
            code=code, execution=execution, analysis=analysis,
            traceback_info=tb, fingerprint=fp,
        )

    execution = execute_python(code, stdin_data=stdin_data, headless=headless)
    tb = parse_traceback(code, execution.stderr)
    return AttemptResult(
        code=code, execution=execution, analysis=analysis,
        traceback_info=tb, fingerprint=fp,
    )


# ============================================================
# OFERTA INTERATIVA
# ============================================================

def offer_interactive_session(code: str) -> None:
    if not OFFER_INTERACTIVE:
        return

    analysis = static_analyze(code)
    has_input = analysis.uses_input
    has_gui = analysis.opens_gui
    has_show = analysis.uses_matplotlib_show
    has_savefig = analysis.uses_savefig

    if has_show and not has_savefig:
        return
    if not has_input and not has_gui:
        return

    print()
    prompt = (
        "Executar novamente de forma interativa? [s/N]: "
        if has_input
        else "Executar novamente mostrando a janela? [s/N]: "
    )
    try:
        answer = input(Fore.WHITE + prompt + Style.RESET_ALL).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if answer not in {"s", "sim", "y", "yes"}:
        return

    log("Executando interativamente. Ctrl+C encerra.")
    try:
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(WORKSPACE),
        )
    except KeyboardInterrupt:
        print()
        warning("Execução interativa interrompida.")
    except Exception as exc:
        error(f"Falha na execução interativa: {exc}")


# ============================================================
# PIPELINE — INTERPRETAÇÃO
# ============================================================

def run_interpretation(
    task: str, model: LunaModel, logger: LunaLogger, intent: Intent,
) -> bool:
    stage("Modo interpretação")
    if intent.module_contents:
        dim(f"   módulos em contexto: {len(intent.module_contents)}")
    agent = LunaAgent(model, logger)
    gen = agent.interpret(task, intent.module_contents)
    logger.log("interpretation", {
        "elapsed": gen.elapsed, "tokens": gen.completion_tokens,
    })
    print()
    success(f"Interpretação concluída em {gen.elapsed:.2f}s.")
    return True


# ============================================================
# PIPELINE — QUICK
# ============================================================

def run_quick(
    task: str, model: LunaModel, logger: LunaLogger, quick_kind: str = "python",
) -> bool:
    agent = LunaAgent(model, logger)
    stage("Modo rápido (python one-liner)")
    quick = agent.generate_quick(task)
    if not quick:
        warning("Fallback para delivery.")
        return run_delivery(
            task, model, logger,
            Intent(wants_delivery=True, wants_interpretation=False,
                   read_requested=False, module_names=[]),
        )

    dim(f"   cmd: {quick.command[:240]}")
    if quick.explanation:
        dim(f"   why: {quick.explanation}")

    stage("Executando comando (python -c)...")
    exec_result = execute_probe(quick.command, kind="python", timeout=SHELL_TIMEOUT)

    logger.log("quick_execution", {
        "command": quick.command,
        "returncode": exec_result.returncode,
        "stdout": truncate(exec_result.stdout, 8000),
        "stderr": truncate(exec_result.stderr, 8000),
    })

    if exec_result.stdout.strip():
        print()
        print(Fore.WHITE + exec_result.stdout.strip() + Style.RESET_ALL)
    if exec_result.stderr.strip():
        print()
        print(Fore.YELLOW + exec_result.stderr.strip() + Style.RESET_ALL)

    print()
    if exec_result.success:
        success(f"Comando concluído em {exec_result.elapsed:.2f}s.")
        return True

    error(f"Comando falhou (returncode={exec_result.returncode}).")

    # Auto-fix e retry
    stage("Tentando corrigir o comando...")
    fix_prompt = f"""
O comando Python abaixo falhou:

COMANDO:
{quick.command}

ERRO:
{truncate(exec_result.stderr or exec_result.stdout, 1500)}

TAREFA ORIGINAL:
{task}

Gere um NOVO comando python one-liner CORRIGIDO.

REGRAS:
- Use requests para HTTP, NUNCA curl.
- Toda requisição com timeout.
- Retorne SOMENTE JSON: {{"command": "...", "explanation": "..."}}
""".strip()

    try:
        result = model.generate(
            system="Você corrige comandos Python one-liner. Retorne JSON.",
            user=fix_prompt,
            max_tokens=300,
            temperature=0.05,
            response_format={"type": "json_object"},
            stream_output=True,
            label="quick-fix",
        )
        data = safe_json(result.text)
        if data and data.get("command"):
            new_cmd = data["command"].strip()
            dim(f"   novo comando: {new_cmd[:200]}")
            stage("Re-executando (python -c)...")
            new_result = execute_probe(new_cmd, kind="python", timeout=SHELL_TIMEOUT)
            if new_result.stdout.strip():
                print()
                print(Fore.WHITE + new_result.stdout.strip() + Style.RESET_ALL)
            if new_result.stderr.strip():
                print(Fore.YELLOW + new_result.stderr.strip() + Style.RESET_ALL)
            print()
            if new_result.success:
                success(f"Corrigido em {new_result.elapsed:.2f}s.")
                return True
            error("Ainda falhou após correção.")
    except Exception as exc:
        warning(f"Falha ao corrigir: {exc}")

    return False


# ============================================================
# PIPELINE — DELIVERY
# ============================================================

def run_delivery(
    task: str, model: LunaModel, logger: LunaLogger, intent: Intent,
) -> bool:

    agent = LunaAgent(model, logger)
    state = SessionState()

    # ---- API DISCOVERY + TESTS ----
    api_endpoints: list[ApiEndpoint] = []
    api_test_hints = ""
    doc_hints = ""

    if intent.involves_api and intent.api_urls:
        stage("Analisando APIs mencionadas...")

        for url in intent.api_urls[:3]:
            # 1. Tenta descobrir spec
            if agent.api_discovery:
                spec = agent.api_discovery.discover(url, state)
                if spec:
                    success(f"OpenAPI spec encontrada: {spec.url} v{spec.version}")
                    base = spec.base_url or url.rstrip("/")
                    for ep in spec.endpoints:
                        ep_url = ep.url if ep.url.startswith("http") else base + ep.url
                        api_endpoints.append(ApiEndpoint(
                            url=ep_url, method=ep.method,
                            parameters=ep.parameters,
                            discovered_via="openapi",
                        ))

                    # Testa alguns endpoints
                    test_results = agent.api_discovery.test_endpoints(
                        base, spec.endpoints[:5], state,
                    )
                    lines = []
                    for tr in test_results:
                        status = tr.status_code if tr.status_code else "?"
                        lines.append(
                            f"[{tr.method}] {tr.url} → {status}, "
                            f"sucesso={tr.success}, {tr.elapsed:.2f}s"
                        )
                        if tr.response_body and tr.success:
                            lines.append(f"  amostra: {truncate(tr.response_body, 300)}")
                    api_test_hints = "\n".join(lines)

            # 2. Testa a URL direta
            dim(f"   testando URL base...")
            test_result = agent.http.request(url, cache=state.api_cache)

            if test_result.success:
                status = test_result.status_code or "?"
                dim(f"   ✓ URL respondeu {status} em {test_result.elapsed:.2f}s")
                api_endpoints.append(ApiEndpoint(
                    url=url, method="GET",
                    sample_response=truncate(test_result.response_body, 500),
                    discovered_via="direct_test",
                ))
                api_test_hints += (
                    f"\n[GET] {url} → {status}, sucesso=True, "
                    f"{test_result.elapsed:.2f}s\n"
                    f"  amostra: {truncate(test_result.response_body, 500)}"
                )
            else:
                dim(f"   ✗ URL falhou: {test_result.error[:150]}")

    # ---- BUILD AUGMENTED TASK ----
    if intent.module_contents or api_endpoints or api_test_hints or doc_hints:
        task = build_augmented_task(
            task, intent.module_contents,
            api_endpoints, api_test_hints, doc_hints,
        )

    log("Analisando tarefa...")
    route = agent.route(task)
    logger.log("route", route)

    complexity = int(route.get("complexity", 3))
    needs_execution = bool(route.get("needs_execution", True))
    needs_planning = bool(
        route.get("needs_planning", complexity >= PLANNER_COMPLEXITY_THRESHOLD)
    )
    task_type = str(route.get("task_type", "python"))
    is_quick = bool(route.get("is_quick", False))
    quick_kind = str(route.get("quick_kind", "python"))
    wants_visual = bool(route.get("wants_visual", False)) or intent.wants_visual
    self_questioning = bool(route.get("self_questioning", False))
    is_api_task = bool(route.get("is_api_task", False)) or intent.involves_api

    needs_verification = bool(route.get("needs_verification", False))
    if task_type in {"visualization", "network"} or is_api_task:
        needs_verification = True

    log(
        f"Complexidade: {complexity}/5 | "
        f"tipo={task_type} | "
        f"planner={'on' if needs_planning else 'off'} | "
        f"execução={'on' if needs_execution else 'off'} | "
        f"verifier={'on' if needs_verification else 'off'} | "
        f"quick={'on' if is_quick else 'off'} | "
        f"api={'on' if is_api_task else 'off'}"
    )

    # ---- QUICK MODE (mas SÓ se não for API complexa) ----
    if is_quick and complexity <= 2 and not api_endpoints:
        stage("Router sugeriu modo rápido. Redirecionando...")
        return run_quick(task, model, logger, quick_kind=quick_kind)

    # ---- SELF-QUESTIONING ----
    if self_questioning:
        stage("Auto-questionamento...")
        q_result = model.generate(
            system=(
                "Você é a Luna. Antes de gerar código, formule 2-3 perguntas "
                "que ajudariam a resolver ambiguidades. Se não houver "
                "ambiguidade, responda 'Nada a questionar'."
            ),
            user=task,
            max_tokens=250,
            temperature=0.10,
            stream_output=True,
            label="self-questioning",
        )
        logger.log("self_questioning", {"raw": q_result.text})
        dim(f"   {truncate(q_result.text, 300)}")

    # ---- ARCHITECT ----
    plan: Optional[TaskPlan] = None
    if ENABLE_PLANNER and needs_planning:
        stage("Arquitetando solução...")
        plan = agent.architect(task, complexity)
        logger.log("architect_plan", {
            "goal": plan.goal,
            "requirements": plan.requirements,
            "checks": plan.checks,
        })
        success("Plano definido.")
        if plan.requirements:
            for req in plan.requirements:
                dim(f"   • {req}")

    before = snapshot_workspace()

    extra_hints = ""
    if intent.forbids_files:
        extra_hints += (
            "\n- Usuário pediu para NÃO salvar arquivos. Use só plt.show()."
        )
    if wants_visual:
        extra_hints += "\n- Usuário quer VER o resultado. Use plt.show()."
    if is_api_task:
        extra_hints += (
            "\n- TAREFA DE API: SEMPRE use `requests` para HTTP. "
            "NUNCA use curl/wget/subprocess. Toda requisição com timeout."
        )
    if api_endpoints:
        extra_hints += (
            "\n- ENDPOINTS testados e funcionais estão listados acima. "
            "Use exatamente esses endpoints."
        )
    if api_test_hints:
        extra_hints += "\n- Os TESTES acima mostram o que funciona. Use isso."
    if doc_hints:
        extra_hints += "\n- As DOCS acima mostram o uso correto. Use isso."

    # ---- CODER ----
    stage("Gerando código (streaming)...")
    generated = agent.generate_code(task, plan, complexity, hints=extra_hints)
    code = normalize_code(generated.text)

    if generated.truncated:
        warning("Geração pode ter sido truncada.")

    logger.log("initial_generation", {
        "elapsed": generated.elapsed,
        "tokens": generated.completion_tokens,
        "tps": generated.tokens_per_second,
    })
    log(
        f"Gerado em {generated.elapsed:.2f}s | "
        f"{generated.completion_tokens} tokens | "
        f"{generated.tokens_per_second:.1f} tok/s"
    )

    if intent.forbids_files and "savefig" in code.lower():
        warning("Usuário pediu sem arquivos — removendo savefig.")
        code = strip_savefig_lines(code)

    # ---- DEPENDENCY RESOLUTION ----
    if agent.deps and needs_execution:
        ok_deps, installed, _ = agent.deps.ensure(code, state)
        if not ok_deps:
            warning("Dependências não resolvidas. Tentando mesmo assim...")

    update_loop_state(code, state)

    # ================================================
    # LOOP PRINCIPAL
    # ================================================

    while True:
        state.attempt_count += 1
        attempt_number = state.attempt_count

        stage(f"Tentativa {attempt_number}")

        if attempt_number > MAX_ATTEMPTS:
            error(f"Limite duro de {MAX_ATTEMPTS} tentativas atingido.")
            logger.log("abort_max_attempts", {"attempt": attempt_number})
            return False

        # ---- Loop detection ----
        loop_signal = detect_loop(code, state)
        if loop_signal:
            warning(
                f"Loop detectado ({loop_signal.kind}, "
                f"{loop_signal.severity}): {loop_signal.evidence}"
            )
            logger.log("loop_detected", {
                "kind": loop_signal.kind,
                "similarity": loop_signal.similarity,
                "evidence": loop_signal.evidence,
            })

        # ---- Analysis + mode decision ----
        analysis = static_analyze(code, agent.deps)
        headless_mode = not should_run_interactive(code, intent, analysis)

        if not headless_mode:
            stage("Modo GUI — a janela do matplotlib será aberta.")
        else:
            dim("   modo headless (Agg)")

        # ---- stdin ----
        stdin_values: list[str] = []
        if plan and plan.test_inputs:
            stdin_values = list(plan.test_inputs)
        elif AUTO_FILL_INPUT and headless_mode:
            prompts = extract_input_prompts(code)
            if prompts:
                stdin_values = [synthesize_input_value(p) for p in prompts]
                dim(f"   input() auto-preenchido: {stdin_values}")

        stdin_payload = build_stdin_payload(stdin_values)

        # ---- EXECUTION ----
        if needs_execution:
            stage("Executando código...")
            attempt = run_attempt(code, stdin_payload, headless=headless_mode, deps=agent.deps)
        else:
            attempt = AttemptResult(
                code=code,
                execution=ExecutionResult(
                    success=True, returncode=0,
                    stdout="", stderr="", elapsed=0.0,
                    mode="headless" if headless_mode else "interactive",
                ),
                analysis=analysis,
                fingerprint=code_fingerprint(code),
            )

        execution = attempt.execution
        state.total_tokens_generated += generated.completion_tokens

        logger.log(f"attempt_{attempt_number:02d}_execution", {
            "returncode": execution.returncode,
            "success": execution.success,
            "elapsed": execution.elapsed,
            "timed_out": execution.timed_out,
            "mode": execution.mode,
            "stdin_used": execution.stdin_used,
            "stdout_tail": truncate(execution.stdout, 4000),
            "stderr_tail": truncate(execution.stderr, 4000),
        })

        # ---- RESULT ----
        if execution.success:
            success(f"Execução OK em {execution.elapsed:.2f}s.")
            if execution.stdout.strip():
                print(Fore.WHITE + execution.stdout.strip() + Style.RESET_ALL)
        else:
            error(f"Falhou (returncode={execution.returncode}).")
            if attempt.traceback_info and not attempt.traceback_info.is_empty:
                print(Fore.RED + describe_traceback(attempt.traceback_info) + Style.RESET_ALL)
            elif execution.stderr.strip():
                print(Fore.RED + execution.stderr.strip() + Style.RESET_ALL)

        # ---- Objective checks ----
        if execution.success and not intent.forbids_files:
            missing_files = verify_expected_files(task)
            if missing_files:
                warning("Arquivos esperados ausentes: " + ", ".join(missing_files))
                execution = ExecutionResult(
                    success=False, returncode=-3,
                    stdout=execution.stdout,
                    stderr="Arquivos esperados ausentes: " + ", ".join(missing_files),
                    elapsed=execution.elapsed,
                    stdin_used=execution.stdin_used,
                    mode=execution.mode,
                )

        if execution.success and attempt.analysis.saves_figures:
            problems = verify_figure_files(attempt.analysis.saves_figures)
            if problems:
                warning("Figuras com problema: " + "; ".join(problems))
                execution = ExecutionResult(
                    success=False, returncode=-4,
                    stdout=execution.stdout,
                    stderr="Figuras com problema: " + "; ".join(problems),
                    elapsed=execution.elapsed,
                    stdin_used=execution.stdin_used,
                    mode=execution.mode,
                )

        # ---- SUCCESS ----
        if execution.success:
            # Reviewer (semi-formal)
            certificate: Optional[ReasoningCertificate] = None
            if (
                ENABLE_REVIEWER
                and agent.reviewer
                and complexity >= REVIEWER_COMPLEXITY_THRESHOLD
            ):
                stage("Revisão semi-formal...")
                certificate = agent.reviewer.review(task, code, complexity)
                attempt.certificate = certificate
                state.review_history.append(certificate)
                logger.log("reviewer_certificate", {
                    "confidence": certificate.confidence,
                    "conclusion": certificate.conclusion,
                })
                dim(f"   confiança: {certificate.confidence}")
                dim(f"   conclusão: {truncate(certificate.conclusion, 200)}")

                if (
                    certificate.confidence == "low"
                    and certificate.counter_evidence
                    and attempt_number < MAX_ATTEMPTS
                ):
                    warning("Revisor com baixa confiança + contra-evidência. Reparando...")
                    execution = ExecutionResult(
                        success=False, returncode=-6,
                        stdout=execution.stdout,
                        stderr=(
                            f"REVISOR SEMI-FORMAL: baixa confiança. "
                            f"Contra-evidência: {certificate.counter_evidence}"
                        ),
                        elapsed=execution.elapsed,
                        stdin_used=execution.stdin_used,
                        mode=execution.mode,
                    )

            # Verifier (external evidence)
            if execution.success and ENABLE_VERIFIER and needs_verification:
                stage("Verificação externa...")
                expected_files = extract_expected_files(task)
                verification = agent.verify(task, code, execution, plan, expected_files)
                logger.log(f"verification_{attempt_number:02d}", verification)

                status = str(verification.get("status", "ok")).lower()
                if status == "retry":
                    warning("Verificador pediu correção.")
                    reason = str(verification.get("reason", ""))
                    correction = str(verification.get("correction", ""))
                    logger.log(f"verification_retry_{attempt_number:02d}", {
                        "reason": reason, "correction": correction,
                    })
                    execution = ExecutionResult(
                        success=False, returncode=-5,
                        stdout=execution.stdout,
                        stderr=f"VERIFICADOR: {reason}\n{correction}",
                        elapsed=execution.elapsed,
                        stdin_used=execution.stdin_used,
                        mode=execution.mode,
                    )

            # Final success
            if execution.success:
                files = changed_files(before)
                if files:
                    success("Arquivos alterados: " + ", ".join(files))
                output_path = choose_output_path(task)
                if output_path:
                    save_program(code, output_path)
                    success(f"Programa salvo em: {output_path}")
                else:
                    dim("   (nenhum arquivo salvo)")

                logger.log("success", {
                    "output_path": str(output_path) if output_path else None,
                    "changed_files": files,
                    "mode": execution.mode,
                    "attempts": attempt_number,
                    "total_tokens": state.total_tokens_generated,
                    "installed_packages": state.installed_packages,
                })
                success(f"Tarefa concluída em {attempt_number} tentativa(s).")
                offer_interactive_session(code)
                return True

        # ---- FAILURE → INVESTIGATION → REPAIR ----
        history_entry = {
            "attempt": attempt_number,
            "result": "execution_failed",
            "error": truncate(execution.combined_output, MAX_TRACEBACK_CHARS),
        }

        # Auto-install em ModuleNotFoundError
        if (
            agent.deps
            and not execution.success
            and "ModuleNotFoundError" in (execution.stderr or "")
            and AUTO_INSTALL_DEPS
        ):
            m = re.search(r"No module named '([^']+)'", execution.stderr)
            if m:
                missing_mod = m.group(1)
                spec = agent.deps.resolve_install_spec(missing_mod)
                if spec and spec not in state.installed_packages:
                    stage(f"Auto-instalando '{spec}'...")
                    ok, log_str = agent.deps.install([spec])
                    if ok:
                        success(f"Instalado: {spec}")
                        state.installed_packages.append(spec)
                        logger.write_text(
                            f"attempt_{attempt_number:02d}_autoinstall.py", code,
                        )
                        continue
                    else:
                        warning(f"Falha ao instalar '{spec}'")

        # Test API calls do código que falhou
        if is_api_task and not execution.success and agent.http:
            urls_in_code = re.findall(r'https?://[^\s"\'<>)]+', code)
            if urls_in_code:
                stage("Testando chamadas de API do código com falha...")
                api_lines = []
                for u in urls_in_code[:3]:
                    tr = agent.http.request(u, cache=state.api_cache)
                    status = tr.status_code if tr.status_code else "?"
                    api_lines.append(f"[GET] {u} → {status}, sucesso={tr.success}")
                    if tr.response_body and tr.success:
                        api_lines.append(f"  amostra: {truncate(tr.response_body, 300)}")
                if api_lines:
                    api_test_hints = "\n".join(api_lines)
                    dim(f"   {len(api_lines)} linhas de teste de API coletadas")

        stage("Investigando causa raiz...")
        investigation = agent.investigate(
            task=task, plan=plan, code=code,
            execution=execution, analysis=attempt.analysis,
            traceback_info=attempt.traceback_info,
            history=[
                f"tentativa {h.get('attempt')}: {h.get('result')} → {truncate(str(h.get('error','')), 200)}"
                for h in [history_entry]
            ],
        )

        stage("Análise estruturada (fase 1/2)...")
        diff = ""
        if len(state.fingerprints) > 1:
            diff = f"(fingerprints: {state.fingerprints[-2][:8]} → {state.fingerprints[-1][:8]})"

        diagnosis = agent.debug_analyze(
            task=task, plan=plan, code=code,
            execution=execution, analysis=attempt.analysis,
            traceback_info=attempt.traceback_info,
            history=[history_entry],
            diff_from_previous=diff,
            investigation=investigation,
            loop_signal=loop_signal,
            api_test_hints=api_test_hints,
            doc_hints=doc_hints,
            reviewer_certificate=attempt.certificate,
        )
        logger.log("failure_diagnosis", {
            "type": diagnosis.root_cause_type,
            "cause": diagnosis.root_cause,
            "location": diagnosis.location,
            "strategy": diagnosis.fix_strategy,
            "must_avoid": diagnosis.must_avoid,
            "hypothesis_key": diagnosis.hypothesis_key,
            "suggested_dependencies": diagnosis.suggested_dependencies,
        })

        dim(f"   tipo: {diagnosis.root_cause_type}")
        dim(f"   causa: {diagnosis.root_cause}")
        if diagnosis.must_avoid:
            dim(f"   must_avoid: {diagnosis.must_avoid}")
        if diagnosis.admissible_alternatives:
            dim(f"   alternativas: {diagnosis.admissible_alternatives}")

        # Install suggested deps
        if (
            diagnosis.suggested_dependencies
            and agent.deps
            and AUTO_INSTALL_DEPS
        ):
            to_install = [
                d for d in diagnosis.suggested_dependencies
                if d and d not in state.installed_packages
            ]
            if to_install:
                stage(f"Instalando deps sugeridas: {to_install}")
                ok, _ = agent.deps.install(to_install)
                if ok:
                    state.installed_packages.extend(to_install)

        # Hypothesis tracking
        hkey = diagnosis.hypothesis_key or "unknown"
        state.hypothesis_counts[hkey] = state.hypothesis_counts.get(hkey, 0) + 1

        if state.hypothesis_counts[hkey] >= MAX_HYPOTHESIS_REPEATS:
            warning(
                f"Hipótese '{hkey}' repetida "
                f"{state.hypothesis_counts[hkey]}×. Forçando inversão."
            )
            diagnosis.must_avoid = list(diagnosis.must_avoid) + [
                f"repetir hipótese '{hkey}'",
                "usar a mesma estrutura da tentativa anterior",
                "insistir na mesma API/comando que falhou",
            ]
            diagnosis.fix_strategy = (
                "INVERSÃO FORÇADA: a hipótese anterior está esgotada. "
                "Proponha uma causa raiz fundamentalmente diferente."
            )

        # Surgical vs normal repair
        if loop_signal and loop_signal.severity == "high":
            warning("Aplicando cirurgia forçada por loop detectado.")
            failing_lines: set[int] = set()
            if attempt.traceback_info and attempt.traceback_info.user_frame:
                failing_lines = find_line_numbers_of_text(
                    code, attempt.traceback_info.user_frame.code_text
                )
            if not failing_lines:
                anti = detect_antipatterns(code)
                for where, _, _ in anti:
                    m = re.match(r"linha (\d+):", where)
                    if m:
                        failing_lines.add(int(m.group(1)))

            surgical_code = comment_out_lines(code, failing_lines) if failing_lines else code

            stage("Reparo cirúrgico (fase 2/2)...")
            surgical = agent.repair_surgical(
                task=task, diagnosis=diagnosis,
                code_with_markers=surgical_code,
                traceback_info=attempt.traceback_info,
            )
            new_code = strip_luna_markers(normalize_code(surgical.text))
        else:
            stage("Gerando código corrigido (fase 2/2)...")
            repaired = agent.repair_code(
                task=task, plan=plan,
                diagnosis=diagnosis,
                previous_code=code, execution=execution,
                traceback_info=attempt.traceback_info,
                investigation=investigation,
                diff_from_previous=diff,
                complexity=complexity,
            )
            new_code = normalize_code(repaired.text)

        if intent.forbids_files and "savefig" in new_code.lower():
            new_code = strip_savefig_lines(new_code)

        update_loop_state(new_code, state)
        code = new_code
        logger.write_text(f"attempt_{attempt_number:02d}_repaired.py", code)

    return False


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def run_agent(task: str, model: LunaModel, logger: LunaLogger) -> bool:

    logger.log("task", {"text": task})

    intent = classify_intent_llm(task, model, logger)
    logger.log("intent", {
        "wants_delivery": intent.wants_delivery,
        "wants_interpretation": intent.wants_interpretation,
        "read_requested": intent.read_requested,
        "module_names": intent.module_names,
        "forbids_files": intent.forbids_files,
        "wants_visual": intent.wants_visual,
        "involves_api": intent.involves_api,
        "api_urls": intent.api_urls,
    })

    if intent.read_requested:
        stage(f"Carregando módulos de ./mols: {intent.module_names or '(auto)'}")
        intent.module_contents = load_modules(intent.module_names, logger)
        if not intent.module_contents:
            warning("Nenhum módulo carregado.")

    # Interpretação pura — só se NÃO é entrega
    if intent.wants_interpretation and not intent.wants_delivery:
        return run_interpretation(task, model, logger, intent)

    return run_delivery(task, model, logger, intent)


# ============================================================
# INTERFACE
# ============================================================

def print_banner() -> None:
    print()
    print(Fore.CYAN + "╔══════════════════════════════════════════════╗")
    print(Fore.CYAN + "║             LUNA AI CODER v10                ║")
    print(Fore.CYAN + "║   multi-agent · semi-formal · api-aware      ║")
    print(Fore.CYAN + "║   robust-http · state-of-the-art 2026        ║")
    print(Fore.CYAN + "╚══════════════════════════════════════════════╝")
    print()
    print(f"{Style.DIM}Comandos: sair | exit | quit{Style.RESET_ALL}")
    print(f"{Style.DIM}Workspace: {WORKSPACE}{Style.RESET_ALL}")
    print(f"{Style.DIM}Módulos:   {MOLS_DIR}{Style.RESET_ALL}")
    print(f"{Style.DIM}Logs:      {LOGS_DIR}{Style.RESET_ALL}")
    print()


def main() -> None:

    print_banner()

    logger = LunaLogger(LOGS_DIR)
    logger.log("session_start", {"cwd": str(BASE_DIR)})
    dim(f"Sessão de log: {logger.session_dir}")

    try:
        model = LunaModel()
    except Exception:
        error("Não foi possível carregar o modelo.")
        print(Fore.RED + traceback.format_exc() + Style.RESET_ALL)
        return

    print()
    print("Digite a tarefa para a Luna.")
    print(f"{Style.DIM}Exemplos:{Style.RESET_ALL}")
    print(f"{Style.DIM}  • qual o preço atual do bitcoin?{Style.RESET_ALL}")
    print(f"{Style.DIM}  • descubra o clima em Tokyo via API{Style.RESET_ALL}")
    print(f"{Style.DIM}  • crie o grafico 3d de uma superficie hiperbolica{Style.RESET_ALL}")
    print(f"{Style.DIM}  • leia o módulo utils e explique o que ele faz{Style.RESET_ALL}")
    print()

    while True:
        try:
            task = input(Fore.WHITE + "Você > " + Style.RESET_ALL).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not task:
            continue

        if task.lower() in {"sair", "exit", "quit"}:
            break

        print()
        started = time.perf_counter()

        try:
            ok = run_agent(task, model, logger)
            elapsed = time.perf_counter() - started
            print()
            if ok:
                success(f"Processo completo em {elapsed:.2f}s.")
            else:
                error(f"Luna não conseguiu concluir em {elapsed:.2f}s.")
        except KeyboardInterrupt:
            print()
            warning("Execução interrompida pelo usuário.")
            try:
                logger.log("interrupted_by_user", {})
            except KeyboardInterrupt:
                pass
        except Exception:
            error("Erro interno inesperado.")
            print(Fore.RED + traceback.format_exc() + Style.RESET_ALL)
            try:
                logger.log("internal_error", {"traceback": traceback.format_exc()})
            except KeyboardInterrupt:
                pass

        print()

    try:
        logger.log("session_end", {})
    except KeyboardInterrupt:
        pass
    dim(f"Logs salvos em: {logger.session_dir}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print(f"{Style.DIM}Encerrado.{Style.RESET_ALL}")