"""
LUNA AI CODING AGENT — v6
=========================

Melhorias desta versão:

- MODO GUI: quando o código tem plt.show() e NÃO salva arquivo, a Luna
  roda com backend nativo (TkAgg/Qt) — a janela do matplotlib abre para
  o usuário ver. Sem MPLBACKEND=Agg nesse caso.
- NÃO GERA ARQUIVOS SEM PEDIDO: nada de luna_output.py automático.
  Arquivos só aparecem quando o usuário dá NOME EXPLÍCITO.
- INTENÇÃO RESPEITADA: "sem salvar", "sem png" → remove savefig antes
  de rodar. "me mostre", "quero ver" → força modo GUI.
- OFERTA INTERATIVA INTELIGENTE: só oferece re-execução quando faz
  sentido (input() pendente), não depois de já mostrar o plot.

Mantido do v5: parser de traceback cirúrgico, anti-padrões, reparo em
duas fases, reparo cirúrgico, quick pipeline, streaming total.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
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

WORKSPACE.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MOLS_DIR.mkdir(parents=True, exist_ok=True)


N_CTX = 8192
N_THREADS = max(1, (os.cpu_count() or 4) - 1)
N_BATCH = 512
N_GPU_LAYERS = 0
USE_FLASH_ATTN = False


TEMPERATURE_ROUTER = 0.03
TEMPERATURE_PLANNER = 0.05
TEMPERATURE_CODER = 0.10
TEMPERATURE_QUICK = 0.04
TEMPERATURE_INVESTIGATOR = 0.03
TEMPERATURE_REPAIR_ANALYSIS = 0.04
TEMPERATURE_REPAIR_CODE = 0.08
TEMPERATURE_REPAIR_SURGICAL = 0.04
TEMPERATURE_VERIFIER = 0.02
TEMPERATURE_INTERPRETER = 0.18

TOP_P = 0.92
TOP_K = 40
MIN_P = 0.05
REPEAT_PENALTY = 1.06
SEED = 42

STREAM = True
SHOW_MODEL_STREAM = True


MAX_ROUTER_TOKENS = 260
MAX_PLAN_TOKENS = 700
MAX_CODE_TOKENS = 2000
MAX_QUICK_TOKENS = 280
MAX_INVESTIGATION_TOKENS = 340
MAX_REPAIR_ANALYSIS_TOKENS = 380
MAX_REPAIR_CODE_TOKENS = 2200
MAX_REPAIR_SURGICAL_TOKENS = 1600
MAX_VERIFY_TOKENS = 400
MAX_INTERPRETER_TOKENS = 1500


MAX_ATTEMPTS = 5
EXECUTION_TIMEOUT = 120
INTERACTIVE_TIMEOUT = 3600
PROBE_TIMEOUT = 25
SHELL_TIMEOUT = 30

MAX_CODE_CHARS = 90000
MAX_TRACEBACK_CHARS = 20000
MAX_PREVIOUS_CODE_CHARS = 70000
MAX_MODULE_BYTES = 500_000


ENABLE_PLANNER = True
ENABLE_VERIFIER = True
ENABLE_INVESTIGATOR = True
PLANNER_COMPLEXITY_THRESHOLD = 4

AUTO_FILL_INPUT = True
OFFER_INTERACTIVE = True


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


@dataclass
class Intent:
    wants_delivery: bool
    wants_interpretation: bool
    read_requested: bool
    module_names: list[str]
    forbids_files: bool = False
    wants_visual: bool = False
    module_contents: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class QuickCommand:
    kind: str
    command: str
    explanation: str = ""


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
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


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

_LIBRARY_PATH_HINTS = (
    "site-packages", "lib/python", "lib\\python",
    "matplotlib", "numpy", "scipy", "pandas", "PIL", "cv2",
    "asyncio", "importlib", "threading",
)


def _is_library_path(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    if lowered.startswith("<") and lowered.endswith(">"):
        return False
    for hint in _LIBRARY_PATH_HINTS:
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
        line = lines[i]
        m = _FRAME_RE.search(line)
        if m:
            file_path = m.group("path")
            try:
                lineno = int(m.group("lineno"))
            except (ValueError, TypeError):
                lineno = 0
            func = (m.group("func") or "").strip()
            code_text = ""
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not next_line.startswith("File \"") \
                        and not next_line.startswith("^"):
                    code_text = next_line
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
# ANTI-PADRÃO DETECTOR
# ============================================================

ANTIPATTERN_RULES: list[tuple[str, str, str]] = [
    (
        r"plot_surface\s*\(\s*[^,]+,\s*[^,]+,\s*[0-9]+(?:\.[0-9]+)?\s*[,)]",
        "plot_surface com Z escalar (int/float)",
        "ax.plot_surface(X, Y, np.zeros_like(X), ...)  ou  ax.plot_surface(X, Y, Z, ...)",
    ),
    (
        r"plot_surface\s*\(\s*[^,]+,\s*[^,]+,\s*(?:True|False|None)\s*[,)]",
        "plot_surface com Z booleano/None",
        "Z deve ser um array 2D com o mesmo shape de X e Y",
    ),
    (
        r"if\s+(?:abs\s*\(\s*)?[A-Za-z_]\w*\s*[<>=!]+\s*[0-9]",
        "comparação de array com escalar usando if",
        "use np.where(...) ou np.errstate(...) para evitar ambiguidade",
    ),
    (
        r"np\.nan\s*==\s*|==\s*np\.nan|!=\s*np\.nan",
        "comparação direta com np.nan (sempre False)",
        "use np.isnan(x) para checar NaN",
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
            is_input = isinstance(func, ast.Name) and func.id == "input"
            if not is_input:
                continue
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
# INTENT
# ============================================================

DELIVER_VERBS = {
    "crie", "criar", "cria", "gere", "gerar", "gera", "faça", "fazer",
    "faz", "escreva", "escrever", "produza", "produzir", "implemente",
    "implementar", "construa", "construir", "desenvolva", "desenvolver",
    "corrija", "corrigir", "conserte", "consertar", "arrume", "arrumar",
    "refatore", "refatorar", "otimize", "otimizar", "converta",
    "converter", "transforme", "transformar", "monte", "montar",
    "roda", "rodar", "execute", "executar", "baixe", "baixar",
    "teste", "testar", "valide", "validar", "resolva", "resolver",
    "plote", "plotar", "desenhe", "desenhar",
}

INTERPRET_VERBS = {
    "explique", "explicar", "explica", "comente", "comentar", "analise",
    "analisar", "descreva", "descrever", "resuma", "resumir", "entenda",
    "entender", "interprete", "interpretar", "diga", "fale",
}

INTERPRET_PHRASES = [
    "o que faz", "o que é", "o que e", "como funciona",
    "por que", "porque", "o que acontece", "me diga", "me explique",
    "me fale sobre", "quero entender", "quero saber", "o que significa",
    "para que serve", "qual a diferença",
]

READ_VERBS = {
    "leia", "ler", "lê", "abrir", "abra", "carregue", "carregar",
    "consuma", "consumir", "importe", "importar",
}

MODULE_NOUNS = {
    "módulo", "modulo", "module", "arquivo", "file", "script",
    "lib", "biblioteca", "package", "pacote", "código", "codigo",
}

VISUAL_VERBS = {
    "mostre", "mostrar", "mostra", "exiba", "exibir", "exibe",
    "visualizar", "visualize", "apresente", "apresentar",
    "veja", "vejamos",
}

VISUAL_PHRASES = [
    "me mostre", "quero ver", "quero que apareça", "mostre o resultado",
    "exiba o resultado", "mostre o grafico", "mostre o gráfico",
    "quero visualizar", "abrir o grafico", "abrir o gráfico",
]

FORBID_FILE_PHRASES = [
    "sem salvar", "sem arquivo", "sem arquivos", "sem png",
    "sem imagem", "sem imagens", "sem gerar arquivo", "sem criar arquivo",
    "não salve", "nao salve", "não salvar", "nao salvar",
    "não crie arquivo", "nao crie arquivo",
    "não crie arquivos", "nao crie arquivos",
    "não grave", "nao grave", "não exporte", "nao exporte",
    "não salve arquivos", "nao salve arquivos",
    "no save", "don't save", "dont save",
    "sem salvar arquivo", "sem salvar arquivos",
    "não gerar arquivo", "nao gerar arquivo",
    "não gerar arquivos", "nao gerar arquivos",
]


def analyze_intent(task: str) -> Intent:
    lower = task.lower()
    words = re.findall(r"\b[a-zà-ÿ0-9_]+\b", lower)
    tokens = set(words)

    wants_delivery = bool(tokens & DELIVER_VERBS) or any(
        v in lower for v in (
            "preciso de", "quero um", "quero uma", "faça um", "faca um",
            "faça uma", "faca uma", "me dê", "me de", "me dê um",
        )
    )
    wants_interpretation = bool(tokens & INTERPRET_VERBS) or any(
        p in lower for p in INTERPRET_PHRASES
    )
    read_requested = bool(tokens & READ_VERBS) and bool(tokens & MODULE_NOUNS)
    wants_visual = bool(tokens & VISUAL_VERBS) or any(
        p in lower for p in VISUAL_PHRASES
    )
    forbids_files = any(p in lower for p in FORBID_FILE_PHRASES)

    module_names: list[str] = []
    if read_requested:
        module_names = extract_module_names(task)

    if not wants_delivery and not wants_interpretation:
        wants_delivery = True

    return Intent(
        wants_delivery=wants_delivery,
        wants_interpretation=wants_interpretation,
        read_requested=read_requested,
        module_names=module_names,
        forbids_files=forbids_files,
        wants_visual=wants_visual,
    )


MODULE_NAME_RE = re.compile(
    r"(?:m[oó]dulo|module|arquivo|file|script|lib|biblioteca|package|pacote|c[oó]digo)"
    r"\s+[\"'\`]?([A-Za-z0-9_][A-Za-z0-9_.\-/]*)[\"'\`]?",
    re.IGNORECASE,
)

FILENAME_RE = re.compile(
    r"\b([A-Za-z0-9_][A-Za-z0-9_-]*"
    r"(?:\.(?:py|pyw|js|mjs|ts|tsx|jsx|rb|go|rs|java|kt|kts|c|cc|cpp|h|hpp|cs|swift|php|lua|sh|bash|ps1|zsh|fish|pl|r|jl|scala|clj|cljs|ex|exs|erl|hs|ml|nim|zig|v|dart|json|txt|md|rst|yaml|yml|toml|cfg|ini|conf|env|lock|sql|html|htm|xml|csv|tsv|log|svg|tex|bib))?)\b"
)

_STOPWORDS = {
    "o", "a", "os", "as", "um", "uma", "de", "do", "da", "em",
    "no", "na", "que", "com", "para", "por", "e", "ou", "se",
    "me", "te", "lhe", "nos", "vos", "lhes",
}


def extract_module_names(task: str) -> list[str]:
    names: list[str] = []
    for m in MODULE_NAME_RE.finditer(task):
        names.append(m.group(1))
    for m in FILENAME_RE.finditer(task):
        candidate = m.group(1)
        if "." in candidate or (
            len(candidate) <= 40 and candidate.lower() not in _STOPWORDS
        ):
            names.append(candidate)
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        stripped = name.strip().strip(".,;:!?\"'`")
        if not stripped:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(stripped)
    return out


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
    if target_stem:
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
            logger.log("module_loaded", {
                "name": name, "path": str(path), "size": len(content),
            })
    return loaded


def build_augmented_task(task: str, module_contents: list[tuple[str, str]]) -> str:
    if not module_contents:
        return task
    parts: list[str] = [
        "A tarefa abaixo refere-se a módulos carregados do diretório ./mols.",
        "Use o conteúdo integral dos módulos como fonte de verdade.", "",
    ]
    for label, content in module_contents:
        parts.append(f"=== MÓDULO: {label} ===")
        parts.append(content)
        parts.append(f"=== FIM DO MÓDULO: {label} ===")
        parts.append("")
    parts.append("=== SOLICITAÇÃO DO USUÁRIO ===")
    parts.append(task)
    return "\n".join(parts)


# ============================================================
# MODELO
# ============================================================

class LunaModel:

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo não encontrado:\n{MODEL_PATH}")
        log("Carregando modelo...")
        kwargs: dict[str, Any] = {
            "model_path": str(MODEL_PATH),
            "n_ctx": N_CTX,
            "n_threads": N_THREADS,
            "n_batch": N_BATCH,
            "n_gpu_layers": N_GPU_LAYERS,
            "verbose": False,
            "use_mmap": True,
            "seed": SEED,
        }
        if USE_FLASH_ATTN:
            kwargs["flash_attn"] = True
        self.llm = Llama(**kwargs)
        success("Modelo carregado.")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:
            return 0

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

        return GenerationResult(
            text=text, elapsed=elapsed,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens, tokens_per_second=tps,
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
  "wants_visual": false
}

complexity: 1 (trivial) a 5 (muito complexa).
task_type: python | coding | file | math | network | visualization | general.
expects_files: true SOMENTE se o usuário pediu EXPLICITAMENTE para
  gravar um arquivo com NOME ("salve em X", "crie o arquivo Y.png").
  Se o usuário apenas disse "crie um gráfico" ou "gere uma imagem" sem
  dizer nome de arquivo, expects_files = false.
is_quick: true se resolve com UM comando.
quick_kind: "python" ou "shell". Só faz sentido se is_quick=true.
needs_network: true se acessa rede.
wants_visual: true se o usuário quer VER o resultado.

Responda SOMENTE com JSON.
""".strip()


PLANNER_SYSTEM = """
Você é o planejador técnico da Luna. Retorne SOMENTE JSON:

{
  "goal": "...",
  "requirements": ["..."],
  "checks": ["..."],
  "test_inputs": ["..."],
  "expected_outputs": ["..."],
  "interactive": false
}

NÃO invente requisitos que o usuário não pediu.
""".strip()


CODER_SYSTEM = """
Você é o engenheiro principal da Luna.

Escreva código Python 3 COMPLETO e EXECUTÁVEL.

═══════════════════════════════════════════════════════════════
REGRA CRÍTICA — NÃO SALVAR ARQUIVOS SEM PEDIDO
═══════════════════════════════════════════════════════════════
- Se o usuário pedir um gráfico/plot SEM mencionar salvar arquivo,
  use SOMENTE `plt.show()`. NÃO chame `plt.savefig()`.
- Só use `plt.savefig('nome.png')` se o usuário pediu explicitamente
  um nome de arquivo, OU se disse "salve", "grave", "exporte".
- Se o usuário disse "sem salvar", "sem png", "não salve" — proibido
  usar savefig.
- Não crie arquivos .py, .txt, .json extras. Não crie logs.
- Se o usuário não pediu arquivo, seu código NÃO deve gravar nada
  em disco.

═══════════════════════════════════════════════════════════════
REGRAS GERAIS
═══════════════════════════════════════════════════════════════
1. Retorne SOMENTE código Python puro. Sem markdown, sem ```.
2. Código completo (todos os imports incluídos).
3. Não dependa de variáveis externas.
4. Nunca peça confirmação. Execute o que foi pedido.
5. O programa DEVE executar do início ao fim sem intervenção.
6. Não explique o código. Apenas o código.

═══════════════════════════════════════════════════════════════
REGRAS CRÍTICAS — MATPLOTLIB 3D
═══════════════════════════════════════════════════════════════
- ax.plot_surface(X, Y, Z, ...) exige X, Y, Z arrays 2D com mesmo shape.

  ❌ ERRADO:   ax.plot_surface(X, Y, 0)              # Z é int
  ❌ ERRADO:   ax.plot_surface(X, Y, 1, alpha=0.3)   # Z é int
  ❌ ERRADO:   ax.plot_surface(X, Y, 5.0)            # Z é float
  ❌ ERRADO:   ax.plot_surface(X, Y, True)           # Z é bool

  ✅ CORRETO:  ax.plot_surface(X, Y, Z)                              # Z é array
  ✅ CORRETO:  ax.plot_surface(X, Y, np.zeros_like(X))               # plano z=0
  ✅ CORRETO:  ax.plot_surface(X, Y, np.ones_like(X) * 5.0)          # plano z=5
  ✅ CORRETO:  ax.plot_surface(X, Y, np.full_like(X, 3.14))          # plano z=3.14

- NUNCA use `if <array> ...:` — use np.where / np.errstate.
- NUNCA use `if abs(denom) < eps` onde denom é array.
- Para evitar divisão por zero em arrays:
      Z = np.where(np.abs(denom) < 1e-9, np.nan, numer / denom)

═══════════════════════════════════════════════════════════════
REGRAS CRÍTICAS — NUMPY
═══════════════════════════════════════════════════════════════
- Comparação com NaN: use np.isnan(x), NUNCA `x == np.nan`.
- Filtragem: np.where, np.ma.masked_where, indexação booleana.

═══════════════════════════════════════════════════════════════
QUALIDADE
═══════════════════════════════════════════════════════════════
- Não adicione complexidade que o usuário não pediu.
- Se a tarefa é gerar um gráfico, apenas gere o gráfico.
- Prefira soluções curtas e corretas.
""".strip()


QUICK_SYSTEM = """
Você é o executor rápido da Luna. Tarefa TRIVIAL — resolva com UM
comando único. Retorne SOMENTE JSON:

{
  "kind": "python",
  "command": "...",
  "explanation": "..."
}

kind = "python": code para `python -c "..."`.
kind = "shell":  comando shell (curl, ls, cat, grep, etc.).

Regras:
- UMA linha. Sem ```.
- Sem instalar nada. Sem alterar arquivos. Sem subprocess.
""".strip()


INVESTIGATOR_SYSTEM = """
Você é o investigador da Luna.

A execução falhou. Analise o traceback ESTRUTURADO. Se o erro aponta
CLARAMENTE para o código do usuário (linha + exceção + código), NÃO
peça probe. Se o erro é ambíguo (ImportError, rede, permissão,
arquivo ausente), PEÇA UM probe.

Retorne SOMENTE JSON:

{
  "need_probe": false,
  "reasoning": "...",
  "kind": "python",
  "probe": ""
}

Erros autoexplicativos (NÃO precisam de probe):
  AttributeError, TypeError, ValueError, KeyError, IndexError,
  NameError, ZeroDivisionError, UnboundLocalError, SyntaxError.

Erros ambíguos (PODEM precisar de probe):
  ImportError, ModuleNotFoundError, ConnectionError,
  FileNotFoundError, PermissionError, timeout.

UM probe. Rápido. Determinístico. Sem alterar arquivos.
""".strip()


REPAIR_ANALYSIS_SYSTEM = """
Você é o analista de causa raiz da Luna.

Retorne SOMENTE JSON:

{
  "root_cause": "explicação técnica precisa",
  "where": "linha N do SEU código (não da biblioteca)",
  "fix_strategy": "como consertar",
  "must_avoid": ["padrão 1", "padrão 2"],
  "alternative_approach": "opcional"
}

REGRA CRÍTICA:
- A causa raiz está no SEU código, não dentro da biblioteca.
  Se a exceção foi levantada dentro de matplotlib/numpy, o problema
  está na CHAMADA que o SEU código fez.

- must_avoid: padrões CONCRETOS que NÃO devem reaparecer.
  Ex.: "ax.plot_surface(X, Y, 0)", "if abs(array) < eps".
""".strip()


REPAIR_CODE_SYSTEM = """
Você é o engenheiro de reparo da Luna.

Recebe análise de causa raiz e código que falhou.
Gere o código Python COMPLETO e CORRIGIDO.

REGRAS ABSOLUTAS:
1. Retorne SOMENTE código Python. Sem markdown.
2. Corrija EXATAMENTE a causa raiz.
3. NÃO use NENHUMA construção listada em must_avoid.
4. Nunca use `if <array> ...:`. Nunca compare com np.nan usando ==.
5. Código executável do início ao fim.
6. Mantenha o mesmo comportamento — só corrija o que está errado.
7. Se o usuário pediu "sem salvar" e o código tinha savefig,
   remova o savefig e use SOMENTE plt.show().
""".strip()


REPAIR_SURGICAL_SYSTEM = """
Você é o engenheiro de reparo cirúrgico da Luna.

O modelo repetiu o código anterior. As linhas culpadas foram
comentadas com `# LUNA-REMOVED`.

Sua tarefa: substituir essas linhas pelas versões CORRETAS.

REGRAS ABSOLUTAS:
1. Retorne SOMENTE o código Python completo. Sem markdown.
2. Cada linha `# LUNA-REMOVED` DEVE ser substituída por código correto.
   Não pode sobrar NENHUM `# LUNA-REMOVED` no output.
3. Mantenha o resto do código intacto.
""".strip()


VERIFIER_SYSTEM = """
Você é o verificador final. Analise se a tarefa foi concluída.

Retorne SOMENTE JSON:
{
  "status": "ok",
  "reason": "...",
  "correction": ""
}

status: "ok" ou "retry".

Use "retry" SOMENTE com evidência clara:
- stdout vazio quando deveria ter saída
- exceção visível
- arquivo esperado ausente
- saída contradiz o esperado

NUNCA exija melhorias que o usuário não pediu.

Nota: se modo=interactive e returncode=0, o usuário JÁ viu o resultado
(janela abriu). status=ok é o correto.
""".strip()


INTERPRETER_SYSTEM = """
Você é a Luna em modo interpretação. O usuário quer ENTENDER algo.

Regras:
- Resposta clara, direta, técnica.
- Se houver módulo carregado, baseie-se EXCLUSIVAMENTE nele.
- Não invente trechos que não estão no módulo.
- Não gere arquivos. Não execute nada.
""".strip()


# ============================================================
# AGENTE
# ============================================================

class LunaAgent:

    def __init__(self, model: LunaModel, logger: LunaLogger) -> None:
        self.model = model
        self.logger = logger

    def route(self, task: str) -> dict[str, Any]:
        result = self.model.generate(
            system=ROUTER_SYSTEM, user=task,
            max_tokens=MAX_ROUTER_TOKENS, temperature=TEMPERATURE_ROUTER,
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
            }
        complexity = int(data.get("complexity", 3))
        data["complexity"] = max(1, min(5, complexity))
        return data

    def plan(self, task: str) -> TaskPlan:
        result = self.model.generate(
            system=PLANNER_SYSTEM, user=task,
            max_tokens=MAX_PLAN_TOKENS, temperature=TEMPERATURE_PLANNER,
            response_format={"type": "json_object"},
            stream_output=True, label="planner",
        )
        self.logger.log("planner_response", {"raw": result.text})
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
            probe_strategy=str(data.get("probe_strategy", "") or ""),
        )

    def generate_code(
        self, task: str, plan: Optional[TaskPlan], hints: str = "",
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

        return self.model.generate(
            system=CODER_SYSTEM, user=user,
            max_tokens=MAX_CODE_TOKENS, temperature=TEMPERATURE_CODER,
            stream_output=True, label="coder",
        )

    def generate_quick(self, task: str) -> Optional[QuickCommand]:
        result = self.model.generate(
            system=QUICK_SYSTEM, user=task,
            max_tokens=MAX_QUICK_TOKENS, temperature=TEMPERATURE_QUICK,
            response_format={"type": "json_object"},
            stream_output=True, label="quick",
        )
        self.logger.log("quick_response", {"raw": result.text})
        data = safe_json(result.text)
        if not data:
            return None
        kind = str(data.get("kind", "python")).lower()
        if kind not in {"python", "shell"}:
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
            "SyntaxError", "UnicodeDecodeError", "UnicodeEncodeError",
        }
        if etype in obvious and info.user_frame:
            return False
        return True

    def investigate(
        self, task: str, plan: Optional[TaskPlan], code: str,
        execution: ExecutionResult, analysis: AnalysisResult,
        traceback_info: Optional[TracebackInfo],
        history: list[dict[str, Any]],
    ) -> str:

        if not ENABLE_INVESTIGATOR:
            return ""
        if not self.should_investigate(traceback_info):
            dim("   investigador: traceback autoexplicativo, pulando probe.")
            return ""

        history_snippet = "\n".join(
            f"- tentativa {h.get('attempt')}: {h.get('result')} → "
            f"{truncate(str(h.get('error','')), 300)}"
            for h in history[-3:]
        )
        tb_desc = describe_traceback(traceback_info) if traceback_info else "(sem traceback)"

        user = f"""
OBJETIVO:
{task}

TRACEBACK ESTRUTURADO:
{tb_desc}

STDERR (bruto):
{truncate(execution.stderr, 3000)}

RETURN CODE: {execution.returncode}
TIMED OUT:   {execution.timed_out}

HISTÓRICO:
{history_snippet}

Decida se precisa de probe.
""".strip()

        result = self.model.generate(
            system=INVESTIGATOR_SYSTEM, user=user,
            max_tokens=MAX_INVESTIGATION_TOKENS,
            temperature=TEMPERATURE_INVESTIGATOR,
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

        kind = str(data.get("kind", "python")).lower()
        if kind not in {"python", "shell"}:
            kind = "python"
        probe = str(data.get("probe", "")).strip()
        reasoning = str(data.get("reasoning", "")).strip()
        if not probe:
            return ""

        stage(f"Investigação — {reasoning or 'probe'}")
        dim(f"   [{kind}] {probe[:260]}")

        probe_exec = execute_probe(probe, kind=kind, timeout=PROBE_TIMEOUT)
        self.logger.log("investigation_probe", {
            "kind": kind, "probe": probe, "reasoning": reasoning,
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
            f"PROBE ({kind}): {probe}\n"
            f"RAZÃO: {reasoning}\n"
            f"RETURN: {probe_exec.returncode}\n"
            f"STDOUT:\n{truncate(probe_exec.stdout, 4000)}\n"
            f"STDERR:\n{truncate(probe_exec.stderr, 4000)}\n"
        )

    def repair_analyze(
        self, task: str, plan: Optional[TaskPlan], code: str,
        execution: ExecutionResult, analysis: AnalysisResult,
        traceback_info: Optional[TracebackInfo],
        history: list[dict[str, Any]],
        diff_from_previous: str, investigation: str,
    ) -> dict[str, Any]:

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

        user = f"""
OBJETIVO:
{task}

==================================================
CÓDIGO QUE FALHOU (linhas do usuário, com marcador)
==================================================
{truncate(annotated_code, MAX_PREVIOUS_CODE_CHARS)}

==================================================
TRACEBACK ESTRUTURADO
==================================================
{tb_desc}

==================================================
STDOUT
==================================================
{truncate(execution.stdout, 4000)}

==================================================
STDERR
==================================================
{truncate(execution.stderr, MAX_TRACEBACK_CHARS)}

==================================================
ANÁLISE ESTÁTICA
==================================================
erros: {json.dumps(analysis.errors, ensure_ascii=False)}
warnings: {json.dumps(analysis.warnings, ensure_ascii=False)}

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

Produza a análise de causa raiz.
- a causa raiz está no CÓDIGO DO USUÁRIO (linha marcada com >>>),
  NÃO na biblioteca.
- must_avoid deve conter PADRÕES CONCRETOS.
""".strip()

        result = self.model.generate(
            system=REPAIR_ANALYSIS_SYSTEM, user=user,
            max_tokens=MAX_REPAIR_ANALYSIS_TOKENS,
            temperature=TEMPERATURE_REPAIR_ANALYSIS,
            response_format={"type": "json_object"},
            stream_output=True, label="repair·analysis",
        )
        self.logger.log("repair_analysis", {"raw": result.text})
        data = safe_json(result.text)
        if not data:
            return {
                "root_cause": "(análise indisponível)",
                "where": "", "fix_strategy": "", "must_avoid": [],
                "alternative_approach": "",
            }
        return data

    def repair_code(
        self, task: str, plan: Optional[TaskPlan],
        analysis_json: dict[str, Any], previous_code: str,
        execution: ExecutionResult,
        traceback_info: Optional[TracebackInfo],
        investigation: str, diff_from_previous: str,
    ) -> GenerationResult:

        must_avoid = analysis_json.get("must_avoid") or []
        must_avoid_block = (
            "\n".join(f"  ❌ {x}" for x in must_avoid) if must_avoid else "  (nada)"
        )
        tb_desc = describe_traceback(traceback_info) if traceback_info else "(sem traceback)"
        investigation_block = investigation.strip() or "(sem investigação)"
        diff_block = diff_from_previous.strip() or "(sem diff)"

        annotated_code = annotate_code_with_marker(
            previous_code,
            traceback_info.user_frame.line if (traceback_info and traceback_info.user_frame) else None,
        )

        user = f"""
OBJETIVO:
{task}

==================================================
ANÁLISE DE CAUSA RAIZ
==================================================
root_cause: {analysis_json.get('root_cause', '')}
where:      {analysis_json.get('where', '')}
fix_strategy: {analysis_json.get('fix_strategy', '')}
alternative_approach: {analysis_json.get('alternative_approach', '')}

MUST_AVOID (NÃO use NENHUMA destas construções):
{must_avoid_block}

==================================================
TRACEBACK (linha do usuário marcada com >>>)
==================================================
{tb_desc}

==================================================
CÓDIGO QUE FALHOU (marcador >>> na linha culpada)
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
- Sem markdown. Sem explicações.
- Corrija a linha marcada com >>>.
- Não use must_avoid.
""".strip()

        return self.model.generate(
            system=REPAIR_CODE_SYSTEM, user=user,
            max_tokens=MAX_REPAIR_CODE_TOKENS,
            temperature=TEMPERATURE_REPAIR_CODE,
            stream_output=True, label="repair·code",
        )

    def repair_surgical(
        self, task: str, analysis_json: dict[str, Any],
        code_with_markers: str,
        traceback_info: Optional[TracebackInfo],
    ) -> GenerationResult:

        tb_desc = describe_traceback(traceback_info) if traceback_info else "(sem traceback)"
        must_avoid = analysis_json.get("must_avoid") or []
        must_avoid_block = "\n".join(f"  ❌ {x}" for x in must_avoid) or "  (nada)"

        user = f"""
OBJETIVO:
{task}

CAUSA RAIZ:
{analysis_json.get('root_cause', '')}

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

        return self.model.generate(
            system=REPAIR_SURGICAL_SYSTEM, user=user,
            max_tokens=MAX_REPAIR_SURGICAL_TOKENS,
            temperature=TEMPERATURE_REPAIR_SURGICAL,
            stream_output=True, label="repair·surgical",
        )

    def verify(
        self, task: str, code: str,
        execution: ExecutionResult,
        plan: Optional[TaskPlan],
    ) -> dict[str, Any]:

        checks: list[str] = []
        expected: list[str] = []
        if plan:
            checks = plan.checks
            expected = plan.expected_outputs

        user = f"""
TAREFA:
{task}

CHECKS:
{json.dumps(checks, ensure_ascii=False)}

SAÍDAS ESPERADAS:
{json.dumps(expected, ensure_ascii=False)}

MODO DE EXECUÇÃO: {execution.mode}
RETURNCODE: {execution.returncode}
TIMED OUT:  {execution.timed_out}

STDOUT:
{truncate(execution.stdout, 8000)}

STDERR:
{truncate(execution.stderr, 6000)}

Nota: se modo=interactive e a execução terminou com sucesso, o usuário
já viu o resultado (janela foi aberta e fechada). status=ok é o correto.
""".strip()

        result = self.model.generate(
            system=VERIFIER_SYSTEM, user=user,
            max_tokens=MAX_VERIFY_TOKENS, temperature=TEMPERATURE_VERIFIER,
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

        return self.model.generate(
            system=INTERPRETER_SYSTEM, user=user,
            max_tokens=MAX_INTERPRETER_TOKENS,
            temperature=TEMPERATURE_INTERPRETER,
            stream_output=True, label="interpreter",
        )


# ============================================================
# HELPERS DE CÓDIGO
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
            out.append(
                f"{indent}# LUNA: savefig removido (usuário pediu sem arquivos)"
            )
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


def static_analyze(code: str) -> AnalysisResult:
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

    if result.uses_input and AUTO_FILL_INPUT:
        result.warnings.append("input() detectado — stdin será preenchido.")

    anti = detect_antipatterns(code)
    if anti:
        for where, reason, suggestion in anti:
            result.warnings.append(f"ANTI-PADRÃO [{where}]: {reason}")
            result.warnings.append(f"    → use: {suggestion}")

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
    started = time.perf_counter()
    env = _build_env(headless=True)
    try:
        if kind == "python":
            proc = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=str(WORKSPACE), capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=timeout, env=env, **_popen_kwargs(headless=True),
            )
        else:
            proc = subprocess.run(
                probe, shell=True,
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


def execute_shell(command: str, timeout: int = SHELL_TIMEOUT) -> ExecutionResult:
    return execute_probe(command, kind="shell", timeout=timeout)


def should_run_interactive(
    code: str,
    intent: Intent,
    analysis: AnalysisResult,
) -> bool:
    """
    Decide se a execução deve ser GUI (interactive) ou headless.

    Critérios para GUI:
    - usa plt.show() e NÃO usa savefig  → usuário quer ver
    - intent.wants_visual e código tem GUI
    - usa input() e AUTO_FILL_INPUT=False (usuário digita)
    """
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
            problems.append(f"{name} existe mas não pôde ser lido")
            continue
        if size < 512:
            problems.append(f"{name} tem apenas {size} bytes (suspeito)")
    return problems


def choose_output_path(task: str) -> Optional[Path]:
    """
    Só devolve um caminho se o usuário pediu arquivo com nome.
    Caso contrário, NÃO salvamos nada.
    """
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
) -> AttemptResult:

    code = normalize_code(code)

    if not code:
        analysis = AnalysisResult(ok=False, errors=["Código vazio."])
        execution = ExecutionResult(
            success=False, returncode=-1, stdout="",
            stderr="Código vazio.", elapsed=0.0,
            mode="headless" if headless else "interactive",
        )
        return AttemptResult(code=code, execution=execution, analysis=analysis)

    if len(code) > MAX_CODE_CHARS:
        analysis = AnalysisResult(ok=False, errors=["Código excedeu o limite."])
        execution = ExecutionResult(
            success=False, returncode=-1, stdout="",
            stderr="Código grande demais.", elapsed=0.0,
            mode="headless" if headless else "interactive",
        )
        return AttemptResult(code=code, execution=execution, analysis=analysis)

    analysis = static_analyze(code)

    if not analysis.ok:
        execution = ExecutionResult(
            success=False, returncode=-2, stdout="",
            stderr="\n".join(analysis.errors), elapsed=0.0,
            stdin_used=stdin_data or "",
            mode="headless" if headless else "interactive",
        )
        tb = parse_traceback(code, execution.stderr)
        return AttemptResult(
            code=code, execution=execution, analysis=analysis, traceback_info=tb,
        )

    execution = execute_python(code, stdin_data=stdin_data, headless=headless)
    tb = parse_traceback(code, execution.stderr)
    return AttemptResult(
        code=code, execution=execution, analysis=analysis, traceback_info=tb,
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

    # Se já mostramos a GUI na execução principal, não oferece de novo
    if has_show and not has_savefig:
        return
    if not has_input and not has_gui:
        return

    print()
    if has_input:
        prompt = "Executar novamente de forma interativa? [s/N]: "
    else:
        prompt = "Executar novamente mostrando a janela? [s/N]: "

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
    stage("Modo rápido (one-liner)")
    quick = agent.generate_quick(task)
    if not quick:
        warning("Fallback para delivery.")
        return run_delivery(
            task, model, logger,
            Intent(wants_delivery=True, wants_interpretation=False,
                   read_requested=False, module_names=[]),
        )

    dim(f"   kind={quick.kind}")
    dim(f"   cmd:  {quick.command[:240]}")
    if quick.explanation:
        dim(f"   why:  {quick.explanation}")

    stage("Executando comando...")
    if quick.kind == "python":
        exec_result = execute_probe(quick.command, kind="python", timeout=SHELL_TIMEOUT)
    else:
        exec_result = execute_shell(quick.command, timeout=SHELL_TIMEOUT)

    logger.log("quick_execution", {
        "kind": quick.kind, "command": quick.command,
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
    return False


# ============================================================
# PIPELINE — DELIVERY
# ============================================================

def run_delivery(
    task: str, model: LunaModel, logger: LunaLogger, intent: Intent,
) -> bool:

    agent = LunaAgent(model, logger)

    if intent.module_contents:
        task = build_augmented_task(task, intent.module_contents)

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

    needs_verification = bool(route.get("needs_verification", False))
    if task_type in {"visualization", "network"}:
        needs_verification = True

    log(
        f"Complexidade: {complexity}/5 | "
        f"tipo={task_type} | "
        f"planner={'on' if needs_planning else 'off'} | "
        f"execução={'on' if needs_execution else 'off'} | "
        f"verifier={'on' if needs_verification else 'off'} | "
        f"quick={'on' if is_quick else 'off'} | "
        f"visual={'on' if wants_visual else 'off'}"
    )

    if is_quick and complexity <= 2:
        stage("Router sugeriu modo rápido. Redirecionando...")
        return run_quick(task, model, logger, quick_kind=quick_kind)

    plan: Optional[TaskPlan] = None
    if ENABLE_PLANNER and needs_planning:
        stage("Planejando solução...")
        plan = agent.plan(task)
        logger.log("plan", {
            "goal": plan.goal, "requirements": plan.requirements,
            "checks": plan.checks, "interactive": plan.interactive,
        })
        success("Plano definido.")
        if plan.requirements:
            for req in plan.requirements:
                dim(f"   • {req}")
    else:
        log("Tarefa simples: pulando planejamento.")

    before = snapshot_workspace()

    # Hints adicionais
    extra_hints = ""
    if intent.forbids_files:
        extra_hints += (
            "\n- O usuário pediu explicitamente para NÃO salvar arquivos. "
            "Use SOMENTE plt.show(). Proibido plt.savefig()."
        )
    if wants_visual:
        extra_hints += (
            "\n- O usuário quer VER o resultado. Use plt.show() para "
            "exibir a janela do matplotlib."
        )

    stage("Gerando código (streaming)...")
    generated = agent.generate_code(task, plan, hints=extra_hints)
    code = normalize_code(generated.text)
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
        warning("Usuário pediu sem arquivos — removendo savefig do código.")
        code = strip_savefig_lines(code)

    previous_codes: list[str] = []
    history: list[dict[str, Any]] = []
    last_code_for_diff = ""

    for attempt_number in range(1, MAX_ATTEMPTS + 1):

        stage(f"Tentativa {attempt_number}/{MAX_ATTEMPTS}")

        is_repeat = previous_codes and any(
            code_fingerprint(code) == code_fingerprint(prev) for prev in previous_codes
        )

        if is_repeat:
            warning("Código idêntico a tentativa anterior.")
            if attempt_number >= MAX_ATTEMPTS:
                error("Limite atingido sem produzir variação.")
                return False

            failing_lines: set[int] = set()
            if history:
                last_err = str(history[-1].get("error", ""))
                tb = parse_traceback(code, last_err)
                if tb.user_frame and tb.user_frame.code_text:
                    failing_lines = find_line_numbers_of_text(
                        code, tb.user_frame.code_text
                    )
                if not failing_lines and tb.library_frame and tb.library_frame.code_text:
                    failing_lines = find_line_numbers_of_text(
                        code, tb.library_frame.code_text
                    )
            if not failing_lines:
                anti = detect_antipatterns(code)
                for where, _, _ in anti:
                    m = re.match(r"linha (\d+):", where)
                    if m:
                        failing_lines.add(int(m.group(1)))

            surgical_code = comment_out_lines(code, failing_lines) if failing_lines else code
            if failing_lines:
                dim(f"   cirurgia: comentando linhas {sorted(failing_lines)}")

            last_analysis: dict[str, Any] = {}
            if isinstance(history[-1].get("analysis"), dict):
                last_analysis = history[-1]["analysis"]

            stage("Reparo cirúrgico (forçando reformulação)...")
            surgical = agent.repair_surgical(
                task=task, analysis_json=last_analysis,
                code_with_markers=surgical_code,
                traceback_info=None,
            )
            new_code = strip_luna_markers(normalize_code(surgical.text))

            if code_fingerprint(new_code) in {
                code_fingerprint(p) for p in previous_codes
            }:
                warning("Modelo insistiu no mesmo código. Abortando tentativa.")
                previous_codes.append(code)
                continue

            code = new_code
            logger.write_text(f"attempt_{attempt_number:02d}_surgical.py", code)

        previous_codes.append(code)
        logger.write_text(f"attempt_{attempt_number:02d}.py", code)

        # Análise + decisão de modo (headless vs GUI)
        analysis = static_analyze(code)
        headless_mode = not should_run_interactive(code, intent, analysis)

        if not headless_mode:
            stage("Modo GUI — a janela do matplotlib será aberta.")
            dim("   feche a janela para continuar.")
        else:
            dim("   modo headless (Agg)")

        # stdin
        stdin_values: list[str] = []
        if plan and plan.test_inputs:
            stdin_values = list(plan.test_inputs)
        elif AUTO_FILL_INPUT and headless_mode:
            prompts = extract_input_prompts(code)
            if prompts:
                stdin_values = [synthesize_input_value(p) for p in prompts]
                dim(
                    f"   input() detectado ({len(prompts)} ocorrência(s)); "
                    f"auto-preenchido: {stdin_values}"
                )
        elif analysis.uses_input and not headless_mode:
            dim("   input() detectado — usuário vai digitar na janela.")

        stdin_payload = build_stdin_payload(stdin_values)

        if needs_execution:
            stage("Executando código...")
            attempt = run_attempt(code, stdin_payload, headless=headless_mode)
        else:
            attempt = AttemptResult(
                code=code,
                execution=ExecutionResult(
                    success=True, returncode=0,
                    stdout="", stderr="", elapsed=0.0,
                    mode="headless" if headless_mode else "interactive",
                ),
                analysis=analysis,
            )

        execution = attempt.execution

        logger.log(f"attempt_{attempt_number:02d}_execution", {
            "returncode": execution.returncode,
            "success": execution.success,
            "elapsed": execution.elapsed,
            "timed_out": execution.timed_out,
            "mode": execution.mode,
            "stdin_used": execution.stdin_used,
            "stdout_tail": truncate(execution.stdout, 4000),
            "stderr_tail": truncate(execution.stderr, 4000),
            "traceback": describe_traceback(attempt.traceback_info) if attempt.traceback_info else "",
        })

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

        # Verificação de arquivos — só se usuário NÃO proibiu
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

        if execution.success:
            history.append({
                "attempt": attempt_number,
                "result": "success", "error": "", "analysis": {},
            })

            if ENABLE_VERIFIER and needs_verification:
                stage("Verificação semântica...")
                verification = agent.verify(task, code, execution, plan)
                logger.log(f"verification_{attempt_number:02d}", verification)

                status = str(verification.get("status", "ok")).lower()
                if status == "retry":
                    warning("Verificador pediu correção.")
                    reason = str(verification.get("reason", ""))
                    correction = str(verification.get("correction", ""))
                    history.append({
                        "attempt": attempt_number,
                        "result": "verification_retry",
                        "error": reason + "\n" + correction,
                        "analysis": {},
                    })

                    if attempt_number < MAX_ATTEMPTS:
                        stage("Reparando com base na verificação...")
                        diff = make_diff(last_code_for_diff, code) if last_code_for_diff else ""
                        analysis_json = agent.repair_analyze(
                            task=task, plan=plan, code=code,
                            execution=execution, analysis=attempt.analysis,
                            traceback_info=None, history=history,
                            diff_from_previous=diff,
                            investigation="(verificação semântica)",
                        )
                        repaired = agent.repair_code(
                            task=task, plan=plan,
                            analysis_json=analysis_json,
                            previous_code=code, execution=execution,
                            traceback_info=None,
                            investigation="(verificação semântica)",
                            diff_from_previous=diff,
                        )
                        last_code_for_diff = code
                        code = normalize_code(repaired.text)
                        if intent.forbids_files and "savefig" in code.lower():
                            code = strip_savefig_lines(code)
                        continue
                    else:
                        warning("Limite atingido na verificação.")

            # Sucesso definitivo
            files = changed_files(before)
            if files:
                success("Arquivos alterados: " + ", ".join(files))

            output_path = choose_output_path(task)
            if output_path:
                save_program(code, output_path)
                success(f"Programa salvo em: {output_path}")
            else:
                dim("   (nenhum arquivo salvo — usuário não pediu)")

            logger.log("success", {
                "output_path": str(output_path) if output_path else None,
                "changed_files": files,
                "mode": execution.mode,
            })
            success("Tarefa concluída com sucesso.")

            offer_interactive_session(code)
            return True

        # Falha → investigação → reparo
        history.append({
            "attempt": attempt_number,
            "result": "execution_failed",
            "error": truncate(execution.combined_output, MAX_TRACEBACK_CHARS),
            "analysis": {},
        })

        if attempt_number >= MAX_ATTEMPTS:
            error("Número máximo de tentativas atingido.")
            output_path = choose_output_path(task)
            if output_path:
                save_program(code, output_path)
                warning(f"Último código salvo em: {output_path}")
            return False

        stage("Investigando causa raiz...")
        investigation = agent.investigate(
            task=task, plan=plan, code=code,
            execution=execution, analysis=attempt.analysis,
            traceback_info=attempt.traceback_info,
            history=history,
        )

        stage("Análise de causa raiz (fase 1/2)...")
        diff = make_diff(last_code_for_diff, code) if last_code_for_diff else ""
        analysis_json = agent.repair_analyze(
            task=task, plan=plan, code=code,
            execution=execution, analysis=attempt.analysis,
            traceback_info=attempt.traceback_info,
            history=history,
            diff_from_previous=diff,
            investigation=investigation,
        )
        logger.log("repair_analysis_json", analysis_json)

        if analysis_json.get("root_cause"):
            dim(f"   causa raiz: {analysis_json.get('root_cause')}")
        if analysis_json.get("must_avoid"):
            dim(f"   must_avoid: {analysis_json.get('must_avoid')}")

        history[-1]["analysis"] = analysis_json

        stage("Gerando código corrigido (fase 2/2)...")
        repaired = agent.repair_code(
            task=task, plan=plan,
            analysis_json=analysis_json,
            previous_code=code, execution=execution,
            traceback_info=attempt.traceback_info,
            investigation=investigation,
            diff_from_previous=diff,
        )
        new_code = normalize_code(repaired.text)

        if any(code_fingerprint(new_code) == code_fingerprint(prev)
               for prev in previous_codes):
            warning("Reparo repetiu. Aplicando cirurgia forçada...")

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
            surgical = agent.repair_surgical(
                task=task, analysis_json=analysis_json,
                code_with_markers=surgical_code,
                traceback_info=attempt.traceback_info,
            )
            new_code = strip_luna_markers(normalize_code(surgical.text))

        if intent.forbids_files and "savefig" in new_code.lower():
            new_code = strip_savefig_lines(new_code)

        last_code_for_diff = code
        code = new_code
        logger.write_text(f"attempt_{attempt_number:02d}_repaired.py", code)

    return False


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def run_agent(task: str, model: LunaModel, logger: LunaLogger) -> bool:

    logger.log("task", {"text": task})

    intent = analyze_intent(task)
    logger.log("intent", {
        "wants_delivery": intent.wants_delivery,
        "wants_interpretation": intent.wants_interpretation,
        "read_requested": intent.read_requested,
        "module_names": intent.module_names,
        "forbids_files": intent.forbids_files,
        "wants_visual": intent.wants_visual,
    })

    if intent.read_requested:
        stage(f"Carregando módulos de ./mols: {intent.module_names or '(auto)'}")
        intent.module_contents = load_modules(intent.module_names, logger)
        if not intent.module_contents:
            warning("Nenhum módulo carregado.")

    # Só interpretação pura
    if intent.wants_interpretation and not intent.wants_delivery:
        return run_interpretation(task, model, logger, intent)

    return run_delivery(task, model, logger, intent)


# ============================================================
# INTERFACE
# ============================================================

def print_banner() -> None:
    print()
    print(Fore.CYAN + "╔══════════════════════════════════════════════╗")
    print(Fore.CYAN + "║              LUNA AI CODER v6                ║")
    print(Fore.CYAN + "║   GUI mode · no silent files · streaming     ║")
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
    print(f"{Style.DIM}  • crie o grafico 3d de uma superficie hiperbolica{Style.RESET_ALL}")
    print(f"{Style.DIM}  • crie o grafico 3d de uma superficie hiperbolica sem salvar{Style.RESET_ALL}")
    print(f"{Style.DIM}  • leia o módulo utils e explique o que ele faz{Style.RESET_ALL}")
    print(f"{Style.DIM}  • faça um curl em https://api.github.com/users/octocat{Style.RESET_ALL}")
    print(f"{Style.DIM}  • qual a versão do numpy instalada?{Style.RESET_ALL}")
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