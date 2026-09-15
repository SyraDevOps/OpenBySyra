"""
LUNA AI CODING AGENT — v2
=========================

Fluxo:

    USER
      ↓
    ROUTER  (classifica complexidade/tipo)
      ↓
    PLANNER (goal + requisitos + checks + test_inputs)
      ↓
    CODER   (geração em streaming)
      ↓
    STATIC ANALYSIS
      ↓
    EXECUTOR (com auto-preenchimento de input() quando possível)
      ↓
    TRACEBACK REAL
      ↓
    REPAIR (com diff entre tentativas)
      ↓
    EXECUTOR
      ↓
    VERIFIER (semântico)
      ↓
    SAVE + OFERTA DE EXECUÇÃO INTERATIVA

Todos os passos são registrados em logs/.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
import shutil
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

WORKSPACE.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Modelo
# -----------------------------

N_CTX = 8192
N_THREADS = max(1, (os.cpu_count() or 4) - 1)
N_BATCH = 512
N_GPU_LAYERS = 0
USE_FLASH_ATTN = False


# -----------------------------
# Geração
# -----------------------------

TEMPERATURE = 0.12
TOP_P = 0.90
TOP_K = 40
MIN_P = 0.05
REPEAT_PENALTY = 1.05
SEED = 42

STREAM = True
SHOW_MODEL_STREAM = True


# -----------------------------
# Limites
# -----------------------------

MAX_ROUTER_TOKENS = 220
MAX_PLAN_TOKENS = 700
MAX_CODE_TOKENS = 1800
MAX_REPAIR_TOKENS = 2000
MAX_VERIFY_TOKENS = 320

MAX_ATTEMPTS = 4
EXECUTION_TIMEOUT = 120


# -----------------------------
# Comportamento
# -----------------------------

ENABLE_PLANNER = True
ENABLE_VERIFIER = True
PLANNER_COMPLEXITY_LEVEL = 2

AUTO_FILL_INPUT = True
OFFER_INTERACTIVE = True

MAX_CODE_CHARS = 60000
MAX_TRACEBACK_CHARS = 18000
MAX_PREVIOUS_CODE_CHARS = 60000


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


@dataclass
class TaskPlan:
    goal: str
    requirements: list[str]
    checks: list[str]
    test_inputs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    interactive: bool = False


@dataclass
class AttemptResult:
    code: str
    execution: ExecutionResult
    analysis: AnalysisResult


# ============================================================
# LOGGER
# ============================================================

class LunaLogger:
    """Logger de sessão. Escreve JSONL de eventos e arquivos auxiliares."""

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
        except OSError:
            pass

    def write_text(self, name: str, content: str) -> Path:
        path = self.session_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def write_json(self, name: str, content: Any) -> Path:
        path = self.session_dir / name
        path.write_text(
            json.dumps(self._serialize(content), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


# ============================================================
# TERMINAL
# ============================================================

def log(message: str) -> None:
    print(f"{Fore.CYAN}[LUNA]{Style.RESET_ALL} {message}")


def success(message: str) -> None:
    print(f"{Fore.GREEN}[LUNA] ✓ {message}{Style.RESET_ALL}")


def warning(message: str) -> None:
    print(f"{Fore.YELLOW}[LUNA] ! {message}{Style.RESET_ALL}")


def error(message: str) -> None:
    print(f"{Fore.RED}[LUNA] ✗ {message}{Style.RESET_ALL}")


def stage(message: str) -> None:
    print(f"{Fore.MAGENTA}[LUNA] ◆ {message}{Style.RESET_ALL}")


def dim(message: str) -> None:
    print(f"{Style.DIM}{message}{Style.RESET_ALL}")


# ============================================================
# UTILITÁRIOS
# ============================================================

def clean_text(text: str) -> str:
    return text.replace("\x00", "").strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()

    # Bloco markdown ```python ... ```
    fence = re.search(
        r"```(?:python|py)?\s*\n(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if fence:
        return fence.group(1).strip()

    # Bloco ``` ... ``` sem linguagem
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)

    return text.strip()


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


def truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[... truncado pela Luna ...]"


def normalize_code(code: str) -> str:
    code = strip_code_fences(code)

    # Remove preâmbulos comuns do modelo.
    lines = code.splitlines()
    while lines and re.match(
        r"^\s*(aqui|here|segue|below|abaixo)\b.*:?\s*$",
        lines[0],
        re.IGNORECASE,
    ):
        lines = lines[1:]

    code = "\n".join(lines)

    if code.lower().startswith("python\n"):
        code = code[7:]

    return code.strip()


def code_fingerprint(code: str) -> str:
    return re.sub(r"\s+", "", code)


def make_diff(old: str, new: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile="attempt_anterior.py",
            tofile="attempt_atual.py",
            lineterm="",
            n=2,
        )
    )


# -----------------------------
# input() — detecção e preenchimento
# -----------------------------

def extract_input_prompts(code: str) -> list[str]:
    prompts: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return prompts

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_input = (
                isinstance(func, ast.Name)
                and func.id == "input"
            )
            if not is_input:
                continue

            if node.args and isinstance(node.args[0], ast.Constant):
                prompts.append(str(node.args[0].value))
            else:
                prompts.append("")

    return prompts


def synthesize_input_value(prompt: str) -> str:
    """Gera valor plausível a partir do texto do prompt."""
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
    return "1"


def build_stdin_payload(values: list[str]) -> Optional[str]:
    if not values:
        return None
    return "\n".join(values) + "\n"


# ============================================================
# MODELO
# ============================================================

class LunaModel:

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado:\n{MODEL_PATH}"
            )

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
        temperature: Optional[float] = None,
        response_format: Optional[dict[str, Any]] = None,
        stream_output: Optional[bool] = None,
        label: str = "modelo",
    ) -> GenerationResult:

        if temperature is None:
            temperature = TEMPERATURE
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
            print(f"{Style.DIM}— início de {label} —{Style.RESET_ALL}")

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

        if stream_output:
            print()
            print(f"{Style.DIM}— fim de {label} —{Style.RESET_ALL}")

        text = "".join(parts)

        elapsed = max(time.perf_counter() - started, 1e-6)
        completion_tokens = self.count_tokens(text)
        tps = completion_tokens / elapsed if completion_tokens else 0.0

        return GenerationResult(
            text=text,
            elapsed=elapsed,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_second=tps,
        )


# ============================================================
# PROMPTS
# ============================================================

ROUTER_SYSTEM = """
Você é o router de um agente de programação local chamado Luna.

Classifique a tarefa antes da execução.

Retorne SOMENTE JSON válido no schema:

{
  "complexity": 1,
  "needs_planning": false,
  "needs_execution": true,
  "needs_verification": false,
  "task_type": "python",
  "expects_files": false
}

complexity:
1 = trivial
2 = simples
3 = média
4 = complexa
5 = muito complexa

task_type: python, coding, file, math, visualization, general.

expects_files: true se a tarefa pede para criar/gravar arquivo.

Responda SOMENTE com JSON.
""".strip()


PLANNER_SYSTEM = """
Você é o planejador técnico da Luna.

Transforme a solicitação do usuário em um plano mínimo, preciso e
executável.

Não escreva código.

Retorne SOMENTE JSON válido no schema:

{
  "goal": "...",
  "requirements": ["..."],
  "checks": ["..."],
  "test_inputs": ["..."],
  "expected_outputs": ["..."],
  "interactive": false
}

Regras:
- "requirements" são condições funcionais que o programa deve cumprir.
- "checks" são verificações objetivas após a execução.
- "test_inputs" são valores de stdin, na ordem, caso o programa use input().
  Se o programa não usa input(), deixe vazio.
- "expected_outputs" são strings que devem aparecer no stdout.
- "interactive" = true se o programa é claramente um REPL/jogo/menu
  interativo.
- NÃO invente requisitos que o usuário não pediu.
""".strip()


CODER_SYSTEM = """
Você é o engenheiro de software principal da Luna.

Escreva código Python 3 completo e executável.

REGRAS ABSOLUTAS:

1. Retorne SOMENTE código Python. Sem markdown, sem ```.
2. O código deve ser completo (todos os imports incluídos).
3. Não dependa de variáveis externas.
4. Não use rede nem subprocess.
5. Arquivos devem ser gravados dentro de ./ (workspace).
6. Não invente bibliotecas.
7. Prefira APIs estáveis e atuais (evite deprecated).
8. Para NumPy/Matplotlib, verifique dimensões e shapes.
9. Para gráficos 3D, X/Y/Z devem ter shapes compatíveis.
10. Não use parâmetros experimentais sem necessidade.
11. Prefira soluções simples e robustas.
12. O programa deve realmente executar do início ao fim.
13. Não explique o código.
14. Gere o programa completo, não fragmentos.
15. Se a tarefa envolve input(), aceite entradas padrão de stdin.
16. Se a tarefa envolve arquivo, use caminho relativo simples.

Se houver erro anterior, use o traceback como fonte principal.
Não repita a mesma solução que já falhou.
""".strip()


REPAIR_SYSTEM = """
Você é o engenheiro de reparo da Luna.

Você recebe: objetivo, plano, código, stdout/stderr, traceback,
análise estática, histórico e diff entre tentativas.

Sua tarefa é corrigir a CAUSA RAIZ.

REGRAS:

1. Retorne SOMENTE o código Python completo.
2. Sem markdown, sem explicações.
3. Preserve o objetivo original.
4. Leia o traceback cuidadosamente.
5. Corrija exatamente a API, tipo, shape ou lógica do erro.
6. Não introduza complexidade desnecessária.
7. Não use pip, subprocess, system, rede ou install.
8. Arquivos dentro do workspace.
9. Código executável do início ao fim.
10. Se a tentativa anterior usou uma API incorreta, substitua-a.
11. Não repita o código anterior.
12. Se o diff mostra que você está voltando a uma versão antiga, mude
    de estratégia.
""".strip()


VERIFIER_SYSTEM = """
Você é o verificador final.

Analise se a tarefa foi realmente concluída.

Retorne SOMENTE JSON no schema:

{
  "status": "ok",
  "reason": "...",
  "correction": ""
}

status: "ok" ou "retry".

Use "retry" somente quando houver evidência clara de que a tarefa
não foi concluída (stdout incompatível com o esperado, arquivos
ausentes, exceção visível, saída vazia quando deveria ter saída).
Não exija melhorias que o usuário não pediu.
""".strip()


# ============================================================
# AGENTE
# ============================================================

class LunaAgent:

    def __init__(self, model: LunaModel, logger: LunaLogger) -> None:
        self.model = model
        self.logger = logger

    # -----------------------------
    # Router
    # -----------------------------

    def route(self, task: str) -> dict[str, Any]:
        result = self.model.generate(
            system=ROUTER_SYSTEM,
            user=task,
            max_tokens=MAX_ROUTER_TOKENS,
            temperature=0.05,
            response_format={"type": "json_object"},
            stream_output=False,
            label="router",
        )

        self.logger.log("router_response", {"raw": result.text})

        data = safe_json(result.text)

        if not data:
            warning("Router retornou JSON inválido. Usando fallback.")
            data = {
                "complexity": 3,
                "needs_planning": True,
                "needs_execution": True,
                "needs_verification": True,
                "task_type": "python",
                "expects_files": False,
            }

        complexity = int(data.get("complexity", 3))
        data["complexity"] = max(1, min(5, complexity))
        return data

    # -----------------------------
    # Planner
    # -----------------------------

    def plan(self, task: str) -> TaskPlan:
        result = self.model.generate(
            system=PLANNER_SYSTEM,
            user=task,
            max_tokens=MAX_PLAN_TOKENS,
            temperature=0.05,
            response_format={"type": "json_object"},
            stream_output=False,
            label="planner",
        )

        self.logger.log("planner_response", {"raw": result.text})

        data = safe_json(result.text)
        if not data:
            warning("Planner retornou JSON inválido.")
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
        )

    # -----------------------------
    # Coder
    # -----------------------------

    def generate_code(
        self,
        task: str,
        plan: Optional[TaskPlan],
    ) -> GenerationResult:

        if plan:
            requirements = "\n".join(f"- {x}" for x in plan.requirements) or "- nenhum"
            checks = "\n".join(f"- {x}" for x in plan.checks) or "- executar sem erro"
            test_inputs = (
                json.dumps(plan.test_inputs, ensure_ascii=False)
                if plan.test_inputs else "[]"
            )
            expected = (
                json.dumps(plan.expected_outputs, ensure_ascii=False)
                if plan.expected_outputs else "[]"
            )

            user = f"""
OBJETIVO:
{plan.goal}

REQUISITOS:
{requirements}

CHECKS:
{checks}

TEST_INPUTS (stdin na ordem, se houver input()):
{test_inputs}

EXPECTED_OUTPUTS (devem aparecer no stdout):
{expected}

SOLICITAÇÃO ORIGINAL:
{task}
""".strip()
        else:
            user = f"SOLICITAÇÃO ORIGINAL:\n\n{task}"

        return self.model.generate(
            system=CODER_SYSTEM,
            user=user,
            max_tokens=MAX_CODE_TOKENS,
            temperature=TEMPERATURE,
            label="coder",
        )

    # -----------------------------
    # Repair
    # -----------------------------

    def repair(
        self,
        task: str,
        plan: Optional[TaskPlan],
        previous_code: str,
        execution: ExecutionResult,
        analysis: AnalysisResult,
        history: list[dict[str, Any]],
        diff_from_previous: str = "",
    ) -> GenerationResult:

        history_text: list[str] = []
        for item in history[-4:]:
            history_text.append(
                f"TENTATIVA {item.get('attempt')}:\n"
                f"  resultado: {item.get('result')}\n"
                f"  erro: {truncate(str(item.get('error', '')), 1200)}\n"
            )
        history_block = "\n".join(history_text)

        plan_text = ""
        if plan:
            plan_text = (
                "PLANO:\n"
                f"  objetivo: {plan.goal}\n"
                f"  requisitos: {json.dumps(plan.requirements, ensure_ascii=False)}\n"
                f"  checks: {json.dumps(plan.checks, ensure_ascii=False)}\n"
                f"  test_inputs: {json.dumps(plan.test_inputs, ensure_ascii=False)}\n"
            )

        diff_block = diff_from_previous.strip() or "(sem diff disponível)"

        user = f"""
OBJETIVO ORIGINAL:
{task}

{plan_text}

==================================================
CÓDIGO QUE FALHOU
==================================================
{truncate(previous_code, MAX_PREVIOUS_CODE_CHARS)}

==================================================
STDOUT
==================================================
{truncate(execution.stdout, MAX_TRACEBACK_CHARS)}

==================================================
STDERR
==================================================
{truncate(execution.stderr, MAX_TRACEBACK_CHARS)}

==================================================
ERRO COMPLETO
==================================================
{truncate(execution.combined_output, MAX_TRACEBACK_CHARS)}

==================================================
ANÁLISE ESTÁTICA
==================================================
erros: {json.dumps(analysis.errors, ensure_ascii=False)}
warnings: {json.dumps(analysis.warnings, ensure_ascii=False)}
imports: {json.dumps(analysis.imports, ensure_ascii=False)}
uses_input: {analysis.uses_input}

==================================================
DIFF vs. TENTATIVA ANTERIOR
==================================================
{diff_block}

==================================================
HISTÓRICO
==================================================
{history_block}

==================================================
STDIN USADO
==================================================
{json.dumps(execution.stdin_used or "", ensure_ascii=False)}

==================================================

IMPORTANTE:
- Identifique a causa raiz.
- Não copie o código anterior.
- Gere uma implementação nova e completa.
""".strip()

        return self.model.generate(
            system=REPAIR_SYSTEM,
            user=user,
            max_tokens=MAX_REPAIR_TOKENS,
            temperature=0.08,
            label="repair",
        )

    # -----------------------------
    # Verifier
    # -----------------------------

    def verify(
        self,
        task: str,
        code: str,
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

CHECKS ESPERADOS:
{json.dumps(checks, ensure_ascii=False)}

SAÍDAS ESPERADAS (devem aparecer no stdout):
{json.dumps(expected, ensure_ascii=False)}

CÓDIGO:
{truncate(code, 30000)}

RESULTADO DA EXECUÇÃO:
returncode = {execution.returncode}
timed_out  = {execution.timed_out}

STDOUT:
{truncate(execution.stdout, 8000)}

STDERR:
{truncate(execution.stderr, 8000)}
""".strip()

        result = self.model.generate(
            system=VERIFIER_SYSTEM,
            user=user,
            max_tokens=MAX_VERIFY_TOKENS,
            temperature=0.03,
            response_format={"type": "json_object"},
            stream_output=False,
            label="verifier",
        )

        data = safe_json(result.text)
        if not data:
            return {
                "status": "ok",
                "reason": "Execução terminou sem erro aparente.",
                "correction": "",
            }
        return data


# ============================================================
# ANÁLISE ESTÁTICA
# ============================================================

BLOCKED_IMPORTS = {
    "socket",
    "requests",
    "urllib",
    "http",
    "httpx",
    "ftplib",
    "telnetlib",
    "ctypes",
}

BLOCKED_CALLS = {
    "system",
    "popen",
}

ALLOWED_FILE_EXTENSIONS = {
    ".py", ".txt", ".json", ".csv", ".tsv", ".md",
    ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html",
    ".xml", ".yaml", ".yml", ".log",
}


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
                name = alias.name
                result.imports.append(name)
                root = name.split(".")[0]
                if root in BLOCKED_IMPORTS:
                    result.ok = False
                    result.errors.append(f"Import bloqueado: {name}")

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            result.imports.append(module)
            root = module.split(".")[0]
            if root in BLOCKED_IMPORTS:
                result.ok = False
                result.errors.append(f"Import bloqueado: {module}")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "input":
                result.uses_input = True
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in BLOCKED_CALLS:
                    result.ok = False
                    result.errors.append(
                        f"Chamada bloqueada: {node.func.attr}()"
                    )
            elif isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_CALLS:
                    result.ok = False
                    result.errors.append(
                        f"Chamada bloqueada: {node.func.id}()"
                    )

    lowered = code.lower()

    if "plt.show(" in lowered:
        result.warnings.append(
            "plt.show() detectado; execução depende de GUI."
        )

    if result.uses_input:
        if AUTO_FILL_INPUT:
            result.warnings.append(
                "input() detectado; entradas serão preenchidas automaticamente."
            )
        else:
            result.warnings.append(
                "input() detectado; entradas não serão preenchidas."
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.match(r"^[A-Za-z]:[\\/]", node.value):
                result.warnings.append(
                    f"Caminho absoluto detectado: {node.value}"
                )

    return result


# ============================================================
# EXECUÇÃO
# ============================================================

def execute_python(
    code: str,
    stdin_data: Optional[str] = None,
    timeout: int = EXECUTION_TIMEOUT,
) -> ExecutionResult:

    started = time.perf_counter()

    env = os.environ.copy()
    env["LUNA_WORKSPACE"] = str(WORKSPACE)

    parts_pythonpath = [str(BASE_DIR), str(WORKSPACE)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts_pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts_pythonpath)
    env["PYTHONIOENCODING"] = "utf-8"

    use_pipe = stdin_data is not None
    stdin_arg = subprocess.PIPE if use_pipe else subprocess.DEVNULL

    process: Optional[subprocess.Popen[str]] = None

    try:
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=str(WORKSPACE),
            stdin=stdin_arg,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        try:
            stdout, stderr = process.communicate(
                input=stdin_data if use_pipe else None,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            elapsed = time.perf_counter() - started
            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout=stdout or "",
                stderr=(stderr or "") + "\nLUNA: timeout.",
                elapsed=elapsed,
                timed_out=True,
                stdin_used=stdin_data or "",
            )

        elapsed = time.perf_counter() - started
        return ExecutionResult(
            success=process.returncode == 0,
            returncode=process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            elapsed=elapsed,
            stdin_used=stdin_data or "",
        )

    except Exception as exc:
        elapsed = time.perf_counter() - started
        return ExecutionResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr=f"Falha ao iniciar processo:\n{type(exc).__name__}: {exc}",
            elapsed=elapsed,
            stdin_used=stdin_data or "",
        )


# ============================================================
# WORKSPACE / ARQUIVOS
# ============================================================

def snapshot_workspace() -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for path in WORKSPACE.rglob("*"):
        if path.is_file():
            try:
                snapshot[str(path.relative_to(WORKSPACE))] = path.stat().st_mtime
            except OSError:
                pass
    return snapshot


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
        r"(?:arquivo|ficheiro|salve|salvar|crie|criar|gere|gerar)"
        r"(?:\s+como|\s+em)?\s+[\"']?([A-Za-z0-9_.-]+\.[A-Za-z0-9]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, task, flags=re.IGNORECASE):
            filename = match.group(1)
            suffix = Path(filename).suffix.lower()
            if suffix in ALLOWED_FILE_EXTENSIONS:
                found.add(filename)
    return sorted(found)


def verify_expected_files(task: str) -> list[str]:
    missing: list[str] = []
    for filename in extract_expected_files(task):
        if not (WORKSPACE / filename).exists():
            missing.append(filename)
    return missing


def choose_output_path(task: str) -> Path:
    expected = extract_expected_files(task)
    if expected:
        return WORKSPACE / expected[0]
    return WORKSPACE / "luna_output.py"


def save_program(code: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code, encoding="utf-8")


# ============================================================
# PROTEÇÃO CONTRA REPETIÇÃO
# ============================================================

def detect_repetition(current_code: str, previous_codes: list[str]) -> bool:
    current = code_fingerprint(current_code)
    return any(current == code_fingerprint(prev) for prev in previous_codes)


# ============================================================
# UMA TENTATIVA
# ============================================================

def run_attempt(
    code: str,
    stdin_data: Optional[str],
) -> AttemptResult:

    code = normalize_code(code)

    if not code:
        analysis = AnalysisResult(
            ok=False, errors=["Modelo retornou código vazio."]
        )
        execution = ExecutionResult(
            success=False, returncode=-1, stdout="",
            stderr="Código vazio.", elapsed=0.0,
        )
        return AttemptResult(code=code, execution=execution, analysis=analysis)

    if len(code) > MAX_CODE_CHARS:
        analysis = AnalysisResult(
            ok=False, errors=["Código excedeu o limite de tamanho."]
        )
        execution = ExecutionResult(
            success=False, returncode=-1, stdout="",
            stderr="Código grande demais.", elapsed=0.0,
        )
        return AttemptResult(code=code, execution=execution, analysis=analysis)

    analysis = static_analyze(code)

    if not analysis.ok:
        execution = ExecutionResult(
            success=False,
            returncode=-2,
            stdout="",
            stderr="\n".join(analysis.errors),
            elapsed=0.0,
            stdin_used=stdin_data or "",
        )
        return AttemptResult(code=code, execution=execution, analysis=analysis)

    execution = execute_python(code, stdin_data=stdin_data)
    return AttemptResult(code=code, execution=execution, analysis=analysis)


# ============================================================
# MODO INTERATIVO
# ============================================================

def offer_interactive_session(file_path: Path) -> None:
    """Após sucesso, oferece rodar o programa interativamente."""
    if not OFFER_INTERACTIVE:
        return

    print()
    print(f"{Fore.CYAN}Programa disponível em:{Style.RESET_ALL} {file_path}")

    try:
        answer = input(
            f"{Fore.WHITE}Executar interativamente agora? [s/N]: "
            f"{Style.RESET_ALL}"
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if answer not in {"s", "sim", "y", "yes"}:
        return

    log("Iniciando execução interativa. Ctrl+C encerra o programa.")
    try:
        subprocess.run(
            [sys.executable, str(file_path)],
            cwd=str(file_path.parent),
        )
    except KeyboardInterrupt:
        print()
        warning("Execução interativa interrompida.")
    except Exception as exc:
        error(f"Falha na execução interativa: {exc}")


# ============================================================
# LOOP PRINCIPAL DO AGENTE
# ============================================================

def run_agent(
    task: str,
    model: LunaModel,
    logger: LunaLogger,
) -> bool:

    agent = LunaAgent(model, logger)

    logger.log("task", {"text": task})

    log("Analisando tarefa...")
    route = agent.route(task)
    logger.log("route", route)

    complexity = int(route.get("complexity", 3))
    needs_execution = bool(route.get("needs_execution", True))
    needs_planning = bool(
        route.get("needs_planning", complexity >= PLANNER_COMPLEXITY_LEVEL)
    )
    needs_verification = bool(
        route.get("needs_verification", complexity >= 2)
    )

    log(f"Complexidade: {complexity}/5  |  "
        f"planner={'on' if needs_planning else 'off'}  |  "
        f"execução={'on' if needs_execution else 'off'}  |  "
        f"verifier={'on' if needs_verification else 'off'}")

    plan: Optional[TaskPlan] = None

    if ENABLE_PLANNER and needs_planning:
        stage("Planejando solução...")
        plan = agent.plan(task)
        logger.log("plan", {
            "goal": plan.goal,
            "requirements": plan.requirements,
            "checks": plan.checks,
            "test_inputs": plan.test_inputs,
            "expected_outputs": plan.expected_outputs,
            "interactive": plan.interactive,
        })
        success("Plano definido.")
        if plan.requirements:
            for req in plan.requirements:
                dim(f"   • {req}")
        if plan.test_inputs:
            dim(f"   · test_inputs: {plan.test_inputs}")
    else:
        log("Tarefa simples: pulando planejamento intermediário.")

    before = snapshot_workspace()

    # ------------------------------------------------
    # Geração inicial
    # ------------------------------------------------

    stage("Gerando código (streaming)...")
    generated = agent.generate_code(task, plan)
    code = normalize_code(generated.text)
    logger.log("initial_generation", {
        "elapsed": generated.elapsed,
        "tokens": generated.completion_tokens,
        "tps": generated.tokens_per_second,
    })
    logger.write_text("attempt_00_initial.py", code)
    log(f"Gerado em {generated.elapsed:.2f}s | "
        f"{generated.completion_tokens} tokens | "
        f"{generated.tokens_per_second:.1f} tok/s")

    previous_codes: list[str] = []
    history: list[dict[str, Any]] = []
    last_code_for_diff = ""

    # ================================================
    # LOOP
    # ================================================

    for attempt_number in range(1, MAX_ATTEMPTS + 1):

        stage(f"Tentativa {attempt_number}/{MAX_ATTEMPTS}")

        if detect_repetition(code, previous_codes):
            warning("O modelo repetiu uma solução anterior.")
            if attempt_number == MAX_ATTEMPTS:
                error("Modelo não conseguiu produzir solução nova.")
                logger.log("abort_repetition", {"attempt": attempt_number})
                return False

        previous_codes.append(code)
        logger.write_text(f"attempt_{attempt_number:02d}.py", code)

        # --------------------------------------------
        # stdin (auto-preenchimento)
        # --------------------------------------------

        stdin_values: list[str] = []
        if plan and plan.test_inputs:
            stdin_values = list(plan.test_inputs)
        elif AUTO_FILL_INPUT:
            prompts = extract_input_prompts(code)
            if prompts:
                stdin_values = [synthesize_input_value(p) for p in prompts]
                dim(f"   input() detectado ({len(prompts)} ocorrência(s)); "
                    f"valores auto-preenchidos: {stdin_values}")

        stdin_payload = build_stdin_payload(stdin_values)

        # --------------------------------------------
        # Execução
        # --------------------------------------------

        if needs_execution:
            stage("Executando código...")
            attempt = run_attempt(code, stdin_payload)
        else:
            attempt = AttemptResult(
                code=code,
                execution=ExecutionResult(
                    success=True, returncode=0,
                    stdout="", stderr="", elapsed=0.0,
                ),
                analysis=static_analyze(code),
            )

        execution = attempt.execution

        logger.log(f"attempt_{attempt_number:02d}_execution", {
            "returncode": execution.returncode,
            "success": execution.success,
            "elapsed": execution.elapsed,
            "timed_out": execution.timed_out,
            "stdin_used": execution.stdin_used,
            "stdout_tail": truncate(execution.stdout, 4000),
            "stderr_tail": truncate(execution.stderr, 4000),
        })

        # --------------------------------------------
        # Resultado
        # --------------------------------------------

        if execution.success:
            success(f"Execução OK em {execution.elapsed:.2f}s.")
            if execution.stdout.strip():
                print(Fore.WHITE + execution.stdout.strip() + Style.RESET_ALL)
        else:
            error(f"Falhou (returncode={execution.returncode}).")
            if execution.stderr.strip():
                print(Fore.RED + execution.stderr.strip() + Style.RESET_ALL)
            elif execution.stdout.strip():
                print(Fore.RED + execution.stdout.strip() + Style.RESET_ALL)

        # --------------------------------------------
        # Arquivos esperados
        # --------------------------------------------

        missing_files = verify_expected_files(task)
        if execution.success and missing_files:
            warning("Arquivos esperados ausentes: " + ", ".join(missing_files))
            execution = ExecutionResult(
                success=False,
                returncode=-3,
                stdout=execution.stdout,
                stderr="Arquivos esperados ausentes: " + ", ".join(missing_files),
                elapsed=execution.elapsed,
                stdin_used=execution.stdin_used,
            )

        # --------------------------------------------
        # Sucesso técnico
        # --------------------------------------------

        if execution.success:
            history.append({
                "attempt": attempt_number,
                "result": "success",
                "error": "",
            })

            if ENABLE_VERIFIER and needs_verification:
                stage("Verificação semântica...")
                verification = agent.verify(task, code, execution, plan)
                logger.log(f"verification_{attempt_number:02d}", verification)

                status = str(verification.get("status", "ok")).lower()
                if status == "retry":
                    warning("Verificador solicitou correção.")
                    reason = str(verification.get("reason", ""))
                    correction = str(verification.get("correction", ""))
                    logger.log(
                        f"verification_retry_{attempt_number:02d}",
                        {"reason": reason, "correction": correction},
                    )
                    history.append({
                        "attempt": attempt_number,
                        "result": "verification_retry",
                        "error": reason + "\n" + correction,
                    })

                    if attempt_number < MAX_ATTEMPTS:
                        stage("Reparando com base na verificação...")
                        diff = make_diff(last_code_for_diff, code) \
                            if last_code_for_diff else ""
                        repaired = agent.repair(
                            task=task,
                            plan=plan,
                            previous_code=code,
                            execution=execution,
                            analysis=attempt.analysis,
                            history=history,
                            diff_from_previous=diff,
                        )
                        last_code_for_diff = code
                        code = normalize_code(repaired.text)
                        logger.write_text(
                            f"attempt_{attempt_number:02d}_repair.py", code
                        )
                        continue
                    else:
                        warning("Limite de tentativas atingido na verificação.")

            # Sucesso definitivo
            files = changed_files(before)
            if files:
                success("Arquivos alterados: " + ", ".join(files))

            output_path = choose_output_path(task)
            save_program(code, output_path)
            logger.log("success", {
                "output_path": str(output_path),
                "changed_files": files,
            })
            success(f"Tarefa concluída com sucesso.")
            success(f"Programa salvo em: {output_path}")

            offer_interactive_session(output_path)
            return True

        # --------------------------------------------
        # Falha → reparo
        # --------------------------------------------

        history.append({
            "attempt": attempt_number,
            "result": "execution_failed",
            "error": truncate(execution.combined_output, MAX_TRACEBACK_CHARS),
        })

        if attempt_number >= MAX_ATTEMPTS:
            error("Número máximo de tentativas atingido.")
            logger.log("abort_max_attempts", {"attempt": attempt_number})
            # ainda salvamos o último código para inspeção
            output_path = choose_output_path(task)
            save_program(code, output_path)
            warning(f"Último código salvo em: {output_path}")
            return False

        stage("Analisando traceback e reparando causa raiz...")

        diff = make_diff(last_code_for_diff, code) if last_code_for_diff else ""
        repaired = agent.repair(
            task=task,
            plan=plan,
            previous_code=code,
            execution=execution,
            analysis=attempt.analysis,
            history=history,
            diff_from_previous=diff,
        )
        new_code = normalize_code(repaired.text)

        if detect_repetition(new_code, previous_codes):
            warning("Reparo repetiu solução anterior. Forçando nova tentativa.")

            forced_history = history + [{
                "attempt": attempt_number,
                "result": "repair_repeated_previous_code",
                "error": (
                    "A solução proposta é idêntica a uma tentativa anterior. "
                    "Você DEVE mudar a estratégia."
                ),
            }]

            if attempt_number + 1 <= MAX_ATTEMPTS:
                repaired = agent.repair(
                    task=task,
                    plan=plan,
                    previous_code=new_code,
                    execution=execution,
                    analysis=attempt.analysis,
                    history=forced_history,
                    diff_from_previous="",
                )
                new_code = normalize_code(repaired.text)

        last_code_for_diff = code
        code = new_code
        logger.write_text(f"attempt_{attempt_number:02d}_repaired.py", code)

    return False


# ============================================================
# INTERFACE
# ============================================================

def print_banner() -> None:
    print()
    print(Fore.CYAN + "╔══════════════════════════════════════════╗")
    print(Fore.CYAN + "║            LUNA AI CODER v2              ║")
    print(Fore.CYAN + "║    Autonomous Local Coding Agent         ║")
    print(Fore.CYAN + "╚══════════════════════════════════════════╝")
    print()
    print(f"{Style.DIM}Comandos: sair | exit | quit{Style.RESET_ALL}")
    print(f"{Style.DIM}Workspace: {WORKSPACE}{Style.RESET_ALL}")
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
    print()

    while True:
        try:
            task = input(
                Fore.WHITE + "Você > " + Style.RESET_ALL
            ).strip()
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
                error(f"Luna não conseguiu concluir a tarefa em {elapsed:.2f}s.")

        except KeyboardInterrupt:
            print()
            warning("Execução interrompida pelo usuário.")
            logger.log("interrupted_by_user", {})
        except Exception:
            error("Erro interno inesperado.")
            print(Fore.RED + traceback.format_exc() + Style.RESET_ALL)
            logger.log("internal_error", {
                "traceback": traceback.format_exc(),
            })

        print()

    logger.log("session_end", {})
    dim(f"Logs salvos em: {logger.session_dir}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()