from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from colorama import Fore, Style, init
from llama_cpp import Llama


# ============================================================
# LUNA AI CODING AGENT
# ============================================================
#
# Arquitetura:
#
# USER
#   ↓
# ROUTER
#   ↓
# PLANNER (somente quando necessário)
#   ↓
# CODER
#   ↓
# STATIC ANALYSIS
#   ↓
# EXECUTOR
#   ↓
# TRACEBACK REAL
#   ↓
# REPAIR
#   ↓
# EXECUTOR
#   ↓
# VERIFIER
#
# O modelo trabalha em streaming.
#
# ============================================================


init(autoreset=True)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.gguf"
WORKSPACE = BASE_DIR / "workspace"

WORKSPACE.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Modelo
# -----------------------------

N_CTX = 8192

# CPU:
N_THREADS = max(1, (os.cpu_count() or 4) - 1)

# Batch maior costuma ajudar no prompt processing.
N_BATCH = 512

# Se tiver GPU compatível com llama.cpp:
# -1 = todas as camadas
N_GPU_LAYERS = 0

# Atenção:
# Em algumas máquinas, flash attention pode melhorar bastante.
# Se sua build suportar:
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

# Streaming SEMPRE ligado.
STREAM = True

# Se True, imprime os tokens/chunks diretamente.
#
# Para o agente de código recomendo False:
# o terminal mostra apenas o progresso.
#
# O modelo continua usando stream=True.
SHOW_MODEL_STREAM = True


# -----------------------------
# Limites
# -----------------------------

MAX_ROUTER_TOKENS = 220
MAX_PLAN_TOKENS = 420
MAX_CODE_TOKENS = 1400
MAX_REPAIR_TOKENS = 1600
MAX_VERIFY_TOKENS = 260

MAX_ATTEMPTS = 3

EXECUTION_TIMEOUT = 120


# -----------------------------
# Comportamento
# -----------------------------

ENABLE_PLANNER = True
ENABLE_VERIFIER = True

# Tarefas muito simples podem ir direto para o coder.
PLANNER_COMPLEXITY_LEVEL = 2

# Não permitir que o modelo gere código infinitamente grande.
MAX_CODE_CHARS = 50000

# Limite do traceback enviado ao modelo.
MAX_TRACEBACK_CHARS = 18000

# Limite do código anterior enviado ao reparador.
MAX_PREVIOUS_CODE_CHARS = 50000


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


@dataclass
class TaskPlan:
    goal: str
    requirements: list[str]
    checks: list[str]


@dataclass
class AttemptResult:
    code: str
    execution: ExecutionResult
    analysis: AnalysisResult


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
    print(f"{Fore.MAGENTA}[LUNA] {message}{Style.RESET_ALL}")


# ============================================================
# UTILITÁRIOS
# ============================================================

def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    return text.strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
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

    # Tentativa de recuperar JSON cercado por texto.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start:end + 1]

        try:
            value = json.loads(candidate)

            if isinstance(value, dict):
                return value

        except json.JSONDecodeError:
            pass

    return None


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    return (
        text[:limit]
        + "\n\n[... conteúdo truncado pela Luna ...]"
    )


def normalize_code(code: str) -> str:
    code = strip_code_fences(code)

    # Alguns modelos colocam "python" no início sem markdown.
    if code.lower().startswith("python\n"):
        code = code[7:]

    return code.strip()


def code_fingerprint(code: str) -> str:
    normalized = re.sub(r"\s+", "", code)
    return normalized


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

    # --------------------------------------------------------
    # Tokenização
    # --------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0

        try:
            return len(
                self.llm.tokenize(
                    text.encode("utf-8"),
                    add_bos=False,
                )
            )
        except Exception:
            return 0

    # --------------------------------------------------------
    # Streaming
    # --------------------------------------------------------

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: Optional[float] = None,
        response_format: Optional[dict[str, Any]] = None,
        stream_output: Optional[bool] = None,
    ) -> GenerationResult:

        if temperature is None:
            temperature = TEMPERATURE

        if stream_output is None:
            stream_output = SHOW_MODEL_STREAM

        messages = [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ]

        # Estimativa do prompt para métricas.
        prompt_text = system + "\n" + user
        prompt_tokens = self.count_tokens(prompt_text)

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

        # ----------------------------------------------------
        # STREAMING REAL
        # ----------------------------------------------------

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
                        print(
                            content,
                            end="",
                            flush=True,
                        )

            except (AttributeError, KeyError, TypeError):
                continue

        if stream_output:
            print()

        text = "".join(parts)

        elapsed = max(
            time.perf_counter() - started,
            0.000001,
        )

        # IMPORTANTE:
        #
        # Não contamos chunks como tokens.
        #
        # O streaming pode entregar pedaços de texto de tamanhos
        # diferentes.
        #
        # Portanto tokenizamos a resposta completa.
        completion_tokens = self.count_tokens(text)

        tokens_per_second = (
            completion_tokens / elapsed
            if completion_tokens
            else 0.0
        )

        return GenerationResult(
            text=text,
            elapsed=elapsed,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_second=tokens_per_second,
        )


# ============================================================
# PROMPTS
# ============================================================

ROUTER_SYSTEM = """
Você é o router de um agente de programação local chamado Luna.

Sua função é classificar a tarefa antes da execução.

Retorne SOMENTE JSON válido.

Schema:

{
  "complexity": 1,
  "needs_planning": false,
  "needs_execution": true,
  "needs_verification": false,
  "task_type": "python"
}

complexity:
1 = muito simples
2 = simples
3 = média
4 = complexa
5 = muito complexa

Use:
python, coding, file, math, visualization ou general.

Não explique nada fora do JSON.
"""


PLANNER_SYSTEM = """
Você é o planejador técnico da Luna.

Transforme a solicitação do usuário em um plano mínimo,
preciso e executável.

Não escreva código.

Retorne SOMENTE JSON válido:

{
  "goal": "...",
  "requirements": ["..."],
  "checks": ["..."]
}

Não invente requisitos que o usuário não pediu.
"""


CODER_SYSTEM = """
Você é o principal engenheiro de software da Luna.

Sua tarefa é escrever código Python executável.

REGRAS ABSOLUTAS:

1. Retorne SOMENTE código Python.
2. Nunca use Markdown.
3. Nunca escreva ```python.
4. O código deve ser completo.
5. Inclua todos os imports necessários.
6. Não dependa de variáveis externas.
7. Não peça entrada interativa ao usuário.
8. Não use input().
14. Arquivos devem ficar dentro de workspace.
15. Não invente bibliotecas.
16. Evite APIs frágeis ou desnecessárias.
17. O código deve funcionar na versão atual das bibliotecas.
18. Para NumPy/Matplotlib, confira cuidadosamente dimensões e shapes.
19. Para gráficos 3D, X, Y e Z devem possuir dimensões compatíveis.
20. Não use parâmetros experimentais sem necessidade.
21. Prefira soluções simples e robustas.
22. O programa deve realmente executar.
23. Não explique o código.
24. Não gere apenas um trecho: gere o programa completo.

IMPORTANTE:

Se houver erro anterior, use o traceback como fonte principal
para descobrir a causa.

Não repita literalmente uma solução que já falhou.
"""


REPAIR_SYSTEM = """
Você é o engenheiro de reparo da Luna.

Você recebeu:

- objetivo original
- código que falhou
- stdout
- stderr
- traceback
- análise estática
- histórico de tentativas

Sua tarefa é corrigir a causa RAIZ.

REGRAS:

1. Retorne SOMENTE o código Python completo.
2. Não use Markdown.
3. Não explique nada.
4. Preserve o objetivo original.
5. Não simplesmente repita o código anterior.
6. Leia o traceback cuidadosamente.
7. Corrija exatamente a API, tipo, shape ou lógica responsável pelo erro.
8. Não introduza uma solução mais complexa sem necessidade.
9. Não use pip.
10. Não instale nada.
11. Não use subprocess.
12. Não use comandos do sistema.
13. Não use rede.
14. Arquivos devem ficar dentro de workspace.
15. O código precisa ser executável do início ao fim.
16. Se a tentativa anterior usou uma API incorreta, substitua-a.
17. Se uma linha específica causou o erro, altere a estratégia quando necessário.
18. Não repita a mesma solução que já falhou.
"""


VERIFIER_SYSTEM = """
Você é o verificador final de um agente de programação.

Analise o resultado da execução.

Retorne SOMENTE JSON:

{
  "status": "ok",
  "reason": "...",
  "correction": ""
}

status deve ser:

ok
ou
retry

Use retry somente quando existir evidência de que a tarefa
não foi realmente concluída ou que existe erro relevante.

Não exija melhorias que o usuário não pediu.
"""


# ============================================================
# ROUTER
# ============================================================

class LunaAgent:

    def __init__(self, model: LunaModel) -> None:
        self.model = model

    # --------------------------------------------------------
    # Router
    # --------------------------------------------------------

    def route(self, task: str) -> dict[str, Any]:

        result = self.model.generate(
            system=ROUTER_SYSTEM,
            user=task,
            max_tokens=MAX_ROUTER_TOKENS,
            temperature=0.05,
            response_format={
                "type": "json_object",
            },
        )

        data = safe_json(result.text)

        if not data:
            warning("Router retornou JSON inválido. Usando fallback.")

            return {
                "complexity": 3,
                "needs_planning": True,
                "needs_execution": True,
                "needs_verification": True,
                "task_type": "python",
            }

        complexity = int(data.get("complexity", 3))

        complexity = max(1, min(5, complexity))

        data["complexity"] = complexity

        return data

    # --------------------------------------------------------
    # Planner
    # --------------------------------------------------------

    def plan(self, task: str) -> TaskPlan:

        result = self.model.generate(
            system=PLANNER_SYSTEM,
            user=task,
            max_tokens=MAX_PLAN_TOKENS,
            temperature=0.05,
            response_format={
                "type": "json_object",
            },
        )

        data = safe_json(result.text)

        if not data:
            warning("Planner retornou JSON inválido.")

            return TaskPlan(
                goal=task,
                requirements=[],
                checks=[],
            )

        return TaskPlan(
            goal=str(data.get("goal", task)),
            requirements=[
                str(x)
                for x in data.get("requirements", [])
                if x
            ],
            checks=[
                str(x)
                for x in data.get("checks", [])
                if x
            ],
        )

    # --------------------------------------------------------
    # Coder
    # --------------------------------------------------------

    def generate_code(
        self,
        task: str,
        plan: Optional[TaskPlan],
    ) -> GenerationResult:

        if plan:

            requirements = "\n".join(
                f"- {item}"
                for item in plan.requirements
            )

            checks = "\n".join(
                f"- {item}"
                for item in plan.checks
            )

            user = f"""
OBJETIVO:
{plan.goal}

REQUISITOS:
{requirements or "- nenhum requisito adicional"}

VERIFICAÇÕES:
{checks or "- executar o programa e verificar erros"}

SOLICITAÇÃO ORIGINAL:
{task}
"""

        else:

            user = f"""
SOLICITAÇÃO ORIGINAL:

{task}
"""

        return self.model.generate(
            system=CODER_SYSTEM,
            user=user,
            max_tokens=MAX_CODE_TOKENS,
            temperature=TEMPERATURE,
        )

    # --------------------------------------------------------
    # Repair
    # --------------------------------------------------------

    def repair(
        self,
        task: str,
        plan: Optional[TaskPlan],
        previous_code: str,
        execution: ExecutionResult,
        analysis: AnalysisResult,
        history: list[dict[str, Any]],
    ) -> GenerationResult:

        history_text = []

        for item in history[-3:]:

            history_text.append(
                f"""
TENTATIVA {item.get("attempt")}:

Resultado:
{item.get("result")}

Erro:
{item.get("error", "")}
"""
            )

        history_block = "\n".join(history_text)

        plan_text = ""

        if plan:

            plan_text = f"""
PLANO:
Objetivo:
{plan.goal}

Requisitos:
{json.dumps(plan.requirements, ensure_ascii=False)}

Checks:
{json.dumps(plan.checks, ensure_ascii=False)}
"""

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

Erros:
{json.dumps(analysis.errors, ensure_ascii=False)}

Warnings:
{json.dumps(analysis.warnings, ensure_ascii=False)}

Imports:
{json.dumps(analysis.imports, ensure_ascii=False)}

==================================================
HISTÓRICO
==================================================

{history_block}

==================================================

IMPORTANTE:

Identifique a causa raiz.

Não copie a tentativa anterior cegamente.

Gere uma implementação nova e completa.
"""

        return self.model.generate(
            system=REPAIR_SYSTEM,
            user=user,
            max_tokens=MAX_REPAIR_TOKENS,
            temperature=0.08,
        )

    # --------------------------------------------------------
    # Verifier
    # --------------------------------------------------------

    def verify(
        self,
        task: str,
        code: str,
        execution: ExecutionResult,
        plan: Optional[TaskPlan],
    ) -> dict[str, Any]:

        checks = []

        if plan:
            checks = plan.checks

        user = f"""
TAREFA:
{task}

CHECKS:
{json.dumps(checks, ensure_ascii=False)}

CÓDIGO:
{truncate(code, 30000)}

RESULTADO:
returncode={execution.returncode}

STDOUT:
{truncate(execution.stdout, 8000)}

STDERR:
{truncate(execution.stderr, 8000)}
"""

        result = self.model.generate(
            system=VERIFIER_SYSTEM,
            user=user,
            max_tokens=MAX_VERIFY_TOKENS,
            temperature=0.03,
            response_format={
                "type": "json_object",
            },
        )

        data = safe_json(result.text)

        if not data:
            return {
                "status": "ok",
                "reason": "Execução terminou sem erro.",
                "correction": "",
            }

        return data


# ============================================================
# ANÁLISE ESTÁTICA
# ============================================================

BLOCKED_IMPORTS = {
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "httpx",
    "ftplib",
    "telnetlib",
    "ctypes",
    "multiprocessing",
}

BLOCKED_CALLS = {
    "system",
    "popen",
    "exec",
    "eval",
}

ALLOWED_FILE_EXTENSIONS = {
    ".py",
    ".txt",
    ".json",
    ".csv",
    ".tsv",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".html",
}


def static_analyze(code: str) -> AnalysisResult:

    result = AnalysisResult(
        ok=True,
    )

    # --------------------------------------------
    # Syntax
    # --------------------------------------------

    try:
        tree = ast.parse(code)

    except SyntaxError as exc:

        result.ok = False

        result.errors.append(
            f"SyntaxError linha {exc.lineno}: {exc.msg}"
        )

        return result

    # --------------------------------------------
    # AST
    # --------------------------------------------

    for node in ast.walk(tree):

        # Imports
        if isinstance(node, ast.Import):

            for alias in node.names:

                name = alias.name

                result.imports.append(name)

                root = name.split(".")[0]

                if root in BLOCKED_IMPORTS:

                    result.ok = False

                    result.errors.append(
                        f"Import bloqueado: {name}"
                    )

        elif isinstance(node, ast.ImportFrom):

            module = node.module or ""

            result.imports.append(module)

            root = module.split(".")[0]

            if root in BLOCKED_IMPORTS:

                result.ok = False

                result.errors.append(
                    f"Import bloqueado: {module}"
                )

        # Chamadas
        elif isinstance(node, ast.Call):

            if isinstance(node.func, ast.Attribute):

                attr = node.func.attr

                if attr in BLOCKED_CALLS:

                    result.ok = False

                    result.errors.append(
                        f"Chamada potencialmente perigosa: {attr}()"
                    )

            elif isinstance(node.func, ast.Name):

                if node.func.id in BLOCKED_CALLS:

                    result.ok = False

                    result.errors.append(
                        f"Chamada potencialmente perigosa: {node.func.id}()"
                    )

    # --------------------------------------------
    # Heurísticas
    # --------------------------------------------

    lowered = code.lower()

    if "plt.show(" in lowered:
        result.warnings.append(
            "plt.show() detectado; execução pode depender de GUI."
        )

    if "input(" in lowered:
        result.ok = False

        result.errors.append(
            "input() não é permitido em execução automática."
        )

    if "os.system(" in lowered:
        result.ok = False

        result.errors.append(
            "os.system() não é permitido."
        )

    if "os.popen(" in lowered:
        result.ok = False

        result.errors.append(
            "os.popen() não é permitido."
        )

    # --------------------------------------------
    # Arquivos absolutos
    # --------------------------------------------

    for node in ast.walk(tree):

        if isinstance(node, ast.Constant):

            if isinstance(node.value, str):

                value = node.value

                # Detecta caminhos Windows absolutos.
                if re.match(
                    r"^[A-Za-z]:[\\/]",
                    value,
                ):

                    result.warnings.append(
                        f"Caminho absoluto detectado: {value}"
                    )

    return result


# ============================================================
# EXECUÇÃO
# ============================================================

def execute_python(code: str) -> ExecutionResult:

    started = time.perf_counter()

    env = os.environ.copy()

    env["LUNA_WORKSPACE"] = str(WORKSPACE)

    # Permite importar módulos do projeto quando necessário.
    existing_pythonpath = env.get("PYTHONPATH", "")

    pythonpath_parts = [
        str(BASE_DIR),
        str(WORKSPACE),
    ]

    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(
        pythonpath_parts
    )

    process: Optional[subprocess.Popen[str]] = None

    try:

        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                code,
            ],
            cwd=str(WORKSPACE),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        try:

            stdout, stderr = process.communicate(
                timeout=EXECUTION_TIMEOUT
            )

        except subprocess.TimeoutExpired:

            process.kill()

            stdout, stderr = process.communicate()

            elapsed = time.perf_counter() - started

            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout=stdout or "",
                stderr=(
                    stderr or ""
                )
                + "\nLUNA: execução excedeu o timeout.",
                elapsed=elapsed,
                timed_out=True,
            )

        elapsed = time.perf_counter() - started

        return ExecutionResult(
            success=process.returncode == 0,
            returncode=process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            elapsed=elapsed,
        )

    except Exception as exc:

        elapsed = time.perf_counter() - started

        return ExecutionResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr=(
                f"Falha ao iniciar processo:\n"
                f"{type(exc).__name__}: {exc}"
            ),
            elapsed=elapsed,
        )


# ============================================================
# SNAPSHOT
# ============================================================

def snapshot_workspace() -> dict[str, float]:

    snapshot: dict[str, float] = {}

    for path in WORKSPACE.rglob("*"):

        if path.is_file():

            try:
                snapshot[str(path.relative_to(WORKSPACE))] = (
                    path.stat().st_mtime
                )

            except OSError:
                pass

    return snapshot


def changed_files(
    before: dict[str, float],
) -> list[str]:

    changed = []

    for path in WORKSPACE.rglob("*"):

        if not path.is_file():
            continue

        try:

            relative = str(
                path.relative_to(WORKSPACE)
            )

            mtime = path.stat().st_mtime

            if (
                relative not in before
                or mtime != before[relative]
            ):

                changed.append(relative)

        except OSError:
            continue

    return sorted(changed)


# ============================================================
# ARQUIVOS ESPERADOS
# ============================================================

def extract_expected_files(task: str) -> list[str]:

    found = set()

    # Exemplos:
    # criar arquivo teste.py
    # salve como resultado.csv
    # gerar arquivo "dados.json"

    patterns = [
        r"(?:arquivo|ficheiro|salve|salvar|crie|criar|gere|gerar)"
        r"(?:\s+como|\s+em)?\s+[\"']?([A-Za-z0-9_.-]+\.[A-Za-z0-9]+)",
    ]

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            task,
            flags=re.IGNORECASE,
        ):

            filename = match.group(1)

            suffix = Path(filename).suffix.lower()

            if suffix in ALLOWED_FILE_EXTENSIONS:

                found.add(filename)

    return sorted(found)


def verify_expected_files(
    task: str,
) -> list[str]:

    expected = extract_expected_files(task)

    missing = []

    for filename in expected:

        path = WORKSPACE / filename

        if not path.exists():

            missing.append(filename)

    return missing


# ============================================================
# PROTEÇÃO CONTRA REPETIÇÃO
# ============================================================

def detect_repetition(
    current_code: str,
    previous_codes: list[str],
) -> bool:

    current = code_fingerprint(current_code)

    for previous in previous_codes:

        if current == code_fingerprint(previous):
            return True

    return False


# ============================================================
# EXECUÇÃO DE UMA TENTATIVA
# ============================================================

def run_attempt(
    code: str,
) -> AttemptResult:

    code = normalize_code(code)

    if not code:
        analysis = AnalysisResult(
            ok=False,
            errors=["Modelo retornou código vazio."],
        )

        execution = ExecutionResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="Código vazio.",
            elapsed=0,
        )

        return AttemptResult(
            code=code,
            execution=execution,
            analysis=analysis,
        )

    if len(code) > MAX_CODE_CHARS:

        analysis = AnalysisResult(
            ok=False,
            errors=[
                "Código excedeu o limite de tamanho."
            ],
        )

        execution = ExecutionResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="Código grande demais.",
            elapsed=0,
        )

        return AttemptResult(
            code=code,
            execution=execution,
            analysis=analysis,
        )

    # --------------------------------------------
    # Static analysis
    # --------------------------------------------

    analysis = static_analyze(code)

    if not analysis.ok:

        execution = ExecutionResult(
            success=False,
            returncode=-2,
            stdout="",
            stderr="\n".join(
                analysis.errors
            ),
            elapsed=0,
        )

        return AttemptResult(
            code=code,
            execution=execution,
            analysis=analysis,
        )

    # --------------------------------------------
    # Execução
    # --------------------------------------------

    execution = execute_python(code)

    return AttemptResult(
        code=code,
        execution=execution,
        analysis=analysis,
    )


# ============================================================
# AGENTE PRINCIPAL
# ============================================================

def run_agent(
    task: str,
    model: LunaModel,
) -> bool:

    agent = LunaAgent(model)

    log("Analisando tarefa...")

    route = agent.route(task)

    complexity = int(
        route.get("complexity", 3)
    )

    needs_execution = bool(
        route.get(
            "needs_execution",
            True,
        )
    )

    needs_verification = bool(
        route.get(
            "needs_verification",
            complexity >= 3,
        )
    )

    needs_planning = bool(
        route.get(
            "needs_planning",
            complexity >= PLANNER_COMPLEXITY_LEVEL,
        )
    )

    log(
        f"Complexidade estimada: {complexity}/5"
    )

    plan: Optional[TaskPlan] = None

    # --------------------------------------------
    # Planejamento
    # --------------------------------------------

    if (
        ENABLE_PLANNER
        and needs_planning
    ):

        stage("Planejando solução...")

        plan = agent.plan(task)

        success("Plano definido.")

    else:

        log(
            "Tarefa simples: planejamento intermediário ignorado."
        )

    # --------------------------------------------
    # Snapshot
    # --------------------------------------------

    before_workspace = snapshot_workspace()

    # --------------------------------------------
    # Primeira geração
    # --------------------------------------------

    stage("Gerando código em streaming...")

    generated = agent.generate_code(
        task,
        plan,
    )

    code = normalize_code(
        generated.text
    )

    log(
        f"Gerado em {generated.elapsed:.2f}s | "
        f"{generated.completion_tokens} tokens | "
        f"{generated.tokens_per_second:.1f} tok/s"
    )

    previous_codes: list[str] = []

    history: list[dict[str, Any]] = []

    # ========================================================
    # LOOP AGENTIVO
    # ========================================================

    for attempt_number in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        stage(
            f"Tentativa {attempt_number}/{MAX_ATTEMPTS}"
        )

        # --------------------------------------------
        # Evitar repetição
        # --------------------------------------------

        if detect_repetition(
            code,
            previous_codes,
        ):

            warning(
                "O modelo repetiu uma solução anterior."
            )

            if attempt_number == MAX_ATTEMPTS:

                error(
                    "Modelo não conseguiu produzir uma nova solução."
                )

                return False

        previous_codes.append(code)

        # --------------------------------------------
        # Execução
        # --------------------------------------------

        if needs_execution:

            stage("Executando código...")

            attempt = run_attempt(code)

        else:

            attempt = AttemptResult(
                code=code,
                execution=ExecutionResult(
                    success=True,
                    returncode=0,
                    stdout="",
                    stderr="",
                    elapsed=0,
                ),
                analysis=static_analyze(code),
            )

        execution = attempt.execution

        # --------------------------------------------
        # Resultado
        # --------------------------------------------

        if execution.success:

            success(
                f"Execução concluída em "
                f"{execution.elapsed:.2f}s."
            )

            if execution.stdout.strip():

                print(
                    Fore.WHITE
                    + execution.stdout.strip()
                    + Style.RESET_ALL
                )

        else:

            error(
                f"Execução falhou "
                f"(returncode={execution.returncode})."
            )

            if execution.stderr.strip():

                print(
                    Fore.RED
                    + execution.stderr.strip()
                    + Style.RESET_ALL
                )

            elif execution.stdout.strip():

                print(
                    Fore.RED
                    + execution.stdout.strip()
                    + Style.RESET_ALL
                )

        # --------------------------------------------
        # Arquivos esperados
        # --------------------------------------------

        missing_files = verify_expected_files(
            task
        )

        if (
            execution.success
            and missing_files
        ):

            warning(
                "Arquivos esperados não foram encontrados: "
                + ", ".join(missing_files)
            )

            execution = ExecutionResult(
                success=False,
                returncode=-3,
                stdout=execution.stdout,
                stderr=(
                    "Arquivos esperados ausentes: "
                    + ", ".join(missing_files)
                ),
                elapsed=execution.elapsed,
            )

        # --------------------------------------------
        # Sucesso técnico
        # --------------------------------------------

        if execution.success:

            history.append(
                {
                    "attempt": attempt_number,
                    "result": "success",
                    "error": "",
                }
            )

            # ----------------------------------------
            # Verificação semântica
            # ----------------------------------------

            if (
                ENABLE_VERIFIER
                and needs_verification
                and complexity >= 3
            ):

                stage("Verificando resultado...")

                verification = agent.verify(
                    task,
                    code,
                    execution,
                    plan,
                )

                status = str(
                    verification.get(
                        "status",
                        "ok",
                    )
                ).lower()

                if status == "retry":

                    warning(
                        "Verificador solicitou uma correção."
                    )

                    reason = verification.get(
                        "reason",
                        "",
                    )

                    correction = verification.get(
                        "correction",
                        "",
                    )

                    history.append(
                        {
                            "attempt": attempt_number,
                            "result": "verification_retry",
                            "error": (
                                str(reason)
                                + "\n"
                                + str(correction)
                            ),
                        }
                    )

                    if attempt_number < MAX_ATTEMPTS:

                        stage(
                            "Gerando correção baseada na verificação..."
                        )

                        repaired = agent.repair(
                            task=task,
                            plan=plan,
                            previous_code=code,
                            execution=execution,
                            analysis=attempt.analysis,
                            history=history,
                        )

                        code = normalize_code(
                            repaired.text
                        )

                        continue

                success(
                    "Verificação final aprovada."
                )

            # ----------------------------------------
            # Sucesso definitivo
            # ----------------------------------------

            files = changed_files(
                before_workspace
            )

            if files:

                success(
                    "Arquivos alterados: "
                    + ", ".join(files)
                )

            success(
                "Tarefa concluída com sucesso."
            )

            return True

        # ====================================================
        # FALHA → REPARO
        # ====================================================

        history.append(
            {
                "attempt": attempt_number,
                "result": "execution_failed",
                "error": truncate(
                    execution.combined_output,
                    MAX_TRACEBACK_CHARS,
                ),
            }
        )

        if attempt_number >= MAX_ATTEMPTS:

            error(
                "Número máximo de tentativas atingido."
            )

            return False

        # --------------------------------------------
        # Reparo
        # --------------------------------------------

        stage(
            "Analisando traceback e corrigindo causa raiz..."
        )

        repaired = agent.repair(
            task=task,
            plan=plan,
            previous_code=code,
            execution=execution,
            analysis=attempt.analysis,
            history=history,
        )

        new_code = normalize_code(
            repaired.text
        )

        # --------------------------------------------
        # Proteção contra repetição
        # --------------------------------------------

        if detect_repetition(
            new_code,
            previous_codes,
        ):

            warning(
                "O reparo gerou novamente uma solução já usada."
            )

            # Faz uma tentativa adicional de reparo
            # somente se ainda houver espaço.
            if attempt_number + 1 <= MAX_ATTEMPTS:

                forced_history = history + [
                    {
                        "attempt": attempt_number,
                        "result": "repair_repeated_previous_code",
                        "error": (
                            "A solução proposta pelo reparador "
                            "é idêntica a uma tentativa anterior. "
                            "É obrigatório mudar a estratégia."
                        ),
                    }
                ]

                repaired = agent.repair(
                    task=task,
                    plan=plan,
                    previous_code=new_code,
                    execution=execution,
                    analysis=attempt.analysis,
                    history=forced_history,
                )

                new_code = normalize_code(
                    repaired.text
                )

        code = new_code

    return False


# ============================================================
# INTERFACE
# ============================================================

def print_banner() -> None:

    print()
    print(
        Fore.CYAN
        + "╔══════════════════════════════════════╗"
    )
    print(
        Fore.CYAN
        + "║            LUNA AI CODER            ║"
    )
    print(
        Fore.CYAN
        + "║       Local Autonomous Agent        ║"
    )
    print(
        Fore.CYAN
        + "╚══════════════════════════════════════╝"
    )
    print()


def main() -> None:

    print_banner()

    try:

        model = LunaModel()

    except Exception as exc:

        error(
            "Não foi possível carregar o modelo."
        )

        print(
            Fore.RED
            + traceback.format_exc()
            + Style.RESET_ALL
        )

        return

    print()
    print(
        "Digite a tarefa para a Luna."
    )
    print(
        "Digite 'sair' para encerrar."
    )
    print()

    while True:

        try:

            task = input(
                Fore.WHITE
                + "Você > "
                + Style.RESET_ALL
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print()
            break

        if not task:
            continue

        if task.lower() in {
            "sair",
            "exit",
            "quit",
        }:

            break

        print()

        started = time.perf_counter()

        try:

            ok = run_agent(
                task,
                model,
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            print()

            if ok:

                success(
                    f"Processo completo em {elapsed:.2f}s."
                )

            else:

                error(
                    f"Luna não conseguiu concluir "
                    f"a tarefa em {elapsed:.2f}s."
                )

        except KeyboardInterrupt:

            print()

            warning(
                "Execução interrompida pelo usuário."
            )

        except Exception:

            error(
                "Erro interno inesperado."
            )

            print(
                Fore.RED
                + traceback.format_exc()
                + Style.RESET_ALL
            )

        print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()