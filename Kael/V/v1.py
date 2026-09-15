import ast
import json
import os
import re
import sys
import time
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any, Optional

from colorama import Fore, Style, init
from llama_cpp import Llama


# ============================================================
# CONFIGURAÇÃO
# ============================================================

init(autoreset=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.gguf"
WORKSPACE = BASE_DIR / "workspace"

WORKSPACE.mkdir(parents=True, exist_ok=True)

N_CTX = 8192

# Se sua CPU tiver poucos núcleos, reduza.
N_THREADS = max(1, (os.cpu_count() or 4) - 1)

N_BATCH = 256

TEMPERATURE = 0.15
TOP_P = 0.90
REPEAT_PENALTY = 1.05

MAX_ROUTER_TOKENS = 180
MAX_PLAN_TOKENS = 300
MAX_COMMAND_TOKENS = 900
MAX_REPAIR_TOKENS = 900
MAX_VERIFY_TOKENS = 350

MAX_ATTEMPTS = 3
EXECUTION_TIMEOUT = 120

MAX_OUTPUT_CHARS = 10000

# Para tarefas muito simples, não chama o LLM para verificação semântica.
SIMPLE_TASK_MAX_COMPLEXITY = 2


# ============================================================
# CORES / UI
# ============================================================

C = {
    "cyan": Fore.CYAN,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "red": Fore.RED,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "white": Fore.WHITE,
    "gray": Fore.LIGHTBLACK_EX,
    "reset": Style.RESET_ALL,
    "bold": Style.BRIGHT,
}


def line(char="─", size=72):
    print(C["gray"] + char * size)


def title(text):
    print()
    line("═")
    print(C["bold"] + C["cyan"] + f"  {text}")
    line("═")


def phase(name):
    print()
    print(C["bold"] + C["cyan"] + f"[ {name.upper()} ]")
    line()


def info(label, value):
    print(
        C["gray"] + f"{label}: " +
        C["white"] + str(value)
    )


def success(text):
    print(C["green"] + "✓ " + text)


def warning(text):
    print(C["yellow"] + "⚠ " + text)


def error(text):
    print(C["red"] + "✗ " + text)


def status(text):
    print(C["blue"] + "→ " + text)


# ============================================================
# UTILITÁRIOS
# ============================================================

def trim(text: str, limit=MAX_OUTPUT_CHARS) -> str:
    text = text or ""

    if len(text) <= limit:
        return text

    return (
        text[:limit]
        + f"\n\n[saída truncada: {len(text) - limit} caracteres omitidos]"
    )


def extract_json(text: str) -> Optional[dict]:
    """
    Tenta encontrar JSON mesmo quando o modelo coloca
    texto ao redor.
    """

    if not text:
        return None

    text = text.strip()

    # Tentativa direta
    try:
        value = json.loads(text)

        if isinstance(value, dict):
            return value
    except Exception:
        pass

    # Markdown ```json
    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL | re.IGNORECASE
    )

    if match:
        try:
            value = json.loads(match.group(1))

            if isinstance(value, dict):
                return value
        except Exception:
            pass

    # Primeiro objeto JSON balanceado
    start = text.find("{")

    if start >= 0:
        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            char = text[i]

            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False

                continue

            if char == '"':
                in_string = True

            elif char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:
                    candidate = text[start:i + 1]

                    try:
                        value = json.loads(candidate)

                        if isinstance(value, dict):
                            return value
                    except Exception:
                        pass

                    break

    return None


def clean_code(text: str) -> str:
    """
    Remove markdown e pequenos resíduos produzidos pelo modelo.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove bloco markdown
    text = re.sub(r"^```(?:python|py)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    # Alguns modelos respondem "Aqui está..."
    lines = text.splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    text = "\n".join(lines).strip()

    return text


def safe_print_model(text):
    """
    Impressão tolerante a caracteres Unicode no Windows.
    """

    try:
        print(text, end="", flush=True)
    except UnicodeEncodeError:
        encoded = text.encode(
            sys.stdout.encoding or "utf-8",
            errors="replace"
        )
        print(encoded.decode(
            sys.stdout.encoding or "utf-8",
            errors="replace"
        ), end="", flush=True)


# ============================================================
# SNAPSHOT DO WORKSPACE
# ============================================================

def workspace_snapshot():
    """
    Mantém o estado somente em RAM.
    Não cria banco nem JSON de histórico.
    """

    snapshot = {}

    for path in WORKSPACE.rglob("*"):
        if not path.is_file():
            continue

        try:
            stat = path.stat()

            relative = str(path.relative_to(WORKSPACE))

            snapshot[relative] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }

        except OSError:
            pass

    return snapshot


def workspace_changes(before, after):
    created = []
    modified = []
    deleted = []

    before_keys = set(before)
    after_keys = set(after)

    for item in sorted(after_keys - before_keys):
        created.append(item)

    for item in sorted(before_keys - after_keys):
        deleted.append(item)

    for item in sorted(before_keys & after_keys):
        if before[item] != after[item]:
            modified.append(item)

    return {
        "created": created,
        "modified": modified,
        "deleted": deleted,
    }


# ============================================================
# AST / ANÁLISE ESTÁTICA
# ============================================================

class NameAnalyzer(ast.NodeVisitor):

    def __init__(self):
        self.loads = []
        self.stores = []
        self.imports = []
        self.calls = []

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.loads.append(node.id)

        elif isinstance(node.ctx, ast.Store):
            self.stores.append(node.id)

        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(
                alias.asname or alias.name.split(".")[0]
            )

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append(
                alias.asname or alias.name
            )

        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)

        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)

        self.generic_visit(node)


def static_analyze(code: str) -> dict:
    """
    Não tenta ser um linter completo.
    Procura problemas úteis para o agente.
    """

    result = {
        "syntax_ok": False,
        "syntax_error": None,
        "imports": [],
        "defined_names": [],
        "used_names": [],
        "possibly_unused": [],
        "warnings": [],
        "signals": [],
    }

    try:
        tree = ast.parse(code)

    except SyntaxError as exc:
        result["syntax_error"] = (
            f"{exc.msg} "
            f"(linha {exc.lineno}, coluna {exc.offset})"
        )

        return result

    result["syntax_ok"] = True

    analyzer = NameAnalyzer()
    analyzer.visit(tree)

    result["imports"] = analyzer.imports
    result["defined_names"] = sorted(set(analyzer.stores))
    result["used_names"] = sorted(set(analyzer.loads))

    # Variáveis atribuídas mas nunca lidas.
    # Ignoramos nomes privados e algumas variáveis comuns.
    ignored = {
        "_",
        "i",
        "j",
        "k",
        "ax",
        "fig",
    }

    for name in set(analyzer.stores):
        if name in ignored:
            continue

        if name.startswith("_"):
            continue

        if name not in analyzer.loads:
            result["possibly_unused"].append(name)

    result["possibly_unused"].sort()

    # Sinais específicos.
    lower = code.lower()

    if "matplotlib" in lower:
        result["signals"].append("matplotlib")

    if "numpy" in lower:
        result["signals"].append("numpy")

    if "projection='3d'" in lower or 'projection="3d"' in lower:
        result["signals"].append("3d_plot")

    if "plot_surface" in lower:
        result["signals"].append("surface_plot")

    if "plot_trisurf" in lower:
        result["signals"].append("trisurf_plot")

    if "plt.show" in lower:
        result["signals"].append("display_plot")

    if "savefig" in lower:
        result["signals"].append("saved_plot")

    if "urlopen" in lower or "urllib" in lower:
        result["signals"].append("network")

    if "path(" in lower or "open(" in lower:
        result["signals"].append("filesystem")

    # Avisos
    if result["possibly_unused"]:
        result["warnings"].append(
            "Variáveis possivelmente definidas mas não utilizadas: "
            + ", ".join(result["possibly_unused"])
        )

    # 3D específico
    if "3d_plot" in result["signals"]:

        if "surface_plot" not in result["signals"] and \
           "trisurf_plot" not in result["signals"]:

            result["warnings"].append(
                "O código cria um eixo 3D, mas não há uma operação "
                "óbvia de superfície."
            )

    # Caso clássico de funil
    funnel_names = {
        "top_radius",
        "bottom_radius",
        "base_radius",
        "height",
    }

    defined = set(analyzer.stores)
    loaded = set(analyzer.loads)

    funnel_defined_unused = (
        defined & funnel_names
    ) - loaded

    if funnel_defined_unused:
        result["warnings"].append(
            "Parâmetros geométricos do funil não utilizados: "
            + ", ".join(sorted(funnel_defined_unused))
        )

    return result


# ============================================================
# MODELO
# ============================================================

class LunaModel:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado:\n{MODEL_PATH}"
            )

        status(f"Carregando modelo: {MODEL_PATH.name}")

        self.llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_batch=N_BATCH,
            verbose=False,
        )

        success("Modelo carregado")

    def generate(
        self,
        messages,
        max_tokens,
        temperature=TEMPERATURE,
        stream=True,
        label="LUNA",
    ):

        started = time.perf_counter()
        first_token_time = None
        token_count = 0
        chunks = []

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=TOP_P,
            repeat_penalty=REPEAT_PENALTY,
            stream=stream,
        )

        if stream:

            for chunk in response:

                choices = chunk.get("choices", [])

                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                token = delta.get("content", "")

                if token:

                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    token_count += 1
                    chunks.append(token)

                    safe_print_model(token)

            print()

        else:

            choices = response.get("choices", [])

            if choices:
                text = choices[0].get("message", {}).get(
                    "content", ""
                )
            else:
                text = ""

            chunks.append(text)

        elapsed = time.perf_counter() - started

        first_token = (
            None
            if first_token_time is None
            else first_token_time - started
        )

        tps = (
            0
            if elapsed <= 0
            else token_count / elapsed
        )

        return {
            "text": "".join(chunks),
            "tokens": token_count,
            "elapsed": elapsed,
            "first_token": first_token,
            "tps": tps,
        }

    def print_metrics(self, result):

        info("Tokens", result["tokens"])
        info(
            "Tempo",
            f"{result['elapsed']:.2f}s"
        )

        if result["first_token"] is not None:
            info(
                "Primeiro token",
                f"{result['first_token']:.2f}s"
            )

        info(
            "Velocidade",
            f"{result['tps']:.2f} tok/s"
        )


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_ROUTER = """
Você é o roteador de uma IA local chamada Luna.

Sua função é classificar a tarefa do usuário.

Não gere código.
Não explique.
Retorne SOMENTE JSON válido.

Formato:

{
  "tool": "python|filesystem|network|mixed|none",
  "action": "descrição curta",
  "complexity": 1,
  "needs_execution": true,
  "needs_verification": true
}

complexity:
1 = simples
2 = moderada
3 = complexa

Escolha python para cálculos, gráficos, simulações e código.
Escolha filesystem para arquivos e diretórios.
Escolha network para requisições HTTP.
Escolha mixed quando houver combinação.
"""


SYSTEM_PLANNER = """
Você é o planejador de um agente Python local.

Receberá uma tarefa do usuário e uma classificação do roteador.

Crie um plano CURTO.

Não gere código.

Retorne somente JSON válido:

{
  "goal": "...",
  "requirements": ["..."],
  "checks": ["..."]
}

Não inclua instalação de pacotes.
Não invente dependências.
"""


SYSTEM_COMMAND = """
Você é o gerador de comandos Python de um agente local.

Sua tarefa é transformar o pedido do usuário em UM ÚNICO COMANDO/PROGRAMA
Python executável diretamente com:

python -c

Regras obrigatórias:

1. Gere somente Python.
2. Não use Markdown.
3. Não escreva explicações.
4. O código deve ser autocontido.
5. Todo import necessário deve estar presente.
6. Toda variável utilizada deve estar definida.
7. Não dependa de código de mensagens anteriores.
8. Não dependa de histórico.
9. Não use variáveis externas.
10. Use bibliotecas instaladas quando apropriado.
11. NÃO execute pip install.
12. Não crie arquivos fora de ./workspace.
13. Se precisar criar arquivos, use pathlib e ./workspace.
14. Para gráficos, utilize corretamente os dados e a geometria.
15. Para matemática/física, utilize equações coerentes com o problema.
16. Não faça uma aproximação visual quando o pedido exigir uma relação
    matemática ou geométrica específica.
17. Todos os parâmetros relevantes definidos devem ser utilizados.
18. Se criar gráfico, mostre o gráfico quando apropriado.
19. Se criar arquivo, imprima o caminho do arquivo criado.
20. Se fizer HTTP, trate erros HTTP e conexão quando apropriado.
21. O resultado deve ser verificável pelo terminal.

IMPORTANTE:

Execução sem erro NÃO significa que a tarefa está conceitualmente correta.
O código precisa representar exatamente o objetivo solicitado.
"""


SYSTEM_REPAIR = """
Você é o módulo de reparação do agente Luna.

Um comando Python foi executado e apresentou um problema.

Você receberá:

- tarefa original
- código anterior
- análise estática
- código de saída
- stdout/stderr
- motivo da verificação

Gere UM NOVO comando Python completo.

Regras:

1. Somente Python.
2. Sem Markdown.
3. Sem explicações.
4. Corrija a causa real.
5. Não apenas esconda o erro.
6. O código deve ser autocontido.
7. Todos os imports devem estar presentes.
8. Todas as variáveis devem estar definidas.
9. Não use histórico.
10. Não execute pip install.
11. Arquivos devem ficar em ./workspace.
12. Preserve o objetivo original.
13. Se o código anterior executou mas estava conceitualmente errado,
    corrija a matemática, geometria ou lógica.
"""


SYSTEM_VERIFY = """
Você é o verificador semântico de um agente Python.

Determine se o resultado REAL corresponde ao pedido original.

Você receberá:

- pedido
- plano
- código executado
- análise estática
- exit code
- saída real
- arquivos criados/modificados

Retorne SOMENTE JSON válido:

{
  "status": "ok|retry",
  "reason": "...",
  "problem": "...",
  "correction": "..."
}

Use "retry" quando o programa executou mas o resultado está
matematicamente, fisicamente, geometricamente ou logicamente errado.

Não gere código.
Não sugira código.
Não invente resultados.

Diferencie:

exit code 0 = execução tecnicamente bem-sucedida

resultado correto = objetivo realmente atendido
"""


# ============================================================
# AGENTE
# ============================================================

class LunaAgent:

    def __init__(self):

        self.model = LunaModel()

    # --------------------------------------------------------
    # ROUTER
    # --------------------------------------------------------

    def route(self, user_request):

        phase("ROTEAMENTO")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_ROUTER,
            },
            {
                "role": "user",
                "content": user_request,
            },
        ]

        result = self.model.generate(
            messages,
            MAX_ROUTER_TOKENS,
            temperature=0.05,
            stream=True,
            label="ROUTER",
        )

        self.model.print_metrics(result)

        data = extract_json(result["text"])

        if not data:

            warning(
                "Router não retornou JSON válido. "
                "Usando Python como padrão."
            )

            data = {
                "tool": "python",
                "action": "execute",
                "complexity": 2,
                "needs_execution": True,
                "needs_verification": True,
            }

        print()

        try:
            print(
                json.dumps(
                    data,
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except Exception:
            print(data)

        return data

    # --------------------------------------------------------
    # PLAN
    # --------------------------------------------------------

    def plan(self, user_request, route):

        phase("PLANEJAMENTO")

        payload = {
            "request": user_request,
            "route": route,
        }

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PLANNER,
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            },
        ]

        result = self.model.generate(
            messages,
            MAX_PLAN_TOKENS,
            temperature=0.05,
            stream=True,
        )

        self.model.print_metrics(result)

        plan = extract_json(result["text"])

        if not plan:

            warning(
                "Planejamento inválido. "
                "Continuando com o pedido original."
            )

            plan = {
                "goal": user_request,
                "requirements": [],
                "checks": [],
            }

        print()

        print(
            json.dumps(
                plan,
                indent=2,
                ensure_ascii=False,
            )
        )

        return plan

    # --------------------------------------------------------
    # GENERATE COMMAND
    # --------------------------------------------------------

    def generate_command(
        self,
        user_request,
        route,
        plan,
        previous_code=None,
        static=None,
        execution=None,
        verification=None,
    ):

        phase("GERAÇÃO DO PYTHON")

        payload = {
            "request": user_request,
            "route": route,
            "plan": plan,
        }

        if previous_code:
            payload["previous_code"] = previous_code

        if static:
            payload["static_analysis"] = static

        if execution:
            payload["execution"] = execution

        if verification:
            payload["verification"] = verification

        messages = [
            {
                "role": "system",
                "content": SYSTEM_REPAIR if previous_code else SYSTEM_COMMAND,
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            },
        ]

        result = self.model.generate(
            messages,
            MAX_REPAIR_TOKENS if previous_code else MAX_COMMAND_TOKENS,
            temperature=0.10,
            stream=True,
        )

        self.model.print_metrics(result)

        code = clean_code(result["text"])

        return code

    # --------------------------------------------------------
    # STATIC VALIDATION
    # --------------------------------------------------------

    def validate_code(self, code):

        phase("ANÁLISE ESTÁTICA")

        result = static_analyze(code)

        if not result["syntax_ok"]:

            error(
                "Erro de sintaxe: "
                + str(result["syntax_error"])
            )

        else:

            success("Sintaxe Python válida")

        if result["imports"]:

            info(
                "Imports",
                ", ".join(result["imports"])
            )

        if result["possibly_unused"]:

            warning(
                "Possíveis variáveis não utilizadas: "
                + ", ".join(result["possibly_unused"])
            )

        for warning_text in result["warnings"]:
            warning(warning_text)

        return result

    # --------------------------------------------------------
    # EXECUTOR
    # --------------------------------------------------------

    def execute(self, code):

        phase("EXECUÇÃO")

        try:

            compile(
                code,
                "<luna>",
                "exec"
            )

        except SyntaxError as exc:

            return {
                "exit_code": -2,
                "stdout": "",
                "stderr": (
                    f"SyntaxError: {exc.msg} "
                    f"(linha {exc.lineno})"
                ),
                "elapsed": 0,
                "timed_out": False,
            }

        started = time.perf_counter()

        process = None

        stdout_chunks = []
        stderr_chunks = []

        try:

            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    code,
                ],

                cwd=str(BASE_DIR),

                stdin=subprocess.DEVNULL,

                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,

                text=True,
                encoding="utf-8",
                errors="replace",

                bufsize=1,

                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if os.name == "nt"
                    else 0
                ),
            )

            assert process.stdout is not None

            while True:

                line_output = process.stdout.readline()

                if line_output:

                    stdout_chunks.append(line_output)

                    # Saída real do Python.
                    print(
                        C["white"]
                        + "│ "
                        + line_output.rstrip()
                    )

                if process.poll() is not None:

                    # Lê o restante
                    remainder = process.stdout.read()

                    if remainder:

                        stdout_chunks.append(remainder)

                        for item in remainder.splitlines():
                            print(
                                C["white"]
                                + "│ "
                                + item
                            )

                    break

                elapsed = time.perf_counter() - started

                if elapsed > EXECUTION_TIMEOUT:

                    process.kill()

                    warning(
                        f"Tempo limite excedido "
                        f"({EXECUTION_TIMEOUT}s)"
                    )

                    return {
                        "exit_code": -3,
                        "stdout": "".join(stdout_chunks),
                        "stderr": "Execution timeout",
                        "elapsed": elapsed,
                        "timed_out": True,
                    }

            exit_code = process.returncode

        except Exception as exc:

            return {
                "exit_code": -4,
                "stdout": "".join(stdout_chunks),
                "stderr": traceback.format_exc(),
                "elapsed": time.perf_counter() - started,
                "timed_out": False,
            }

        elapsed = time.perf_counter() - started

        stdout = "".join(stdout_chunks)

        if exit_code == 0:

            success(
                f"Execução concluída em {elapsed:.2f}s"
            )

        else:

            error(
                f"Python terminou com código {exit_code}"
            )

        return {
            "exit_code": exit_code,
            "stdout": trim(stdout),
            "stderr": "",
            "elapsed": elapsed,
            "timed_out": False,
        }

    # --------------------------------------------------------
    # VERIFICAÇÃO SEMÂNTICA
    # --------------------------------------------------------

    def should_semantic_verify(
        self,
        route,
        static,
        execution,
    ):

        if execution["exit_code"] != 0:
            return True

        complexity = int(
            route.get("complexity", 2)
            or 2
        )

        if complexity >= 3:
            return True

        if static["warnings"]:
            return True

        signals = set(static["signals"])

        important = {
            "3d_plot",
            "surface_plot",
            "network",
            "filesystem",
        }

        if signals & important:
            return True

        return False

    def verify(
        self,
        user_request,
        route,
        plan,
        code,
        static,
        execution,
        changes,
    ):

        phase("VERIFICAÇÃO")

        payload = {
            "request": user_request,
            "route": route,
            "plan": plan,
            "code": code,
            "static_analysis": static,
            "execution": execution,
            "workspace_changes": changes,
        }

        messages = [
            {
                "role": "system",
                "content": SYSTEM_VERIFY,
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            },
        ]

        result = self.model.generate(
            messages,
            MAX_VERIFY_TOKENS,
            temperature=0.05,
            stream=True,
        )

        self.model.print_metrics(result)

        verification = extract_json(result["text"])

        if not verification:

            warning(
                "Verificador não retornou JSON válido."
            )

            verification = {
                "status": (
                    "ok"
                    if execution["exit_code"] == 0
                    else "retry"
                ),
                "reason": "Fallback baseado no exit code.",
                "problem": "",
                "correction": "",
            }

        print()

        print(
            json.dumps(
                verification,
                indent=2,
                ensure_ascii=False,
            )
        )

        return verification

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    def final_result(
        self,
        code,
        execution,
        changes,
        attempts,
        verification=None,
    ):

        phase("RESULTADO")

        if execution["exit_code"] == 0:

            success("Tarefa executada com sucesso.")

        else:

            error(
                f"A tarefa terminou com erro "
                f"(código {execution['exit_code']})."
            )

        print()

        info("Tentativas", attempts)
        info(
            "Tempo de execução",
            f"{execution['elapsed']:.2f}s"
        )

        # ----------------------------------------------------
        # ARQUIVOS
        # ----------------------------------------------------

        created = changes.get("created", [])
        modified = changes.get("modified", [])
        deleted = changes.get("deleted", [])

        if created or modified or deleted:

            print()
            print(C["bold"] + "Alterações no workspace:")

            for item in created:
                print(
                    C["green"]
                    + "  + "
                    + item
                )

            for item in modified:
                print(
                    C["yellow"]
                    + "  ~ "
                    + item
                )

            for item in deleted:
                print(
                    C["red"]
                    + "  - "
                    + item
                )

        # ----------------------------------------------------
        # SAÍDA REAL
        # ----------------------------------------------------

        if execution["stdout"].strip():

            print()
            print(C["bold"] + "Saída do Python:")
            line()

            print(
                execution["stdout"].rstrip()
            )

            line()

        if execution["stderr"].strip():

            print()
            print(C["bold"] + C["red"] + "Erro:")

            print(
                execution["stderr"].rstrip()
            )

        # ----------------------------------------------------
        # PROBLEMA SEMÂNTICO
        # ----------------------------------------------------

        if verification:

            if verification.get("status") == "retry":

                print()
                warning(
                    "A verificação detectou uma inconsistência:"
                )

                print(
                    verification.get(
                        "problem",
                        verification.get(
                            "reason",
                            ""
                        )
                    )
                )

        print()

        if execution["exit_code"] == 0:
            success("Luna concluiu a tarefa.")
        else:
            warning(
                "Luna não conseguiu concluir a tarefa "
                "dentro das tentativas disponíveis."
            )


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def main():

    title("SYRA AI / LUNA")

    print(
        C["gray"]
        + f"Modelo: {MODEL_PATH.name}"
    )

    print(
        C["gray"]
        + f"Workspace: {WORKSPACE}"
    )

    print(
        C["gray"]
        + f"Threads: {N_THREADS}"
    )

    print(
        C["gray"]
        + "Sem histórico persistente"
    )

    try:

        agent = LunaAgent()

    except Exception as exc:

        error("Não foi possível iniciar o modelo.")

        print()
        print(str(exc))

        return

    title("AGENTE PRONTO")

    print(
        C["gray"]
        + "Digite uma tarefa ou 'sair'."
    )

    while True:

        print()

        try:

            user_request = input(
                C["bold"]
                + C["white"]
                + "Você › "
            ).strip()

        except KeyboardInterrupt:

            print()
            break

        except EOFError:

            print()
            break

        if not user_request:
            continue

        if user_request.lower() in {
            "sair",
            "exit",
            "quit",
        }:
            break

        task_started = time.perf_counter()

        try:

            # =================================================
            # 1. ROUTER
            # =================================================

            route = agent.route(
                user_request
            )

            if not route.get(
                "needs_execution",
                True
            ):

                phase("RESPOSTA")

                print(
                    "A tarefa não requer execução Python."
                )

                continue

            # =================================================
            # 2. PLANEJAMENTO
            # =================================================

            plan = agent.plan(
                user_request,
                route,
            )

            # =================================================
            # 3. SNAPSHOT
            # =================================================

            before = workspace_snapshot()

            previous_code = None
            previous_static = None
            previous_execution = None
            previous_verification = None

            final_execution = None
            final_changes = {
                "created": [],
                "modified": [],
                "deleted": [],
            }

            attempts = 0

            # =================================================
            # 4. LOOP REACT
            # =================================================

            while attempts < MAX_ATTEMPTS:

                attempts += 1

                if attempts > 1:

                    phase(
                        f"REPARAÇÃO — TENTATIVA {attempts}"
                    )

                # ---------------------------------------------
                # Geração
                # ---------------------------------------------

                code = agent.generate_command(
                    user_request=user_request,
                    route=route,
                    plan=plan,
                    previous_code=previous_code,
                    static=previous_static,
                    execution=previous_execution,
                    verification=previous_verification,
                )

                if not code:

                    error(
                        "O modelo não gerou código."
                    )

                    final_execution = {
                        "exit_code": -5,
                        "stdout": "",
                        "stderr": "Empty model output",
                        "elapsed": 0,
                        "timed_out": False,
                    }

                    break

                # ---------------------------------------------
                # Código
                # ---------------------------------------------

                phase("COMANDO GERADO")

                line()

                print(code)

                line()

                # ---------------------------------------------
                # Análise estática
                # ---------------------------------------------

                static = agent.validate_code(
                    code
                )

                # Se sintaxe estiver errada,
                # não desperdiçamos execução.
                if not static["syntax_ok"]:

                    previous_code = code
                    previous_static = static

                    previous_execution = {
                        "exit_code": -2,
                        "stdout": "",
                        "stderr": static[
                            "syntax_error"
                        ],
                        "elapsed": 0,
                        "timed_out": False,
                    }

                    previous_verification = {
                        "status": "retry",
                        "problem": (
                            "Erro de sintaxe Python."
                        ),
                        "reason": static[
                            "syntax_error"
                        ],
                    }

                    if attempts < MAX_ATTEMPTS:

                        warning(
                            "Código inválido. "
                            "Solicitando correção."
                        )

                        continue

                    final_execution = previous_execution
                    break

                # ---------------------------------------------
                # Execução
                # ---------------------------------------------

                execution = agent.execute(
                    code
                )

                after = workspace_snapshot()

                changes = workspace_changes(
                    before,
                    after,
                )

                final_execution = execution
                final_changes = changes

                # ---------------------------------------------
                # Erro técnico
                # ---------------------------------------------

                if execution["exit_code"] != 0:

                    previous_code = code
                    previous_static = static
                    previous_execution = execution

                    previous_verification = {
                        "status": "retry",
                        "problem": (
                            "A execução Python falhou."
                        ),
                        "reason": execution[
                            "stderr"
                        ],
                    }

                    if attempts < MAX_ATTEMPTS:

                        warning(
                            "Execução falhou. "
                            "O Luna vai analisar o erro "
                            "e tentar novamente."
                        )

                        continue

                    break

                # ---------------------------------------------
                # Sucesso técnico
                # ---------------------------------------------

                needs_verify = agent.should_semantic_verify(
                    route,
                    static,
                    execution,
                )

                if not needs_verify:

                    success(
                        "Execução concluída e nenhuma "
                        "verificação semântica adicional "
                        "é necessária."
                    )

                    break

                # ---------------------------------------------
                # Verificação semântica
                # ---------------------------------------------

                verification = agent.verify(
                    user_request=user_request,
                    route=route,
                    plan=plan,
                    code=code,
                    static=static,
                    execution=execution,
                    changes=changes,
                )

                if verification.get("status") == "ok":

                    success(
                        "Resultado considerado coerente "
                        "com a tarefa."
                    )

                    previous_verification = verification

                    break

                # ---------------------------------------------
                # Resultado semanticamente incorreto
                # ---------------------------------------------

                previous_code = code
                previous_static = static
                previous_execution = execution
                previous_verification = verification

                if attempts < MAX_ATTEMPTS:

                    warning(
                        "O código executou, mas o resultado "
                        "foi considerado conceitualmente "
                        "incorreto."
                    )

                    continue

                break

            # =================================================
            # 5. RESULTADO
            # =================================================

            agent.final_result(
                code=previous_code,
                execution=final_execution,
                changes=final_changes,
                attempts=attempts,
                verification=previous_verification,
            )

            total = time.perf_counter() - task_started

            print()
            info(
                "Tempo total da tarefa",
                f"{total:.2f}s"
            )

        except KeyboardInterrupt:

            print()
            warning(
                "Tarefa interrompida pelo usuário."
            )

        except Exception:

            error("Erro interno no agente.")

            print()

            traceback.print_exc()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()