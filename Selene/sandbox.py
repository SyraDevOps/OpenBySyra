"""
Luna Sandbox Pro · V5
=====================

Runtime local de ferramentas para agentes/LLMs com:
- compatibilidade com a Luna V4
- workspace isolado em sandbox_data/
- sessões independentes
- eventos JSON em tempo real
- auditoria SQLite
- perfis de capacidades
- shell/exec controlado dentro do workspace
- Python inline e scripts
- filesystem completo no workspace
- HTTP público com proteção contra SSRF
- downloads
- SQL
- grep/glob/stat
- hash
- zip/unzip
- Git controlado no workspace
- processos de longa duração com start/status/stop
- módulos core/
- agentes JSON encadeáveis
- retries, condições, continue_on_error
- variáveis {{ ... }} entre etapas
- resolução fuzzy de módulos/agentes
- backend host ou Docker
- quotas de tempo, saída e arquivo
- política fail-closed para caminhos e rede
- nenhum acesso arbitrário fora do workspace

A ideia é dar MUITA autonomia ao modelo dentro da caixa, sem transformar o
sandbox em acesso irrestrito ao computador do usuário.

Exemplos:
    python sandbox_pro.py --catalog --json
    python sandbox_pro.py --list agents --json
    python sandbox_pro.py --module diagnostico --module-args "{}" --json
    python sandbox_pro.py --agent pesquisa --context "pesquise Gemma 4" --json

Tool direta:
    python sandbox_pro.py --tool exec --tool-args "{\"command\":[\"python\",\"-V\"]}" --json

Perfis:
    SANDBOX_PROFILE=safe|builder|autonomous

Backend:
    SANDBOX_BACKEND=host|docker

No backend Docker, o workspace é montado em /workspace e o processo é isolado.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import ipaddress
import json
import os
import queue
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
import zipfile

from dataclasses import dataclass
from difflib import get_close_matches, SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
AGENTS = ROOT / "agents"
WORK = ROOT / "sandbox_data"
SCRIPTS = AGENTS / "scripts"
ENV_FILE = ROOT / ".env"
AUDIT_DB = WORK / "_sandbox" / "audit.db"
PROC_DIR = WORK / "_sandbox" / "processes"
SESSION_DIR = WORK / "sessions"

MAX_OUTPUT = int(os.getenv("SANDBOX_MAX_OUTPUT", "50000"))
MAX_FILE_BYTES = int(os.getenv("SANDBOX_MAX_FILE_BYTES", str(8 * 1024 * 1024)))
MAX_DOWNLOAD_BYTES = int(os.getenv("SANDBOX_MAX_DOWNLOAD_BYTES", str(32 * 1024 * 1024)))
TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "60"))
MAX_AGENT_STEPS = int(os.getenv("SANDBOX_MAX_AGENT_STEPS", "40"))
MAX_RETRIES = int(os.getenv("SANDBOX_MAX_RETRIES", "3"))
PROFILE = os.getenv("SANDBOX_PROFILE", "builder").strip().casefold()
BACKEND = os.getenv("SANDBOX_BACKEND", "host").strip().casefold()
DOCKER_IMAGE = os.getenv("SANDBOX_DOCKER_IMAGE", "python:3.11-slim")
NETWORK_ENABLED = os.getenv("SANDBOX_NETWORK", "1") != "0"
ALLOW_GIT = os.getenv("SANDBOX_GIT", "1") != "0"
ALLOW_PROCESS = os.getenv("SANDBOX_PROCESS", "1") != "0"

TOKEN = re.compile(r"{{\s*([^{}]+?)\s*}}")
VALID_SESSION = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")

# Ferramentas por perfil.
PROFILE_TOOLS = {
    "safe": {
        "http", "curl", "download",
        "read", "list", "glob", "grep", "stat", "hash",
        "sql",
        "module",
        "ask", "llm_transform",
    },
    "builder": {
        "http", "curl", "download",
        "read", "write", "append", "mkdir", "copy", "move", "delete",
        "list", "glob", "grep", "stat", "hash",
        "python", "exec",
        "sql",
        "module",
        "zip", "unzip",
        "git",
        "ask", "llm_transform",
    },
    "autonomous": {
        "http", "curl", "download",
        "read", "write", "append", "mkdir", "copy", "move", "delete",
        "list", "glob", "grep", "stat", "hash",
        "python", "exec",
        "process_start", "process_status", "process_stop",
        "sql",
        "module",
        "zip", "unzip",
        "git",
        "ask", "llm_transform",
    },
}

if PROFILE not in PROFILE_TOOLS:
    PROFILE = "builder"

# Alguns executáveis do host são explicitamente proibidos mesmo no perfil
# autonomous. Isso impede o agente de transformar a caixa em administração
# irrestrita do sistema operacional.
DENIED_EXECUTABLES = {
    "powershell", "powershell.exe", "pwsh", "pwsh.exe",
    "cmd", "cmd.exe",
    "reg", "reg.exe",
    "schtasks", "schtasks.exe",
    "sc", "sc.exe",
    "wmic", "wmic.exe",
    "net", "net.exe", "netsh", "netsh.exe",
    "diskpart", "diskpart.exe",
    "shutdown", "shutdown.exe",
    "reboot",
    "sudo", "su",
    "mount", "umount",
}

# Binários normalmente úteis em desenvolvimento. Outros podem ser usados no
# backend Docker, mas no host exigem esta allowlist.
HOST_EXEC_ALLOWLIST = {
    "python", "python.exe", "py", "py.exe",
    "git", "git.exe",
    "node", "node.exe",
    "npm", "npm.cmd", "npm.exe",
    "npx", "npx.cmd", "npx.exe",
    "pip", "pip.exe", "pip3", "pip3.exe",
    "pytest", "pytest.exe",
    "ruff", "ruff.exe",
    "black", "black.exe",
    "mypy", "mypy.exe",
    "cargo", "cargo.exe",
    "rustc", "rustc.exe",
    "go", "go.exe",
    "java", "java.exe", "javac", "javac.exe",
    "gcc", "g++", "clang", "clang++",
    "cmake", "cmake.exe",
    "make", "make.exe",
    "curl", "curl.exe",
}

# O agente pode pedir comandos adicionais via variável de ambiente.
_extra_exec = os.getenv("SANDBOX_EXEC_ALLOW", "").strip()
if _extra_exec:
    HOST_EXEC_ALLOWLIST |= {
        x.strip().casefold()
        for x in _extra_exec.split(",")
        if x.strip()
    }


# ============================================================
# EVENTOS
# ============================================================

_OUTPUT_AS_JSON = False


def emit(event: str, message: str, **data: Any) -> None:
    """Evento intermediário compatível com a Luna V4."""
    if not _OUTPUT_AS_JSON:
        return

    payload = {
        "event": event,
        "message": message,
        "time": time.time(),
    }
    payload.update(data)
    print(json.dumps(payload, ensure_ascii=True), flush=True)


# ============================================================
# AUDITORIA
# ============================================================

def init_audit() -> None:
    AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(AUDIT_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                session_id TEXT,
                action TEXT NOT NULL,
                target TEXT,
                ok INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                detail TEXT
            )
            """
        )
        conn.commit()


def audit(
    action: str,
    target: str,
    ok: bool,
    started: float,
    *,
    session_id: str = "",
    detail: Any = None,
) -> None:
    try:
        init_audit()

        if isinstance(detail, (dict, list)):
            detail_text = json.dumps(detail, ensure_ascii=False)
        else:
            detail_text = str(detail or "")

        # Nunca grava payload gigantesco no banco de auditoria.
        detail_text = detail_text[:10000]

        with sqlite3.connect(AUDIT_DB) as conn:
            conn.execute(
                """
                INSERT INTO audit(
                    created_at, session_id, action, target,
                    ok, duration_ms, detail
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    session_id,
                    action,
                    target[:500],
                    1 if ok else 0,
                    int((time.perf_counter() - started) * 1000),
                    detail_text,
                ),
            )
            conn.commit()
    except Exception:
        # Auditoria nunca pode derrubar a ferramenta principal.
        pass


def recent_audit(limit: int = 30) -> dict[str, Any]:
    init_audit()
    limit = max(1, min(int(limit), 200))

    with sqlite3.connect(AUDIT_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, session_id, action, target,
                   ok, duration_ms, detail
            FROM audit
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return {
        "ok": True,
        "rows": [dict(r) for r in rows],
    }


# ============================================================
# SESSÃO / WORKSPACE
# ============================================================

def ensure_dirs() -> None:
    for p in (CORE, AGENTS, WORK, SCRIPTS, SESSION_DIR, PROC_DIR):
        p.mkdir(parents=True, exist_ok=True)


def sanitize_session(raw: str | None) -> str:
    value = (raw or "default").strip()

    if not VALID_SESSION.fullmatch(value):
        raise ValueError(
            "ID de sessão inválido. Use letras, números, ponto, hífen ou underscore."
        )

    return value


def session_root(session_id: str) -> Path:
    ensure_dirs()
    sid = sanitize_session(session_id)
    root = (SESSION_DIR / sid).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def work_path(raw: str, session_id: str = "default") -> Path:
    """
    Resolve caminhos exclusivamente dentro do workspace da sessão.

    O usuário pode usar:
      "."                     -> raiz da sessão
      "arquivo.txt"
      "projeto/app.py"

    Nenhum ../ consegue escapar.
    """
    base = session_root(session_id).resolve()
    raw = str(raw or ".").replace("\\", "/").lstrip("/")

    candidate = (base / raw).resolve()

    if candidate == base or base in candidate.parents:
        return candidate

    raise ValueError("Caminho fora do workspace da sessão.")


def relative_work_path(path: Path, session_id: str) -> str:
    return str(path.resolve().relative_to(session_root(session_id).resolve()))


# ============================================================
# NOMES / RESOLUÇÃO FUZZY
# ============================================================

def normalized(raw: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", str(raw).casefold())
        if not unicodedata.combining(c)
        and (c.isalnum() or c == "_")
    )


def safe_name(raw: str, label: str) -> str:
    n = (
        str(raw)
        .strip()
        .removesuffix(".py")
        .removesuffix(".json")
    )

    if not n or not n.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Nome de {label} inválido.")

    return n


def resolve_file(
    folder: Path,
    suffix: str,
    raw: str,
    label: str,
) -> tuple[str, Path]:
    requested = safe_name(raw, label)
    direct = (folder / f"{requested}{suffix}").resolve()

    if direct.parent == folder.resolve() and direct.is_file():
        return requested, direct

    options = {
        normalized(p.stem): (p.stem, p.resolve())
        for p in folder.glob(f"*{suffix}")
        if not p.name.startswith("_")
    }

    close = get_close_matches(
        normalized(requested),
        list(options),
        n=1,
        cutoff=0.68,
    )

    if close:
        resolved = options[close[0]]
        emit(
            "info",
            f"Nome aproximado resolvido: {requested} → {resolved[0]}",
        )
        return resolved

    available = ", ".join(
        sorted(p.stem for p in folder.glob(f"*{suffix}"))
    ) or "nenhum"

    raise ValueError(
        f"{label.capitalize()} '{requested}' não encontrado. "
        f"Disponíveis: {available}."
    )


# ============================================================
# SEGREDOS / ENV
# ============================================================

def sandbox_env() -> dict[str, str]:
    """
    Ambiente mínimo. Não entrega o os.environ inteiro ao processo executado.
    Secrets do .env ficam disponíveis apenas para substituição explícita em HTTP.
    """
    values = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "NO_COLOR": "1",
    }

    if os.name == "nt":
        for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT"):
            if key in os.environ:
                values[key] = os.environ[key]

    return values


def secrets() -> dict[str, str]:
    values: dict[str, str] = {}

    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            stripped = line.strip()

            if (
                not stripped
                or stripped.startswith("#")
                or "=" not in stripped
            ):
                continue

            key, value = stripped.split("=", 1)

            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
                values[key.strip()] = value.strip().strip("'\"")

    return values


def substitute_secret(text: str) -> str:
    values = secrets()

    return re.sub(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
        lambda m: values.get(m.group(1), ""),
        str(text),
    )


# ============================================================
# REDE / SSRF
# ============================================================

def validated_public_target(raw: str) -> tuple[str, str, int, str]:
    if not NETWORK_ENABLED:
        raise ValueError("Rede desabilitada pela política do sandbox.")

    value = str(raw).strip()

    if "://" not in value:
        value = "https://" + value

    parsed = urlparse(value)

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Use URL HTTP/HTTPS pública válida.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in {80, 443}:
        raise ValueError("Somente portas HTTP 80/443 são permitidas.")

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as e:
        raise ValueError(
            f"Domínio não resolvido: {parsed.hostname}"
        ) from e

    target_ip = None
    for entry in addresses:
        ip = ipaddress.ip_address(entry[4][0])

        if not ip.is_global:
            raise ValueError(
                f"Destino privado/local bloqueado: {ip}"
            )
        if target_ip is None:
            target_ip = str(ip)

    if target_ip is None:
        raise ValueError(f"Nenhum IP público encontrado para {parsed.hostname}")
    return value, parsed.hostname, port, target_ip


def validated_public_url(raw: str) -> str:
    return validated_public_target(raw)[0]


def http_request(
    raw: str,
    method: str = "GET",
    headers: dict[str, Any] | None = None,
    body: Any = None,
    *,
    data: Any = None,
    json_body: Any = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    url, hostname, port, pinned_ip = validated_public_target(raw)
    method = str(method).upper()

    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
        raise ValueError("Método HTTP não permitido.")

    if body is None and data is not None:
        body = data
    if body is None and json_body is not None:
        body = json_body

    headers = headers or {}
    header_args: list[str] = []

    for key, value in headers.items():
        key = str(key)

        if not re.fullmatch(r"[A-Za-z0-9-]+", key):
            raise ValueError(f"Header inválido: {key}")

        rendered = substitute_secret(str(value))
        header_args += ["-H", f"{key}: {rendered}"]

    if body is not None and "content-type" not in {str(k).lower() for k in headers}:
        if isinstance(body, (dict, list)) or json_body is not None:
            header_args += ["-H", "Content-Type: application/json"]

    curl = "curl.exe" if os.name == "nt" else "curl"

    cmd = [
        curl,
        "--silent",
        "--show-error",
        "--max-redirs",
        "0",
        "--max-time",
        str(TIMEOUT),
        "--max-filesize",
        str(MAX_OUTPUT),
        "--resolve",
        f"{hostname}:{port}:{pinned_ip}",
        "--request",
        method,
        *header_args,
    ]

    if body is not None:
        if isinstance(body, (dict, list)):
            payload = json.dumps(body, ensure_ascii=False)
        else:
            payload = str(body)
        cmd += ["--data-raw", payload]

    cmd += [
        "--output",
        "-",
        "--write-out",
        "\n__STATUS__:%{http_code}",
        "--",
        url,
    ]

    emit("progress", f"HTTP {method} {url}")

    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT + 5,
            shell=False,
            env=sandbox_env(),
        )
    except FileNotFoundError as e:
        raise RuntimeError("curl não foi encontrado.") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Requisição excedeu o timeout.") from e

    response_body, _, status_raw = p.stdout[-(MAX_OUTPUT + 200):].rpartition(
        "\n__STATUS__:"
    )

    status = (
        int(status_raw.strip())
        if status_raw.strip().isdigit()
        else None
    )

    result = {
        "ok": (
            p.returncode == 0
            and (status is None or status < 400)
        ),
        "url": url,
        "method": method,
        "status": status,
        "body": response_body[:MAX_OUTPUT],
        "error": p.stderr[:4000] or None,
    }

    audit(
        "http",
        url,
        result["ok"],
        started,
        detail={"method": method, "status": status},
    )

    return result


def download_file(
    raw_url: str,
    target: str,
    *,
    session_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    url = validated_public_url(raw_url)
    dest = work_path(target, session_id)
    dest.parent.mkdir(parents=True, exist_ok=True)

    curl = "curl.exe" if os.name == "nt" else "curl"

    cmd = [
        curl,
        "--fail",
        "--silent",
        "--show-error",
        "--max-redirs",
        "0",
        "--max-time",
        str(TIMEOUT),
        "--max-filesize",
        str(MAX_DOWNLOAD_BYTES),
        "--output",
        str(dest),
        "--",
        url,
    ]

    emit("progress", f"Baixando {url}")

    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT + 5,
            shell=False,
            env=sandbox_env(),
        )
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    if p.returncode != 0:
        dest.unlink(missing_ok=True)

    size = dest.stat().st_size if dest.is_file() else 0

    result = {
        "ok": p.returncode == 0 and dest.is_file(),
        "url": url,
        "path": relative_work_path(dest, session_id) if dest.exists() else target,
        "bytes": size,
        "stderr": p.stderr[:4000] or None,
    }

    audit(
        "download",
        url,
        result["ok"],
        started,
        session_id=session_id,
        detail={"target": target, "bytes": size},
    )

    return result


# ============================================================
# PROCESS EXECUTION
# ============================================================

def _safe_command_list(command: Any) -> list[str]:
    if isinstance(command, str):
        # Não usa shell=True. shlex serve apenas para tokenizar.
        cmd = shlex.split(command, posix=(os.name != "nt"))
    elif isinstance(command, list):
        cmd = [str(x) for x in command]
    else:
        raise ValueError("command deve ser string ou lista.")

    if not cmd:
        raise ValueError("Comando vazio.")

    exe = Path(cmd[0]).name.casefold()

    if exe in DENIED_EXECUTABLES:
        raise ValueError(
            f"Executável bloqueado pela política do sandbox: {exe}"
        )

    if BACKEND == "host" and exe not in {
        x.casefold() for x in HOST_EXEC_ALLOWLIST
    }:
        raise ValueError(
            f"Executável '{exe}' não está na allowlist do backend host. "
            "Adicione-o em SANDBOX_EXEC_ALLOW ou use SANDBOX_BACKEND=docker."
        )

    return cmd


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _docker_wrap(
    command: list[str],
    *,
    session_id: str,
    network: bool,
) -> list[str]:
    if not _docker_available():
        raise RuntimeError(
            "SANDBOX_BACKEND=docker foi selecionado, mas Docker não está disponível."
        )

    root = session_root(session_id).resolve()

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--pids-limit",
        "128",
        "--memory",
        os.getenv("SANDBOX_DOCKER_MEMORY", "2g"),
        "--cpus",
        os.getenv("SANDBOX_DOCKER_CPUS", "2"),
        "--workdir",
        "/workspace",
        "--mount",
        f"type=bind,source={root},target=/workspace",
    ]

    if not network:
        docker_cmd += ["--network", "none"]

    docker_cmd += [
        DOCKER_IMAGE,
        *command,
    ]

    return docker_cmd


def exec_command(
    command: Any,
    *,
    session_id: str,
    cwd: str = ".",
    stdin: Any = "",
    timeout: int | None = None,
    network: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    cmd = _safe_command_list(command)
    local_cwd = work_path(cwd, session_id)
    local_cwd.mkdir(parents=True, exist_ok=True)

    timeout = max(1, min(int(timeout or TIMEOUT), TIMEOUT * 5))

    if BACKEND == "docker":
        run_cmd = _docker_wrap(
            cmd,
            session_id=session_id,
            network=bool(network and NETWORK_ENABLED),
        )
        run_cwd = ROOT
    else:
        run_cmd = cmd
        run_cwd = local_cwd

    emit(
        "progress",
        "Executando: " + " ".join(shlex.quote(x) for x in cmd[:12]),
    )

    payload = (
        json.dumps(stdin, ensure_ascii=False)
        if isinstance(stdin, (dict, list))
        else str(stdin or "")
    )

    try:
        p = subprocess.run(
            run_cmd,
            cwd=run_cwd,
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            env=sandbox_env() if BACKEND == "host" else None,
        )
    except subprocess.TimeoutExpired as e:
        result = {
            "ok": False,
            "error": f"Comando excedeu {timeout}s.",
            "stdout": (e.stdout or "")[-MAX_OUTPUT:] if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "")[-8000:] if isinstance(e.stderr, str) else "",
        }
        audit(
            "exec",
            cmd[0],
            False,
            started,
            session_id=session_id,
            detail=result,
        )
        return result

    result = {
        "ok": p.returncode == 0,
        "command": cmd,
        "returncode": p.returncode,
        "stdout": p.stdout[-MAX_OUTPUT:],
        "stderr": p.stderr[-8000:] or None,
        "backend": BACKEND,
    }

    audit(
        "exec",
        cmd[0],
        result["ok"],
        started,
        session_id=session_id,
        detail={
            "returncode": p.returncode,
            "command": cmd[:20],
        },
    )

    return result


# ============================================================
# PROCESSOS LONGOS
# ============================================================

def _proc_meta_path(proc_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", proc_id):
        raise ValueError("ID de processo inválido.")
    return PROC_DIR / f"{proc_id}.json"


def process_start(
    command: Any,
    *,
    session_id: str,
    cwd: str = ".",
) -> dict[str, Any]:
    if not ALLOW_PROCESS:
        raise ValueError("Processos persistentes desabilitados.")

    if BACKEND != "host":
        raise ValueError(
            "process_start persistente atualmente requer SANDBOX_BACKEND=host."
        )

    started = time.perf_counter()
    cmd = _safe_command_list(command)
    local_cwd = work_path(cwd, session_id)
    local_cwd.mkdir(parents=True, exist_ok=True)

    proc_id = uuid.uuid4().hex[:16]
    log_dir = work_path(f"_process_logs/{proc_id}", session_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    out = stdout_path.open("ab")
    err = stderr_path.open("ab")

    try:
        creationflags = 0
        start_new_session = True

        if os.name == "nt":
            creationflags = getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0,
            )
            start_new_session = False

        p = subprocess.Popen(
            cmd,
            cwd=local_cwd,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            shell=False,
            env=sandbox_env(),
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
    finally:
        out.close()
        err.close()

    meta = {
        "id": proc_id,
        "pid": p.pid,
        "session_id": session_id,
        "command": cmd,
        "cwd": relative_work_path(local_cwd, session_id),
        "stdout": relative_work_path(stdout_path, session_id),
        "stderr": relative_work_path(stderr_path, session_id),
        "started_at": time.time(),
    }

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    _proc_meta_path(proc_id).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    audit(
        "process_start",
        cmd[0],
        True,
        started,
        session_id=session_id,
        detail=meta,
    )

    return {"ok": True, **meta}


def _pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            p = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                shell=False,
            )
            return str(pid) in p.stdout

        os.kill(pid, 0)
        return True
    except Exception:
        return False


def process_status(proc_id: str) -> dict[str, Any]:
    meta_path = _proc_meta_path(proc_id)

    if not meta_path.is_file():
        raise ValueError("Processo não encontrado.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sid = meta["session_id"]

    stdout_path = work_path(meta["stdout"], sid)
    stderr_path = work_path(meta["stderr"], sid)

    stdout = (
        stdout_path.read_text(
            encoding="utf-8",
            errors="replace",
        )[-MAX_OUTPUT:]
        if stdout_path.is_file()
        else ""
    )
    stderr = (
        stderr_path.read_text(
            encoding="utf-8",
            errors="replace",
        )[-8000:]
        if stderr_path.is_file()
        else ""
    )

    alive = _pid_alive(int(meta["pid"]))

    return {
        "ok": True,
        **meta,
        "alive": alive,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
    }


def process_stop(proc_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    meta_path = _proc_meta_path(proc_id)

    if not meta_path.is_file():
        raise ValueError("Processo não encontrado.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pid = int(meta["pid"])

    if not _pid_alive(pid):
        return {
            "ok": True,
            "id": proc_id,
            "pid": pid,
            "already_stopped": True,
        }

    if os.name == "nt":
        p = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            shell=False,
        )
        ok = p.returncode == 0
        detail = p.stderr or p.stdout
    else:
        try:
            os.kill(pid, 15)
            ok = True
            detail = "SIGTERM enviado."
        except OSError as e:
            ok = False
            detail = str(e)

    audit(
        "process_stop",
        str(pid),
        ok,
        started,
        session_id=meta["session_id"],
        detail=detail,
    )

    return {
        "ok": ok,
        "id": proc_id,
        "pid": pid,
        "message": detail,
    }


# ============================================================
# FILESYSTEM
# ============================================================

def fs_read(path: str, *, session_id: str) -> dict[str, Any]:
    p = work_path(path, session_id)

    if not p.is_file():
        raise ValueError("Arquivo não encontrado.")

    if p.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(
            f"Arquivo excede limite de leitura ({MAX_FILE_BYTES} bytes)."
        )

    data = p.read_text(encoding="utf-8", errors="replace")

    return {
        "ok": True,
        "path": relative_work_path(p, session_id),
        "content": data[:MAX_OUTPUT],
        "truncated": len(data) > MAX_OUTPUT,
        "bytes": p.stat().st_size,
    }


def fs_write(
    path: str,
    content: Any,
    *,
    session_id: str,
    append: bool = False,
) -> dict[str, Any]:
    p = work_path(path, session_id)
    p.parent.mkdir(parents=True, exist_ok=True)

    text = str(content)

    encoded = text.encode("utf-8")
    if len(encoded) > MAX_FILE_BYTES:
        raise ValueError(
            f"Conteúdo excede limite ({MAX_FILE_BYTES} bytes)."
        )

    if append:
        with p.open("a", encoding="utf-8") as f:
            f.write(text)
    else:
        p.write_text(text, encoding="utf-8")

    return {
        "ok": True,
        "path": relative_work_path(p, session_id),
        "bytes": p.stat().st_size,
        "mode": "append" if append else "write",
    }


def fs_list(
    path: str = ".",
    *,
    session_id: str,
    recursive: bool = False,
    limit: int = 500,
) -> dict[str, Any]:
    p = work_path(path, session_id)

    if not p.exists():
        raise ValueError("Caminho não encontrado.")

    limit = max(1, min(int(limit), 2000))

    if p.is_file():
        items = [p]
    elif recursive:
        items = list(p.rglob("*"))
    else:
        items = list(p.iterdir())

    rows = []

    for item in sorted(items, key=lambda x: str(x).casefold())[:limit]:
        try:
            rows.append(
                {
                    "path": relative_work_path(item, session_id),
                    "type": "dir" if item.is_dir() else "file",
                    "bytes": item.stat().st_size if item.is_file() else None,
                }
            )
        except OSError:
            pass

    return {
        "ok": True,
        "path": relative_work_path(p, session_id),
        "items": rows,
        "truncated": len(items) > limit,
    }


def fs_glob(
    pattern: str,
    *,
    session_id: str,
    limit: int = 500,
) -> dict[str, Any]:
    root = session_root(session_id)
    limit = max(1, min(int(limit), 2000))

    # Path.glob não deixa pattern absoluto escapar, mas bloqueamos .. também.
    if ".." in Path(pattern).parts:
        raise ValueError("Glob com '..' não é permitido.")

    matches = list(root.glob(pattern))[:limit]

    return {
        "ok": True,
        "pattern": pattern,
        "matches": [
            relative_work_path(p, session_id)
            for p in matches
        ],
    }


def fs_grep(
    query: str,
    *,
    session_id: str,
    pattern: str = "**/*",
    regex: bool = False,
    case_sensitive: bool = False,
    limit: int = 200,
) -> dict[str, Any]:
    root = session_root(session_id)
    limit = max(1, min(int(limit), 1000))

    flags = 0 if case_sensitive else re.IGNORECASE

    if regex:
        rx = re.compile(query, flags)
    else:
        rx = re.compile(re.escape(query), flags)

    matches = []

    for p in root.glob(pattern):
        if not p.is_file():
            continue

        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue

            text = p.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        for line_no, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                matches.append(
                    {
                        "path": relative_work_path(p, session_id),
                        "line": line_no,
                        "text": line[:1000],
                    }
                )

                if len(matches) >= limit:
                    return {
                        "ok": True,
                        "query": query,
                        "matches": matches,
                        "truncated": True,
                    }

    return {
        "ok": True,
        "query": query,
        "matches": matches,
        "truncated": False,
    }


def fs_stat(path: str, *, session_id: str) -> dict[str, Any]:
    p = work_path(path, session_id)

    if not p.exists():
        raise ValueError("Caminho não encontrado.")

    st = p.stat()

    return {
        "ok": True,
        "path": relative_work_path(p, session_id),
        "type": "dir" if p.is_dir() else "file",
        "bytes": st.st_size if p.is_file() else None,
        "mtime": st.st_mtime,
        "ctime": st.st_ctime,
    }


def fs_hash(
    path: str,
    *,
    session_id: str,
    algorithm: str = "sha256",
) -> dict[str, Any]:
    p = work_path(path, session_id)

    if not p.is_file():
        raise ValueError("Arquivo não encontrado.")

    try:
        h = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError("Algoritmo de hash inválido.") from e

    with p.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)

    return {
        "ok": True,
        "path": relative_work_path(p, session_id),
        "algorithm": algorithm,
        "digest": h.hexdigest(),
    }


def fs_delete(path: str, *, session_id: str) -> dict[str, Any]:
    p = work_path(path, session_id)

    if p == session_root(session_id).resolve():
        raise ValueError("Não é permitido apagar a raiz da sessão.")

    existed = p.exists()

    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)

    return {
        "ok": True,
        "path": path,
        "existed": existed,
    }


def fs_copy_move(
    source: str,
    target: str,
    *,
    session_id: str,
    move: bool,
) -> dict[str, Any]:
    src = work_path(source, session_id)
    dst = work_path(target, session_id)

    if not src.exists():
        raise ValueError("Origem não encontrada.")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if move:
        shutil.move(str(src), str(dst))
    else:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    return {
        "ok": True,
        "source": source,
        "path": relative_work_path(dst, session_id),
        "operation": "move" if move else "copy",
    }


# ============================================================
# ZIP
# ============================================================

def zip_create(
    source: str,
    target: str,
    *,
    session_id: str,
) -> dict[str, Any]:
    src = work_path(source, session_id)
    dst = work_path(target, session_id)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise ValueError("Origem não encontrada.")

    with zipfile.ZipFile(
        dst,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        if src.is_file():
            z.write(src, arcname=src.name)
        else:
            for p in src.rglob("*"):
                if p.is_file():
                    z.write(
                        p,
                        arcname=str(p.relative_to(src)),
                    )

    return {
        "ok": True,
        "path": relative_work_path(dst, session_id),
        "bytes": dst.stat().st_size,
    }


def zip_extract(
    source: str,
    target: str,
    *,
    session_id: str,
) -> dict[str, Any]:
    src = work_path(source, session_id)
    dst = work_path(target, session_id)
    dst.mkdir(parents=True, exist_ok=True)

    if not src.is_file():
        raise ValueError("ZIP não encontrado.")

    total = 0
    extracted = 0

    with zipfile.ZipFile(src, "r") as z:
        for info in z.infolist():
            # Zip slip protection
            candidate = (dst / info.filename).resolve()

            if not (
                candidate == dst.resolve()
                or dst.resolve() in candidate.parents
            ):
                raise ValueError(
                    f"Entrada ZIP insegura: {info.filename}"
                )

            total += int(info.file_size)

            if total > MAX_DOWNLOAD_BYTES * 4:
                raise ValueError("ZIP expandido excede quota.")

        z.extractall(dst)
        extracted = len(z.infolist())

    return {
        "ok": True,
        "path": relative_work_path(dst, session_id),
        "files": extracted,
        "expanded_bytes": total,
    }


# ============================================================
# SQL
# ============================================================

def sql_query(
    database: str,
    query: str,
    params: list[Any] | tuple[Any, ...] | None,
    *,
    session_id: str,
) -> dict[str, Any]:
    db = work_path(database or "workspace.db", session_id)
    db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db, timeout=15) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(str(query), params or [])

        rows = (
            [dict(r) for r in cursor.fetchmany(1000)]
            if cursor.description
            else []
        )

        conn.commit()

        return {
            "ok": True,
            "database": relative_work_path(db, session_id),
            "rows": rows,
            "rowcount": cursor.rowcount,
            "truncated": bool(
                cursor.description and len(rows) >= 1000
            ),
        }


# ============================================================
# MODULES
# ============================================================

def run_module(
    raw: str,
    args: Any = None,
    *,
    session_id: str = "default",
) -> dict[str, Any]:
    started = time.perf_counter()
    name, path = resolve_file(
        CORE,
        ".py",
        raw,
        "módulo",
    )

    emit("progress", f"Executando módulo {name}")

    try:
        module_env = sandbox_env()
        module_env["LUNA_WORKSPACE"] = str(session_root(session_id))
        module_env["SANDBOX_SESSION"] = session_id

        p = subprocess.run(
            [
                sys.executable,
                "-I",
                str(path),
            ],
            # Compatibilidade com o sandbox anterior: módulos confiáveis de
            # core/ continuam vendo core/ como diretório de trabalho.
            cwd=CORE,
            input=json.dumps(
                args or {},
                ensure_ascii=False,
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT,
            shell=False,
            env=module_env,
        )
    except subprocess.TimeoutExpired:
        result = {
            "ok": False,
            "module": name,
            "error": f"Módulo excedeu {TIMEOUT}s.",
        }
        audit(
            "module",
            name,
            False,
            started,
            session_id=session_id,
            detail=result,
        )
        return result

    stdout = p.stdout[-MAX_OUTPUT:]

    try:
        data = json.loads(stdout.strip() or "{}")
    except json.JSONDecodeError:
        data = None

    result = {
        "ok": p.returncode == 0,
        "module": name,
        "returncode": p.returncode,
        "data": data,
        "stdout": stdout,
        "stderr": p.stderr[-8000:] or None,
    }

    audit(
        "module",
        name,
        result["ok"],
        started,
        session_id=session_id,
        detail={
            "returncode": p.returncode,
        },
    )

    return result


# ============================================================
# GIT
# ============================================================

ALLOWED_GIT_SUBCOMMANDS = {
    "status", "diff", "log", "show",
    "init", "add", "commit", "branch",
    "checkout", "switch", "restore",
    "rev-parse", "ls-files",
}


def git_tool(
    args: list[Any],
    *,
    session_id: str,
    cwd: str = ".",
) -> dict[str, Any]:
    if not ALLOW_GIT:
        raise ValueError("Git desabilitado pela política.")

    args = [str(x) for x in (args or [])]

    if not args:
        raise ValueError("Informe argumentos do git.")

    if args[0] not in ALLOWED_GIT_SUBCOMMANDS:
        raise ValueError(
            f"Subcomando git bloqueado: {args[0]}"
        )

    return exec_command(
        ["git", *args],
        session_id=session_id,
        cwd=cwd,
        network=False,
    )


# ============================================================
# TOOL DISPATCH
# ============================================================

def ensure_tool_allowed(tool_type: str) -> None:
    allowed = PROFILE_TOOLS[PROFILE]

    if tool_type not in allowed:
        raise ValueError(
            f"Ferramenta '{tool_type}' bloqueada no perfil '{PROFILE}'. "
            f"Permitidas: {', '.join(sorted(allowed))}"
        )


def tool_step(
    spec: dict[str, Any],
    *,
    session_id: str,
) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("Especificação de ferramenta deve ser objeto.")

    tool_type = str(spec.get("type", "")).strip()
    ensure_tool_allowed(tool_type)

    if tool_type in {"http", "curl"}:
        payload = spec.get("body")
        if payload is None and "data" in spec:
            payload = spec.get("data")
        if payload is None and "json" in spec:
            payload = spec.get("json")

        return http_request(
            str(spec["url"]),
            str(spec.get("method", "GET")),
            spec.get("headers"),
            payload,
            data=spec.get("data"),
            json_body=spec.get("json"),
        )

    if tool_type in {"list_directory", "list_dir"}:
        return fs_list(
            str(spec.get("path", ".")),
            session_id=session_id,
            recursive=bool(spec.get("recursive", False)),
            limit=int(spec.get("limit", 500)),
        )

    if tool_type in {"read_file", "read"}:
        return fs_read(
            str(spec.get("path", ".")),
            session_id=session_id,
        )

    if tool_type in {"write_file", "write"}:
        return fs_write(
            str(spec.get("path", "")),
            spec.get("content", ""),
            session_id=session_id,
            append=bool(spec.get("append", False)),
        )

    if tool_type == "download":
        return download_file(
            str(spec["url"]),
            str(spec["path"]),
            session_id=session_id,
        )

    if tool_type == "module":
        return run_module(
            str(spec["name"]),
            spec.get("args", spec.get("input", {})),
            session_id=session_id,
        )

    if tool_type == "mkdir":
        p = work_path(str(spec["path"]), session_id)
        p.mkdir(parents=True, exist_ok=True)
        return {
            "ok": True,
            "path": relative_work_path(p, session_id),
        }

    if tool_type == "write":
        return fs_write(
            str(spec["path"]),
            spec.get("content", ""),
            session_id=session_id,
            append=False,
        )

    if tool_type == "append":
        return fs_write(
            str(spec["path"]),
            spec.get("content", ""),
            session_id=session_id,
            append=True,
        )

    if tool_type == "read":
        return fs_read(
            str(spec["path"]),
            session_id=session_id,
        )

    if tool_type == "list":
        return fs_list(
            str(spec.get("path", ".")),
            session_id=session_id,
            recursive=bool(spec.get("recursive", False)),
            limit=int(spec.get("limit", 500)),
        )

    if tool_type == "glob":
        return fs_glob(
            str(spec["pattern"]),
            session_id=session_id,
            limit=int(spec.get("limit", 500)),
        )

    if tool_type == "grep":
        return fs_grep(
            str(spec["query"]),
            session_id=session_id,
            pattern=str(spec.get("pattern", "**/*")),
            regex=bool(spec.get("regex", False)),
            case_sensitive=bool(spec.get("case_sensitive", False)),
            limit=int(spec.get("limit", 200)),
        )

    if tool_type == "stat":
        return fs_stat(
            str(spec["path"]),
            session_id=session_id,
        )

    if tool_type == "hash":
        return fs_hash(
            str(spec["path"]),
            session_id=session_id,
            algorithm=str(spec.get("algorithm", "sha256")),
        )

    if tool_type == "copy":
        return fs_copy_move(
            str(spec["source"]),
            str(spec["target"]),
            session_id=session_id,
            move=False,
        )

    if tool_type == "move":
        return fs_copy_move(
            str(spec["source"]),
            str(spec["target"]),
            session_id=session_id,
            move=True,
        )

    if tool_type == "delete":
        return fs_delete(
            str(spec["path"]),
            session_id=session_id,
        )

    if tool_type == "exec":
        return exec_command(
            spec.get("command"),
            session_id=session_id,
            cwd=str(spec.get("cwd", ".")),
            stdin=spec.get("input", ""),
            timeout=int(spec.get("timeout", TIMEOUT)),
            network=bool(spec.get("network", False)),
        )

    if tool_type == "python":
        # Pode usar script em agents/scripts ou código inline.
        if "script" in spec:
            script = (SCRIPTS / str(spec["script"])).resolve()

            if (
                script.parent != SCRIPTS.resolve()
                or not script.is_file()
                or script.suffix != ".py"
            ):
                raise ValueError(
                    "Script deve estar em agents/scripts/."
                )

            return exec_command(
                [
                    sys.executable,
                    "-I",
                    str(script),
                    *map(str, spec.get("args", [])),
                ],
                session_id=session_id,
                cwd=str(spec.get("cwd", ".")),
                stdin=spec.get("input", {}),
                timeout=int(spec.get("timeout", TIMEOUT)),
                network=bool(spec.get("network", False)),
            )

        code = str(spec.get("code", ""))

        if not code.strip():
            raise ValueError("Informe code ou script para python.")

        tmp = work_path(
            f"_tmp/inline_{uuid.uuid4().hex[:10]}.py",
            session_id,
        )
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(code, encoding="utf-8")

        try:
            return exec_command(
                [sys.executable, "-I", str(tmp)],
                session_id=session_id,
                cwd=str(spec.get("cwd", ".")),
                stdin=spec.get("input", {}),
                timeout=int(spec.get("timeout", TIMEOUT)),
                network=bool(spec.get("network", False)),
            )
        finally:
            tmp.unlink(missing_ok=True)

    if tool_type == "process_start":
        return process_start(
            spec.get("command"),
            session_id=session_id,
            cwd=str(spec.get("cwd", ".")),
        )

    if tool_type == "process_status":
        return process_status(str(spec["id"]))

    if tool_type == "process_stop":
        return process_stop(str(spec["id"]))

    if tool_type == "sql":
        return sql_query(
            str(spec.get("database", "workspace.db")),
            str(spec["query"]),
            spec.get("params", []),
            session_id=session_id,
        )

    if tool_type == "zip":
        return zip_create(
            str(spec["source"]),
            str(spec["target"]),
            session_id=session_id,
        )

    if tool_type == "unzip":
        return zip_extract(
            str(spec["source"]),
            str(spec["target"]),
            session_id=session_id,
        )

    if tool_type == "git":
        return git_tool(
            spec.get("args", []),
            session_id=session_id,
            cwd=str(spec.get("cwd", ".")),
        )

    if tool_type == "ask":
        return {
            "ok": True,
            "needs_user_input": True,
            "question": str(spec["question"]),
        }

    if tool_type == "llm_transform":
        return {
            "ok": False,
            "needs_llm": True,
            "prompt": str(spec["prompt"]),
            "error": (
                "A etapa LLM deve ser resolvida pelo ReAct da Luna."
            ),
        }

    raise ValueError(
        f"Tipo de ferramenta não implementado: {tool_type}"
    )


# ============================================================
# AGENT STATE / TEMPLATES / CONDITIONS
# ============================================================

def get_state(data: Any, key: str) -> Any:
    current = data

    for part in str(key).split("."):
        if isinstance(current, dict):
            current = current.get(part, "")
        elif (
            isinstance(current, list)
            and part.isdigit()
            and int(part) < len(current)
        ):
            current = current[int(part)]
        else:
            return ""

    return current


def render(value: Any, state: dict[str, Any]) -> Any:
    if isinstance(value, str):
        match = TOKEN.fullmatch(value)

        if match:
            return get_state(state, match.group(1))

        return TOKEN.sub(
            lambda m: str(get_state(state, m.group(1))),
            value,
        )

    if isinstance(value, list):
        return [render(x, state) for x in value]

    if isinstance(value, dict):
        return {
            k: render(v, state)
            for k, v in value.items()
        }

    return value


def initial_state(context: str, session_id: str) -> dict[str, Any]:
    match = re.search(
        r"\b(?:sobre|acerca de|a respeito de)\s+(.+)$",
        context,
        re.I,
    )

    topic = (
        match.group(1).strip()
        if match
        else context
    )

    topic = re.sub(
        r"^\s*(?:pesquise|pesquisar|pesquisa|consulte|consultar|notícias sobre|noticias sobre)\s*",
        "",
        topic,
        flags=re.I,
    ).strip(" .,:;!?\t\r\n") or context.strip()

    return {
        "context": context,
        "context_url": quote_plus(context),
        "topic": topic,
        "topic_url": quote_plus(topic),
        "session_id": session_id,
        "steps": {},
        "vars": {},
    }


def condition_matches(
    condition: Any,
    state: dict[str, Any],
) -> bool:
    """
    Condições simples e declarativas:
      {"path":"steps.fetch.ok","equals":true}
      {"path":"steps.fetch.status","in":[200,201]}
      {"path":"steps.fetch.body","contains":"abc"}
      {"path":"steps.x.ok","truthy":true}
    """
    if condition in (None, "", {}):
        return True

    if not isinstance(condition, dict):
        return bool(condition)

    value = get_state(
        state,
        str(condition.get("path", "")),
    )

    if "equals" in condition:
        return value == condition["equals"]

    if "not_equals" in condition:
        return value != condition["not_equals"]

    if "in" in condition:
        return value in condition["in"]

    if "contains" in condition:
        return str(condition["contains"]) in str(value)

    if "truthy" in condition:
        return bool(value) is bool(condition["truthy"])

    return bool(value)


# ============================================================
# AGENT DEFINITIONS
# ============================================================

def validate_agent(definition: Any) -> None:
    if (
        not isinstance(definition, dict)
        or not isinstance(definition.get("steps"), list)
        or not definition["steps"]
    ):
        raise ValueError(
            "Agente precisa de uma lista steps não vazia."
        )

    if len(definition["steps"]) > MAX_AGENT_STEPS:
        raise ValueError(
            f"Agente excede {MAX_AGENT_STEPS} etapas."
        )

    for idx, step in enumerate(definition["steps"], 1):
        if not isinstance(step, dict):
            raise ValueError(f"Etapa {idx} deve ser objeto.")

        tool_type = step.get("type")

        if not isinstance(tool_type, str):
            raise ValueError(
                f"Etapa {idx} não possui type."
            )

        # Valida contra todas as tools conhecidas, não apenas o perfil atual.
        known = set().union(*PROFILE_TOOLS.values())

        if tool_type not in known:
            raise ValueError(
                f"Etapa {idx}: tipo desconhecido '{tool_type}'."
            )

        retries = int(step.get("retries", 0))

        if retries < 0 or retries > MAX_RETRIES:
            raise ValueError(
                f"Etapa {idx}: retries deve estar entre 0 e {MAX_RETRIES}."
            )


def create_agent(
    raw: str,
    definition: Any,
    overwrite: bool = False,
) -> dict[str, Any]:
    name = safe_name(raw, "agente")

    data = (
        json.loads(definition)
        if isinstance(definition, str)
        else definition
    )

    validate_agent(data)

    AGENTS.mkdir(parents=True, exist_ok=True)
    path = (AGENTS / f"{name}.json").resolve()

    if path.parent != AGENTS.resolve():
        raise ValueError("Destino de agente inválido.")

    if path.exists() and not overwrite:
        raise ValueError(
            "Agente já existe; use overwrite=true."
        )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    return {
        "ok": True,
        "agent": name,
        "path": f"agents/{name}.json",
    }


def run_agent(
    raw: str,
    context: str = "",
    *,
    session_id: str = "default",
) -> dict[str, Any]:
    started = time.perf_counter()
    name, path = resolve_file(
        AGENTS,
        ".json",
        raw,
        "agente",
    )

    definition = json.loads(
        path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )
    validate_agent(definition)

    state = initial_state(
        context,
        session_id,
    )
    results = []

    log_dir = work_path(
        f"_agent_logs/{name}/{int(time.time())}_{uuid.uuid4().hex[:6]}",
        session_id,
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    emit(
        "info",
        f"Agente {name} iniciado com {len(definition['steps'])} etapas.",
    )

    for idx, raw_step in enumerate(definition["steps"], 1):
        rendered = render(raw_step, state)
        ident = str(
            rendered.get("id", f"step_{idx}")
        )

        if not condition_matches(
            rendered.get("if"),
            state,
        ):
            item = {
                "step": idx,
                "id": ident,
                "type": rendered.get("type"),
                "skipped": True,
                "result": {
                    "ok": True,
                    "skipped": True,
                    "reason": "condition_false",
                },
            }
            results.append(item)
            state["steps"][ident] = item["result"]
            emit(
                "step",
                f"Etapa {idx}/{len(definition['steps'])} ignorada pela condição.",
            )
            continue

        retries = int(rendered.get("retries", 0))
        attempts = 0
        result: dict[str, Any] = {
            "ok": False,
            "error": "Etapa não executada.",
        }

        while attempts <= retries:
            attempts += 1

            emit(
                "progress",
                f"Etapa {idx}/{len(definition['steps'])} · "
                f"{rendered.get('type')} · tentativa {attempts}/{retries + 1}",
            )

            try:
                result = tool_step(
                    rendered,
                    session_id=session_id,
                )
            except (
                KeyError,
                OSError,
                RuntimeError,
                ValueError,
                sqlite3.Error,
                subprocess.TimeoutExpired,
            ) as e:
                result = {
                    "ok": False,
                    "error": str(e),
                }

            if (
                result.get("ok")
                or result.get("needs_user_input")
                or result.get("needs_llm")
            ):
                break

            if attempts <= retries:
                delay = min(
                    float(rendered.get("retry_delay", 0.5))
                    * attempts,
                    5.0,
                )
                emit(
                    "warning",
                    f"Etapa {idx} falhou; nova tentativa em {delay:.1f}s.",
                )
                time.sleep(delay)

        state["steps"][ident] = result

        step_assignments = raw_step.get("set", rendered.get("set"))
        if isinstance(step_assignments, dict):
            for key, value in step_assignments.items():
                state["vars"][str(key)] = render(
                    value,
                    state,
                )

        item = {
            "step": idx,
            "id": ident,
            "type": rendered.get("type"),
            "attempts": attempts,
            "result": result,
        }

        results.append(item)

        (log_dir / f"step_{idx:02}.json").write_text(
            json.dumps(
                item,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if result.get("ok"):
            emit(
                "step",
                f"Etapa {idx}/{len(definition['steps'])} concluída.",
            )
        else:
            emit(
                "warning",
                f"Etapa {idx}/{len(definition['steps'])} não concluiu.",
            )

        if result.get("needs_user_input"):
            break

        # needs_llm é devolvido ao ReAct para que a Luna faça a transformação.
        if result.get("needs_llm"):
            break

        if (
            not result.get("ok")
            and not rendered.get("continue_on_error")
        ):
            break

    # Sucesso apenas se todas as etapas executadas relevantes concluíram.
    relevant = [
        x for x in results
        if not x.get("skipped")
    ]
    ok = bool(relevant) and all(
        x["result"].get("ok")
        for x in relevant
    )

    # Se ficou esperando LLM/user, é parcial, não uma falha definitiva.
    needs_llm = any(
        x["result"].get("needs_llm")
        for x in relevant
    )
    needs_user_input = any(
        x["result"].get("needs_user_input")
        for x in relevant
    )

    if needs_llm or needs_user_input:
        status = "partial"
    elif ok:
        status = "success"
    else:
        status = "failure"

    output = {
        "ok": ok,
        "status": status,
        "agent": name,
        "steps": results,
        "state": state,
        "needs_llm": needs_llm,
        "needs_user_input": needs_user_input,
        "log_dir": relative_work_path(
            log_dir,
            session_id,
        ),
    }

    (log_dir / "final.json").write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit(
        "agent",
        name,
        ok,
        started,
        session_id=session_id,
        detail={
            "status": status,
            "steps": len(results),
        },
    )

    return output


# ============================================================
# CATALOG / LISTING / CAPABILITIES
# ============================================================

def listing(
    area: str,
    *,
    session_id: str = "default",
) -> dict[str, Any]:
    area = str(area).lower()

    if area == "agents":
        AGENTS.mkdir(parents=True, exist_ok=True)
        files = [
            p.stem
            for p in AGENTS.glob("*.json")
            if not p.name.startswith("_")
        ]

    elif area in {"core", "modules", "modulos", "módulos"}:
        CORE.mkdir(parents=True, exist_ok=True)
        files = [
            p.stem
            for p in CORE.glob("*.py")
            if not p.name.startswith("_")
        ]

    elif area == "workspace":
        result = fs_list(
            ".",
            session_id=session_id,
            recursive=True,
            limit=500,
        )
        files = [
            x["path"]
            for x in result["items"]
            if x["type"] == "file"
        ]

    elif area == "sessions":
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(
            p.name
            for p in SESSION_DIR.iterdir()
            if p.is_dir()
        )

    else:
        raise ValueError(
            "Área inválida: agents, core, workspace ou sessions."
        )

    return {
        "ok": True,
        "area": area,
        "files": sorted(files)[:500],
    }


def catalog() -> dict[str, Any]:
    agents = []
    modules = []

    for p in AGENTS.glob("*.json"):
        if p.name.startswith("_"):
            continue

        try:
            definition = json.loads(
                p.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

            agents.append(
                {
                    "name": p.stem,
                    "description": definition.get("description", ""),
                    "step_types": [
                        x.get("type")
                        for x in definition.get("steps", [])
                        if isinstance(x, dict)
                    ],
                    "steps": definition.get("steps", [])[:12],
                }
            )
        except Exception:
            pass

    for p in CORE.glob("*.py"):
        if p.name.startswith("_"):
            continue

        try:
            description = ast.get_docstring(
                ast.parse(
                    p.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                )
            ) or ""
        except Exception:
            description = ""

        modules.append(
            {
                "name": p.stem,
                "description": description,
            }
        )

    return {
        "ok": True,
        "version": "5.1",
        "profile": PROFILE,
        "backend": BACKEND,
        "network": NETWORK_ENABLED,
        "agents": agents,
        "modules": modules,
        "step_types": sorted(PROFILE_TOOLS[PROFILE]),
        "all_step_types": sorted(
            set().union(*PROFILE_TOOLS.values())
        ),
        "limits": {
            "timeout": TIMEOUT,
            "max_output": MAX_OUTPUT,
            "max_file_bytes": MAX_FILE_BYTES,
            "max_download_bytes": MAX_DOWNLOAD_BYTES,
            "max_agent_steps": MAX_AGENT_STEPS,
            "max_retries": MAX_RETRIES,
        },
    }


def capabilities() -> dict[str, Any]:
    return {
        "ok": True,
        "profile": PROFILE,
        "backend": BACKEND,
        "workspace_root": "sandbox_data/sessions/<session>",
        "network_enabled": NETWORK_ENABLED,
        "process_enabled": ALLOW_PROCESS,
        "git_enabled": ALLOW_GIT,
        "docker_available": _docker_available(),
        "tools": sorted(PROFILE_TOOLS[PROFILE]),
        "limits": catalog()["limits"],
        "security": {
            "workspace_only": True,
            "public_http_only": True,
            "shell_true": False,
            "host_exec_allowlist": BACKEND == "host",
            "docker_cap_drop_all": BACKEND == "docker",
            "docker_no_new_privileges": BACKEND == "docker",
        },
    }


# ============================================================
# CLI
# ============================================================

def parse_json_arg(raw: str, label: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{label} precisa ser JSON válido: {e}"
        ) from e


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Luna Sandbox Pro V5",
    )

    parser.add_argument("url", nargs="?")
    parser.add_argument("--json", action="store_true", dest="as_json")

    parser.add_argument("--session", default="default")

    parser.add_argument("--module")
    parser.add_argument("--module-args", default="{}")

    parser.add_argument("--agent")
    parser.add_argument("--context", default="")

    parser.add_argument("--list")

    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--capabilities", action="store_true")
    parser.add_argument("--audit", type=int)

    parser.add_argument("--create-agent")
    parser.add_argument("--definition")
    parser.add_argument("--overwrite", action="store_true")

    # Tool direta, muito útil para o ReAct moderno.
    parser.add_argument("--tool")
    parser.add_argument("--tool-args", default="{}")

    return parser.parse_args()


def main() -> None:
    global _OUTPUT_AS_JSON

    ensure_dirs()
    args = cli()
    _OUTPUT_AS_JSON = bool(args.as_json)

    try:
        session_id = sanitize_session(args.session)

        if args.catalog:
            result = catalog()

        elif args.capabilities:
            result = capabilities()

        elif args.audit is not None:
            result = recent_audit(args.audit)

        elif args.list:
            result = listing(
                args.list,
                session_id=session_id,
            )

        elif args.agent:
            result = run_agent(
                args.agent,
                args.context,
                session_id=session_id,
            )

        elif args.module:
            result = run_module(
                args.module,
                parse_json_arg(
                    args.module_args,
                    "--module-args",
                ),
                session_id=session_id,
            )

        elif args.create_agent:
            result = create_agent(
                args.create_agent,
                args.definition or "{}",
                args.overwrite,
            )

        elif args.tool:
            spec = parse_json_arg(
                args.tool_args,
                "--tool-args",
            )

            if not isinstance(spec, dict):
                raise ValueError(
                    "--tool-args deve ser um objeto JSON."
                )

            spec = {
                "type": args.tool,
                **spec,
            }

            result = tool_step(
                spec,
                session_id=session_id,
            )

        elif args.url:
            result = http_request(args.url)

        else:
            raise ValueError(
                "Informe URL, --tool, --module, --agent, --list, "
                "--catalog, --capabilities ou --create-agent."
            )

    except (
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        sqlite3.Error,
        OSError,
        KeyError,
    ) as e:
        result = {
            "ok": False,
            "status": "failure",
            "error": str(e),
        }

    if args.as_json:
        print(
            json.dumps(
                result,
                ensure_ascii=True,
            ),
            flush=True,
        )
    else:
        if isinstance(result, dict) and result.get("body"):
            print(result["body"])
        elif isinstance(result, dict) and result.get("error"):
            print(result["error"])
        else:
            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )

    raise SystemExit(
        0
        if isinstance(result, dict) and result.get("ok")
        else 1
    )


if __name__ == "__main__":
    main()
