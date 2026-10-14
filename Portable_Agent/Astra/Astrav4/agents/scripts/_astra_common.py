"""
Utilidades compartilhadas pelos agentes da Astra (somente biblioteca padrão).

Os scripts em agents/scripts/ rodam com `python -I` (modo isolado) e com um
ambiente mínimo, então este módulo cuida do que costuma quebrar:
  - saída UTF-8 no Windows;
  - descobrir a pasta do usuário sem depender de HOME/USERPROFILE;
  - HTTP com certificado VERIFICADO, retry, gzip e limite de tamanho;
  - bloqueio de endereços internos quando a URL vem do usuário (SSRF);
  - abrir links/arquivos no sistema (sem nunca executar programas);
  - atalhos e "últimos resultados" compartilhados entre agentes.

Contrato dos scripts: leem JSON do stdin ({"contexto": "..."}), escrevem um
relatório de texto e imprimem UMA linha JSON no stdout. Código de saída 0 =
a tarefa foi atendida (mesmo que a resposta seja "não encontrei"); código 1 =
falha de infraestrutura (rede fora do ar etc.), que o ReAct da Astra trata.
O campo "interpretado" do JSON diz o que o script entendeu do pedido.
"""

from __future__ import annotations

import difflib
import gzip
import ipaddress
import json
import os
import re
import socket
import ssl
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import zlib
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- básico

def setup_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


setup_utf8()


def read_input() -> dict[str, Any]:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8", errors="replace").strip()
        data = json.loads(raw) if raw else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def contexto() -> str:
    data = read_input()
    return str(data.get("contexto") or data.get("context") or "").strip()


def norm(text: Any) -> str:
    """minúsculas, sem acentos, espaços colapsados."""
    plain = "".join(
        c for c in unicodedata.normalize("NFKD", str(text).casefold())
        if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", " ", plain).strip()


def norm_1to1(text: str) -> str:
    """Normaliza SEM mudar o comprimento (cada caractere vira um), para que
    posições achadas no texto normalizado valham no texto original."""
    out = []
    for c in text:
        d = unicodedata.normalize("NFKD", c)
        base = next((x for x in d if not unicodedata.combining(x)), c)
        out.append(base.lower()[:1] or " ")
    return "".join(out)


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def finish(report_path: str, text: str, **fields: Any) -> None:
    """Sucesso: grava o relatório e imprime o JSON de resultado."""
    write_text(report_path, text)
    emit({"ok": True, "status": "sucesso", **fields})


def fail(message: str, report_path: str | None = None, **fields: Any) -> None:
    """Falha de infraestrutura: relatório com o erro, JSON ok=false, saída 1."""
    if report_path:
        try:
            write_text(report_path, message)
        except Exception:
            pass
    emit({"ok": False, "error": message, **fields})
    sys.exit(1)


def clean_command(text: str, extra_words: set[str] | None = None) -> str:
    """Tira só palavras de COMANDO do começo/fim do pedido ("abra", "por favor",
    "usando o agente X"), sem mexer no assunto no meio da frase."""
    words = {
        "astra", "por", "favor", "pfv", "pf", "usando", "use", "usa", "usar", "utilize",
        "agente", "agent", "o", "a", "os", "as", "um", "uma", "me", "mim", "pra", "para",
        "quero", "queria", "gostaria", "preciso", "pode", "poderia", "consegue", "voce",
        "rode", "rodar", "execute", "executar", "faca", "fazer", "faz",
    } | (extra_words or set())
    tokens = text.strip().split()
    while tokens and norm(tokens[0].strip(".,:;!?")) in words:
        tokens.pop(0)
    while tokens and norm(tokens[-1].strip(".,:;!?")) in {"por", "favor", "pfv", "pf", "agora", "ai", "ae"}:
        tokens.pop()
    return " ".join(tokens).strip(" .,:;!?\t\r\n")


# ---------------------------------------------------------------- pastas

def user_home() -> Path:
    """Pasta do usuário, mesmo com ambiente mínimo (sem HOME/USERPROFILE)."""
    for key in ("USERPROFILE", "HOME"):
        value = os.environ.get(key)
        if value and Path(value).is_dir():
            return Path(value)
    try:
        import pwd  # type: ignore

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        pass
    if os.name == "nt":
        drive, path = os.environ.get("HOMEDRIVE", ""), os.environ.get("HOMEPATH", "")
        if drive and path and Path(drive + path).is_dir():
            return Path(drive + path)
        try:
            import ctypes
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            # CSIDL_PROFILE = 40
            if ctypes.windll.shell32.SHGetFolderPathW(None, 40, None, 0, buf) == 0:  # type: ignore[attr-defined]
                return Path(buf.value)
        except Exception:
            pass
    return Path.cwd()


_FOLDER_ALIASES = {
    "downloads": ("Downloads", "downloads", "baixados", "transferencias", "download"),
    "documentos": ("Documents", "documentos", "documents", "docs", "meus documentos"),
    "desktop": ("Desktop", "area de trabalho", "desktop", "ambiente de trabalho", "mesa"),
    "imagens": ("Pictures", "imagens", "fotos", "pictures", "figuras"),
    "musicas": ("Music", "musicas", "musica", "music"),
    "videos": ("Videos", "videos", "filmes", "movies"),
}


def known_folder(name: str) -> Path | None:
    """Resolve "downloads", "documentos", "área de trabalho"... para a pasta real."""
    key = norm(name)
    home = user_home()
    for _canon, aliases in _FOLDER_ALIASES.items():
        if key in {norm(a) for a in aliases}:
            candidates = [home / aliases[0]]
            if os.name == "nt":
                one = home / "OneDrive" / aliases[0]
                candidates.append(one)
            candidates.append(home / aliases[1].title())
            for c in candidates:
                if c.is_dir():
                    return c
            return candidates[0]
    if key in {"home", "pasta pessoal", "pasta do usuario", "minha pasta"}:
        return home
    return None


def enable_user_site() -> None:
    """`python -I` ignora o site-packages do usuário (onde o pip --user instala).
    Reanexa esse caminho para permitir pacotes opcionais (psutil, ddgs)."""
    try:
        import site

        candidates = []
        try:
            candidates.append(site.getusersitepackages())
        except Exception:
            pass
        home = user_home()
        if os.name == "nt":
            base = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming") / "Python"
            if base.is_dir():
                candidates += [str(p / "site-packages") for p in base.glob("Python*")]
        for c in candidates:
            if c and Path(c).is_dir() and c not in sys.path:
                sys.path.append(c)
    except Exception:
        pass


# ---------------------------------------------------------------- HTTP

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def ssl_context() -> ssl.SSLContext:
    if os.environ.get("ASTRA_AGENTS_INSECURE_SSL") == "1":
        # Só por escolha explícita do usuário (ex.: antivírus que intercepta HTTPS).
        return ssl._create_unverified_context()  # noqa: S323
    return ssl.create_default_context()


def is_public_host(host: str) -> bool:
    """False para localhost/redes privadas/link-local (proteção SSRF)."""
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified
        ):
            return False
    return True


class HttpError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def http_get(
    url: str,
    *,
    timeout: float = 12.0,
    headers: dict[str, str] | None = None,
    max_bytes: int = 3_000_000,
    retries: int = 2,
    guard: bool = False,
) -> tuple[bytes, dict[str, str], str]:
    """GET com certificado verificado, gzip, retry e limite de tamanho.
    Devolve (corpo, cabeçalhos, url_final). guard=True bloqueia hosts internos
    (use sempre que a URL veio do usuário/da web)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HttpError("Só links http/https são permitidos.")
    if guard and not is_public_host(parsed.hostname or ""):
        raise HttpError("Endereço bloqueado (rede interna ou inexistente).")

    hdrs = {
        "User-Agent": DEFAULT_UA,
        "Accept": "*/*",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
        "Accept-Encoding": "gzip, deflate",
    }
    hdrs.update(headers or {})
    ctx = ssl_context()
    last: Exception | None = None

    class _Guard(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers_, newurl):  # noqa: ANN001
            target = urllib.parse.urlparse(newurl)
            if target.scheme not in ("http", "https"):
                raise HttpError("Redirecionamento para esquema não permitido.")
            if guard and not is_public_host(target.hostname or ""):
                raise HttpError("Redirecionamento para endereço bloqueado.")
            return super().redirect_request(req, fp, code, msg, headers_, newurl)

    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx), _Guard())

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with opener.open(req, timeout=timeout) as resp:
                body = resp.read(max_bytes + 1)[:max_bytes]
                info = {k.lower(): v for k, v in resp.headers.items()}
                enc = info.get("content-encoding", "").lower()
                try:
                    if enc == "gzip":
                        body = gzip.decompress(body)
                    elif enc == "deflate":
                        body = zlib.decompress(body)
                except Exception:
                    pass
                return body[:max_bytes], info, resp.geturl()
        except HttpError:
            raise
        except urllib.error.HTTPError as exc:
            last = HttpError(f"HTTP {exc.code} em {parsed.netloc}", exc.code)
            if exc.code < 500 and exc.code != 429:
                break
        except ssl.SSLError as exc:
            raise HttpError(
                f"Falha de certificado SSL em {parsed.netloc}: {exc}. "
                "Se um antivírus/proxy intercepta HTTPS, defina ASTRA_AGENTS_INSECURE_SSL=1."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = HttpError(f"Sem resposta de {parsed.netloc}: {getattr(exc, 'reason', exc)}")
        if attempt < retries:
            time.sleep(0.6 * (attempt + 1))
    raise last if last else HttpError("Falha de rede.")


def decode_body(body: bytes, headers: dict[str, str] | None = None) -> str:
    charset = None
    if headers:
        m = re.search(r"charset=([\w-]+)", headers.get("content-type", ""), re.I)
        if m:
            charset = m.group(1)
    if not charset:
        m = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", body[:4096], re.I)
        if m:
            charset = m.group(1).decode("ascii", "ignore")
    for enc in (charset, "utf-8", "latin-1"):
        if not enc:
            continue
        try:
            return body.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


def http_json(url: str, **kw: Any) -> Any:
    body, headers, _ = http_get(url, **kw)
    return json.loads(decode_body(body, headers))


# ---------------------------------------------------------------- abrir

_BLOCKED_EXT = {
    ".exe", ".bat", ".cmd", ".com", ".msi", ".ps1", ".vbs", ".vbe", ".js", ".jse", ".wsf",
    ".scr", ".lnk", ".reg", ".sh", ".jar", ".dll", ".hta", ".cpl", ".pif", ".app", ".command",
    ".appimage", ".desktop",
}


def safe_url(url: str) -> str | None:
    """Normaliza para http(s). "cnn.com" vira https://cnn.com. None se não for um link."""
    url = url.strip().strip("<>\"'")
    if not url:
        return None
    if re.match(r"^[a-z][a-z0-9+.-]*:", url, re.I):
        return url if url.lower().startswith(("http://", "https://")) else None
    if re.match(r"^(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:[/:?#]\S*)?$", url, re.I):
        return "https://" + url
    return None


def open_url(url: str) -> bool:
    link = safe_url(url)
    if not link:
        return False
    try:
        if webbrowser.open(link, new=2):
            return True
    except Exception:
        pass
    if os.name == "nt":
        try:
            os.startfile(link)  # type: ignore[attr-defined]
            return True
        except Exception:
            return False
    return False


def open_path(path: str | Path) -> bool:
    """Abre arquivo/pasta no programa padrão. NUNCA abre executáveis/scripts."""
    p = Path(path).expanduser()
    if not p.exists():
        return False
    if p.is_file() and p.suffix.lower() in _BLOCKED_EXT:
        return False
    try:
        if os.name == "nt":
            os.startfile(str(p))  # type: ignore[attr-defined]
            return True
        import subprocess

        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- estado compartilhado

STATE_DIR = Path("estado")


def save_last(origem: str, itens: list[dict[str, Any]]) -> None:
    """Guarda a última lista de links (busca, notícias, vídeos...) para o
    agente "abrir" poder atender "abra o 2º resultado"."""
    clean = [
        {"titulo": str(i.get("titulo") or i.get("title") or "")[:200], "url": str(i.get("url") or i.get("link") or "")}
        for i in itens
        if (i.get("url") or i.get("link"))
    ]
    if not clean:
        return
    try:
        write_text(STATE_DIR / "ultimos.json", json.dumps({"origem": origem, "ts": time.time(), "itens": clean}, ensure_ascii=False, indent=1))
    except Exception:
        pass


def load_last() -> dict[str, Any]:
    try:
        return json.loads((STATE_DIR / "ultimos.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


_ORDINALS = {
    "primeiro": 1, "primeira": 1, "1o": 1, "um": 1, "1": 1,
    "segundo": 2, "segunda": 2, "2o": 2, "dois": 2, "2": 2,
    "terceiro": 3, "terceira": 3, "3o": 3, "tres": 3, "3": 3,
    "quarto": 4, "quarta": 4, "4o": 4, "quatro": 4, "4": 4,
    "quinto": 5, "quinta": 5, "5o": 5, "cinco": 5, "5": 5,
    "sexto": 6, "sexta": 6, "6o": 6, "seis": 6, "6": 6,
    "setimo": 7, "setima": 7, "7o": 7, "sete": 7, "7": 7,
    "oitavo": 8, "oitava": 8, "8o": 8, "oito": 8, "8": 8,
    "nono": 9, "nona": 9, "9o": 9, "nove": 9, "9": 9,
    "decimo": 10, "decima": 10, "10o": 10, "dez": 10, "10": 10,
}


def parse_ordinal(text: str) -> int | None:
    """"abra o segundo resultado", "link 3", "a 2ª notícia" → 2 / 3 / 2."""
    n = norm(text).replace("º", "o").replace("ª", "a")
    if re.search(r"\b(ultimo|ultima)\b", n):
        return -1
    m = re.search(
        r"\b(?:resultado|link|noticia|video|item|opcao|fonte|pagina|numero|n|#)\s*(?:de\s*)?(?:numero\s*)?#?(\d{1,2})\b", n
    ) or re.search(r"#(\d{1,2})\b", n) or re.search(r"\b(\d{1,2})\s*[oa]?\s*(?:resultado|link|noticia|video|item|opcao|fonte|pagina)\b", n)
    if m:
        return int(m.group(1))
    for word, value in _ORDINALS.items():
        if len(word) > 2 and re.search(rf"\b{word}\b", n) and re.search(r"\b(resultado|link|noticia|video|item|opcao|imagem|pesquisa|fonte|pagina)", n):
            return value
    return None


# ---------------------------------------------------------------- atalhos

SHORTCUTS_FILE = Path("atalhos") / "atalhos.json"

DEFAULT_SHORTCUTS: dict[str, dict[str, Any]] = {
    "cnn": {"alvos": ["https://www.cnnbrasil.com.br/"], "aliases": ["cnn brasil", "cnnbrasil"]},
    "g1": {"alvos": ["https://g1.globo.com/"], "aliases": ["globo news", "g um"]},
    "globo": {"alvos": ["https://www.globo.com/"], "aliases": []},
    "uol": {"alvos": ["https://www.uol.com.br/"], "aliases": []},
    "folha": {"alvos": ["https://www.folha.uol.com.br/"], "aliases": ["folha de sao paulo"]},
    "bbc": {"alvos": ["https://www.bbc.com/portuguese"], "aliases": ["bbc brasil"]},
    "youtube": {"alvos": ["https://www.youtube.com/"], "aliases": ["yt", "you tube"]},
    "gmail": {"alvos": ["https://mail.google.com/"], "aliases": ["email", "e-mail", "meu email"]},
    "google": {"alvos": ["https://www.google.com/"], "aliases": ["pesquisa google"]},
    "maps": {"alvos": ["https://maps.google.com/"], "aliases": ["mapas", "google maps"]},
    "drive": {"alvos": ["https://drive.google.com/"], "aliases": ["google drive"]},
    "agenda": {"alvos": ["https://calendar.google.com/"], "aliases": ["google agenda", "google calendar"]},
    "github": {"alvos": ["https://github.com/"], "aliases": ["git hub"]},
    "whatsapp": {"alvos": ["https://web.whatsapp.com/"], "aliases": ["zap", "whatsapp web", "zapzap"]},
    "claude": {"alvos": ["https://claude.ai/"], "aliases": []},
    "chatgpt": {"alvos": ["https://chatgpt.com/"], "aliases": ["gpt", "chat gpt"]},
    "gemini": {"alvos": ["https://gemini.google.com/"], "aliases": []},
    "wikipedia": {"alvos": ["https://pt.wikipedia.org/"], "aliases": ["wiki"]},
    "netflix": {"alvos": ["https://www.netflix.com/"], "aliases": []},
    "spotify": {"alvos": ["https://open.spotify.com/"], "aliases": []},
    "linkedin": {"alvos": ["https://www.linkedin.com/"], "aliases": []},
    "reddit": {"alvos": ["https://www.reddit.com/"], "aliases": []},
    "twitter": {"alvos": ["https://x.com/"], "aliases": ["x", "x twitter"]},
    "instagram": {"alvos": ["https://www.instagram.com/"], "aliases": ["insta"]},
    "tradutor": {"alvos": ["https://translate.google.com/"], "aliases": ["google tradutor", "traduzir"]},
}

_SHORTCUT_FILLER = {
    "abra", "abre", "abrir", "abrindo", "acesse", "acessar", "acessa", "va", "vai", "ir", "leve",
    "para", "pro", "pra", "ate", "o", "a", "os", "as", "no", "na", "nos", "nas", "de", "do", "da",
    "site", "pagina", "pag", "link", "atalho", "atalhos", "usando", "use", "usa", "agent", "agente",
    "por", "favor", "quero", "ver", "mostre", "me", "meu", "minha", "aplicativo", "app", "portal",
    "um", "uma", "e", "em", "navegador", "chrome", "browser", "astra", "pfv", "pf", "rode", "execute",
}


def load_shortcuts() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(SHORTCUTS_FILE.read_text(encoding="utf-8"))
        items = data.get("atalhos")
        if isinstance(items, dict):
            return items
    except Exception:
        pass
    items = json.loads(json.dumps(DEFAULT_SHORTCUTS))
    save_shortcuts(items)
    return items


def save_shortcuts(items: dict[str, dict[str, Any]]) -> None:
    write_text(SHORTCUTS_FILE, json.dumps({"versao": 1, "atalhos": items}, ensure_ascii=False, indent=2))


def shortcut_query_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9.\-]*", norm(text)) if t not in _SHORTCUT_FILLER]


def find_shortcut(text: str, items: dict[str, dict[str, Any]] | None = None) -> tuple[str | None, list[str]]:
    """Acha o atalho citado em `text`. Devolve (nome, candidatos_parecidos).
    Casa por nome/alias exato (palavra inteira) e, em seguida, por
    semelhança de grafia (cnn ≈ cnnn, gitub ≈ github)."""
    items = items if items is not None else load_shortcuts()
    tokens = shortcut_query_tokens(text)
    if not tokens:
        return None, []
    phrase = " ".join(tokens)
    names: dict[str, str] = {}  # nome-ou-alias normalizado -> nome do atalho
    for name, spec in items.items():
        names[norm(name)] = name
        for alias in spec.get("aliases", []):
            names[norm(alias)] = name

    # 1) a frase inteira, ou um nome/alias contido nela como palavra(s) inteira(s)
    if phrase in names:
        return names[phrase], []
    best: tuple[int, str] | None = None
    for key, name in names.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", phrase):
            if best is None or len(key) > best[0]:
                best = (len(key), name)
    if best:
        return best[1], []

    # 2) semelhança de grafia
    scored: list[tuple[float, str]] = []
    for key, name in names.items():
        score = max(difflib.SequenceMatcher(None, key, cand).ratio() for cand in [phrase, *tokens])
        scored.append((score, name))
    scored.sort(reverse=True)
    uniq: list[tuple[float, str]] = []
    for score, name in scored:
        if all(name != n for _s, n in uniq):
            uniq.append((score, name))
    if uniq and uniq[0][0] >= 0.84 and (len(uniq) == 1 or uniq[0][0] - uniq[1][0] >= 0.05):
        return uniq[0][1], []
    return None, [n for s, n in uniq[:4] if s >= 0.55]


def open_target(target: str) -> tuple[bool, str]:
    """Abre um alvo de atalho: link http(s) ou caminho local seguro."""
    link = safe_url(target)
    if link:
        return open_url(link), link
    expanded = known_folder(target) or Path(target).expanduser()
    return open_path(expanded), str(expanded)
