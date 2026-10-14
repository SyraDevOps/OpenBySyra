"""Agente organizador: organiza arquivos de uma pasta em subpastas — com segurança.

  organize meus downloads            → mostra o PLANO (nada é movido ainda)
  confirmar organização              → executa o último plano (até 60 min depois)
  organize a área de trabalho por data   → subpastas AAAA-MM
  organize downloads e aplique       → executa direto
  desfazer organização               → devolve tudo ao lugar original

Segurança: só MOVE (nunca apaga nem sobrescreve; nome repetido vira "nome (1)"),
só mexe nos arquivos soltos da pasta (subpastas ficam como estão), ignora
arquivos ocultos, downloads em andamento e atalhos, e só trabalha dentro da
sua pasta de usuário. Cada execução grava um histórico para o "desfazer".
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "organizador/relatorio.txt"
PLANO = Path("organizador/plano.json")
HIST = Path("organizador/historico")
VALIDADE_PLANO = 60 * 60
MAX_ARQUIVOS = 3000

CATEGORIAS: dict[str, set[str]] = {
    "Imagens": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".heic", ".tiff", ".tif", ".ico", ".raw", ".psd"},
    "Vídeos": {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp"},
    "Músicas": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus", ".mid"},
    "Documentos": {".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt", ".md", ".tex", ".pages"},
    "Planilhas": {".xls", ".xlsx", ".csv", ".ods", ".tsv", ".numbers"},
    "Apresentações": {".ppt", ".pptx", ".odp", ".key"},
    "Livros": {".epub", ".mobi", ".azw3", ".djvu"},
    "Compactados": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".iso"},
    "Programas": {".exe", ".msi", ".dmg", ".pkg", ".deb", ".rpm", ".apk", ".appimage"},
    "Código": {".py", ".js", ".ts", ".html", ".css", ".json", ".xml", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".php", ".sql", ".sh", ".bat", ".ipynb", ".yml", ".yaml"},
    "Fontes": {".ttf", ".otf", ".woff", ".woff2"},
}
EXT_PARA_CAT = {e: c for c, exts in CATEGORIAS.items() for e in exts}
IGNORAR_EXT = {".crdownload", ".part", ".tmp", ".download", ".opdownload", ".partial", ".lnk", ".url", ".ini", ".lock", ".db", ".ds_store", ".desktop"}
IGNORAR_NOME = {"desktop.ini", "thumbs.db", ".ds_store"}


# ------------------------------------------------------------------ pasta alvo

def _bloqueada(p: Path) -> str | None:
    """Motivo de recusa se a pasta não deve ser organizada."""
    try:
        r = p.resolve()
        home = C.user_home().resolve()
    except Exception:
        return "caminho inválido"
    if r == home or r == Path(r.anchor):
        return "pasta pessoal/raiz do disco é ampla demais"
    if home not in r.parents and r != home and Path.cwd().resolve() not in (r, *r.parents):
        return "só organizo pastas dentro da sua pasta de usuário"
    partes = {x.lower() for x in r.parts}
    if partes & {"windows", "system32", "program files", "program files (x86)", "appdata", ".ssh", ".git", "library", "node_modules", "$recycle.bin"}:
        return "pasta de sistema/programas — não mexo nisso"
    return None


def achar_pasta(texto: str) -> Path | None:
    n = C.norm(texto)
    m = re.search(r"([A-Za-z]:[\\/][^\"“”\n,;]+|~[\\/][^\"“”\n,;]*|(?<![\w/])/[A-Za-z][^\"“”\n,;]*)", texto)
    if m:
        bruto = re.sub(r"\s+(?:por\s+\w+|e\s+(?:aplique|execute|organize).*|sem\s+.*|direto)$", "", m.group(1).strip().rstrip(".!?"), flags=re.I)
        return Path(bruto).expanduser()
    if re.search(r"\b(?:pasta pessoal|minha pasta|pasta do usuario|home)\b", n):
        return C.user_home()
    for alias in sorted({a for v in C._FOLDER_ALIASES.values() for a in v}, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(C.norm(alias))}(?![a-z0-9])", n):
            return C.known_folder(alias)
    return None


# ------------------------------------------------------------------ plano

def modo_de(texto: str) -> str:
    n = C.norm(texto)
    if re.search(r"\bpor (?:data|mes|ano|periodo)\b|\bmensal\b", n):
        return "data"
    return "tipo"


def montar_plano(raiz: Path, modo: str, agora: float | None = None) -> dict:
    agora = agora or time.time()
    movs, pulados = [], defaultdict(int)
    n = 0
    for f in sorted(raiz.iterdir(), key=lambda x: x.name.lower()):
        try:
            if not f.is_file() or f.is_symlink():
                continue
        except OSError:
            continue
        nome, ext = f.name, f.suffix.lower()
        if nome.startswith(".") or nome.lower() in IGNORAR_NOME or ext in IGNORAR_EXT:
            pulados["ocultos/temporários/atalhos"] += 1
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        if agora - st.st_mtime < 120:
            pulados["modificados há menos de 2 min"] += 1
            continue
        n += 1
        if n > MAX_ARQUIVOS:
            pulados["acima do limite por execução"] += 1
            continue
        sub = EXT_PARA_CAT.get(ext, "Outros") if modo == "tipo" else datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m")
        movs.append({"origem": str(f), "destino_pasta": sub, "tamanho": st.st_size})
    return {"raiz": str(raiz), "modo": modo, "criado": agora, "movimentos": movs, "pulados": dict(pulados)}


def _tam(b: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024 or u == "GB":
            return f"{b:.0f} {u}" if u == "B" else f"{b:.1f} {u}"
        b /= 1024
    return ""


def texto_plano(plano: dict) -> str:
    por: dict[str, list[dict]] = defaultdict(list)
    for m in plano["movimentos"]:
        por[m["destino_pasta"]].append(m)
    linhas = [f"🗂️ Plano para organizar {plano['raiz']} (por {'tipo' if plano['modo'] == 'tipo' else 'data'})", ""]
    if not por:
        linhas.append("Nada para organizar: não há arquivos soltos elegíveis.")
    for pasta in sorted(por, key=lambda p: -len(por[p])):
        itens = por[pasta]
        linhas.append(f"📁 {pasta}/  — {len(itens)} arquivo(s), {_tam(sum(i['tamanho'] for i in itens))}")
        for i in itens[:3]:
            linhas.append(f"     • {Path(i['origem']).name}")
        if len(itens) > 3:
            linhas.append(f"     … e mais {len(itens) - 3}")
    if plano["pulados"]:
        linhas += ["", "Ignorados: " + "; ".join(f"{k} ({v})" for k, v in plano["pulados"].items())]
    if por:
        linhas += ["", f"Total: {len(plano['movimentos'])} arquivo(s). NADA foi movido ainda.", "👉 Para executar diga: “confirmar organização”. Para cancelar, ignore."]
    return "\n".join(linhas)


# ------------------------------------------------------------------ execução

def destino_livre(pasta: Path, nome: str) -> Path:
    alvo = pasta / nome
    if not alvo.exists():
        return alvo
    stem, ext = Path(nome).stem, Path(nome).suffix
    i = 1
    while (pasta / f"{stem} ({i}){ext}").exists():
        i += 1
    return pasta / f"{stem} ({i}){ext}"


def executar(plano: dict) -> tuple[list[dict], list[str]]:
    raiz = Path(plano["raiz"])
    feitos, erros = [], []
    for m in plano["movimentos"]:
        origem = Path(m["origem"])
        try:
            if origem.parent != raiz or not origem.is_file() or origem.is_symlink():
                erros.append(f"{origem.name}: mudou desde o plano (ignorado)")
                continue
            pasta = raiz / m["destino_pasta"]
            pasta.mkdir(exist_ok=True)
            dest = destino_livre(pasta, origem.name)
            shutil.move(str(origem), str(dest))
            feitos.append({"de": str(origem), "para": str(dest)})
        except Exception as exc:  # noqa: BLE001
            erros.append(f"{origem.name}: {exc}")
    return feitos, erros


def desfazer() -> tuple[str, dict]:
    arqs = sorted(HIST.glob("*.json")) if HIST.is_dir() else []
    for arq in reversed(arqs):
        dados = json.loads(arq.read_text(encoding="utf-8"))
        if dados.get("desfeito"):
            continue
        ok, erros, pastas = 0, [], set()
        for mv in reversed(dados["movimentos"]):
            para, de = Path(mv["para"]), Path(mv["de"])
            try:
                if not para.exists():
                    erros.append(f"{para.name}: não está mais lá")
                    continue
                destino = destino_livre(de.parent, de.name) if de.exists() else de
                shutil.move(str(para), str(destino))
                pastas.add(para.parent)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                erros.append(f"{para.name}: {exc}")
        for p in pastas:
            try:
                p.rmdir()  # só remove se ficou vazia
            except OSError:
                pass
        dados["desfeito"] = True
        arq.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
        msg = f"↩️ Desfeito: {ok} arquivo(s) voltaram para {dados['raiz']}."
        if erros:
            msg += "\n⚠️ " + "; ".join(erros[:5])
        return msg, {"desfeito": ok}
    return "ℹ️ Não há nenhuma organização para desfazer.", {"desfeito": 0}


def main() -> None:
    bruto = C.contexto()
    n = C.norm(bruto)
    try:
        if re.search(r"\b(?:desfaz\w*|desfeit\w*|reverta|reverter|volt\w+\s+(?:tudo|como\s+estava)|desorganiz\w*)\b", n):
            msg, meta = desfazer()
            C.finish(REPORT, msg, resposta="direta", interpretado="organizador: desfazer", **meta)
            return

        confirma = bool(re.search(r"\b(?:confirm\w*|aplic\w*|execut\w*|pode\s+(?:organizar|mover|fazer)|sim|prossiga|manda\s+ver|fa[cç]a)\b", n))
        direto = bool(re.search(r"\b(?:e\s+aplique|e\s+execute|direto|sem\s+(?:previa|preview|perguntar)|ja\s+(?:aplique|organize))\b", n))
        pasta = achar_pasta(bruto)

        plano = None
        if confirma and not direto and PLANO.exists():
            try:
                plano = json.loads(PLANO.read_text(encoding="utf-8"))
            except Exception:
                plano = None
            if plano and (time.time() - plano.get("criado", 0) > VALIDADE_PLANO):
                C.finish(REPORT, "⏳ O plano expirou (mais de 60 min). Peça de novo: “organize meus downloads”.", resposta="direta", interpretado="organizador: plano expirado")
                return
            if plano and pasta and Path(plano["raiz"]) != pasta:
                plano = None  # confirmação de outra pasta: gera plano novo
        if plano is None:
            if pasta is None:
                C.finish(REPORT, "⚠️ Qual pasta organizar? Ex.: “organize meus downloads”, “organize a área de trabalho por data”, “organize ~/Documentos/Fotos”.\n(Se queria executar um plano, diga “confirmar organização”.)", resposta="direta", interpretado="organizador: sem pasta")
                return
            motivo = _bloqueada(pasta)
            if motivo:
                C.finish(REPORT, f"⛔ Não vou organizar {pasta}: {motivo}.", resposta="direta", interpretado="organizador: pasta recusada")
                return
            if not pasta.is_dir():
                C.finish(REPORT, f"⚠️ A pasta {pasta} não existe.", resposta="direta", interpretado="organizador: pasta inexistente")
                return
            plano = montar_plano(pasta, modo_de(bruto))
            if not (confirma and direto) and not direto:
                if plano["movimentos"]:
                    C.write_text(PLANO, json.dumps(plano, ensure_ascii=False))
                else:
                    PLANO.unlink(missing_ok=True)
                C.finish(REPORT, texto_plano(plano), resposta="direta", arquivos=len(plano["movimentos"]), interpretado=f"organizador: plano {pasta.name}", aplicado=False)
                return

        feitos, erros = executar(plano)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        HIST.mkdir(parents=True, exist_ok=True)
        (HIST / f"{stamp}.json").write_text(json.dumps({"raiz": plano["raiz"], "modo": plano["modo"], "movimentos": feitos}, ensure_ascii=False, indent=1), encoding="utf-8")
        try:
            PLANO.unlink()
        except OSError:
            pass
        por: dict[str, int] = defaultdict(int)
        for f in feitos:
            por[Path(f["para"]).parent.name] += 1
        linhas = [f"✅ Organizei {len(feitos)} arquivo(s) em {plano['raiz']}:"] + [f"   📁 {k}/ — {v}" for k, v in sorted(por.items(), key=lambda kv: -kv[1])]
        if erros:
            linhas.append("⚠️ " + "; ".join(erros[:5]))
        linhas.append("\nSe não gostou: “desfazer organização”.")
        C.finish(REPORT, "\n".join(linhas), resposta="direta", movidos=len(feitos), erros=len(erros), aplicado=True, interpretado=f"organizador: aplicado {Path(plano['raiz']).name}")
    except Exception as exc:  # noqa: BLE001
        C.fail(f"Erro ao organizar: {exc}", REPORT)


if __name__ == "__main__":
    main()
