"""Agente arquivos: encontra e abre arquivos/pastas do computador.

  ache o arquivo contrato.pdf        | onde está meu currículo
  últimos arquivos baixados          | arquivos de hoje nos downloads
  arquivos maiores que 200 MB        → o que está ocupando espaço
  abra a pasta downloads             | abra o arquivo relatório
Procura em Downloads, Documentos, Desktop, Imagens, Vídeos e Músicas (ou na
pasta citada). Nunca abre programas/scripts. Guarda a lista: “abra o 2º”.
"""

from __future__ import annotations

import difflib
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "arquivos/resultado.txt"
MAX_VARRIDOS = 60_000
MAX_SEGUNDOS = 12
PROFUNDIDADE = 5
PULAR_DIRS = {"node_modules", ".git", "appdata", "__pycache__", ".cache", ".venv", "venv", "site-packages", "$recycle.bin", "library", ".trash", "windows"}

TIPOS = {
    "pdf": {".pdf"}, "foto": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".bmp"}, "imagem": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".bmp", ".svg"},
    "video": {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv"}, "musica": {".mp3", ".wav", ".flac", ".m4a", ".ogg"},
    "planilha": {".xls", ".xlsx", ".csv", ".ods"}, "documento": {".doc", ".docx", ".pdf", ".txt", ".odt", ".rtf"},
    "word": {".doc", ".docx"}, "excel": {".xls", ".xlsx"}, "apresentacao": {".ppt", ".pptx", ".odp"}, "slide": {".ppt", ".pptx"},
    "zip": {".zip", ".rar", ".7z"}, "compactado": {".zip", ".rar", ".7z", ".tar", ".gz"}, "codigo": {".py", ".js", ".ts", ".html", ".css", ".json", ".java", ".c", ".cpp"},
}
_STOP = {
    "ache", "achar", "acha", "procure", "procurar", "busque", "buscar", "encontre", "encontrar", "onde", "esta", "estao", "fica", "esta", "arquivo", "arquivos", "meu", "minha", "meus", "minhas",
    "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das", "no", "na", "nos", "nas", "em", "que", "e", "para", "pra", "por", "me", "mostre", "mostrar", "liste", "listar",
    "abra", "abre", "abrir", "pasta", "ultimos", "ultimas", "ultimo", "ultima", "recentes", "recente", "novos", "baixados", "baixado", "baixei", "salvei", "criei", "hoje", "ontem", "semana", "mes",
    "maiores", "maior", "pesados", "grandes", "ocupando", "espaco", "mais", "tipo", "extensao", "nome", "chamado", "chamada", "com", "ver", "veja", "tenho", "tem", "qual", "quais", "esse", "essa",
    "computador", "pc", "disco", "agente", "agent", "usando", "use", "arquivos", "download", "downloads",
}


def interpretar(texto: str) -> dict:
    n = C.norm(texto)
    p: dict = {"acao": "buscar", "pastas": [], "termos": [], "exts": set(), "min_mb": None, "ordem": "relevancia", "desde": None, "pasta_alvo": None}
    if re.search(r"\babr\w*\b", n):
        p["acao"] = "abrir_pasta" if re.search(r"\bpasta\b", n) else "abrir"
    m = re.search(r"\b(?:maiores?|mais pesad\w+|acima de|mais de|maior que|maiores que)\s*(?:que\s+)?(\d+(?:[.,]\d+)?)\s*(kb|mb|gb)\b", n)
    if m:
        v = float(m.group(1).replace(",", "."))
        p["min_mb"] = v * {"kb": 1 / 1024, "mb": 1, "gb": 1024}[m.group(2)]
        p["ordem"] = "tamanho"
    elif re.search(r"\b(?:maiores|mais pesados|ocupando espaco|mais grandes|pesados)\b", n):
        p["ordem"] = "tamanho"
        p["min_mb"] = 50
    if re.search(r"\b(?:ultimos?|ultimas?|recentes?|novos|baixei|salvei|criei)\b", n):
        p["ordem"] = "recentes"
    if re.search(r"\bhoje\b", n):
        p["desde"] = time.mktime(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timetuple())
        p["ordem"] = "recentes"
    elif re.search(r"\bontem\b", n):
        p["desde"] = time.time() - 2 * 86400
        p["ordem"] = "recentes"
    elif re.search(r"\b(?:esta|nesta|essa|dessa) semana\b|\bultimos 7 dias\b", n):
        p["desde"] = time.time() - 7 * 86400
        p["ordem"] = "recentes"
    for chave, exts in TIPOS.items():
        if re.search(rf"\b{chave}s?\b", n):
            p["exts"] |= exts
    for ext in re.findall(r"\.([a-z0-9]{2,5})\b", n):
        if ext not in {"com", "br"}:
            p["exts"].add("." + ext)
    # pasta citada
    for alias in sorted({a for v in C._FOLDER_ALIASES.values() for a in v}, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(C.norm(alias))}(?![a-z0-9])", n):
            f = C.known_folder(alias)
            if f:
                p["pasta_alvo"] = f
                break
    # termos de nome: o que sobrar
    toks = re.findall(r"[a-z0-9][a-z0-9_.\-]*", n)
    tipo_words = {c for c in TIPOS} | {c + "s" for c in TIPOS}
    alias_words = {w for v in C._FOLDER_ALIASES.values() for a in v for w in C.norm(a).split()}
    p["termos"] = [t for t in toks if t not in _STOP and t not in tipo_words and t not in alias_words and not re.fullmatch(r"\d+(?:[.,]\d+)?|kb|mb|gb|\.\w+", t)]
    termos = []
    for t in (x.strip(".") for x in p["termos"]):
        m2 = re.fullmatch(r"(.+)\.([a-z0-9]{2,5})", t)
        if m2 and m2.group(2) not in {"com", "br", "org", "net"}:
            termos.append(m2.group(1))
            p["exts"].add("." + m2.group(2))
        elif t:
            termos.append(t)
    p["termos"] = termos
    p["tipos"] = [c for c in TIPOS if re.search(rf"\b{c}s?\b", n)]
    return p


def varrer(raizes: list[Path]):
    t0, total = time.time(), 0
    for raiz in raizes:
        base = len(raiz.parts)
        for dirpath, dirs, files in os.walk(raiz, followlinks=False):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in PULAR_DIRS]
            if len(Path(dirpath).parts) - base >= PROFUNDIDADE:
                dirs[:] = []
            for f in files:
                total += 1
                if total > MAX_VARRIDOS or time.time() - t0 > MAX_SEGUNDOS:
                    return
                if f.startswith("."):
                    continue
                yield Path(dirpath) / f


def pontuar(nome: str, termos: list[str]) -> float:
    if not termos:
        return 1.0
    n = C.norm(Path(nome).stem)
    toks = re.findall(r"[a-z0-9]+", n)
    total = 0.0
    for t in termos:
        if t == n:
            total += 2.0
        elif t in n:
            total += 1.0 + (0.3 if t in toks else 0)
        else:
            melhor = max((difflib.SequenceMatcher(None, t, w).ratio() for w in toks), default=0)
            if melhor >= 0.82 and len(t) > 3:
                total += melhor * 0.8
            else:
                return 0.0
    return total / len(termos)


def tam(b: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024 or u == "GB":
            return f"{b:.0f} {u}" if u == "B" else f"{b:.1f} {u}"
        b /= 1024
    return ""


def buscar(p: dict) -> list[dict]:
    raizes = [p["pasta_alvo"]] if p["pasta_alvo"] else [d for d in (C.known_folder(x) for x in ("downloads", "documentos", "desktop", "imagens", "videos", "musicas")) if d and d.is_dir()]
    achados = []
    for f in varrer([r for r in raizes if r.is_dir()]):
        if p["exts"] and f.suffix.lower() not in p["exts"]:
            continue
        sc = pontuar(f.name, p["termos"])
        if sc <= 0:
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        if p["desde"] and st.st_mtime < p["desde"]:
            continue
        if p["min_mb"] and st.st_size < p["min_mb"] * 1024 * 1024:
            continue
        achados.append({"caminho": str(f), "nome": f.name, "tamanho": st.st_size, "mtime": st.st_mtime, "score": sc})
    chave = {"tamanho": lambda a: -a["tamanho"], "recentes": lambda a: -a["mtime"]}.get(p["ordem"], lambda a: (-a["score"], -a["mtime"]))
    return sorted(achados, key=chave)


def main() -> None:
    bruto = C.contexto()
    p = interpretar(bruto)
    try:
        if p["acao"] == "abrir_pasta":
            pasta = p["pasta_alvo"]
            if pasta is None:
                C.finish(REPORT, "⚠️ Qual pasta abrir? Ex.: “abra a pasta downloads”.", resposta="direta", interpretado="arquivos: pasta?")
                return
            ok = C.open_path(pasta)
            C.finish(REPORT, f"{'📂 Abri' if ok else '⚠️ Não consegui abrir'}: {pasta}", resposta="direta", aberto=ok, interpretado=f"arquivos: abrir pasta {pasta.name}")
            return
        # "abra o 2º arquivo" cai no agente abrir; aqui "abra o arquivo X" procura pelo nome.
        if not p["termos"] and not p["exts"] and p["ordem"] == "relevancia" and not p["pasta_alvo"]:
            C.finish(REPORT, "⚠️ Qual arquivo procurar? Ex.: “ache o contrato.pdf”, “últimos arquivos baixados”, “arquivos maiores que 200 MB”.", resposta="direta", interpretado="arquivos: vazio")
            return
        achados = buscar(p)
    except Exception as exc:  # noqa: BLE001
        C.fail(f"Erro ao procurar arquivos: {exc}", REPORT)
        return
    desc = " ".join(p["termos"]) or ", ".join(p["tipos"]) or ("recentes" if p["ordem"] == "recentes" else "grandes" if p["ordem"] == "tamanho" else "tudo")
    if not achados:
        C.finish(REPORT, f"🔎 Não achei arquivos para “{desc}”" + (f" em {p['pasta_alvo']}" if p["pasta_alvo"] else " em Downloads/Documentos/Desktop/Imagens/Vídeos/Músicas") + ".", resposta="direta", quantidade=0, interpretado=f"arquivos: {desc}")
        return
    top = achados[:10]
    C.save_last("arquivos", [{"titulo": a["nome"], "url": a["caminho"]} for a in top])
    linhas = [f"🔎 {len(achados)} arquivo(s) para “{desc}” (mostrando {len(top)}):", ""]
    for i, a in enumerate(top, 1):
        linhas += [f"{i}. {a['nome']}  ·  {tam(a['tamanho'])}  ·  {datetime.fromtimestamp(a['mtime']):%d/%m/%Y %H:%M}", f"   {a['caminho']}"]
    if p["acao"] == "abrir":
        ok = C.open_path(top[0]["caminho"])
        linhas.append(f"\n{'✅ Abri' if ok else '⚠️ Não consegui abrir (tipo bloqueado ou sem programa)'} o 1º.")
    else:
        linhas.append("\nDiga “abra o 2º” para abrir um deles.")
    C.finish(REPORT, "\n".join(linhas), resposta="direta", quantidade=len(achados), interpretado=f"arquivos: {desc}")


if __name__ == "__main__":
    main()
