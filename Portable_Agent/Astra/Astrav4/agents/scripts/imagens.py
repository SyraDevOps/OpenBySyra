"""Agente imagens: busca imagens, lista os links e abre/baixa a escolhida.

  imagens de jaguatirica         → lista e abre a 1ª no navegador
  mostre fotos do Pão de Açúcar
  baixe uma imagem de capivara   → salva em imagens/ (no workspace)
Fontes: DuckDuckGo Imagens (se o pacote ddgs existir) e Wikimedia Commons
(sem chave). Sem rede, abre a busca de imagens do navegador.
"""

from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "imagens/resultado.txt"
COMMONS = "https://commons.wikimedia.org/w/api.php"
_EXT_OK = (".jpg", ".jpeg", ".png", ".webp", ".gif")


_VERBOS = {"busque", "buscar", "busca", "pesquise", "pesquisar", "pesquisa", "procure", "procurar", "mostre", "mostrar", "mostra", "abra", "abre", "abrir",
           "baixe", "baixar", "baixa", "ache", "achar", "ver", "veja", "salve", "salvar", "me", "mim", "quero", "queria"}
_FILLER = {"uma", "um", "as", "os", "a", "o", "algumas", "alguns", "imagem", "imagens", "foto", "fotos", "figura", "figuras", "wallpaper", "wallpapers",
           "de", "do", "da", "dos", "das", "sobre", "para", "pra", "com", "e"}


def limpar_consulta(texto: str) -> str:
    t = re.sub(r"\s+(?:na|da|pela|em)\s+(?:internet|web)\b.*$", "", C.clean_command(texto), flags=re.I)
    tokens = t.split()
    while tokens and C.norm(tokens[0].strip(".,:;!?")) in (_VERBOS | _FILLER):
        tokens.pop(0)
    return " ".join(tokens).strip(" .,:;!?\t\r\n")


def via_commons(q: str, maximo: int) -> list[dict]:
    try:
        url = COMMONS + "?" + urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "search", "gsrsearch": f"{q} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": maximo * 2,
            "prop": "imageinfo", "iiprop": "url|size|mime", "iiurlwidth": 1000,
        })
        dados = C.http_json(url, timeout=10, retries=1)
    except Exception:  # noqa: BLE001
        return []
    return parse_commons(dados, maximo)


def parse_commons(dados: dict, maximo: int) -> list[dict]:
    paginas = sorted((dados.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for p in paginas:
        info = (p.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        if not url or not str(info.get("mime", "")).startswith("image/") or info.get("mime") == "image/svg+xml":
            continue
        titulo = re.sub(r"^File:|\.\w{3,4}$", "", p.get("title", "")).replace("_", " ")
        out.append({"titulo": titulo[:120], "url": url, "pagina": info.get("descriptionurl", ""), "fonte": "Wikimedia Commons"})
        if len(out) >= maximo:
            break
    return out


def via_ddgs(q: str, maximo: int) -> list[dict]:
    try:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore
        res = list(DDGS().images(q, region="br-pt", max_results=maximo, safesearch="moderate"))
    except Exception:  # noqa: BLE001
        return []
    return [{"titulo": str(r.get("title", ""))[:120], "url": r.get("image", ""), "pagina": r.get("url", ""), "fonte": r.get("source", "web")} for r in res if r.get("image")]


def baixar_imagem(url: str, destino: Path) -> Path | None:
    try:
        corpo, hdr, _ = C.http_get(url, timeout=15, max_bytes=12_000_000, guard=True)
    except Exception:  # noqa: BLE001
        return None
    tipo = hdr.get("content-type", "").lower()
    if not tipo.startswith("image/") or len(corpo) < 3000:
        return None
    ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}.get(tipo.split(";")[0], ".jpg")
    nome = re.sub(r"[^\w-]+", "_", C.norm(destino.stem))[:50] or "imagem"
    destino = destino.with_name(nome + ext)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(corpo)
    return destino


def main() -> None:
    bruto = C.contexto()
    q = limpar_consulta(bruto)
    n = C.norm(bruto)
    quer_baixar = bool(re.search(r"\b(?:baix\w*|salv\w*|download)\b", n))
    quer_abrir = not quer_baixar and not re.search(r"\b(?:list\w*|quais|links?)\b", n)
    if not q:
        C.finish(REPORT, "⚠️ De que devo buscar imagens? Ex.: “imagens de jaguatirica”.", resposta="direta", interpretado="imagens: vazio")
        return
    res = via_ddgs(q, 6) or via_commons(q, 6)
    if not res:
        busca = "https://www.bing.com/images/search?q=" + urllib.parse.quote(q)
        ok = C.open_url(busca) if quer_abrir else False
        C.finish(REPORT, f"⚠️ Não consegui buscar as imagens de “{q}” agora. {'Abri a busca de imagens no navegador.' if ok else 'Use:'}\n{busca}", resposta="direta", aberto=ok, url=busca, interpretado=f"imagens: {q} (sem lista)")
        return
    C.save_last("imagens", res)
    linhas = [f"🖼️ Imagens de “{q}”", ""]
    for i, r in enumerate(res, 1):
        linhas += [f"{i}. {r['titulo'] or 'Sem título'}  [{r['fonte']}]", f"   {r['url']}"]
    extra: dict = {}
    if quer_baixar:
        salvo = None
        for r in res:
            salvo = baixar_imagem(r["url"], Path("imagens") / q)
            if salvo:
                break
        linhas.append(f"\n💾 Salva em: {salvo}" if salvo else "\n⚠️ Não consegui baixar nenhuma (links bloqueados).")
        extra["arquivo"] = str(salvo) if salvo else None
    elif quer_abrir:
        ok = C.open_url(res[0]["url"])
        linhas.append(f"\n{'✅ Abri' if ok else '⚠️ Não consegui abrir sozinho'} a 1ª imagem. Diga “abra a 3ª” para outra.")
        extra["aberto"] = ok
    C.finish(REPORT, "\n".join(linhas), resposta="direta", quantidade=len(res), interpretado=f"imagens: {q}", **extra)


if __name__ == "__main__":
    main()
