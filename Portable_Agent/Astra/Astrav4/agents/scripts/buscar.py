"""Agente buscar: pesquisa na web (DuckDuckGo) e devolve resultados limpos.

Usa o pacote ddgs se estiver instalado; sem ele (ou se falhar) cai para a
página HTML do DuckDuckGo, só com a biblioteca padrão. Guarda os links em
estado/ultimos.json para “abra o 2º resultado”.
"""

from __future__ import annotations

import html
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "buscar/resultados.txt"


def limpar_consulta(texto: str) -> str:
    t = C.clean_command(texto, {"astra"})
    t = re.sub(r"^(?:pesquis(?:e|ar|a)|busque|buscar|busca|busc[ao]|procur(?:e|ar|a)|consult(?:e|ar)|ache|achar|descubra|veja|ver)\s+(?:por\s+|sobre\s+|acerca\s+de\s+|a\s+respeito\s+de\s+|informa[cç][oõ]es\s+(?:sobre|de)\s+)?", "", t, flags=re.I)
    t = re.sub(r"\s+(?:no|na|pelo|pela|em)\s+(?:duck\s*duck\s*go|internet|web|google|net)\b.*$", "", t, flags=re.I)
    t = re.sub(r"^(?:para\s+mim\s+)?", "", t, flags=re.I)
    return t.strip(" .,:;!?\t\r\n")


def encurtar(v: object, n: int) -> str:
    v = re.sub(r"\s+", " ", html.unescape(str(v or ""))).strip()
    return v if len(v) <= n else v[: n - 1].rstrip() + "…"


def via_ddgs(q: str, quer_noticias: bool, maximo: int) -> tuple[list[dict], str]:
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return [], "pacote ddgs ausente"
    cliente, erro = DDGS(), ""
    for modo in (["news"] if quer_noticias else []) + ["text"]:
        for regiao in ("br-pt", "wt-wt"):
            try:
                bruto = list(cliente.news(q, region=regiao, max_results=maximo) if modo == "news" else cliente.text(q, region=regiao, max_results=maximo))
            except Exception as exc:  # noqa: BLE001
                erro = str(exc)
                continue
            res = []
            for it in bruto:
                url = str(it.get("href") or it.get("url") or "").strip()
                if not url:
                    continue
                e = {"titulo": encurtar(it.get("title"), 160), "url": url, "resumo": encurtar(it.get("body"), 300)}
                if it.get("date"):
                    e["data"] = str(it["date"])[:10]
                if it.get("source"):
                    e["fonte"] = encurtar(it["source"], 60)
                res.append(e)
            if res:
                return res, modo
    return [], erro or "sem resultados"


def parse_html_ddg(corpo: str, maximo: int) -> list[dict]:
    """Extrai resultados de html.duckduckgo.com/html/ (testável com HTML fixo)."""
    res = []
    for bloco in re.split(r'<div[^>]+class="[^"]*\bresult\b[^"]*"', corpo)[1:]:
        a = re.search(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', bloco, re.S)
        if not a:
            continue
        url = html.unescape(a.group(1))
        if "uddg=" in url:  # link de redirecionamento do DDG
            url = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(url if url.startswith("http") else "https:" + url).query).get("uddg", [url])[0])
        if url.startswith("//"):
            url = "https:" + url
        if not url.startswith("http") or "duckduckgo.com/y.js" in url:  # anúncio
            continue
        titulo = encurtar(re.sub(r"<[^>]+>", "", a.group(2)), 160)
        s = re.search(r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|td)>', bloco, re.S)
        res.append({"titulo": titulo, "url": url, "resumo": encurtar(re.sub(r"<[^>]+>", "", s.group(1)) if s else "", 300)})
        if len(res) >= maximo:
            break
    return res


def via_html(q: str, maximo: int) -> tuple[list[dict], str]:
    try:
        corpo, hdr, _ = C.http_get("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q, "kl": "br-pt"}), timeout=12)
        return parse_html_ddg(C.decode_body(corpo, hdr), maximo), ""
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


def main() -> None:
    bruto = C.contexto()
    q = limpar_consulta(bruto)
    if not q:
        C.finish(REPORT, "⚠️ O que devo pesquisar? Ex.: “pesquise melhores notebooks 2026”.", resposta="direta", ok_busca=False, interpretado="buscar: vazio")
        return
    quer_noticias = bool(re.search(r"\b(?:not[ií]cias?|news|[uú]ltimas|[uú]ltimos|hoje|agora)\b", bruto, re.I))
    res, via = via_ddgs(q, quer_noticias, 6)
    if not res:
        res, erro_html = via_html(q, 6)
        via = "html" if res else (erro_html or via)
    if not res:
        link = "https://duckduckgo.com/?q=" + urllib.parse.quote(q)
        C.fail(f"Não consegui pesquisar agora (sem internet ou bloqueio): {via}. Você pode pesquisar manualmente: {link}", REPORT, query=q)
        return
    C.save_last("busca", res)
    linhas = [f"🔎 Resultados para “{q}”", ""]
    for i, r in enumerate(res, 1):
        extra = " · ".join(x for x in (r.get("fonte"), r.get("data")) if x)
        linhas.append(f"{i}. {r['titulo']}" + (f"  ({extra})" if extra else ""))
        if r.get("resumo"):
            linhas.append(f"   {r['resumo']}")
        linhas.append(f"   {r['url']}")
    linhas.append("\nDiga “abra o 2º resultado” ou “leia o 1º” para continuar.")
    C.write_text(REPORT, "\n".join(linhas))
    # stdout: JSON único com os resultados (o passo de resumo do LLM lê isto).
    C.emit({"ok": True, "status": "sucesso", "query": q, "kind": via, "results": [{"title": r["titulo"], "url": r["url"], "snippet": r.get("resumo", ""), **({"source": r["fonte"]} if r.get("fonte") else {}), **({"date": r["data"]} if r.get("data") else {})} for r in res], "interpretado": f"buscar: {q}"})


if __name__ == "__main__":
    main()
