"""Agente notícias: manchetes de RSS (g1, CNN Brasil, Folha, UOL, BBC, Tecnoblog...)
e busca de notícias por assunto no Google Notícias. Sem chave de API.

  notícias                         → principais do Brasil (várias fontes)
  notícias de tecnologia / economia / esportes / mundo / ciência / ia
  notícias sobre o dólar           → busca por assunto
  notícias da cnn / do g1          → só dessa fonte
Guarda os links: depois diga “abra a 2ª notícia” ou “leia a 3ª”.
"""

from __future__ import annotations

import html
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "noticias/relatorio.txt"

FONTES = {
    "g1": ("g1", {"geral": "https://g1.globo.com/rss/g1/", "tecnologia": "https://g1.globo.com/rss/g1/tecnologia/", "economia": "https://g1.globo.com/rss/g1/economia/", "mundo": "https://g1.globo.com/rss/g1/mundo/", "ciencia": "https://g1.globo.com/rss/g1/ciencia-e-saude/", "esportes": "https://ge.globo.com/rss/ge/"}),
    "cnn": ("CNN Brasil", {"geral": "https://www.cnnbrasil.com.br/feed/", "tecnologia": "https://www.cnnbrasil.com.br/tecnologia/feed/", "economia": "https://www.cnnbrasil.com.br/economia/feed/", "esportes": "https://www.cnnbrasil.com.br/esportes/feed/", "mundo": "https://www.cnnbrasil.com.br/internacional/feed/"}),
    "folha": ("Folha", {"geral": "https://feeds.folha.uol.com.br/emcimadahora/rss091.xml", "tecnologia": "https://feeds.folha.uol.com.br/tec/rss091.xml", "economia": "https://feeds.folha.uol.com.br/mercado/rss091.xml", "mundo": "https://feeds.folha.uol.com.br/mundo/rss091.xml", "esportes": "https://feeds.folha.uol.com.br/esporte/rss091.xml", "ciencia": "https://feeds.folha.uol.com.br/ciencia/rss091.xml"}),
    "uol": ("UOL", {"geral": "https://rss.uol.com.br/feed/noticias.xml", "tecnologia": "https://rss.uol.com.br/feed/tecnologia.xml", "economia": "https://rss.uol.com.br/feed/economia.xml", "esportes": "https://rss.uol.com.br/feed/esporte.xml"}),
    "bbc": ("BBC Brasil", {"geral": "https://feeds.bbci.co.uk/portuguese/rss.xml"}),
    "tecnoblog": ("Tecnoblog", {"tecnologia": "https://tecnoblog.net/feed/"}),
    "olhardigital": ("Olhar Digital", {"tecnologia": "https://olhardigital.com.br/feed/"}),
    "techcrunch": ("TechCrunch", {"tecnologia": "https://techcrunch.com/feed/"}),
}
GERAL = ["g1", "cnn", "folha", "bbc"]
POR_CATEGORIA = {
    "tecnologia": ["g1", "tecnoblog", "olhardigital", "cnn"],
    "economia": ["g1", "folha", "cnn", "uol"],
    "esportes": ["g1", "cnn", "folha", "uol"],
    "mundo": ["g1", "bbc", "folha", "cnn"],
    "ciencia": ["g1", "folha"],
}
CATEGORIAS = {
    "tecnologia": r"tecnolog\w*|tech|gadgets?|celulares?|apps?",
    "economia": r"economi\w*|mercado|financ\w*|bolsa|d[oó]lar|investiment\w*",
    "esportes": r"esportes?|futebol|esportiv\w*",
    "mundo": r"mundo|internacional|exterior",
    "ciencia": r"ci[eê]ncia|sa[uú]de|espa[cç]o",
}
GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
GOOGLE_TOP = "https://news.google.com/rss?hl=pt-BR&gl=BR&ceid=BR:pt-419"


def _txt(el: ET.Element | None) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", el.text or "")).strip() if el is not None and el.text else ""


def parse_feed(corpo: str, fonte: str) -> list[dict]:
    """Lê RSS 2.0 ou Atom; recusa XML com DTD/entidades (billion laughs)."""
    if "<!ENTITY" in corpo or "<!DOCTYPE" in corpo and "ENTITY" in corpo:
        return []
    try:
        raiz = ET.fromstring(corpo.encode("utf-8"))
    except ET.ParseError:
        return []
    itens = []
    for it in raiz.iter():
        tag = it.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        d: dict = {"fonte": fonte}
        for ch in it:
            t = ch.tag.split("}")[-1]
            if t == "title" and "titulo" not in d:
                d["titulo"] = re.sub(r"\s+", " ", _txt(ch))
            elif t == "link" and "url" not in d:
                d["url"] = (ch.get("href") or _txt(ch)).strip()
            elif t in ("pubDate", "published", "updated", "date") and "data" not in d:
                try:
                    dt = parsedate_to_datetime(_txt(ch)) if "," in _txt(ch) else datetime.fromisoformat(_txt(ch).replace("Z", "+00:00"))
                    d["data"] = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except Exception:  # noqa: BLE001
                    pass
            elif t == "source" and _txt(ch):
                d["fonte"] = _txt(ch)
            elif t in ("description", "summary") and "resumo" not in d:
                r = re.sub(r"\s+", " ", _txt(ch))
                d["resumo"] = r[:220]
        # Google Notícias põe " - Fonte" no fim do título
        if d.get("titulo") and d.get("url", "").startswith("http"):
            m = re.match(r"^(.*?)\s+-\s+([^-]{2,40})$", d["titulo"])
            if m and "news.google.com" in d["url"]:
                d["titulo"], d["fonte"] = m.group(1), m.group(2)
            itens.append(d)
    return itens


def baixar(url: str, fonte: str) -> list[dict]:
    try:
        corpo, hdr, _ = C.http_get(url, timeout=10, retries=1, max_bytes=2_000_000)
        return parse_feed(C.decode_body(corpo, hdr), fonte)
    except Exception:  # noqa: BLE001
        return []


def interpretar(texto: str) -> dict:
    n = C.norm(texto)
    ped = {"categoria": "geral", "fonte": None, "assunto": "", "quantidade": 8}
    for chave, (nome, _) in FONTES.items():
        if re.search(rf"\b(?:d[aoe]s?|na|no|pela|pelo|da|do)\s+{re.escape(chave)}\b|\b{re.escape(C.norm(nome))}\b", n) and chave not in ("tecnoblog",) or re.search(rf"\b{chave}\b", n):
            ped["fonte"] = chave
            break
    m = re.search(r"\b(?:sobre|acerca de|a respeito de|referente a)\s+(.+)$", texto, re.I)
    if m:
        ped["assunto"] = re.sub(r"^(?:o|a|os|as)\s+", "", re.sub(r"\b(?:hoje|agora|recentes?|ultimas?|últimas?)\b", "", m.group(1), flags=re.I).strip(), flags=re.I).strip(" ?.!,")
    else:
        for cat, pat in CATEGORIAS.items():
            if re.search(rf"\b(?:{pat})\b", n):
                ped["categoria"] = cat
                break
        else:
            if re.search(r"\b(?:ia|inteligencia artificial|chatgpt|openai|claude|gemini)\b", n):
                ped["assunto"] = "inteligência artificial"
    mq = re.search(r"\b(?:as\s+|os\s+)?(\d{1,2})\s+(?:principais|ultimas|primeiras|noticias|manchetes)\b|\btop\s*(\d{1,2})\b", n)
    if mq:
        ped["quantidade"] = max(1, min(int(mq.group(1) or mq.group(2)), 20))
    return ped


def coletar(ped: dict) -> tuple[list[dict], str]:
    jobs: list[tuple[str, str]] = []
    if ped["assunto"]:
        jobs.append((GOOGLE_NEWS.format(q=urllib.parse.quote(ped["assunto"] + " when:3d")), "Google Notícias"))
        titulo = f"sobre “{ped['assunto']}”"
    else:
        cat = ped["categoria"]
        chaves = [ped["fonte"]] if ped["fonte"] else (POR_CATEGORIA.get(cat) or GERAL)
        for ch in chaves:
            nome, feeds = FONTES[ch]
            url = feeds.get(cat) or feeds.get("geral") or next(iter(feeds.values()))
            jobs.append((url, nome))
        if not any(True for _ in jobs):
            jobs.append((GOOGLE_TOP, "Google Notícias"))
        titulo = ("de " + cat if cat != "geral" else "do Brasil") + (f" — {FONTES[ped['fonte']][0]}" if ped["fonte"] else "")
    with ThreadPoolExecutor(max_workers=6) as ex:
        listas = list(ex.map(lambda j: baixar(*j), jobs))
    if not any(listas) and not ped["assunto"]:  # reserva
        listas = [baixar(GOOGLE_TOP, "Google Notícias")]
    return intercalar(listas), titulo


def intercalar(listas: list[list[dict]]) -> list[dict]:
    """Alterna entre as fontes (round-robin), tirando títulos repetidos."""
    vistos, out = set(), []
    for i in range(max((len(l) for l in listas), default=0)):
        for l in listas:
            if i < len(l):
                chave = re.sub(r"\W+", "", C.norm(l[i]["titulo"]))[:60]
                if chave and chave not in vistos:
                    vistos.add(chave)
                    out.append(l[i])
    return out


def formatar(itens: list[dict], titulo: str, agora: datetime) -> str:
    linhas = [f"📰 Notícias {titulo}", ""]
    for i, d in enumerate(itens, 1):
        quando = ""
        if d.get("data"):
            h = (agora - d["data"]).total_seconds() / 3600
            quando = f" · há {int(h * 60)} min" if 0 <= h < 1 else f" · há {int(h)}h" if 0 <= h < 24 else f" · {d['data'].astimezone():%d/%m}" if h >= 24 else ""
        linhas.append(f"{i}. {d['titulo']}  [{d['fonte']}{quando}]")
        linhas.append(f"   {d['url']}")
    linhas.append("\nDiga “abra a 2ª notícia” ou “leia a 3ª” para continuar.")
    return "\n".join(linhas)


def main() -> None:
    bruto = C.contexto()
    ped = interpretar(C.clean_command(bruto, {"noticias", "noticia", "manchetes"}) or bruto)
    itens, titulo = coletar(ped)
    if not itens:
        C.fail("Não consegui baixar notícias agora (sem internet ou fontes fora do ar).", REPORT)
        return
    itens.sort(key=lambda d: d.get("data") or datetime.min.replace(tzinfo=timezone.utc), reverse=True) if ped["assunto"] else None
    itens = itens[: ped["quantidade"]]
    C.save_last("notícias", itens)
    C.finish(REPORT, formatar(itens, titulo, datetime.now(timezone.utc)), resposta="direta", quantidade=len(itens), interpretado=f"noticias: {titulo}")


if __name__ == "__main__":
    main()
