"""Agente wiki: resumo da Wikipédia (pt, com reserva en) para “quem foi…”, “o que é…”.

  quem foi Santos Dumont | o que é blockchain | wikipedia sobre a Revolução Francesa
Guarda o link: “abra o artigo” / “leia o 1º” continua na página completa.
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
REPORT = "wiki/resultado.txt"


def termo(texto: str) -> str:
    t = texto.strip()
    t = re.sub(r"^(?:astra[, ]+)?(?:por favor[, ]+)?(?:usando o agente \w+[, ]+)?", "", t, flags=re.I)
    t = re.sub(r"^(?:me\s+)?(?:diga|fale|conte|explique|explica|pesquis\w+|busqu\w+|procur\w+|mostre)\s+(?:sobre|a respeito de|de|do|da)?\s*", "", t, flags=re.I)
    t = re.sub(r"^(?:quem\s+(?:foi|[eé]|era|s[aã]o|foram)|o\s+que\s+(?:[eé]|foi|s[aã]o|significa|quer dizer)|qual\s+(?:[eé]|foi)|quando\s+(?:foi|nasceu|morreu)|onde\s+(?:fica|[eé]))\s+", "", t, flags=re.I)
    t = re.sub(r"^(?:(?:na|no|da|em)\s+)?(?:wikip[eé]dia|wiki)\s*(?:sobre|de|do|da)?\s*", "", t, flags=re.I)
    for _ in range(3):
        t = re.sub(r"^(?:sobre|a|o|as|os|um|uma|de|do|da)\s+", "", t.strip(), flags=re.I)
    return t.strip(" ?!.,:;")


def buscar(q: str, lang: str) -> dict | None:
    base = f"https://{lang}.wikipedia.org"
    ops = C.http_json(f"{base}/w/api.php?action=opensearch&format=json&limit=5&namespace=0&search={urllib.parse.quote(q)}", timeout=10, retries=1)
    titulos = ops[1] if isinstance(ops, list) and len(ops) > 1 else []
    if not titulos:
        return None
    for t in titulos[:2]:
        d = C.http_json(f"{base}/api/rest_v1/page/summary/{urllib.parse.quote(t.replace(' ', '_'), safe='')}", timeout=10, retries=1)
        if d.get("type") == "disambiguation":
            continue
        if d.get("extract"):
            return {"titulo": d.get("title", t), "desc": d.get("description", ""), "texto": d["extract"], "url": (d.get("content_urls", {}).get("desktop", {}) or {}).get("page", f"{base}/wiki/{urllib.parse.quote(t)}"), "lang": lang, "outros": titulos}
    return {"titulo": titulos[0], "desc": "", "texto": "", "url": f"{base}/wiki/{urllib.parse.quote(titulos[0].replace(' ', '_'))}", "lang": lang, "outros": titulos, "ambigua": True}


def main() -> None:
    q = termo(C.contexto())
    if not q:
        C.finish(REPORT, "⚠️ Sobre o que? Ex.: “quem foi Santos Dumont”, “o que é blockchain”.", resposta="direta", interpretado="wiki: vazio")
        return
    try:
        r = buscar(q, "pt") or buscar(q, "en")
    except (C.HttpError, OSError, ValueError) as exc:
        C.fail(f"Não consegui consultar a Wikipédia agora: {exc}", REPORT)
        return
    if not r:
        C.finish(REPORT, f"🔎 Não achei um artigo da Wikipédia para “{q}”. Tente “pesquise {q}”.", resposta="direta", interpretado=f"wiki: {q} (nada)")
        return
    C.save_last("wikipédia", [{"titulo": t, "url": f"https://{r['lang']}.wikipedia.org/wiki/{urllib.parse.quote(t.replace(' ', '_'))}"} for t in r["outros"]])
    if r.get("ambigua"):
        corpo = f"🤔 “{q}” pode ser vários assuntos: " + ", ".join(r["outros"]) + ".\nDiga qual (ou “abra o 2º”)."
    else:
        texto = r["texto"]
        if len(texto) > 900:
            texto = texto[:900].rsplit(". ", 1)[0] + "."
        corpo = f"📚 {r['titulo']}" + (f" — {r['desc']}" if r["desc"] else "") + f"\n\n{texto}\n\n{r['url']}" + ("\n(artigo em inglês)" if r["lang"] == "en" else "")
        if len(r["outros"]) > 1:
            corpo += "\nVeja também: " + ", ".join(r["outros"][1:4])
    C.finish(REPORT, corpo, resposta="direta", interpretado=f"wiki: {q}", artigo=r["titulo"])


if __name__ == "__main__":
    main()
