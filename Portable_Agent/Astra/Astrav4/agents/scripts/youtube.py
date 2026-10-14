"""Agente youtube: pesquisa vídeos, lista com canal/duração e abre o escolhido.

  toque lofi hip hop        → abre o 1º resultado
  pesquise vídeos de receita de pão   → só lista (sem abrir)
  abra o 3º vídeo           → (agente abrir) usa a última lista
Sem internet, abre a página de busca do YouTube mesmo assim.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "youtube/resultado.txt"


def limpar_consulta(texto: str) -> str:
    t = C.clean_command(texto, {"youtube", "yt"})
    t = re.sub(r"\b(?:(?:no|do|pelo|na|em)\s+)?(?:youtube|you tube|yt)\b", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    verbo = r"(?:abr(?:a|e|ir)|toque|toca|tocar|coloque|coloca|colocar|ponha|p[oô]r|reproduz(?:a|ir)|d[eê]\s+play(?:\s+(?:em|no|na))?|assist(?:a|ir)|ver|veja|pesquis(?:e|ar|a)|busque|buscar|busca|procur(?:e|ar|a))"
    t = re.sub(rf"^{verbo}\s+", "", t, flags=re.I)
    t = re.sub(r"^(?:o\s+|a\s+|os\s+|as\s+)?(?:v[ií]deos?|m[uú]sicas?|clipes?|canal|filmes?)\s+(?:d[aeo]s?\s+|sobre\s+)", "", t, flags=re.I)
    t = re.sub(r"^(?:o\s+|a\s+)?(?:v[ií]deo|m[uú]sica|clipe)\s+", "", t, flags=re.I)
    return t.strip(" .,:;!?\t\r\n")


def quer_abrir(texto: str) -> bool:
    n = C.norm(texto)
    if re.search(r"\b(?:pesquis\w*|busc\w*|procur\w*|list\w*|mostr\w*|quais|opcoes|sugest\w*)\b", n) and not re.search(r"\b(?:abr\w*|toc\w*|coloc\w*|reproduz\w*|play)\b", n):
        return False
    return True


def _walk(o):
    if isinstance(o, dict):
        if "videoRenderer" in o:
            yield o["videoRenderer"]
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def _runs(d: dict | None) -> str:
    if not d:
        return ""
    return d.get("simpleText") or "".join(r.get("text", "") for r in d.get("runs", []))


def parse_resultados(pagina: str, maximo: int = 6) -> list[dict]:
    m = re.search(r"ytInitialData\s*=\s*", pagina)
    videos: list[dict] = []
    if m:
        try:
            dados, _ = json.JSONDecoder().raw_decode(pagina[m.end():])
            for vr in _walk(dados):
                vid = vr.get("videoId")
                if not vid:
                    continue
                videos.append({
                    "titulo": _runs(vr.get("title")), "canal": _runs(vr.get("ownerText")), "duracao": _runs(vr.get("lengthText")) or "AO VIVO",
                    "views": _runs(vr.get("viewCountText")), "quando": _runs(vr.get("publishedTimeText")), "url": f"https://www.youtube.com/watch?v={vid}",
                })
                if len(videos) >= maximo:
                    break
        except Exception:  # noqa: BLE001
            videos = []
    if not videos:  # reserva: só IDs
        vistos: list[str] = []
        for vid in re.findall(r'"videoId":"([\w-]{11})"', pagina):
            if vid not in vistos:
                vistos.append(vid)
        videos = [{"titulo": f"Vídeo {i}", "canal": "", "duracao": "", "views": "", "quando": "", "url": f"https://www.youtube.com/watch?v={v}"} for i, v in enumerate(vistos[:maximo], 1)]
    return videos


def main() -> None:
    bruto = C.contexto()
    q = limpar_consulta(bruto)
    abrir = quer_abrir(bruto)
    if not q:
        C.finish(REPORT, "⚠️ O que devo procurar no YouTube? Ex.: “toque lofi hip hop”.", resposta="direta", interpretado="youtube: vazio")
        return
    busca = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(q)
    try:
        corpo, hdr, _ = C.http_get(busca, timeout=12, headers={"Accept-Language": "pt-BR,pt;q=0.9"}, max_bytes=4_000_000)
        videos = parse_resultados(C.decode_body(corpo, hdr))
    except Exception as exc:  # noqa: BLE001
        videos = []
        erro = str(exc)
    if not videos:
        ok = C.open_url(busca) if abrir else False
        C.finish(REPORT, f"⚠️ Não consegui listar os vídeos de “{q}”. {'Abri a página de busca do YouTube.' if ok else 'Use a busca:'}\n{busca}", resposta="direta", aberto=ok, interpretado=f"youtube: {q} (sem lista)", url=busca)
        return
    C.save_last("youtube", videos)
    linhas = [f"▶️ YouTube — “{q}”", ""]
    for i, v in enumerate(videos, 1):
        det = " · ".join(x for x in (v["canal"], v["duracao"], v["views"], v["quando"]) if x)
        linhas += [f"{i}. {v['titulo']}", f"   {det}" if det else "", f"   {v['url']}"]
    ok = False
    if abrir:
        ok = C.open_url(videos[0]["url"])
        linhas.append(f"\n{'✅ Abri' if ok else '⚠️ Não consegui abrir sozinho'} o 1º vídeo. Diga “abra o 3º vídeo” para outro.")
    else:
        linhas.append("\nDiga “abra o 2º vídeo” para assistir.")
    C.finish(REPORT, "\n".join(l for l in linhas if l != ""), resposta="direta", aberto=ok, quantidade=len(videos), url=videos[0]["url"], interpretado=f"youtube: {q}")


if __name__ == "__main__":
    main()
