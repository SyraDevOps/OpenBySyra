"""Agente abrir: abre um link específico no navegador (ou um arquivo/pasta).

  abra https://exemplo.com   | abra exemplo.com.br
  abra o 2º resultado        → usa a última lista (busca, notícias, YouTube, arquivos)
  abra o link da primeira notícia
  abra o site do nubank      → se existir atalho, usa; senão pesquisa o endereço
Só abre http(s), pastas e documentos comuns: nunca executa programas/scripts.
"""

from __future__ import annotations

import re
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "abrir/resultado.txt"

_URL = r"(https?://[^\s<>\"“”]+|(?:www\.)?[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:com|org|net|gov|edu|io|br|app|dev|tv|co|ai|me|info|xyz|pt|uk|de|fr|es|it|jp)(?:\.[a-z]{2})?(?:/[^\s<>\"“”]*)?)"


def main() -> None:
    bruto = C.contexto()
    texto = C.clean_command(bruto, {"abra", "abre", "abrir", "abrindo", "acesse", "acessar", "va", "link", "site", "pagina"}) or bruto
    n = C.norm(bruto)

    # 1) link explícito
    m = re.search(_URL, bruto, re.I)
    if m:
        alvo = m.group(1).rstrip(".,;:!?)")
        link = C.safe_url(alvo)
        if link:
            ok = C.open_url(link)
            C.finish(REPORT, f"{'🌐 Abri' if ok else '⚠️ Não consegui abrir sozinho; use'}: {link}", resposta="direta", aberto=ok, url=link, interpretado=f"abrir url: {link}")
            return

    # 2) ordinal da última lista
    ultimos = C.load_last()
    itens = ultimos.get("itens") or []
    idx = C.parse_ordinal(bruto)
    if idx is not None:
        if not itens:
            C.finish(REPORT, "⚠️ Não tenho uma lista recente para escolher. Pesquise algo antes (notícias, YouTube, busca) e depois diga “abra o 2º”.", resposta="direta", aberto=False, interpretado="abrir ordinal sem lista")
            return
        pos = len(itens) - 1 if idx == -1 else idx - 1
        if not 0 <= pos < len(itens):
            C.finish(REPORT, f"⚠️ A última lista ({ultimos.get('origem', '?')}) tem {len(itens)} itens; o número {idx} não existe.", resposta="direta", aberto=False, interpretado=f"abrir ordinal fora da lista: {idx}")
            return
        item = itens[pos]
        ok, mostrado = C.open_target(item["url"])
        C.finish(REPORT, f"{'🌐 Abri' if ok else '⚠️ Não consegui abrir sozinho; use'} o item {pos + 1} de {ultimos.get('origem', 'lista')}:\n  {item.get('titulo') or ''}\n  {mostrado}", resposta="direta", aberto=ok, url=item["url"], interpretado=f"abrir item {pos + 1} de {ultimos.get('origem')}")
        return

    # 3) atalho conhecido ("abra o site da cnn")
    itens_atalhos = C.load_shortcuts()
    nome, cands = C.find_shortcut(bruto, itens_atalhos)
    if nome:
        resultados = [C.open_target(a) for a in itens_atalhos[nome].get("alvos", [])]
        C.finish(REPORT, f"{'🌐 Abri' if all(o for o, _ in resultados) else '⚠️ Tentei abrir'} “{nome}”: " + ", ".join(l for _, l in resultados), resposta="direta", aberto=any(o for o, _ in resultados), interpretado=f"abrir atalho {nome}")
        return

    # 4) sem link nem atalho: abre a busca do que foi pedido, dizendo isso claramente
    if len(C.norm(texto)) >= 2 and not cands:
        busca = "https://duckduckgo.com/?q=" + urllib.parse.quote(re.sub(r"\b(?:o|a)?\s*site\s+(?:d[aeo]s?\s+)?", "", texto, flags=re.I))
        ok = C.open_url(busca)
        C.finish(REPORT, f"🔎 Não sei o endereço exato de “{texto}”, então abri uma busca: {busca}\nDica: diga o link ou crie um atalho (“crie o atalho {texto} para https://…”).", resposta="direta", aberto=ok, interpretado=f"abrir busca: {texto}", aproximado=True)
        return
    sug = f" Quis dizer: {', '.join(cands)}?" if cands else ""
    C.finish(REPORT, f"⚠️ Não entendi o que abrir.{sug} Diga um link, “abra o 2º resultado” ou o nome de um atalho.", resposta="direta", aberto=False, interpretado="abrir: nada reconhecido")


if __name__ == "__main__":
    main()
