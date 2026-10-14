"""Agente bomdia: roda agenda, tarefas, clima e notícias e junta num resumo.
Se uma parte falhar (sem internet, por ex.), as outras continuam."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
HERE = Path(__file__).resolve().parent
REPORT = "bomdia/resumo.txt"
DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
PARTES = {
    "agenda": ("gerenciar_calendario.py", "o que tenho hoje", "agenda/relatorio.txt", 20),
    "tarefas": ("tarefas.py", "minhas tarefas", "tarefas/relatorio.txt", 20),
    "clima": ("clima.py", "clima hoje", "clima/previsao.txt", 45),
    "noticias": ("noticias.py", "as 5 principais notícias", "noticias/relatorio.txt", 45),
}


def rodar(chave: str) -> str | None:
    script, pedido, relatorio, limite = PARTES[chave]
    Path(relatorio).unlink(missing_ok=True)  # nunca usar relatório velho
    try:
        subprocess.run([sys.executable, "-I", str(HERE / script)], input=json.dumps({"contexto": pedido}).encode("utf-8"), capture_output=True, timeout=limite, check=False)
        return Path(relatorio).read_text(encoding="utf-8").strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    agora = datetime.now()
    saudacao = "Bom dia" if agora.hour < 12 else "Boa tarde" if agora.hour < 18 else "Boa noite"
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = dict(zip(PARTES, ex.map(rodar, PARTES)))
    partes = [f"☀️ {saudacao}! Hoje é {DIAS[agora.weekday()]}, {agora:%d/%m/%Y}.", ""]
    partes += ["📅 AGENDA DE HOJE", res["agenda"] or "   não consegui ler a agenda agora.", ""]
    tf = res["tarefas"]
    if tf and "Nenhuma tarefa" not in tf:
        partes += ["✅ TAREFAS PENDENTES", tf, ""]
    cl = res["clima"]
    if cl and not cl.startswith("Não consegui"):
        partes += ["🌤️ CLIMA", cl.split("\n\n")[0], ""]
    else:
        partes += ["🌤️ CLIMA", "   indisponível agora (sem internet?).", ""]
    nt = res["noticias"]
    if nt and "Diga “abra" in nt:
        linhas = nt.split("\n\nDiga")[0].split("\n")[2:]
        partes += ["📰 MANCHETES", "\n".join(linhas), ""]
    C.finish(REPORT, "\n".join(partes).strip(), resposta="direta", interpretado=f"bomdia {agora:%Y-%m-%d}")


if __name__ == "__main__":
    main()
