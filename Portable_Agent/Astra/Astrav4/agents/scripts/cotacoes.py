"""Agente cotacoes: dólar, euro, libra, peso argentino, iene e bitcoin em reais (AwesomeAPI).

  dólar hoje | cotação do euro | quanto está o dólar | cotações
Mostra compra, máxima/mínima do dia e variação. Para converter valores use o conversor.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
REPORT = "cotacoes/resultado.txt"
PARES = {
    "dolar": ("USD", "Dólar", r"d[oó]lar|usd"), "euro": ("EUR", "Euro", r"euro|eur"), "libra": ("GBP", "Libra esterlina", r"libra|gbp"),
    "peso": ("ARS", "Peso argentino", r"peso|ars"), "iene": ("JPY", "Iene", r"iene|jpy"), "bitcoin": ("BTC", "Bitcoin", r"bitcoin|btc"),
    "dolarcad": ("CAD", "Dólar canadense", r"canadense|cad"), "franco": ("CHF", "Franco suíço", r"franco|chf"),
}


def num(s) -> str:
    v = float(s)
    t = f"{v:,.4f}" if v < 100 else f"{v:,.2f}"
    return t.replace(",", "X").replace(".", ",").replace("X", ".")


def main() -> None:
    n = C.norm(C.contexto())
    escolhidas = [k for k, (_c, _n, pat) in PARES.items() if re.search(rf"\b(?:{C.norm(pat)})\b", n)]
    escolhidas = escolhidas or ["dolar", "euro", "libra", "bitcoin"]
    codigos = ",".join(f"{PARES[k][0]}-BRL" for k in escolhidas)
    try:
        d = C.http_json(f"https://economia.awesomeapi.com.br/json/last/{codigos}", timeout=10, retries=1)
    except (C.HttpError, OSError, ValueError) as exc:
        C.fail(f"Não consegui as cotações agora: {exc}", REPORT)
        return
    linhas = ["💱 Cotações em reais (AwesomeAPI)", ""]
    for k in escolhidas:
        cod, nome, _ = PARES[k]
        r = d.get(f"{cod}BRL")
        if not r:
            continue
        var = float(r.get("pctChange", 0))
        var_txt = f"{var:+.2f}".replace(".", ",")
        linhas.append(f"{nome} ({cod}): R$ {num(r['bid'])}   {'🟢▲' if var >= 0 else '🔴▼'} {var_txt}%")
        linhas.append(f"   máx {num(r['high'])} · mín {num(r['low'])} · atualizado {r.get('create_date', '')[11:16]}")
    C.finish(REPORT, "\n".join(linhas), resposta="direta", interpretado=f"cotacoes: {','.join(escolhidas)}")


if __name__ == "__main__":
    main()
