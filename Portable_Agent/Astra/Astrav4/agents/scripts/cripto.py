"""Agente cripto: preço, variação e tendência de criptomoedas (CoinGecko, sem chave).

  preço do bitcoin | quanto está o ethereum em dólar
  bitcoin e solana | top 10 criptomoedas
  gráfico do bitcoin (7 dias) | tendência do ETH em 30 dias
Mostra BRL e USD. Moedas desconhecidas são procuradas na busca do CoinGecko.
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
REPORT = "crypto/relatorio.txt"
API = "https://api.coingecko.com/api/v3"

MOEDAS = {
    "bitcoin": "bitcoin", "btc": "bitcoin", "ethereum": "ethereum", "eth": "ethereum", "ether": "ethereum", "solana": "solana", "sol": "solana",
    "cardano": "cardano", "ada": "cardano", "ripple": "ripple", "xrp": "ripple", "dogecoin": "dogecoin", "doge": "dogecoin", "shiba": "shiba-inu",
    "bnb": "binancecoin", "binance coin": "binancecoin", "litecoin": "litecoin", "ltc": "litecoin", "polkadot": "polkadot", "dot": "polkadot",
    "avalanche": "avalanche-2", "avax": "avalanche-2", "chainlink": "chainlink", "link": "chainlink", "polygon": "matic-network", "matic": "matic-network",
    "tether": "tether", "usdt": "tether", "usdc": "usd-coin", "tron": "tron", "trx": "tron", "toncoin": "the-open-network", "ton": "the-open-network",
    "pepe": "pepe", "stellar": "stellar", "xlm": "stellar", "monero": "monero", "xmr": "monero", "uniswap": "uniswap", "uni": "uniswap",
}
_RUIDO = {"preco", "valor", "cotacao", "quanto", "esta", "vale", "custa", "do", "da", "de", "o", "a", "os", "as", "em", "no", "na", "e", "hoje", "agora", "atual", "cripto", "criptomoeda",
          "criptomoedas", "moeda", "moedas", "grafico", "tendencia", "historico", "evolucao", "variacao", "dias", "dia", "semana", "mes", "reais", "real", "dolar", "dolares", "usd", "brl",
          "top", "principais", "maiores", "ranking", "mostre", "veja", "ver", "me", "diga", "qual", "como", "esta", "ta", "valor", "ultimos", "ultimas", "dos", "das", "sobre", "para", "pra",
          "mercado", "capitalizacao", "preços", "usando", "use", "agente", "agent", "astra", "por", "favor", "estao", "tao", "ai", "ate", "ultimo", "cada", "uma", "um"}
BLOCOS = "▁▂▃▄▅▆▇█"


def interpretar(texto: str) -> dict:
    n = C.norm(texto)
    ped = {"top": None, "grafico": False, "dias": 7, "ids": [], "buscar": []}
    mt = re.search(r"\btop\s*(\d{1,2})\b|\b(\d{1,2})\s+(?:maiores|principais)\b", n)
    if mt:
        ped["top"] = max(1, min(int(mt.group(1) or mt.group(2)), 20))
    elif re.search(r"\b(?:principais|maiores|ranking|top)\b.*\b(?:cripto\w*|moedas)\b|\b(?:cripto\w*)\b.*\b(?:principais|ranking)\b", n):
        ped["top"] = 10
    if re.search(r"\b(?:grafico|tendencia|historico|evolucao|sparkline|ultimos \d+ dias)\b", n):
        ped["grafico"] = True
    md = re.search(r"\b(\d{1,3})\s*(?:dias|d)\b", n)
    if md:
        ped["dias"] = max(2, min(int(md.group(1)), 365))
    elif re.search(r"\bmes\b|\b30\b", n):
        ped["dias"] = 30
    elif re.search(r"\bano\b", n):
        ped["dias"] = 365
    # procura as moedas citadas (mais longas primeiro)
    resto = n
    for nome in sorted(MOEDAS, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(nome)}(?![a-z0-9])", resto):
            cid = MOEDAS[nome]
            if cid not in ped["ids"]:
                ped["ids"].append(cid)
            resto = re.sub(rf"(?<![a-z0-9]){re.escape(nome)}(?![a-z0-9])", " ", resto)
    desconhecidas = [t for t in re.findall(r"[a-z0-9]{2,}", resto) if t not in _RUIDO and not t.isdigit()]
    if desconhecidas and not ped["ids"] and not ped["top"]:
        ped["buscar"] = [" ".join(desconhecidas[:2])]
    return ped


def sparkline(valores: list[float], largura: int = 40) -> str:
    if not valores:
        return ""
    if len(valores) > largura:
        passo = len(valores) / largura
        valores = [valores[int(i * passo)] for i in range(largura)]
    lo, hi = min(valores), max(valores)
    if hi == lo:
        return BLOCOS[3] * len(valores)
    return "".join(BLOCOS[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in valores)


def brl(v: float) -> str:
    s = f"{v:,.2f}" if v >= 1 else f"{v:,.6f}".rstrip("0")
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def seta(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{'🟢▲' if p >= 0 else '🔴▼'} {p:+.2f}%".replace(".", ",")


def grande(v: float | None) -> str:
    if not v:
        return "—"
    for lim, suf in ((1e12, " tri"), (1e9, " bi"), (1e6, " mi")):
        if v >= lim:
            return f"{v / lim:.1f}{suf}".replace(".", ",")
    return brl(v)


def buscar_id(termo: str) -> tuple[str, str] | None:
    d = C.http_json(f"{API}/search?query={urllib.parse.quote(termo)}", timeout=10)
    coins = d.get("coins") or []
    if not coins:
        return None
    c = coins[0]
    return c["id"], c.get("name", c["id"])


def main() -> None:
    bruto = C.contexto()
    ped = interpretar(C.clean_command(bruto) or bruto)
    try:
        if ped["top"]:
            d = C.http_json(f"{API}/coins/markets?vs_currency=brl&order=market_cap_desc&per_page={ped['top']}&page=1&price_change_percentage=24h", timeout=12)
            usd = C.http_json(f"{API}/simple/price?ids=bitcoin&vs_currencies=usd,brl", timeout=10)
            cambio = usd["bitcoin"]["brl"] / usd["bitcoin"]["usd"] if usd.get("bitcoin") else None
            linhas = [f"🏆 Top {len(d)} criptomoedas por valor de mercado", ""]
            for c in d:
                u = f" (US$ {brl(c['current_price'] / cambio)})" if cambio else ""
                linhas.append(f"{c['market_cap_rank']:>2}. {c['name']} ({c['symbol'].upper()})  R$ {brl(c['current_price'])}{u}  {seta(c.get('price_change_percentage_24h_in_currency'))}")
            C.finish(REPORT, "\n".join(linhas) + "\n\nFonte: CoinGecko", resposta="direta", interpretado=f"cripto: top {ped['top']}")
            return
        ids = ped["ids"][:5]
        for termo in ped["buscar"]:
            achado = buscar_id(termo)
            if achado:
                ids.append(achado[0])
        if not ids:
            ids = ["bitcoin"] if not re.search(r"\b\w+\b", C.norm(bruto).replace("cripto", "")) or re.search(r"cripto|bitcoin", C.norm(bruto)) else []
        if not ids:
            C.finish(REPORT, "⚠️ Qual criptomoeda? Ex.: “preço do bitcoin”, “ethereum e solana”, “top 10 criptomoedas”.", resposta="direta", interpretado="cripto: sem moeda")
            return
        d = C.http_json(f"{API}/coins/markets?vs_currency=brl&ids={','.join(ids)}&price_change_percentage=24h,7d", timeout=12)
        usd = C.http_json(f"{API}/simple/price?ids={','.join(ids)}&vs_currencies=usd", timeout=10)
        if not d:
            C.finish(REPORT, "⚠️ Não encontrei essa criptomoeda no CoinGecko.", resposta="direta", interpretado="cripto: não encontrada")
            return
        linhas = ["💰 Criptomoedas — CoinGecko", ""]
        for c in d:
            u = (usd.get(c["id"]) or {}).get("usd")
            linhas += [
                f"{c['name']} ({c['symbol'].upper()})  #{c.get('market_cap_rank') or '?'}",
                f"   R$ {brl(c['current_price'])}" + (f"   ·   US$ {brl(u)}" if u else ""),
                f"   24h {seta(c.get('price_change_percentage_24h_in_currency'))}   7d {seta(c.get('price_change_percentage_7d_in_currency'))}",
                f"   Máx/mín 24h: R$ {brl(c.get('high_24h') or 0)} / R$ {brl(c.get('low_24h') or 0)}   ·   Valor de mercado: R$ {grande(c.get('market_cap'))}",
            ]
            if ped["grafico"]:
                try:
                    h = C.http_json(f"{API}/coins/{c['id']}/market_chart?vs_currency=brl&days={ped['dias']}", timeout=12)
                    pr = [p[1] for p in h.get("prices", [])]
                    if pr:
                        linhas += [f"   {ped['dias']} dias: {sparkline(pr)}", f"   mín R$ {brl(min(pr))}  ·  máx R$ {brl(max(pr))}  ·  variação {seta((pr[-1] / pr[0] - 1) * 100)}"]
                except Exception:
                    linhas.append("   (gráfico indisponível agora)")
            linhas.append("")
        C.finish(REPORT, "\n".join(linhas).strip(), resposta="direta", moedas=[c["id"] for c in d], interpretado=f"cripto: {','.join(ids)}")
    except (C.HttpError, OSError, KeyError, ValueError) as exc:
        C.fail(f"Não consegui consultar o CoinGecko agora (limite de acesso ou sem internet): {exc}", REPORT)


if __name__ == "__main__":
    main()
