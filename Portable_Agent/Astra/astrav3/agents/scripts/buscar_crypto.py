import sys
import json
import re
import ssl
import urllib.request
import urllib.parse
from pathlib import Path

# Configura codificação segura para o Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

ssl_context = ssl._create_unverified_context()
HEADERS = {"User-Agent": "Astra-Crypto-Agent/1.0"}

# Mapeamento rápido de apelidos comuns para os IDs oficiais da API do CoinGecko
MAPA_MOEDAS = {
    "btc": "bitcoin",
    "bitcoin": "bitcoin",
    "eth": "ethereum",
    "ethereum": "ethereum",
    "sol": "solana",
    "solana": "solana",
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    "ada": "cardano",
    "cardano": "cardano",
    "xrp": "ripple",
    "ripple": "ripple",
    "pepe": "pepe",
    "bnb": "binancecoin"
}

def identificar_moeda(frase: str) -> str:
    frase_lower = frase.lower()
    for sigla, id_cg in MAPA_MOEDAS.items():
        if re.search(rf"\b{sigla}\b", frase_lower):
            return id_cg
    return "bitcoin"  # Padrão se não especificar

def quer_grafico(frase: str) -> bool:
    frase_lower = frase.lower()
    return any(palavra in frase_lower for palavra in ["grafico", "gráfico", "historico", "histórico", "tendencia", "evolução"])

def buscar_dados_coingecko(coin_id: str, incluir_grafico: bool):
    # Endpoint de preço atual, volume e variação de 24h
    url_price = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd,brl&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"
    
    req = urllib.request.Request(url_price, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10, context=ssl_context) as resp:
        data_price = json.loads(resp.read().decode("utf-8"))
        
    if coin_id not in data_price:
        raise ValueError(f"Moeda '{coin_id}' não encontrada na base do CoinGecko.")
        
    info = data_price[coin_id]
    
    precos_historicos = []
    if incluir_grafico:
        # Pega os preços dos últimos 7 dias (pontos diários) para desenhar o gráfico
        url_chart = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=7"
        try:
            req_c = urllib.request.Request(url_chart, headers=HEADERS)
            with urllib.request.urlopen(req_c, timeout=10, context=ssl_context) as resp_c:
                data_chart = json.loads(resp_c.read().decode("utf-8"))
                precos_historicos = [p[1] for p in data_chart.get("prices", [])]
        except Exception:
            pass
            
    return info, precos_historicos

def gerar_grafico_ascii(precos: list) -> str:
    if not precos:
        return "[Gráfico indisponível no momento]"
    
    # Reduz para 15 pontos para caber bem no terminal
    passo = max(1, len(precos) // 15)
    amostras = [precos[i] for i in range(0, len(precos), passo)][-15:]
    
    min_p, max_p = min(amostras), max(amostras)
    altura = 5
    canvas = [[" " for _ in range(len(amostras))] for _ in range(altura)]
    
    for col, val in enumerate(amostras):
        if max_p == min_p:
            linha = altura // 2
        else:
            linha = int((val - min_p) / (max_p - min_p) * (altura - 1))
        linha = altura - 1 - linha  # Inverte eixo Y do terminal
        canvas[linha][col] = "●"
        
    # Liga os pontos visualmente com traços simples se desejar ou deixa os pontos
    linhas_str = []
    for r in range(altura):
        # Adiciona escala de preço aproximada na lateral esquerda
        linhas_str.append("".join(canvas[r]))
        
    grafico_desenhado = "\n".join(linhas_str)
    return f"\nTendência dos últimos 7 dias (USD):\n{grafico_desenhado}\nMin: ${min_p:,.2f} | Máx: ${max_p:,.2f}"

def main():
    try:
        raw_bytes = sys.stdin.buffer.read()
        data = json.loads(raw_bytes.decode("utf-8", errors="replace")) if raw_bytes else {}
    except Exception:
        data = {}

    contexto = str(data.get("contexto", ""))
    coin_id = identificar_moeda(contexto)
    mostrar_grafico = quer_grafico(contexto)

    arquivo_saida = Path("crypto/relatorio.txt")
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)

    try:
        info, historico = buscar_dados_coingecko(coin_id, mostrar_grafico)
        
        nome_bonito = coin_id.capitalize()
        preco_usd = info.get("usd", 0)
        preco_brl = info.get("brl", 0)
        variacao_24h = info.get("usd_24h_change", 0)
        market_cap = info.get("usd_market_cap", 0)
        vol_24h = info.get("usd_24h_vol", 0)

        emoji_var = "📈" if variacao_24h >= 0 else "📉"
        sinal_var = "+" if variacao_24h >= 0 else ""

        relatorio = (
            f"Relatório de Mercado Cripto: {nome_bonito} ({coin_id.upper()})\n"
            f"----------------------------------------\n"
            f"• Preço atual (USD): ${preco_usd:,.2f}\n"
            f"• Preço atual (BRL): R$ {preco_brl:,.2f}\n"
            f"• Variação em 24h: {emoji_var} {sinal_var}{variacao_24h:.2f}%\n"
            f"• Valor de Mercado: ${market_cap:,.0f}\n"
            f"• Volume (24h): ${vol_24h:,.0f}\n"
        )

        if mostrar_grafico:
            relatorio += gerar_grafico_ascii(historico)

        arquivo_saida.write_text(relatorio, encoding="utf-8")

        print(json.dumps({
            "ok": True,
            "coin": coin_id,
            "price_usd": preco_usd,
            "change_24h": variacao_24h,
            "status": "sucesso"
        }, ensure_ascii=True))

    except Exception as e:
        msg_erro = f"Erro ao consultar dados de criptomoedas: {e}"
        arquivo_saida.write_text(msg_erro, encoding="utf-8")
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=True))

if __name__ == "__main__":
    main()