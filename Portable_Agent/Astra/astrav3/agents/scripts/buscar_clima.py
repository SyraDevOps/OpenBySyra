import sys
import json
import re
import ssl
import unicodedata
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

# Força a saída do Windows para aceitar UTF-8 sem travar no charmap/cp1252
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

ssl_context = ssl._create_unverified_context()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

TABELA_CLIMA = {
    0: "Céu limpo e ensolarado ☀️",
    1: "Predominantemente ensolarado 🌤️",
    2: "Parcialmente nublado ⛅",
    3: "Nublado / Encoberto ☁️",
    45: "Nevoeiro 🌫️",
    48: "Nevoeiro denso 🌫️",
    51: "Garoa leve 🌧️",
    53: "Garoa moderada 🌧️",
    55: "Garoa densa 🌧️",
    61: "Chuva fraca 🌦️",
    63: "Chuva moderada 🌧️",
    65: "Chuva forte 🌧️",
    80: "Pancadas de chuva passageiras 🌦️",
    81: "Pancadas de chuva moderadas 🌧️",
    82: "Pancadas de chuva violentas ⛈️",
    95: "Tempestade com trovoadas ⚡",
    96: "Tempestade com granizo ⛈️",
    99: "Tempestade severa 🌪️"
}

def remover_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))

def extrair_nome_cidade(frase: str) -> str:
    """Extrai o nome da cidade de forma flexível a partir de preposições ou limpando ruídos."""
    frase_limpa = str(frase).strip()
    if not frase_limpa:
        return "São Paulo"

    # Tenta capturar após preposições comuns (ex: 'em', 'na', 'no', 'para', 'de', 'sobre')
    match = re.search(r"\b(?:em|na|no|para|de|sobre)\s+([A-Za-zÀ-ÿ0-9\s'-]+)(?:\?|$)", frase_limpa, re.IGNORECASE)
    if match:
        cidade = match.group(1).strip()
    else:
        # Se não houver preposição, remove palavras-comando comuns da Astra/Linguagem natural
        cidade = re.sub(r"\b(usando|o|a|os|as|um|uma|agente|clima|me|fala|como|está|esta|tempo|cidade|previsao|previsão|pesquise|por|favor|hoje|agora)\b", "", frase_limpa, flags=re.IGNORECASE)
        cidade = cidade.strip()
    
    # Remove pontuações residuais
    cidade = re.sub(r"[.,:;!?\t\r\n]", "", cidade).strip()
    
    return cidade if len(cidade) > 1 else "São Paulo"

def geolocalizar(cidade: str):
    candidatos = [cidade, remover_acentos(cidade)]
    for q in candidatos:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(q)}&count=5&language=pt&format=json"
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10, context=ssl_context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "results" in data and len(data["results"]) > 0:
                    resultados = data["results"]
                    # Prioriza resultados do Brasil se o nome for ambíguo
                    br = [r for r in resultados if r.get("country_code") == "BR"]
                    item = br[0] if br else resultados[0]
                    return {
                        "nome": item.get("name"),
                        "estado": item.get("admin1", ""),
                        "pais": item.get("country", ""),
                        "lat": item.get("latitude"),
                        "lon": item.get("longitude")
                    }
        except Exception:
            continue
    return None

def obter_previsao(lat: float, lon: float):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
        f"&timezone=auto"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10, context=ssl_context) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    try:
        raw_bytes = sys.stdin.buffer.read()
        raw_in = raw_bytes.decode("utf-8", errors="replace").strip()
        data = json.loads(raw_in) if raw_in else {}
    except Exception:
        data = {}

    contexto = str(data.get("contexto", "")).strip()
    cidade = extrair_nome_cidade(contexto)

    arquivo_saida = Path("clima/previsao.txt")
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)

    try:
        local = geolocalizar(cidade)
        if not local:
            msg_erro = f"Não consegui encontrar as coordenadas geográficas para a localidade '{cidade}'."
            arquivo_saida.write_text(msg_erro, encoding="utf-8")
            print(json.dumps({"ok": False, "error": msg_erro}, ensure_ascii=True))
            return

        dados = obter_previsao(local["lat"], local["lon"])
        current = dados.get("current", {})
        daily = dados.get("daily", {})

        cod_clima = current.get("weather_code", 0)
        condicao = TABELA_CLIMA.get(cod_clima, "Tempo estável")

        temp = current.get("temperature_2m")
        sensacao = current.get("apparent_temperature")
        umidade = current.get("relative_humidity_2m")
        vento = current.get("wind_speed_10m")

        # Dados diários para os próximos dias (Hoje, Amanhã, Depois de amanhã)
        dias_max = daily.get("temperature_2m_max", [temp, temp, temp])
        dias_min = daily.get("temperature_2m_min", [temp, temp, temp])
        dias_chuva = daily.get("precipitation_probability_max", [0, 0, 0])
        dias_codigo = daily.get("weather_code", [0, 0, 0])

        local_formatado = f"{local['nome']}"
        if local['estado']:
            local_formatado += f" ({local['estado']})"
        local_formatado += f", {local['pais']}"

        # Montagem do relatório detalhado
        resultado = (
            f"🌍 Previsão do Tempo para: {local_formatado}\n"
            f"----------------------------------------\n"
            f"📍 CONDIÇÃO ATUAL:\n"
            f"• Situação: {condicao}\n"
            f"• Temperatura: {temp}°C (Sensação térmica: {sensacao}°C)\n"
            f"• Umidade do ar: {umidade}%\n"
            f"• Vento: {vento} km/h\n\n"
            f"📅 PRÓXIMOS DIAS:\n"
        )

        # Adiciona os próximos 3 dias de forma dinâmica
        hoje = datetime.now()
        nomes_dias = ["Hoje", "Amanhã", (hoje + timedelta(days=2)).strftime("%d/%m")]

        for i in range(min(3, len(dias_max))):
            nome_dia = nomes_dias[i] if i < 2 else f"Dia ({nomes_dias[i]})"
            cond_dia = TABELA_CLIMA.get(dias_codigo[i], "Variável")
            resultado += (
                f"• {nome_dia}: Máx {dias_max[i]}°C | Mín {dias_min[i]}°C | "
                f"Chuva: {dias_chuva[i]}% | {cond_dia}\n"
            )

        arquivo_saida.write_text(resultado, encoding="utf-8")

        print(json.dumps({
            "ok": True,
            "cidade": local_formatado,
            "temperatura_atual": temp,
            "condicao": condicao,
            "status": "sucesso"
        }, ensure_ascii=True))

    except Exception as e:
        msg_falha = f"Erro ao consultar o serviço meteorológico: {e}"
        arquivo_saida.write_text(msg_falha, encoding="utf-8")
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=True))

if __name__ == "__main__":
    main()