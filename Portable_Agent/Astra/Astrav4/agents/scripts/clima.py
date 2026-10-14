"""Agente clima: previsão detalhada via Open-Meteo (sem chave de API).

Entende: "clima em Curitiba", "vai chover amanhã no Rio?", "previsão da semana
em Lisboa, Portugal", "preciso de guarda-chuva?", "qualidade do ar", "defina
minha cidade como Campinas". Sem cidade no pedido usa a cidade padrão salva
(clima/config.json) ou, na falta dela, a localização aproximada pelo IP.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
C.enable_user_site()

REPORT = "clima/previsao.txt"
CONFIG = Path("clima/config.json")
GEO = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"

CODIGOS = {
    0: ("Céu limpo", "☀️"), 1: ("Predominantemente limpo", "🌤️"), 2: ("Parcialmente nublado", "⛅"),
    3: ("Nublado", "☁️"), 45: ("Nevoeiro", "🌫️"), 48: ("Nevoeiro com geada", "🌫️"),
    51: ("Garoa fraca", "🌦️"), 53: ("Garoa moderada", "🌦️"), 55: ("Garoa forte", "🌧️"),
    56: ("Garoa congelante", "🌧️"), 57: ("Garoa congelante forte", "🌧️"),
    61: ("Chuva fraca", "🌦️"), 63: ("Chuva moderada", "🌧️"), 65: ("Chuva forte", "🌧️"),
    66: ("Chuva congelante", "🌧️"), 67: ("Chuva congelante forte", "🌧️"),
    71: ("Neve fraca", "🌨️"), 73: ("Neve moderada", "🌨️"), 75: ("Neve forte", "❄️"), 77: ("Grãos de neve", "🌨️"),
    80: ("Pancadas de chuva fracas", "🌦️"), 81: ("Pancadas de chuva", "🌧️"), 82: ("Pancadas violentas", "⛈️"),
    85: ("Pancadas de neve", "🌨️"), 86: ("Pancadas de neve fortes", "❄️"),
    95: ("Trovoadas", "⛈️"), 96: ("Trovoadas com granizo", "⛈️"), 99: ("Trovoadas com granizo forte", "⛈️"),
}
DIAS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]

_RUIDO = {
    "clima", "tempo", "previsao", "previsão", "temperatura", "chuva", "chover", "chovendo", "frio", "calor", "vento", "uv",
    "como", "esta", "esta", "está", "ta", "tá", "vai", "vem", "fazer", "faz", "fara", "ficar", "fica", "sera", "será",
    "hoje", "amanha", "amanhã", "agora", "semana", "fim", "de", "do", "da", "dos", "das", "no", "na", "em", "para", "pra",
    "por", "o", "a", "os", "as", "um", "uma", "que", "qual", "quais", "quanto", "bem", "bom", "ai", "aí", "la", "lá", "ali",
    "depois", "proximos", "próximos", "proximo", "próximo", "dias", "dia", "preciso", "precisa", "levar", "guarda-chuva",
    "guarda", "chuva", "casaco", "agasalho", "sol", "protetor", "solar", "ar", "qualidade", "poluicao", "poluição",
    "detalhado", "detalhada", "completo", "completa", "e", "me", "diga", "fale", "informe", "mostre", "ver", "veja",
    "consulta", "consultar", "consulte", "pesquisar", "pesquise", "astra", "usando", "use", "agente", "agent", "meu",
    "minha", "cidade", "sera", "tera", "terá", "ter", "tenho", "esta", "estar", "estou", "estamos", "previsao", "umidade",
    "sensacao", "sensação", "termica", "térmica", "nublado", "ensolarado", "limpo", "tarde", "noite", "manha", "manhã",
    "madrugada", "terca", "terça", "quarta", "quinta", "sexta", "sabado", "sábado", "domingo", "segunda", "feira",
    "se", "ha", "há", "tem", "tera", "verao", "verão", "inverno", "outono", "primavera", "essa", "esta", "nesta", "neste",
}


# ------------------------------------------------------------------ pedido

def _config() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _salvar_config(cfg: dict) -> None:
    C.write_text(CONFIG, json.dumps(cfg, ensure_ascii=False, indent=2))


_LIGACAO = {"de", "do", "da", "dos", "das", "d'", "e"}
_RUIDO_N = {C.norm(x) for x in _RUIDO}


def extrair_cidade(texto: str) -> str:
    """Pega a cidade depois de em/no/na/para; sem preposição, descarta as
    palavras de clima e usa o que sobrar ('clima curitiba' → 'Curitiba').
    Mantém 'Rio de Janeiro' e 'Lisboa, Portugal'."""
    t = re.sub(r"[?!]+", " ", texto).strip()
    achados = [m.group(1) for m in re.finditer(r"\b(?:em|no|na|nos|nas|para|pra|pro|sobre|de|do|da)\s+(.+)$", t, re.I)]
    for cand in [*achados[::-1], t] if achados else [t]:
        tokens = re.findall(r"[A-Za-zÀ-ÿ0-9'’.-]+|,", cand)
        nome: list[str] = []
        for i, tok in enumerate(tokens):
            n = C.norm(tok)
            if tok == ",":
                if nome:
                    nome.append(",")
                continue
            if n in _LIGACAO:
                prox = C.norm(tokens[i + 1]) if i + 1 < len(tokens) else ""
                if nome and prox and prox not in _RUIDO_N and prox != ",":
                    nome.append(tok)
                    continue
                if nome:
                    break
                continue
            if n in _RUIDO_N:
                if nome:
                    break
                continue
            nome.append(tok)
        cidade = re.sub(r"\s+,", ",", " ".join(nome)).strip(" ,.-")
        if len(cidade) >= 2:
            return cidade
    return ""


def interpretar_pedido(texto: str) -> dict:
    n = C.norm(texto)
    pedido = {"definir": None, "dias": 7, "foco": "geral", "alvo": None, "cidade": ""}
    m = re.search(r"\b(?:defin\w+|configur\w+|mud\w+|troc\w+|salv\w+|fix\w+)\b.*?\b(?:cidade|local|localizacao)\b.*?\b(?:como|para|pra|em|:)\s+(.+)$", texto, re.I) \
        or re.search(r"\bminha\s+cidade\s+(?:e|é|eh)\s+(.+)$", texto, re.I) \
        or re.search(r"\bmoro\s+(?:em|na|no)\s+(.+)$", texto, re.I)
    if m:
        pedido["definir"] = m.group(1).strip(" .!?")
        return pedido
    if re.search(r"\b(?:chuv|chov|guarda\s*chuva|molh|pancada|garoa|temporal|trovoad)\w*", n):
        pedido["foco"] = "chuva"
    elif re.search(r"\b(?:qualidade do ar|poluicao|aqi|fumaca|respirar)\b", n):
        pedido["foco"] = "ar"
    elif re.search(r"\b(?:uv|protetor solar|bronz|indice ultravioleta)\b", n):
        pedido["foco"] = "uv"
    elif re.search(r"\b(?:casaco|agasalho|frio|friozinho|calor|quente|gelado|temperatura)\b", n):
        pedido["foco"] = "temperatura"
    hoje = date.today()
    if re.search(r"\bdepois de amanha\b", n):
        pedido["alvo"] = hoje + timedelta(days=2)
    elif re.search(r"\bamanha\b", n):
        pedido["alvo"] = hoje + timedelta(days=1)
    elif re.search(r"\b(?:hoje|agora|neste momento|nesse momento)\b", n):
        pedido["alvo"] = hoje
    else:
        for i, nome in enumerate(["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]):
            if re.search(rf"\b{nome}(?:\s*feira)?\b", n):
                delta = (i - hoje.weekday()) % 7
                pedido["alvo"] = hoje + timedelta(days=delta or 7)
                break
    if re.search(r"\b(?:fim de semana|fds|final de semana)\b", n):
        sab = hoje + timedelta(days=(5 - hoje.weekday()) % 7)
        pedido["alvo"] = sab if sab >= hoje else hoje
        pedido["fds"] = True
    mm = re.search(r"\bproximos?\s+(\d{1,2})\s+dias\b", n)
    if mm:
        pedido["dias"] = max(1, min(int(mm.group(1)), 16))
    pedido["cidade"] = extrair_cidade(C.clean_command(texto, {"clima", "tempo"}))
    return pedido


# ------------------------------------------------------------------ rede

def geocodificar(cidade: str) -> dict | None:
    partes = [p.strip() for p in re.split(r"\s*,\s*", cidade) if p.strip()]
    nome = partes[0]
    dica = C.norm(partes[1]) if len(partes) > 1 else ""
    # "Campinas SP" sem vírgula: tenta separar a sigla/país no fim.
    if not dica:
        m = re.match(r"^(.+?)\s+([A-Za-z]{2})$", nome)
        if m and m.group(2).upper() in {"AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"}:
            nome, dica = m.group(1), m.group(2).lower()
    url = f"{GEO}?name={urllib.parse.quote(nome)}&count=8&language=pt&format=json"
    dados = C.http_json(url, timeout=10)
    achados = dados.get("results") or []
    if not achados:
        return None
    if dica:
        for r in achados:
            alvo = " ".join(C.norm(r.get(k, "")) for k in ("admin1", "country", "country_code", "admin2"))
            if dica in alvo.split() or dica in alvo:
                return r
    # Sem dica: prefere a maior população (evita "São Paulo" virar uma vila).
    return max(achados, key=lambda r: r.get("population") or 0)


def localizacao_por_ip() -> dict | None:
    try:
        d = C.http_json("https://ipwho.is/", timeout=6, retries=1)
        if d.get("success") and d.get("latitude") is not None:
            return {"name": d.get("city") or "Sua região", "admin1": d.get("region"), "country": d.get("country"),
                    "latitude": d["latitude"], "longitude": d["longitude"], "timezone": (d.get("timezone") or {}).get("id")}
    except Exception:
        pass
    return None


def baixar(lat: float, lon: float, dias: int) -> tuple[dict, dict | None]:
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": max(dias, 2),
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,wind_gusts_10m,is_day",
        "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,uv_index_max,sunrise,sunset,wind_speed_10m_max",
    })
    prev = C.http_json(f"{FORECAST}?{q}", timeout=14)
    try:
        ar = C.http_json(f"{AIR}?" + urllib.parse.urlencode({"latitude": lat, "longitude": lon, "current": "us_aqi,pm2_5,pm10", "timezone": "auto"}), timeout=8, retries=1)
    except Exception:
        ar = None
    return prev, ar


# ------------------------------------------------------------------ texto

def desc(code: int | None) -> tuple[str, str]:
    return CODIGOS.get(int(code) if code is not None else -1, ("Condição desconhecida", "🌡️"))


def rotulo_uv(v: float | None) -> str:
    if v is None:
        return "—"
    faixas = [(3, "baixo"), (6, "moderado"), (8, "alto"), (11, "muito alto")]
    nome = next((n for lim, n in faixas if v < lim), "extremo")
    return f"{v:.0f} ({nome})"


def rotulo_aqi(v: float | None) -> str:
    if v is None:
        return "—"
    faixas = [(50, "boa"), (100, "moderada"), (150, "ruim para sensíveis"), (200, "ruim"), (300, "muito ruim")]
    nome = next((n for lim, n in faixas if v <= lim), "perigosa")
    return f"{v:.0f} ({nome})"


def dia_nome(d: date, hoje: date) -> str:
    if d == hoje:
        return "Hoje"
    if d == hoje + timedelta(days=1):
        return "Amanhã"
    return f"{DIAS[d.weekday()].capitalize()} {d:%d/%m}"


def janelas_de_chuva(horas: list[tuple[datetime, float, float]], limite: float = 40) -> list[str]:
    """Agrupa horas seguidas com probabilidade ≥ limite em faixas legíveis."""
    faixas, ini, ult, pico = [], None, None, 0.0
    for dt, prob, _mm in horas:
        if prob >= limite:
            if ini is None:
                ini = dt
            ult, pico = dt, max(pico, prob)
        elif ini is not None:
            faixas.append((ini, ult, pico))
            ini = None
            pico = 0.0
    if ini is not None:
        faixas.append((ini, ult, pico))
    return [f"{a:%H}h–{(b + timedelta(hours=1)):%H}h (até {p:.0f}%)" for a, b, p in faixas]


def montar(local: dict, prev: dict, ar: dict | None, pedido: dict, agora: datetime | None = None) -> tuple[str, str]:
    cur, daily, hourly = prev.get("current", {}), prev.get("daily", {}), prev.get("hourly", {})
    agora = agora or datetime.fromisoformat(cur["time"]) if cur.get("time") else datetime.now()
    hoje = agora.date()
    nome = local.get("name", "?")
    extra = ", ".join(x for x in (local.get("admin1"), local.get("country")) if x and x != nome)
    titulo = f"{nome}" + (f" — {extra}" if extra else "")

    dias = []
    for i, ds in enumerate(daily.get("time", [])):
        dias.append({
            "data": date.fromisoformat(ds), "cod": daily["weather_code"][i], "max": daily["temperature_2m_max"][i],
            "min": daily["temperature_2m_min"][i], "mm": daily["precipitation_sum"][i] or 0,
            "prob": (daily.get("precipitation_probability_max") or [None] * 99)[i], "uv": (daily.get("uv_index_max") or [None] * 99)[i],
            "vento": (daily.get("wind_speed_10m_max") or [None] * 99)[i], "nasce": daily["sunrise"][i][-5:], "poe": daily["sunset"][i][-5:],
        })
    horas = []
    for i, ts in enumerate(hourly.get("time", [])):
        horas.append((datetime.fromisoformat(ts), hourly["precipitation_probability"][i] or 0, hourly["precipitation"][i] or 0, hourly["temperature_2m"][i], hourly["weather_code"][i]))

    linhas: list[str] = []
    resumo = ""
    aqi = (ar or {}).get("current", {}).get("us_aqi") if ar else None

    # --- cabeçalho com a resposta direta ---
    alvo = pedido.get("alvo")
    dia_alvo = next((d for d in dias if d["data"] == alvo), None) if alvo else None
    atual_txt, atual_emoji = desc(cur.get("weather_code"))
    if dia_alvo and dia_alvo["data"] != hoje:
        t, e = desc(dia_alvo["cod"])
        resumo = f"{dia_nome(dia_alvo['data'], hoje)} em {nome}: {t.lower()}, {dia_alvo['min']:.0f}° a {dia_alvo['max']:.0f}°C"
        if dia_alvo["prob"] is not None:
            resumo += f", chance de chuva {dia_alvo['prob']:.0f}%"
        if pedido["foco"] == "chuva":
            p = dia_alvo["prob"] or 0
            resumo = ("Sim, leve guarda-chuva. " if p >= 50 else "Talvez — leve um guarda-chuva por garantia. " if p >= 30 else "Provavelmente não vai chover. ") + resumo
    else:
        resumo = f"Agora em {nome}: {cur.get('temperature_2m', 0):.0f}°C (sensação {cur.get('apparent_temperature', 0):.0f}°C), {atual_txt.lower()}"
        if dias:
            resumo += f". Hoje: {dias[0]['min']:.0f}° a {dias[0]['max']:.0f}°C"
            if dias[0]["prob"] is not None:
                resumo += f", chance de chuva {dias[0]['prob']:.0f}%"
        if pedido["foco"] == "chuva":
            futuras = [(dt, p, mm) for dt, p, mm, *_ in horas if agora - timedelta(hours=1) <= dt <= agora + timedelta(hours=24)]
            jan = janelas_de_chuva(futuras)
            resumo = (("Sim, há chance de chuva nas próximas 24h: " + "; ".join(jan[:3]) + ". Leve guarda-chuva. ") if jan else "Sem previsão de chuva relevante nas próximas 24h. ") + resumo

    linhas += [f"{atual_emoji} {titulo}", resumo, ""]

    # --- agora ---
    linhas += [
        "AGORA",
        f"  {atual_emoji} {atual_txt} · {cur.get('temperature_2m', 0):.1f}°C (sensação {cur.get('apparent_temperature', 0):.1f}°C)",
        f"  💧 Umidade {cur.get('relative_humidity_2m', 0):.0f}% · 💨 Vento {cur.get('wind_speed_10m', 0):.0f} km/h (rajadas {cur.get('wind_gusts_10m', 0):.0f}) · 🌧️ {cur.get('precipitation', 0):.1f} mm agora",
    ]
    if dias:
        d0 = dias[0]
        linhas.append(f"  🌅 Nascer {d0['nasce']} · 🌇 pôr do sol {d0['poe']} · ☀️ UV máx. {rotulo_uv(d0['uv'])}")
    if aqi is not None:
        linhas.append(f"  🫁 Qualidade do ar (AQI): {rotulo_aqi(aqi)}")

    # --- chuva nas próximas horas ---
    prox = [(dt, p, mm) for dt, p, mm, *_ in horas if agora - timedelta(hours=1) < dt <= agora + timedelta(hours=24)]
    if prox:
        jan = janelas_de_chuva(prox)
        linhas += ["", "CHUVA NAS PRÓXIMAS 24H", "  " + ("; ".join(jan) if jan else "Sem chance relevante de chuva (<40%).")]

    # --- hora a hora (foco no alvo, ou próximas 12h) ---
    if pedido["foco"] in ("geral", "temperatura") and horas:
        base = [h for h in horas if h[0].date() == (alvo or hoje)]
        if (alvo or hoje) == hoje:
            base = [h for h in horas if h[0] >= agora.replace(minute=0, second=0, microsecond=0)][:12]
        passo = base[::3] if base else []
        if passo:
            linhas += ["", "HORA A HORA"]
            for dt, p, mm, temp, cod in passo:
                linhas.append(f"  {dt:%H}h  {desc(cod)[1]} {temp:.0f}°C  · chuva {p:.0f}%")

    # --- semana ---
    mostrar = dias[: pedido.get("dias", 7)]
    if mostrar:
        linhas += ["", f"PRÓXIMOS {len(mostrar)} DIAS"]
        for d in mostrar:
            t, e = desc(d["cod"])
            prob = f"{d['prob']:.0f}%" if d["prob"] is not None else "—"
            linhas.append(f"  {dia_nome(d['data'], hoje):<14} {e} {d['min']:>3.0f}° / {d['max']:>3.0f}°  chuva {prob:>4} ({d['mm']:.1f} mm)  {t}")

    # --- dicas ---
    dicas = []
    if dias:
        d0 = dias[0]
        if (d0["prob"] or 0) >= 50:
            dicas.append("☔ Leve guarda-chuva.")
        if (d0["uv"] or 0) >= 6:
            dicas.append("🧴 UV alto: protetor solar.")
        if d0["max"] >= 33:
            dicas.append("🥤 Calor forte: hidrate-se.")
        if d0["min"] <= 12:
            dicas.append("🧥 Frio: leve casaco.")
    if aqi is not None and aqi > 100:
        dicas.append("😷 Ar ruim: evite exercício ao ar livre.")
    if dicas:
        linhas += ["", "DICAS", "  " + "  ".join(dicas)]
    linhas += ["", "Fonte: Open-Meteo"]
    return "\n".join(linhas).strip(), resumo


# ------------------------------------------------------------------ main

def main() -> None:
    texto = C.contexto()
    pedido = interpretar_pedido(texto)
    cfg = _config()

    if pedido["definir"]:
        try:
            achado = geocodificar(pedido["definir"])
        except Exception as exc:
            C.fail(f"Não consegui validar a cidade agora ({exc}).", REPORT)
            return
        if not achado:
            msg = f"⚠️ Não encontrei a cidade “{pedido['definir']}”. Tente “Cidade, Estado” ou “Cidade, País”."
            C.finish(REPORT, msg, resposta="direta", interpretado=f"definir cidade: {pedido['definir']}", encontrado=False)
            return
        cfg["cidade_padrao"] = {k: achado.get(k) for k in ("name", "admin1", "country", "latitude", "longitude", "timezone")}
        _salvar_config(cfg)
        quem = ", ".join(x for x in (achado.get("name"), achado.get("admin1"), achado.get("country")) if x)
        C.finish(REPORT, f"📍 Cidade padrão definida: {quem}. Agora “clima” sem cidade usa essa.", resposta="direta", interpretado=f"definir cidade: {quem}")
        return

    try:
        if pedido["cidade"]:
            local = geocodificar(pedido["cidade"])
            if not local:
                msg = f"⚠️ Não encontrei a cidade “{pedido['cidade']}”. Tente “Cidade, Estado” ou “Cidade, País”."
                C.finish(REPORT, msg, resposta="direta", interpretado=f"clima: {pedido['cidade']}", encontrado=False)
                return
        else:
            local = cfg.get("cidade_padrao") or localizacao_por_ip()
            if not local:
                C.finish(REPORT, "⚠️ Não sei sua cidade. Diga “clima em <cidade>” ou “defina minha cidade como <cidade>”.", resposta="direta", interpretado="clima: sem cidade", encontrado=False)
                return
        prev, ar = baixar(local["latitude"], local["longitude"], max(pedido["dias"], 2))
    except (C.HttpError, OSError, ValueError, KeyError) as exc:
        C.fail(f"Não consegui consultar o clima agora (sem internet ou serviço fora do ar): {exc}", REPORT)
        return

    corpo, resumo = montar(local, prev, ar, pedido)
    C.finish(REPORT, corpo, resposta="direta", cidade=local.get("name"), resumo=resumo, interpretado=f"clima: {local.get('name')} · foco={pedido['foco']}")


if __name__ == "__main__":
    main()
