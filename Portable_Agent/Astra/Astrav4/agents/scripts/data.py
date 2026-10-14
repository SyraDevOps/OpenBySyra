"""Agente data: data/hora, contas com datas, feriados e fusos.

  que dia é hoje | que horas são | que horas são em Tóquio
  quantos dias faltam para o Natal | quantos dias até 25/12 | daqui a 45 dias
  que dia da semana foi 7/9/1822 | quantos dias entre 1/1/2026 e 31/12/2026
  próximo feriado | feriados de dezembro | qual a semana do ano
  quantos anos tem quem nasceu em 15/03/1990
"""

from __future__ import annotations

import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402
import _astra_datas as D  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "data/resultado.txt"

FUSOS = {
    "toquio": ("Asia/Tokyo", 9), "tokyo": ("Asia/Tokyo", 9), "londres": ("Europe/London", 0), "paris": ("Europe/Paris", 1), "lisboa": ("Europe/Lisbon", 0),
    "madri": ("Europe/Madrid", 1), "berlim": ("Europe/Berlin", 1), "roma": ("Europe/Rome", 1), "moscou": ("Europe/Moscow", 3), "dubai": ("Asia/Dubai", 4),
    "nova york": ("America/New_York", -5), "new york": ("America/New_York", -5), "los angeles": ("America/Los_Angeles", -8), "san francisco": ("America/Los_Angeles", -8),
    "chicago": ("America/Chicago", -6), "miami": ("America/New_York", -5), "toronto": ("America/Toronto", -5), "mexico": ("America/Mexico_City", -6),
    "buenos aires": ("America/Argentina/Buenos_Aires", -3), "santiago": ("America/Santiago", -4), "pequim": ("Asia/Shanghai", 8), "xangai": ("Asia/Shanghai", 8),
    "seul": ("Asia/Seoul", 9), "sydney": ("Australia/Sydney", 10), "mumbai": ("Asia/Kolkata", 5.5), "india": ("Asia/Kolkata", 5.5), "cairo": ("Africa/Cairo", 2),
    "utc": ("UTC", 0), "gmt": ("UTC", 0), "brasilia": ("America/Sao_Paulo", -3), "sao paulo": ("America/Sao_Paulo", -3), "manaus": ("America/Manaus", -4),
}
_BONITO = {"toquio": "Tóquio", "mexico": "Cidade do México", "brasilia": "Brasília", "sao paulo": "São Paulo", "seul": "Seul", "xangai": "Xangai", "madri": "Madri", "moscou": "Moscou", "india": "Índia"}
FERIADOS_NOMES_EXTRA = {"natal": (12, 25), "ano novo": (1, 1), "reveillon": (12, 31), "halloween": (10, 31), "dia dos namorados": (6, 12), "dia das criancas": (10, 12), "tiradentes": (4, 21),
                        "independencia": (9, 7), "proclamacao da republica": (11, 15), "finados": (11, 2), "dia do trabalho": (5, 1), "black friday": None}


def _fmt(d: date) -> str:
    return f"{D.DIAS_NOMES[d.weekday()]}, {d:%d/%m/%Y}"


def _dias(n: int) -> str:
    return f"{abs(n)} dia" + ("" if abs(n) == 1 else "s")


def hora_em(local: str, agora_utc: datetime) -> tuple[str, str] | None:
    chave = C.norm(local)
    if chave not in FUSOS:
        return None
    tz, off = FUSOS[chave]
    try:
        from zoneinfo import ZoneInfo

        dt = agora_utc.astimezone(ZoneInfo(tz))
    except Exception:  # sem tzdata (Windows): usa deslocamento fixo, sem horário de verão
        dt = agora_utc.astimezone(timezone(timedelta(hours=off)))
    return _BONITO.get(chave, local.title()), f"{dt:%H:%M} ({D.DIAS_NOMES[dt.weekday()]}, {dt:%d/%m}) · {dt.tzname() or tz}"


def proximo_feriado(hoje: date, quantos: int = 5) -> list[tuple[date, str]]:
    out: list[tuple[date, str]] = []
    for ano in (hoje.year, hoje.year + 1):
        out += [(d, n) for d, n in sorted(D.feriados_nacionais(ano).items()) if d >= hoje]
    return out[:quantos]


def alvo_nomeado(n: str, hoje: date) -> tuple[str, date] | None:
    for nome, md in FERIADOS_NOMES_EXTRA.items():
        if md and re.search(rf"\b{re.escape(nome)}\b", n):
            d = date(hoje.year, *md)
            return nome.title(), d if d >= hoje else date(hoje.year + 1, *md)
    for ano in (hoje.year, hoje.year + 1):
        for d, nome in D.feriados_nacionais(ano).items():
            if re.search(rf"\b{re.escape(C.norm(nome))}\b", n) and d >= hoje:
                return nome, d
    if re.search(r"\bcarnaval\b", n):
        for ano in (hoje.year, hoje.year + 1):
            c = D.pascoa(ano) - timedelta(days=47)
            if c >= hoje:
                return "Carnaval", c
    if re.search(r"\bpascoa\b", n):
        for ano in (hoje.year, hoje.year + 1):
            if D.pascoa(ano) >= hoje:
                return "Páscoa", D.pascoa(ano)
    return None


def responder(texto: str, agora: datetime) -> tuple[str, dict]:
    n = C.norm(texto)
    hoje = agora.date()
    agora_utc = agora.astimezone(timezone.utc) if agora.tzinfo else agora.replace(tzinfo=timezone.utc)

    # fuso de outra cidade
    m = re.search(r"\b(?:que horas|hora|horario)\b.*\b(?:em|no|na|de)\s+([a-z ]+?)\s*[?!.]*$", n)
    if m:
        r = hora_em(m.group(1).strip(), datetime.now(timezone.utc))
        if r:
            return f"🕒 Agora em {r[0]}: {r[1]}\n   Aqui: {agora:%H:%M} ({D.DIAS_NOMES[agora.weekday()]}, {agora:%d/%m})", {"tipo": "fuso"}

    # feriados
    if re.search(r"\bproximos? feriados?\b|\bquando (?:e|eh|sera) o proximo feriado\b|\bproximo feriado\b", n):
        lista = proximo_feriado(hoje)
        linhas = ["🎌 Próximos feriados:"] + [f"  • {_fmt(d)} — {nome}  (em {_dias((d - hoje).days)})" if d != hoje else f"  • HOJE — {nome}" for d, nome in lista]
        return "\n".join(linhas), {"tipo": "feriados"}
    per = D.periodo_da_consulta(texto, agora) if re.search(r"\bferiados?\b", n) else None
    if per:
        _t, ini, fim = per
        fer = {}
        for ano in range(ini.year, fim.year + 1):
            fer.update({d: nm for d, nm in D.feriados_nacionais(ano).items() if ini <= d <= fim})
        if not fer:
            return f"🎌 Nenhum feriado nacional em {_t}.", {"tipo": "feriados"}
        return f"🎌 Feriados — {_t}:\n" + "\n".join(f"  • {_fmt(d)} — {nm}" for d, nm in sorted(fer.items())), {"tipo": "feriados"}

    # idade
    m = re.search(r"\b(?:nasce\w*|nascimento|nasci)\b", n)
    if m or re.search(r"\bquantos anos\b", n):
        it = D.interpretar(texto, agora, preferir_futuro=False)
        if it.data and it.data <= hoje:
            nasc = it.data
            anos = hoje.year - nasc.year - ((hoje.month, hoje.day) < (nasc.month, nasc.day))
            prox = date(hoje.year, nasc.month, min(nasc.day, 28)) if (nasc.month, nasc.day) != (2, 29) else date(hoje.year, 3, 1)
            if prox < hoje:
                prox = prox.replace(year=hoje.year + 1)
            return f"🎂 {anos} anos (nascimento em {_fmt(nasc)}). Próximo aniversário em {_dias((prox - hoje).days)}.", {"tipo": "idade"}

    # faltam N dias para X / até X
    m = re.search(r"\b(?:quantos dias|quanto tempo|quantas semanas|quantos meses|falta\w*|faltam)\b.*\b(?:para|pra|ate|pro)\s+(?:o\s+|a\s+)?(.+?)\s*[?!.]*$", n)
    if m:
        alvo_txt = m.group(1)
        nom = alvo_nomeado(alvo_txt, hoje)
        if nom:
            nome, d = nom
        else:
            it = D.interpretar(alvo_txt, agora)
            d, nome = (it.data, "") if it.data else (None, "")
        if d:
            dias = (d - hoje).days
            semanas, resto = divmod(dias, 7)
            quando = "é HOJE!" if dias == 0 else f"faltam {_dias(dias)}" + (f" (≈ {semanas} semana(s) e {resto} dia(s))" if dias >= 14 else "")
            if dias < 0:
                quando = f"já passou há {_dias(dias)}"
            return (f"📅 {nome} — {_fmt(d)}: {quando}." if nome else f"📅 Até {_fmt(d)}: {quando}."), {"tipo": "contagem", "dias": dias}

    # entre duas datas
    m = re.search(r"\bentre\s+(.+?)\s+e\s+(.+?)\s*[?!.]*$", n)
    if m:
        a, b = D.interpretar(m.group(1), agora, preferir_futuro=False).data, D.interpretar(m.group(2), agora, preferir_futuro=False).data
        if a and b:
            dias = (b - a).days
            return f"📅 Entre {a:%d/%m/%Y} e {b:%d/%m/%Y}: {_dias(dias)} (≈ {abs(dias) // 7} semanas e {abs(dias) % 7} dia(s)).", {"tipo": "diferenca", "dias": dias}

    # daqui a N dias / N dias atrás / N dias depois de DATA
    m = re.search(rf"\b(?:daqui a|daqui)\s+(\d+)\s*(dias?|semanas?|meses|mes)\b|\b(\d+)\s*(dias?|semanas?|meses|mes)\s+(?:atras|antes)\b|\bha\s+(\d+)\s*(dias?|semanas?|meses|mes)\b", n)
    if m:
        passado = bool(m.group(3) or m.group(5))
        qtd = int(m.group(1) or m.group(3) or m.group(5))
        unidade = m.group(2) or m.group(4) or m.group(6)
        if unidade.startswith("mes"):
            d = D._add_meses(hoje, -qtd if passado else qtd)
        else:
            d = hoje + timedelta(days=(-1 if passado else 1) * qtd * (7 if unidade.startswith("s") else 1))
        return f"📅 {'Há' if passado else 'Daqui a'} {qtd} {unidade}: {_fmt(d)}.", {"tipo": "deslocamento"}

    # semana/dia do ano
    if re.search(r"\bsemana do ano\b|\bdia do ano\b|\bque semana\b", n):
        iso = hoje.isocalendar()
        return f"📆 Hoje é o dia {hoje.timetuple().tm_yday} do ano · semana ISO {iso.week} de {iso.year}. Faltam {(date(hoje.year, 12, 31) - hoje).days} dias para acabar {hoje.year}.", {"tipo": "ano"}

    # dia da semana de uma data específica
    it = D.interpretar(texto, agora, preferir_futuro=False)
    if it.data and it.data != hoje and re.search(r"\b(?:dia da semana|que dia|foi|sera|cai|caiu)\b", n):
        verbo = "caiu/cai" if it.data != hoje else "é"
        return f"📅 {it.data:%d/%m/%Y} {'foi' if it.data < hoje else 'será'} {D.DIAS_NOMES[it.data.weekday()]}.", {"tipo": "dia_semana"}

    # padrão: agora
    quer_hora = re.search(r"\b(?:horas?|horario)\b", n)
    quer_data = re.search(r"\b(?:data|dia|hoje|mes|ano)\b", n)
    linhas = []
    if quer_hora and not quer_data:
        linhas.append(f"🕒 Agora são {agora:%H:%M}.")
    elif quer_data and not quer_hora:
        linhas.append(f"📅 Hoje é {_fmt(hoje)}.")
    else:
        linhas.append(f"📅 Hoje é {_fmt(hoje)} · 🕒 {agora:%H:%M:%S}")
    fer = D.feriados_nacionais(hoje.year).get(hoje)
    if fer:
        linhas.append(f"🎌 Hoje é feriado: {fer}.")
    return "\n".join(linhas), {"tipo": "agora"}


def main() -> None:
    bruto = C.contexto()
    texto = C.clean_command(bruto) or bruto
    try:
        corpo, meta = responder(texto, datetime.now().astimezone())
    except Exception as exc:  # noqa: BLE001
        C.fail(f"Erro ao calcular a data: {exc}", REPORT)
        return
    C.finish(REPORT, corpo, resposta="direta", interpretado=f"data/{meta.get('tipo')}: {bruto[:90]}", **meta)


if __name__ == "__main__":
    main()
