"""
Interpretação de datas e horas em português do Brasil, só com a biblioteca
padrão. Usada pelos agentes calendario, tarefas e data.

    >>> interpretar("marque dentista amanhã às 15h30 por 1 hora").data
    (amanhã), .hora = 15:30, .duracao_min = 60

As posições (spans) valem no TEXTO ORIGINAL (a normalização preserva o
comprimento), então dá para remover o que foi entendido e sobrar só o título.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

try:  # import tanto como pacote do script quanto isolado
    from _astra_common import norm_1to1
except Exception:  # pragma: no cover
    import unicodedata

    def norm_1to1(text: str) -> str:
        out = []
        for c in text:
            d = unicodedata.normalize("NFKD", c)
            base = next((x for x in d if not unicodedata.combining(x)), c)
            out.append(base.lower()[:1] or " ")
        return "".join(out)


MESES = {
    "janeiro": 1, "jan": 1, "fevereiro": 2, "fev": 2, "marco": 3, "mar": 3, "abril": 4, "abr": 4,
    "maio": 5, "mai": 5, "junho": 6, "jun": 6, "julho": 7, "jul": 7, "agosto": 8, "ago": 8,
    "setembro": 9, "set": 9, "outubro": 10, "out": 10, "novembro": 11, "nov": 11, "dezembro": 12, "dez": 12,
}
MESES_NOMES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
               "setembro", "outubro", "novembro", "dezembro"]
DIAS_NOMES = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
DIAS = {"segunda": 0, "terca": 1, "quarta": 2, "quinta": 3, "sexta": 4, "sabado": 5, "domingo": 6,
        "seg": 0, "ter": 1, "qua": 2, "qui": 3, "sex": 4, "sab": 5, "dom": 6}
BYDAY = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]

_MES_RE = "|".join(sorted(MESES, key=len, reverse=True))
_DIA_RE = "segunda|terca|quarta|quinta|sexta|sabado|domingo"
_NUM_PALAVRA = {"um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6,
                "sete": 7, "oito": 8, "nove": 9, "dez": 10, "onze": 11, "doze": 12, "quinze": 15,
                "vinte": 20, "trinta": 30, "meia": 0.5, "meio": 0.5}


@dataclass
class Interpretado:
    data: date | None = None
    hora: time | None = None
    fim: time | None = None
    duracao_min: int | None = None
    recorrencia: dict | None = None
    dia_inteiro: bool = False
    erro: str | None = None
    spans: list[tuple[int, int]] = field(default_factory=list)
    periodo: str | None = None  # "semana", "mes", "hoje"... quando o pedido é sobre um intervalo


def _fim_do_mes(ano: int, mes: int) -> int:
    return calendar.monthrange(ano, mes)[1]


def _proxima_ocorrencia_dia(base: date, dia: int) -> date | None:
    """Próximo dia N do mês a partir de `base` (hoje conta)."""
    ano, mes = base.year, base.month
    for _ in range(14):
        if dia <= _fim_do_mes(ano, mes):
            cand = date(ano, mes, dia)
            if cand >= base:
                return cand
        mes += 1
        if mes > 12:
            mes, ano = 1, ano + 1
    return None


def _add_meses(d: date, n: int) -> date:
    m = d.month - 1 + n
    ano, mes = d.year + m // 12, m % 12 + 1
    return date(ano, mes, min(d.day, _fim_do_mes(ano, mes)))


def _num(token: str) -> float | None:
    token = token.strip()
    if token.isdigit():
        return float(token)
    return _NUM_PALAVRA.get(token)


def interpretar(texto: str, base: datetime | None = None, *, preferir_futuro: bool = True) -> Interpretado:
    base = base or datetime.now()
    hoje = base.date()
    n = norm_1to1(texto)
    r = Interpretado()
    usados: list[tuple[int, int]] = []

    def livre(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in usados)

    def achar(padrao: str, flags: int = 0):
        for m in re.finditer(padrao, n, flags):
            if livre(m.start(), m.end()):
                return m
        return None

    def usar(m, *, grupo: int = 0) -> None:
        usados.append((m.start(grupo), m.end(grupo)))

    # ------------------------------------------------------- recorrência
    m = achar(r"\b(?:toda|todas|todo|todos)\s+(?:as?\s+|os?\s+)?(" + _DIA_RE + r")s?(?:[- ]feiras?)?\b")
    if m:
        r.recorrencia = {"freq": "WEEKLY", "interval": 1, "byday": [BYDAY[DIAS[m.group(1)]]]}
        r.data = r.data
        r._dia_recorrencia = DIAS[m.group(1)]  # type: ignore[attr-defined]
        usar(m)
    if r.recorrencia is None:
        m = achar(r"\b(?:toda|todas)\s+(?:as\s+)?semanas?\b|\bsemanalmente\b")
        if m:
            r.recorrencia = {"freq": "WEEKLY", "interval": 1}
            usar(m)
    if r.recorrencia is None:
        m = achar(r"\b(?:todo|todos)\s+(?:os\s+)?dias\b|\b(?:todo\s+dia)\b(?!\s+\d)|\bdiariamente\b")
        if m:
            r.recorrencia = {"freq": "DAILY", "interval": 1}
            usar(m)
    if r.recorrencia is None:
        m = achar(r"\b(?:todo|todos)\s+(?:os\s+)?(?:mes|meses)\b(?:\s+(?:no\s+)?dia\s+(\d{1,2}))?|\bmensalmente\b|\btodo\s+dia\s+(\d{1,2})\b")
        if m:
            dia = next((g for g in m.groups() if g), None)
            r.recorrencia = {"freq": "MONTHLY", "interval": 1}
            if dia:
                r.recorrencia["bymonthday"] = int(dia)
                if 1 <= int(dia) <= 31:
                    r.data = _proxima_ocorrencia_dia(hoje, int(dia))
            usar(m)
    if r.recorrencia is None:
        m = achar(r"\b(?:todo|todos)\s+(?:os\s+)?anos?\b|\banualmente\b")
        if m:
            r.recorrencia = {"freq": "YEARLY", "interval": 1}
            usar(m)
    if r.recorrencia is None:
        m = achar(r"\ba\s+cada\s+(\d+|" + "|".join(_NUM_PALAVRA) + r")\s+(dias?|semanas?|meses|mes|anos?)\b")
        if m:
            qtd = int(_num(m.group(1)) or 1)
            freq = {"d": "DAILY", "s": "WEEKLY", "m": "MONTHLY", "a": "YEARLY"}[m.group(2)[0]]
            r.recorrencia = {"freq": freq, "interval": max(1, qtd)}
            usar(m)

    # ------------------------------------------------------- duração
    m = achar(r"\b(?:por|durante)\s+(?:(\d+|uma|um)\s+)?horas?\s+e\s+meia\b")
    if m:
        q = m.group(1)
        r.duracao_min = int(60 * (int(q) if q and q.isdigit() else 1) + 30)
        usar(m)
    m = None if r.duracao_min else achar(r"\b(?:por|durante|duracao\s+de|com\s+duracao\s+de)\s+(\d+(?:[.,]\d+)?|" + "|".join(_NUM_PALAVRA) + r")\s*(h|hr|hrs|horas?|min|mins|minutos?)\b")
    if m:
        valor = m.group(1).replace(",", ".")
        qtd = float(valor) if re.match(r"^\d", valor) else float(_NUM_PALAVRA.get(valor, 1))
        unidade = m.group(2)
        r.duracao_min = int(round(qtd * (60 if unidade.startswith("h") else 1)))
        usar(m)
    if r.duracao_min is None:
        m = achar(r"\b(?:por\s+|durante\s+)?(?:meia\s+hora)\b")
        if m:
            r.duracao_min = 30
            usar(m)

    # ------------------------------------------------------- horário
    def hora_valida(h: int, mi: int) -> bool:
        return 0 <= h <= 23 and 0 <= mi <= 59

    m = achar(r"\b(?:das?|de)\s+(\d{1,2})(?:\s*(?:h|:)\s*(\d{2})?)?\s*(?:h|horas?)?\s+(?:as?|ate\s+as?|ate|a)\s+(\d{1,2})(?:\s*(?:h|:)\s*(\d{2})?)?\s*(?:h|horas?)?\b")
    if m and hora_valida(int(m.group(1)), int(m.group(2) or 0)) and hora_valida(int(m.group(3)), int(m.group(4) or 0)):
        r.hora = time(int(m.group(1)), int(m.group(2) or 0))
        r.fim = time(int(m.group(3)), int(m.group(4) or 0))
        usar(m)

    if r.hora is None:
        m = achar(r"\bmeio[- ]?dia\b")
        if m:
            r.hora = time(12, 0)
            usar(m)
    if r.hora is None:
        m = achar(r"\bmeia[- ]?noite\b")
        if m:
            r.hora = time(0, 0)
            usar(m)
    if r.hora is None:
        m = achar(r"\b(\d{1,2})\s*(?:h|horas?|:)\s*(\d{2})?\b(?!\s*(?:/|de\s+(?:" + _MES_RE + r")))")
        if m and hora_valida(int(m.group(1)), int(m.group(2) or 0)):
            h, mi = int(m.group(1)), int(m.group(2) or 0)
            ap = re.match(r"\s*(?:da|de|a|as)?\s*(manha|tarde|noite|madrugada)\b", n[m.end():])
            if ap:
                if ap.group(1) in ("tarde", "noite") and h < 12:
                    h += 12
                usados.append((m.end(), m.end() + ap.end()))
            r.hora = time(h, mi)
            usar(m)
    if r.hora is None:
        m = achar(r"\b(?:as|a)\s+(\d{1,2})(?:\s*(?:h|horas?))?\b(?!\s*(?:/|de\s+(?:" + _MES_RE + r")|dias?))(?:\s+(?:da|de)\s+(manha|tarde|noite|madrugada))?")
        if m and hora_valida(int(m.group(1)), 0):
            h = int(m.group(1))
            if m.group(2) in ("tarde", "noite") and h < 12:
                h += 12
            r.hora = time(h, 0)
            usar(m)
    if r.hora is None:
        m = achar(r"\b(\d{1,2})\s+(?:da|de)\s+(manha|tarde|noite|madrugada)\b")
        if m and hora_valida(int(m.group(1)), 0):
            h = int(m.group(1))
            if m.group(2) in ("tarde", "noite") and h < 12:
                h += 12
            r.hora = time(h, 0)
            usar(m)
    if r.hora is None:
        m = achar(r"\b(?:de\s+manha|pela\s+manha|a\s+tarde|de\s+tarde|pela\s+tarde|a\s+noite|de\s+noite|pela\s+noite)\b")
        if m:
            txt = m.group(0)
            r.hora = time(9, 0) if "manha" in txt else time(14, 0) if "tarde" in txt else time(19, 0)
            r.periodo = "turno"
            usar(m)

    # ------------------------------------------------------- data
    m = achar(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
    if m:
        try:
            r.data = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            r.erro = "data inválida"
        usar(m)
    if r.data is None and not r.erro:
        m = achar(r"\b(\d{1,2})[/.\-](\d{1,2})(?:[/.\-](\d{2,4}))?\b")
        if m:
            d_, mo = int(m.group(1)), int(m.group(2))
            ano = int(m.group(3)) if m.group(3) else hoje.year
            if ano < 100:
                ano += 2000
            try:
                r.data = date(ano, mo, d_)
                if not m.group(3) and preferir_futuro and r.data < hoje:
                    r.data = date(ano + 1, mo, d_)
            except ValueError:
                r.erro = f"data inválida ({m.group(0)})"
            usar(m)
    if r.data is None and not r.erro:
        m = achar(r"\b(?:dia\s+)?(\d{1,2})\s*(?:de\s+)?(" + _MES_RE + r")\b(?:\s*(?:de\s+)?(\d{4}))?")
        if m:
            d_, mo = int(m.group(1)), MESES[m.group(2)]
            ano = int(m.group(3)) if m.group(3) else hoje.year
            try:
                r.data = date(ano, mo, d_)
                if not m.group(3) and preferir_futuro and r.data < hoje:
                    r.data = date(ano + 1, mo, d_)
            except ValueError:
                r.erro = f"data inválida ({d_}/{mo})"
            usar(m)
    if r.data is None and not r.erro:
        m = achar(r"\bdepois\s+de\s+amanha\b")
        if m:
            r.data = hoje + timedelta(days=2)
            usar(m)
    if r.data is None and not r.erro:
        m = achar(r"\bamanha\b")
        if m:
            r.data = hoje + timedelta(days=1)
            usar(m)
    if r.data is None and not r.erro:
        m = achar(r"\bontem\b")
        if m:
            r.data = hoje - timedelta(days=1)
            usar(m)
    if r.data is None and not r.erro:
        m = achar(r"\bhoje\b")
        if m:
            r.data = hoje
            usar(m)
    if r.data is None and not r.erro:
        m = achar(r"\b(?:daqui\s+a|dentro\s+de|em)\s+(\d+|" + "|".join(k for k in _NUM_PALAVRA if k not in ("meia", "meio")) + r")\s+(dias?|semanas?|meses|mes)\b")
        if m:
            qtd = int(_num(m.group(1)) or 1)
            un = m.group(2)
            if un.startswith("d"):
                r.data = hoje + timedelta(days=qtd)
            elif un.startswith("s"):
                r.data = hoje + timedelta(weeks=qtd)
            else:
                r.data = _add_meses(hoje, qtd)
            usar(m)
    if r.data is None and not r.erro:
        m = achar(r"\b(?:(proxima|proximo|esta|este|nesta|neste|na\s+proxima|no\s+proximo)\s+)?(" + _DIA_RE + r")(?:[- ]feira)?(\s+que\s+vem|\s+proxima)?\b")
        if m:
            alvo = DIAS[m.group(2)]
            delta = (alvo - hoje.weekday()) % 7
            if delta == 0 and (m.group(1) and "prox" in m.group(1) or m.group(3)):
                delta = 7
            elif m.group(3) or (m.group(1) and "prox" in m.group(1) and delta == 0):
                delta = delta or 7
            r.data = hoje + timedelta(days=delta)
            usar(m)
    if r.data is None and not r.erro:
        m = achar(r"\b(?:dia\s+)(\d{1,2})\b(?!\s*[/:h])")
        if m and 1 <= int(m.group(1)) <= 31:
            cand = _proxima_ocorrencia_dia(hoje, int(m.group(1)))
            if cand:
                r.data = cand
                usar(m)
    if r.data is None and not r.erro:
        m = achar(r"\b(?:fim|final)\s+de\s+semana\b|\bfds\b")
        if m:
            r.data = hoje + timedelta(days=(5 - hoje.weekday()) % 7)
            usar(m)
    if r.data is None and not r.erro:
        m = achar(r"\b(?:semana\s+que\s+vem|proxima\s+semana)\b")
        if m:
            r.data = hoje + timedelta(days=7 - hoje.weekday())
            r.periodo = "semana"
            usar(m)

    if r.recorrencia and r.recorrencia.get("freq") == "WEEKLY" and r.data is None and hasattr(r, "_dia_recorrencia"):
        alvo = r._dia_recorrencia  # type: ignore[attr-defined]
        r.data = hoje + timedelta(days=(alvo - hoje.weekday()) % 7)

    r.spans = sorted(usados)
    return r


def remover_spans(texto: str, spans: list[tuple[int, int]]) -> str:
    out, last = [], 0
    for a, b in sorted(spans):
        out.append(texto[last:a])
        out.append(" ")
        last = b
    out.append(texto[last:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


def periodo_da_consulta(texto: str, base: datetime | None = None) -> tuple[str, date, date] | None:
    """Para listagens: "hoje", "amanhã", "esta semana", "outubro", "mês que vem",
    "próximos 7 dias" → (rótulo, início, fim) inclusivos."""
    base = base or datetime.now()
    hoje = base.date()
    n = norm_1to1(texto)
    if re.search(r"\bhoje\b", n):
        return "hoje", hoje, hoje
    if re.search(r"\bamanha\b", n) and not re.search(r"depois\s+de\s+amanha", n):
        d = hoje + timedelta(days=1)
        return "amanhã", d, d
    if re.search(r"\bdepois\s+de\s+amanha\b", n):
        d = hoje + timedelta(days=2)
        return "depois de amanhã", d, d
    m = re.search(r"\bproximos?\s+(\d{1,3})\s+dias\b", n)
    if m:
        return f"próximos {m.group(1)} dias", hoje, hoje + timedelta(days=int(m.group(1)))
    if re.search(r"\b(?:semana\s+que\s+vem|proxima\s+semana)\b", n):
        ini = hoje + timedelta(days=7 - hoje.weekday())
        return "semana que vem", ini, ini + timedelta(days=6)
    if re.search(r"\b(?:esta|essa|nesta|nessa)\s+semana\b|\bda\s+semana\b|\bsemana\b", n):
        ini = hoje - timedelta(days=hoje.weekday())
        return "esta semana", ini, ini + timedelta(days=6)
    if re.search(r"\b(?:mes\s+que\s+vem|proximo\s+mes)\b", n):
        ini = _add_meses(hoje.replace(day=1), 1)
        return f"{MESES_NOMES[ini.month - 1]} de {ini.year}", ini, ini.replace(day=_fim_do_mes(ini.year, ini.month))
    m = re.search(r"\b(" + _MES_RE + r")\b(?:\s*(?:de\s+)?(\d{4}))?", n)
    if m and not re.search(r"\b\d{1,2}\s*(?:de\s+)?" + m.group(1), n):
        mes = MESES[m.group(1)]
        ano = int(m.group(2)) if m.group(2) else hoje.year
        return f"{MESES_NOMES[mes - 1]} de {ano}", date(ano, mes, 1), date(ano, mes, _fim_do_mes(ano, mes))
    if re.search(r"\b(?:este|esse|neste|nesse)\s+mes\b|\bdo\s+mes\b|\bmes\b", n):
        return f"{MESES_NOMES[hoje.month - 1]} de {hoje.year}", hoje.replace(day=1), hoje.replace(day=_fim_do_mes(hoje.year, hoje.month))
    return None


def descrever_data(d: date, base: date | None = None) -> str:
    base = base or date.today()
    delta = (d - base).days
    rel = {0: "hoje", 1: "amanhã", 2: "depois de amanhã", -1: "ontem"}.get(delta)
    txt = f"{DIAS_NOMES[d.weekday()]}, {d.day:02d}/{d.month:02d}/{d.year}"
    return f"{rel} ({txt})" if rel else txt


# ---------------------------------------------------------------- feriados (Brasil)

def pascoa(ano: int) -> date:
    """Algoritmo de Meeus/Jones/Butcher (calendário gregoriano)."""
    a, b, c = ano % 19, ano // 100, ano % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = (h + l - 7 * m + 114) % 31 + 1
    return date(ano, mes, dia)


def feriados_nacionais(ano: int) -> dict[date, str]:
    p = pascoa(ano)
    fixos = {
        (1, 1): "Confraternização Universal", (4, 21): "Tiradentes", (5, 1): "Dia do Trabalho",
        (9, 7): "Independência do Brasil", (10, 12): "Nossa Senhora Aparecida", (11, 2): "Finados",
        (11, 15): "Proclamação da República", (12, 25): "Natal",
    }
    out = {date(ano, m, d): nome for (m, d), nome in fixos.items()}
    if ano >= 2024:
        out[date(ano, 11, 20)] = "Consciência Negra"
    out[p - timedelta(days=47)] = "Carnaval (terça-feira)"
    out[p - timedelta(days=48)] = "Carnaval (segunda-feira)"
    out[p - timedelta(days=2)] = "Sexta-feira Santa"
    out[p] = "Páscoa"
    out[p + timedelta(days=60)] = "Corpus Christi"
    return dict(sorted(out.items()))
