"""Agente calendario: agenda em arquivo iCal (.ics), em português natural.

Entende: marcar/agendar, listar (hoje, amanhã, semana, mês, próximos N dias),
próximo compromisso, buscar ("quando é a prova?"), remarcar/adiar, apagar,
eventos recorrentes (toda segunda às 9h, todo dia 5...), duração e conflitos.
O arquivo agenda/meu_calendario.ics abre direto no Google Agenda/Outlook.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import uuid
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402
import _astra_datas as D  # noqa: E402

ICS_PATH = Path("agenda/meu_calendario.ics")
REPORT = "agenda/relatorio.txt"
CABECALHO = "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//Astra Assistant//Calendar Manager//PT\r\nCALSCALE:GREGORIAN\r\n"
RODAPE = "END:VCALENDAR\r\n"
DURACAO_PADRAO = 60


# ------------------------------------------------------------------ iCal

def _unfold(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _fold(line: str) -> str:
    """Quebra linhas longas em 75 bytes (RFC 5545) sem cortar caractere UTF-8."""
    out, cur, size = [], "", 0
    for ch in line:
        b = len(ch.encode("utf-8"))
        if size + b > 75:
            out.append(cur)
            cur, size = " " + ch, 1 + b
        else:
            cur += ch
            size += b
    out.append(cur)
    return "\r\n".join(out)


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\r", "").replace("\n", "\\n")


def _unesc(text: str) -> str:
    return text.replace("\\n", "\n").replace("\\N", "\n").replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")


def _parse_dt(value: str, params: str) -> tuple[datetime | None, bool]:
    value = value.strip()
    try:
        if "VALUE=DATE" in params.upper() or re.fullmatch(r"\d{8}", value):
            return datetime.strptime(value[:8], "%Y%m%d"), True
        return datetime.strptime(value.rstrip("Z")[:15], "%Y%m%dT%H%M%S"), False
    except ValueError:
        return None, False


def _parse_rrule(value: str) -> dict:
    parts = dict(p.split("=", 1) for p in value.split(";") if "=" in p)
    rule = {"freq": parts.get("FREQ", "").upper(), "interval": int(parts.get("INTERVAL", "1") or 1)}
    if "COUNT" in parts:
        rule["count"] = int(parts["COUNT"])
    if "UNTIL" in parts:
        dt, _ = _parse_dt(parts["UNTIL"], "")
        rule["until"] = dt.date() if dt else None
    if "BYDAY" in parts:
        rule["byday"] = [re.sub(r"^[+-]?\d+", "", d) for d in parts["BYDAY"].split(",")]
    if "BYMONTHDAY" in parts:
        try:
            rule["bymonthday"] = int(parts["BYMONTHDAY"].split(",")[0])
        except ValueError:
            pass
    return rule


def carregar() -> tuple[list[dict], str | None]:
    """Lê o .ics. Devolve (eventos, aviso). Blocos ilegíveis são preservados."""
    if not ICS_PATH.exists():
        return [], None
    texto = ICS_PATH.read_text(encoding="utf-8", errors="replace")
    eventos: list[dict] = []
    ignorados = 0
    for bloco in re.findall(r"BEGIN:VEVENT.*?END:VEVENT", "\n".join(_unfold(texto)), re.S):
        ev: dict = {"raw": bloco, "uid": "", "summary": "", "location": "", "rrule": None, "valarm": "BEGIN:VALARM" in bloco}
        dentro_alarme = False
        for linha in bloco.split("\n"):
            if linha.startswith("BEGIN:VALARM"):
                dentro_alarme = True
            if linha.startswith("END:VALARM"):
                dentro_alarme = False
                continue
            if dentro_alarme or ":" not in linha:
                continue
            chave, valor = linha.split(":", 1)
            nome, _, params = chave.partition(";")
            nome = nome.upper()
            if nome == "UID":
                ev["uid"] = valor.strip()
            elif nome == "SUMMARY":
                ev["summary"] = _unesc(valor).strip()
            elif nome == "LOCATION":
                ev["location"] = _unesc(valor).strip()
            elif nome == "DTSTART":
                ev["start"], ev["all_day"] = _parse_dt(valor, params)
            elif nome == "DTEND":
                ev["end"], _ = _parse_dt(valor, params)
            elif nome == "DURATION":
                m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", valor.strip())
                if m:
                    ev["duration"] = timedelta(days=int(m.group(1) or 0), hours=int(m.group(2) or 0), minutes=int(m.group(3) or 0))
            elif nome == "RRULE":
                try:
                    ev["rrule"] = _parse_rrule(valor)
                except Exception:
                    ev["rrule"] = None
        if not ev.get("start") or not ev["summary"]:
            ignorados += 1
            ev["invalido"] = True
        if not ev["uid"]:
            ev["uid"] = "legacy-" + hashlib.sha1(bloco.encode("utf-8")).hexdigest()[:12]
        ev["id"] = hashlib.sha1(ev["uid"].encode("utf-8")).hexdigest()[:6]
        eventos.append(ev)
    aviso = f"{ignorados} bloco(s) do arquivo não puderam ser lidos e foram preservados." if ignorados else None
    return eventos, aviso


def duracao(ev: dict) -> timedelta:
    if ev.get("all_day"):
        return timedelta(days=1)
    if ev.get("end") and ev["end"] > ev["start"]:
        return ev["end"] - ev["start"]
    return ev.get("duration") or timedelta(minutes=DURACAO_PADRAO)


def bloco_novo(titulo: str, inicio: datetime, *, dia_inteiro: bool, dur_min: int | None, rec: dict | None, local: str = "") -> tuple[str, str]:
    uid = f"{uuid.uuid4()}@astra.local"
    linhas = ["BEGIN:VEVENT", f"UID:{uid}", "DTSTAMP:" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")]
    if dia_inteiro:
        linhas += [f"DTSTART;VALUE=DATE:{inicio:%Y%m%d}", f"DTEND;VALUE=DATE:{inicio + timedelta(days=1):%Y%m%d}"]
    else:
        fim = inicio + timedelta(minutes=dur_min or DURACAO_PADRAO)
        linhas += [f"DTSTART:{inicio:%Y%m%dT%H%M%S}", f"DTEND:{fim:%Y%m%dT%H%M%S}"]
    linhas.append(f"SUMMARY:{_esc(titulo)}")
    if local:
        linhas.append(f"LOCATION:{_esc(local)}")
    if rec:
        regra = f"FREQ={rec['freq']}"
        if rec.get("interval", 1) > 1:
            regra += f";INTERVAL={rec['interval']}"
        if rec.get("byday"):
            regra += ";BYDAY=" + ",".join(rec["byday"])
        if rec.get("bymonthday"):
            regra += f";BYMONTHDAY={rec['bymonthday']}"
        linhas.append("RRULE:" + regra)
    if not dia_inteiro:
        linhas += ["BEGIN:VALARM", "ACTION:DISPLAY", "DESCRIPTION:Lembrete", "TRIGGER:-PT15M", "END:VALARM"]
    linhas.append("END:VEVENT")
    return "\r\n".join(_fold(l) for l in linhas), uid


def salvar(eventos: list[dict]) -> None:
    """Regrava o .ics de forma atômica, com cópia de segurança (.bak)."""
    ICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ICS_PATH.exists():
        try:
            ICS_PATH.with_suffix(".ics.bak").write_bytes(ICS_PATH.read_bytes())
        except Exception:
            pass
    corpo = "".join(ev["raw"].replace("\r\n", "\n").replace("\n", "\r\n") + "\r\n" for ev in eventos)
    tmp = ICS_PATH.with_suffix(".ics.tmp")
    tmp.write_text(CABECALHO + corpo + RODAPE, encoding="utf-8", newline="")
    os.replace(tmp, ICS_PATH)


def reescrever_inicio(ev: dict, novo_inicio: datetime, nova_dur: timedelta | None = None) -> str:
    """Troca DTSTART/DTEND de um bloco existente, preservando o resto."""
    dur = nova_dur or duracao(ev)
    out = []
    for linha in ev["raw"].replace("\r\n", "\n").split("\n"):
        up = linha.upper()
        if up.startswith("DTSTART"):
            out.append(f"DTSTART;VALUE=DATE:{novo_inicio:%Y%m%d}" if ev.get("all_day") else f"DTSTART:{novo_inicio:%Y%m%dT%H%M%S}")
        elif up.startswith("DTEND"):
            out.append(f"DTEND;VALUE=DATE:{novo_inicio + timedelta(days=1):%Y%m%d}" if ev.get("all_day") else f"DTEND:{novo_inicio + dur:%Y%m%dT%H%M%S}")
        else:
            out.append(linha)
    if not any(l.upper().startswith("DTEND") for l in out) and not ev.get("all_day"):
        idx = next(i for i, l in enumerate(out) if l.upper().startswith("DTSTART"))
        out.insert(idx + 1, f"DTEND:{novo_inicio + dur:%Y%m%dT%H%M%S}")
    return "\n".join(out)


# ------------------------------------------------------------------ recorrência

def ocorrencias(ev: dict, ini: date, fim: date) -> list[datetime]:
    """Datas em que o evento acontece dentro de [ini, fim]."""
    start: datetime = ev["start"]
    regra = ev.get("rrule")
    if not regra or not regra.get("freq"):
        return [start] if ini <= start.date() <= fim else []
    freq, passo = regra["freq"], max(1, regra.get("interval", 1))
    limite = regra.get("until") or fim
    limite = min(limite, fim)
    out: list[datetime] = []
    contador = 0
    base = start.date()

    def juntar(d: date) -> datetime:
        return datetime.combine(d, start.time())

    candidatos: list[date] = []
    if freq == "DAILY":
        d = base
        while d <= limite and len(candidatos) < 4000:
            candidatos.append(d)
            d += timedelta(days=passo)
    elif freq == "WEEKLY":
        dias = [D.BYDAY.index(x) for x in regra.get("byday", []) if x in D.BYDAY] or [base.weekday()]
        semana0 = base - timedelta(days=base.weekday())
        s = semana0
        while s <= limite and len(candidatos) < 4000:
            for wd in sorted(dias):
                d = s + timedelta(days=wd)
                if d >= base:
                    candidatos.append(d)
            s += timedelta(weeks=passo)
    elif freq == "MONTHLY":
        dia = regra.get("bymonthday") or base.day
        ano, mes = base.year, base.month
        while date(ano, mes, 1) <= limite and len(candidatos) < 1200:
            if dia <= D._fim_do_mes(ano, mes):
                d = date(ano, mes, dia)
                if d >= base:
                    candidatos.append(d)
            mes += passo
            while mes > 12:
                mes -= 12
                ano += 1
    elif freq == "YEARLY":
        ano = base.year
        while ano <= limite.year and len(candidatos) < 200:
            try:
                candidatos.append(date(ano, base.month, base.day))
            except ValueError:
                pass
            ano += passo
    for d in candidatos:
        contador += 1
        if regra.get("count") and contador > regra["count"]:
            break
        if d > limite:
            break
        if d >= ini:
            out.append(juntar(d))
    return out


def expandir(eventos: list[dict], ini: date, fim: date) -> list[tuple[datetime, dict]]:
    itens = []
    for ev in eventos:
        if ev.get("invalido"):
            continue
        for dt in ocorrencias(ev, ini, fim):
            itens.append((dt, ev))
    itens.sort(key=lambda x: (x[0], x[1]["summary"].casefold()))
    return itens


# ------------------------------------------------------------------ apresentação

def hora_txt(dt: datetime, ev: dict) -> str:
    if ev.get("all_day"):
        return "dia todo"
    fim = dt + duracao(ev)
    return f"{dt:%H:%M}–{fim:%H:%M}" if fim.date() == dt.date() else f"{dt:%H:%M}"


def linha_evento(dt: datetime, ev: dict) -> str:
    extra = " 🔁" if ev.get("rrule") else ""
    local = f" 📍 {ev['location']}" if ev.get("location") else ""
    return f"  • {hora_txt(dt, ev)}  {ev['summary']}{extra}{local}   #{ev['id']}"


def formatar_lista(titulo: str, itens: list[tuple[datetime, dict]], ini: date, fim: date) -> str:
    feriados = {}
    if (fim - ini).days <= 62:
        for ano in range(ini.year, fim.year + 1):
            feriados.update({d: n for d, n in D.feriados_nacionais(ano).items() if ini <= d <= fim})
    por_dia: dict[date, list[str]] = {}
    for dt, ev in itens:
        por_dia.setdefault(dt.date(), []).append(linha_evento(dt, ev))
    for d, nome in feriados.items():
        por_dia.setdefault(d, []).insert(0, f"  🎌 Feriado: {nome}")
    n = len(itens)
    linhas = [f"📅 Agenda — {titulo}: " + (f"{n} compromisso(s)" if n else "nenhum compromisso")]
    for d in sorted(por_dia):
        linhas.append(f"\n{D.descrever_data(d)}")
        linhas.extend(por_dia[d])
    if not n and not feriados:
        linhas.append("\nTudo livre por aqui. Para marcar: “marque dentista amanhã às 15h”.")
    return "\n".join(linhas)


# ------------------------------------------------------------------ intenções

_VERBO_APAGAR = r"\b(?:apag\w*|remov\w*|cancel\w*|exclu\w*|delet\w*|tire|tira)\b"
_VERBO_REMARCAR = r"\b(?:remarc\w*|reagend\w*|adi(?:e|ar|a)|mov\w*|transfir\w*|mud(?:e|ar)|alter\w*)\b"
_VERBO_ADICIONAR = r"\b(?:marc(?:a|ar|e)|agend(?:e|ar|ei)|adicion\w*|cri(?:e|ar|a)|anot\w*|coloc\w*|inclu\w*|registr\w*|salv\w*|lembr(?:e|ar)(?:-?me)?|bota|po[ee])\b"
_VERBO_BUSCAR = r"\b(?:busc\w*|procur\w*|ach(?:e|ar|a)|pesquis\w*|encontr\w*)\b"
_CONSULTA = r"\b(?:quando|que\s+dia|qual|quais|o\s+que|quantos?|liste|listar|mostr\w*|ver|veja|tenho|tem)\b"
_LIXO_TITULO = {
    "marque", "marcar", "marca", "agende", "agendar", "adicione", "adicionar", "crie", "criar", "anote", "anotar",
    "coloque", "colocar", "inclua", "incluir", "registre", "registrar", "salve", "lembre", "lembrar", "me", "mim",
    "um", "uma", "o", "a", "os", "as", "compromisso", "evento", "lembrete", "tarefa", "novo", "nova", "na", "no",
    "agenda", "calendario", "minha", "meu", "de", "do", "da", "para", "pra", "pro", "em", "dia", "as", "a", "e",
    "tenho", "preciso", "quero", "por", "favor", "usando", "use", "agente", "calendario", "que", "sobre", "com",
}
_PREP_FINAL = {"as", "a", "de", "do", "da", "no", "na", "em", "para", "pra", "e", "ate", "dia", "das", "dos", "por"}


def limpar_titulo(texto: str, spans: list[tuple[int, int]], extra: set[str] | None = None) -> str:
    resto = D.remover_spans(texto, spans)
    tokens = resto.split()
    stop = _LIXO_TITULO | (extra or set())

    def n(t: str) -> str:
        return C.norm(t.strip(".,:;!?-–—"))

    def verbo(t: str) -> bool:
        return any(re.fullmatch(v[2:-2], n(t)) for v in (_VERBO_APAGAR, _VERBO_REMARCAR, _VERBO_BUSCAR)) if t else False

    while tokens and (n(tokens[0]) in stop or verbo(tokens[0])):
        tokens.pop(0)
    while tokens and n(tokens[-1]) in _PREP_FINAL | stop:
        tokens.pop()
    titulo = " ".join(tokens).strip(" .,:;!?-–—\"'")
    return titulo[:1].upper() + titulo[1:] if titulo else ""


def achar_alvos(eventos: list[dict], termo: str, *, dia: date | None = None) -> list[dict]:
    """Eventos cujo título contém TODAS as palavras do termo (sem acento/caixa)."""
    if not termo and dia is None:
        return []
    m = re.search(r"#([0-9a-f]{6})\b", termo.casefold())
    if m:
        return [e for e in eventos if e["id"] == m.group(1)]
    palavras = [p for p in C.norm(termo).split() if p not in _PREP_FINAL and len(p) > 1]
    out = []
    for ev in eventos:
        if ev.get("invalido"):
            continue
        titulo = C.norm(ev["summary"])
        if palavras and not all(p in titulo or any(_parecido(p, t) for t in titulo.split()) for p in palavras):
            continue
        if dia is not None and not ocorrencias(ev, dia, dia):
            continue
        out.append(ev)
    return out


def _parecido(a: str, b: str) -> bool:
    import difflib

    return len(a) > 3 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.86


# ------------------------------------------------------------------ ações

def conflitos(eventos: list[dict], inicio: datetime, fim: datetime, ignorar_uid: str = "") -> list[tuple[datetime, dict]]:
    achados = []
    for dt, ev in expandir(eventos, inicio.date(), inicio.date()):
        if ev["uid"] == ignorar_uid or ev.get("all_day"):
            continue
        if dt < fim and dt + duracao(ev) > inicio:
            achados.append((dt, ev))
    return achados


def acao_adicionar(texto: str, eventos: list[dict], agora: datetime) -> tuple[str, dict]:
    it = D.interpretar(texto, agora)
    if it.erro:
        return f"⚠️ Não consegui marcar: {it.erro}. Tente algo como “23 de outubro às 14h”.", {"acao": "adicionar", "ok": False}
    if it.data is None:
        return ("⚠️ Não entendi a data. Diga o dia, por exemplo: “amanhã”, “sexta às 15h”, “23/10” ou “dia 23 de outubro às 14h”.", {"acao": "adicionar", "ok": False})
    titulo = limpar_titulo(texto, it.spans)
    if not titulo:
        return "⚠️ Qual é o nome do compromisso? Ex.: “marque dentista amanhã às 15h”.", {"acao": "adicionar", "ok": False}
    dia_inteiro = it.hora is None and it.periodo is None
    inicio = datetime.combine(it.data, it.hora or time(0, 0))
    dur = it.duracao_min
    if it.fim and it.hora:
        dur = max(15, int((datetime.combine(it.data, it.fim) - inicio).total_seconds() // 60))
    bloco, uid = bloco_novo(titulo, inicio, dia_inteiro=dia_inteiro, dur_min=dur, rec=it.recorrencia)
    ev_novo = {"raw": bloco, "uid": uid}
    eventos.append(ev_novo)
    salvar(eventos)

    quando = D.descrever_data(it.data, agora.date())
    hora = "dia todo" if dia_inteiro else f"{inicio:%H:%M}" + (f"–{(inicio + timedelta(minutes=dur or DURACAO_PADRAO)):%H:%M}" if not dia_inteiro else "")
    linhas = ["✅ Compromisso agendado!", f"📌 {titulo}", f"📅 {quando} · ⏰ {hora}"]
    if it.recorrencia:
        nomes = {"DAILY": "todos os dias", "WEEKLY": "toda semana", "MONTHLY": "todo mês", "YEARLY": "todo ano"}
        linhas.append(f"🔁 Repete {nomes.get(it.recorrencia['freq'], '')}")
    if inicio < agora and not it.recorrencia:
        linhas.append("⚠️ Essa data/hora já passou — confirme se era isso mesmo.")
    if not dia_inteiro:
        fim = inicio + timedelta(minutes=dur or DURACAO_PADRAO)
        lista = [(dt, e) for dt, e in conflitos(eventos[:-1], inicio, fim)]
        for dt, e in lista[:3]:
            linhas.append(f"⚠️ Conflito com “{e['summary']}” ({hora_txt(dt, e)}).")
    return "\n".join(linhas), {"acao": "adicionar", "ok": True, "titulo": titulo, "quando": inicio.isoformat()}


def acao_apagar(texto: str, eventos: list[dict], agora: datetime) -> tuple[str, dict]:
    it = D.interpretar(texto, agora)
    termo = limpar_titulo(texto, it.spans, {"todos", "todas", "tudo", "compromissos", "eventos"})
    todos = bool(re.search(r"\b(?:todos|todas)\b", C.norm(texto)))
    alvos = achar_alvos(eventos, termo, dia=it.data if (termo or it.data) else None)
    if not termo and it.data is None:
        return "⚠️ Qual compromisso apagar? Diga o nome (ou o #id da lista), ex.: “apague dentista”.", {"acao": "apagar", "ok": False}
    if not alvos:
        return f"⚠️ Não encontrei nenhum compromisso com “{termo or texto}”" + (f" em {it.data:%d/%m/%Y}" if it.data else "") + ".", {"acao": "apagar", "ok": True, "removidos": 0}
    if len(alvos) > 1 and not todos:
        linhas = [f"🤔 Encontrei {len(alvos)} compromissos parecidos. Diga qual (use o #id) ou “apague todos {termo}”:"]
        for ev in alvos[:10]:
            linhas.append(f"  • {ev['start']:%d/%m/%Y %H:%M}  {ev['summary']}   #{ev['id']}")
        return "\n".join(linhas), {"acao": "apagar", "ok": True, "removidos": 0, "ambiguo": True}
    ids = {e["uid"] for e in alvos}
    restantes = [e for e in eventos if e["uid"] not in ids]
    salvar(restantes)
    nomes = ", ".join(f"“{e['summary']}”" for e in alvos[:5])
    return f"🗑️ Apaguei {len(alvos)} compromisso(s): {nomes}.", {"acao": "apagar", "ok": True, "removidos": len(alvos)}


def acao_remarcar(texto: str, eventos: list[dict], agora: datetime) -> tuple[str, dict]:
    it = D.interpretar(texto, agora)
    # "remarque a prova para dia 25": o alvo é o texto SEM a parte da nova data.
    termo = limpar_titulo(texto, it.spans)
    if it.data is None and it.hora is None:
        return "⚠️ Para qual dia/hora devo mover? Ex.: “remarque a prova para dia 25 às 10h”.", {"acao": "remarcar", "ok": False}
    alvos = achar_alvos(eventos, termo)
    if not alvos:
        return f"⚠️ Não encontrei compromisso com “{termo}”.", {"acao": "remarcar", "ok": False}
    if len(alvos) > 1:
        linhas = ["🤔 Mais de um compromisso combina. Use o #id:"]
        linhas += [f"  • {e['start']:%d/%m/%Y %H:%M}  {e['summary']}   #{e['id']}" for e in alvos[:10]]
        return "\n".join(linhas), {"acao": "remarcar", "ok": False, "ambiguo": True}
    ev = alvos[0]
    antigo: datetime = ev["start"]
    nova_data = it.data or antigo.date()
    nova_hora = it.hora or (antigo.time() if not ev.get("all_day") else time(0, 0))
    novo = datetime.combine(nova_data, nova_hora)
    dur = duracao(ev)
    if it.fim and it.hora:
        dur = datetime.combine(nova_data, it.fim) - novo
    ev["raw"] = reescrever_inicio(ev, novo, dur)
    salvar(eventos)
    return (f"🔄 Remarcado: “{ev['summary']}”\n  de {antigo:%d/%m/%Y %H:%M}\n  para {novo:%d/%m/%Y %H:%M}", {"acao": "remarcar", "ok": True})


def acao_proximo(eventos: list[dict], agora: datetime) -> tuple[str, dict]:
    itens = [(dt, ev) for dt, ev in expandir(eventos, agora.date(), agora.date() + timedelta(days=400)) if ev.get("all_day") or dt >= agora]
    if not itens:
        return "📅 Não há nenhum compromisso futuro na sua agenda.", {"acao": "proximo", "ok": True, "quantidade": 0}
    dt, ev = itens[0]
    falta = dt - agora
    horas = int(falta.total_seconds() // 3600)
    quando = f"em {falta.days} dia(s)" if falta.days >= 1 else (f"em {horas}h{int(falta.total_seconds() // 60) % 60:02d}" if horas else f"em {int(falta.total_seconds() // 60)} min")
    return (f"📅 Próximo compromisso:\n📌 {ev['summary']}\n⏰ {D.descrever_data(dt.date(), agora.date())} · {hora_txt(dt, ev)} ({quando})", {"acao": "proximo", "ok": True, "quantidade": 1})


def acao_buscar(texto: str, eventos: list[dict], agora: datetime) -> tuple[str, dict]:
    termo = limpar_titulo(texto, [], {"quando", "que", "dia", "qual", "quais", "quanto", "tenho", "tem", "foi", "sera", "eh", "e", "esta", "marcado", "marcada"})
    termo = re.sub(r"\b(?:quando|que dia)\b", "", termo, flags=re.I).strip()
    if not termo:
        return acao_listar(texto, eventos, agora)
    achados = achar_alvos(eventos, termo)
    if not achados:
        return f"📅 Não achei nada sobre “{termo}” na agenda.", {"acao": "buscar", "ok": True, "quantidade": 0}
    linhas = [f"🔎 Sobre “{termo}”:"]
    for ev in sorted(achados, key=lambda e: e["start"])[:10]:
        prox = next(iter(ocorrencias(ev, agora.date(), agora.date() + timedelta(days=400))), ev["start"])
        linhas.append(linha_evento(prox, ev).replace("  • ", f"  • {prox:%d/%m/%Y} "))
    return "\n".join(linhas), {"acao": "buscar", "ok": True, "quantidade": len(achados)}


def acao_listar(texto: str, eventos: list[dict], agora: datetime) -> tuple[str, dict]:
    per = D.periodo_da_consulta(texto, agora)
    if per:
        titulo, ini, fim = per
    else:
        it = D.interpretar(texto, agora, preferir_futuro=False)
        if it.data:
            titulo, ini, fim = D.descrever_data(it.data, agora.date()), it.data, it.data
        else:
            titulo, ini, fim = "próximos 30 dias", agora.date(), agora.date() + timedelta(days=30)
    itens = expandir(eventos, ini, fim)
    return formatar_lista(titulo, itens, ini, fim), {"acao": "listar", "ok": True, "quantidade": len(itens), "periodo": titulo}


def decidir(texto: str, agora: datetime) -> str:
    n = C.norm(texto)
    if re.search(_VERBO_APAGAR, n):
        return "apagar"
    if re.search(_VERBO_REMARCAR, n):
        return "remarcar"
    consulta = re.search(_CONSULTA, n)
    if re.search(r"\bproxim[oa]s?\b", n) and re.search(r"\b(?:compromisso|evento|reuniao|prova|tarefa)s?\b|\bmeu\b|\bqual\b", n) and not re.search(r"\bproximos?\s+\d+\s+dias\b", n):
        if not re.search(r"\bproxim[oa]\s+(?:semana|mes)\b|\b" + D._DIA_RE + r"\b", n):
            return "proximo"
    if re.search(_VERBO_ADICIONAR, n) and not (consulta and not re.search(r"\b(?:marc|agend|adicion|cri|anot)\w*", n)):
        return "adicionar"
    if re.search(_VERBO_BUSCAR, n) or re.search(r"\b(?:quando|que\s+dia)\b", n):
        return "buscar"
    if not consulta and "?" not in texto:
        it = D.interpretar(texto, agora)
        if it.data and (it.recorrencia or it.hora) and limpar_titulo(texto, it.spans):
            return "adicionar"
    if re.search(r"^\s*tenho\b", n) and D.interpretar(texto, agora).data and not consulta.group(0) == "tem" and "?" not in texto:
        return "adicionar"
    return "listar"


def interpretar_e_executar(contexto: str) -> tuple[str, dict]:
    agora = datetime.now()
    texto = C.clean_command(contexto, {"calendario", "agenda"}) or contexto
    eventos, aviso = carregar()
    acao = decidir(texto, agora)
    funcoes = {
        "adicionar": lambda: acao_adicionar(texto, eventos, agora),
        "apagar": lambda: acao_apagar(texto, eventos, agora),
        "remarcar": lambda: acao_remarcar(texto, eventos, agora),
        "proximo": lambda: acao_proximo(eventos, agora),
        "buscar": lambda: acao_buscar(texto, eventos, agora),
        "listar": lambda: acao_listar(texto, eventos, agora),
    }
    corpo, meta = funcoes[acao]()
    if aviso:
        corpo += f"\n\nℹ️ {aviso}"
    meta["interpretado"] = f"{acao}: {texto[:120]}"
    return corpo, meta


def main() -> None:
    texto = C.contexto()
    try:
        corpo, meta = interpretar_e_executar(texto)
    except Exception as exc:  # erro inesperado: relata em vez de fingir sucesso
        C.fail(f"Erro ao processar a agenda: {exc}", REPORT)
        return
    C.finish(REPORT, corpo, arquivo=str(ICS_PATH), **meta)


if __name__ == "__main__":
    main()
