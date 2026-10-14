"""Agente lembrete: avisos e timers que disparam sozinhos (notificação + som).

  me lembre em 10 minutos de tirar o bolo | timer de 5 minutos
  me lembre às 15h de ligar para a Ana    | lembrete amanhã às 9h: consulta
  meus lembretes | cancele o lembrete 2 | cancele todos os lembretes
Roda em segundo plano: continua valendo se você fechar a Astra, mas NÃO sobrevive
a desligar/reiniciar o computador. Para compromissos duradouros use o calendário.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402
import _astra_datas as D  # noqa: E402

C.setup_utf8()
REPORT = "lembretes/relatorio.txt"
PASTA = Path("lembretes").resolve()
ARQ = PASTA / "ativos.json"
WORKER = Path(__file__).resolve().parent / "lembrete_worker.py"
UN = {"s": 1, "seg": 1, "segundo": 1, "segundos": 1, "min": 60, "minuto": 60, "minutos": 60, "h": 3600, "hora": 3600, "horas": 3600, "dia": 86400, "dias": 86400}


def carregar() -> dict:
    try:
        d = json.loads(ARQ.read_text(encoding="utf-8"))
        d.setdefault("itens", [])
        d.setdefault("prox", 1)
        return d
    except Exception:
        return {"itens": [], "prox": 1}


def salvar(d: dict) -> None:
    PASTA.mkdir(parents=True, exist_ok=True)
    tmp = ARQ.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(ARQ)


def duracao_s(n: str) -> int | None:
    total, achou = 0, False
    meia = re.search(r"\b(\d+|uma|um)\s*(hora|horas|h)\s*e\s*meia\b", n)
    if meia:
        v = 1 if meia.group(1) in ("uma", "um") else int(meia.group(1))
        return v * 3600 + 1800
    for m in re.finditer(r"(\d+(?:[.,]\d+)?|meia|meio|uma|um)\s*(segundos?|seg|minutos?|min|horas?|h|dias?)\b", n):
        v = {"meia": 0.5, "meio": 0.5, "uma": 1, "um": 1}.get(m.group(1)) or float(m.group(1).replace(",", "."))
        total += v * UN[m.group(2)]
        achou = True
    return int(total) if achou else None


def texto_do_lembrete(texto: str, spans_remover: list[str]) -> str:
    t = texto
    for pat in spans_remover:
        t = re.sub(pat, " ", t, flags=re.I)
    t = re.sub(r"\b(?:me\s+)?(?:lembr\w+|avis\w+|cham\w+|notifi\w+)\b|\b(?:crie|cria|criar|coloque|defina|programe|marque|configure)\b|\b(?:um|uma)\s+(?:lembrete|timer|alarme|cron[oô]metro|aviso)\b|\b(?:lembrete|timer|alarme)\b", " ", t, flags=re.I)
    t = re.sub(r"\b(?:às|as|a)\s*(?=[:,\-]|$)", " ", t, flags=re.I)
    t = re.sub(r"(?:^|\s)(?:às|as)\s+(?=(?:de|da|do|dos|das|para|pra|que|sobre)\b)", " ", t, flags=re.I)
    t = re.sub(r"^\W*(?:de|para|pra|que|sobre|:|-)\s+", "", re.sub(r"\s+", " ", t).strip(), flags=re.I)
    t = re.sub(r"^(?:de|para|pra|que|sobre|da|do)\s+", "", t.strip(" :-,.?!"), flags=re.I)
    t = re.sub(r"\s+(?:de|para|em|as|às|a)$", "", t.strip())
    return t.strip(" :-,.?!")


def iniciar_worker(rid: int) -> bool:
    cmd = [sys.executable, "-I", str(WORKER), str(PASTA), str(rid)]
    kw: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "close_fds": True}
    try:
        if os.name == "nt":
            base = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            try:
                subprocess.Popen(cmd, creationflags=base | 0x01000000, **kw)  # + BREAKAWAY_FROM_JOB
            except OSError:
                subprocess.Popen(cmd, creationflags=base, **kw)
        else:
            subprocess.Popen(cmd, start_new_session=True, **kw)
        return True
    except Exception:
        return False


def fmt_quando(ts: float, agora: datetime) -> str:
    dt = datetime.fromtimestamp(ts)
    falta = int(ts - time.time())
    rel = f"em {falta // 3600}h{falta % 3600 // 60:02d}" if falta >= 3600 else f"em {max(falta // 60, 0)} min" if falta >= 60 else f"em {max(falta, 0)} s"
    return f"{D.descrever_data(dt.date(), agora.date())} às {dt:%H:%M} ({rel})"


def main() -> None:
    bruto = C.contexto()
    texto = C.clean_command(bruto) or bruto
    n = C.norm(texto)
    agora = datetime.now()
    d = carregar()
    # limpa disparados/cancelados antigos (> 1 dia)
    d["itens"] = [x for x in d["itens"] if x.get("estado") == "ativo" or time.time() - x["quando"] < 86400]
    try:
        if re.search(r"\b(?:cancel\w+|apag\w+|remov\w+|delet\w+|desativ\w+|para(?:r)?)\b", n) and re.search(r"\b(?:lembretes?|timers?|alarmes?|avisos?)\b", n):
            m = re.search(r"#?\b(\d+)\b", n)
            ativos = [x for x in d["itens"] if x.get("estado") == "ativo"]
            alvo = ativos if re.search(r"\b(?:todos|todas|tudo)\b", n) else [x for x in ativos if m and x["id"] == int(m.group(1))]
            if not alvo and not m and len(ativos) == 1:
                alvo = ativos
            for x in alvo:
                x["estado"] = "cancelado"
            salvar(d)
            corpo = f"🗑️ Cancelei {len(alvo)} lembrete(s)." if alvo else "⚠️ Não achei esse lembrete. Veja: “meus lembretes”."
            meta = {"acao": "cancelar"}
        elif re.search(r"\b(?:meus|quais|liste|mostre|ver|veja|ativos|tenho)\b.*\b(?:lembretes?|timers?|alarmes?)\b|^(?:lembretes?|timers?)$", n):
            ativos = sorted((x for x in d["itens"] if x.get("estado") == "ativo"), key=lambda x: x["quando"])
            corpo = ("⏰ Lembretes ativos:\n" + "\n".join(f"  #{x['id']}  {fmt_quando(x['quando'], agora)}  —  {x['texto']}" for x in ativos)) if ativos else "⏰ Nenhum lembrete ativo."
            meta = {"acao": "listar"}
        else:
            seg = duracao_s(n) if re.search(r"\b(?:em|daqui a|daqui|de|por)\s+(?:\d|meia|meio|uma?\b)", n) else None
            alvo_dt, remover = None, []
            if seg:
                alvo_dt = agora + timedelta(seconds=seg)
                remover = [r"\b(?:daqui a|daqui|em|de|por)\s+(?:\d+(?:[.,]\d+)?|meia|meio|uma|um)\s*(?:horas?\s*e\s*meia|segundos?|seg|minutos?|min|horas?|h|dias?)\b(?:\s*e\s*(?:meia|\d+\s*min\w*))?"]
            else:
                it = D.interpretar(texto, agora)
                if it.hora and not it.erro:
                    alvo_dt = datetime.combine(it.data or agora.date(), it.hora)
                    if alvo_dt <= agora and alvo_dt.date() == agora.date():
                        alvo_dt += timedelta(days=1)  # "às 8h" já passou hoje → amanhã
                    remover = [re.escape(texto[a:b]) for a, b in sorted(it.spans, reverse=True)]
                elif it.data and not it.hora:
                    alvo_dt = datetime.combine(it.data, datetime.min.time().replace(hour=9))
                    remover = [re.escape(texto[a:b]) for a, b in sorted(it.spans, reverse=True)]
            if alvo_dt is None:
                C.finish(REPORT, "⚠️ Para quando? Ex.: “me lembre em 10 minutos de tirar o bolo”, “me lembre às 15h de ligar para a Ana”.", resposta="direta", interpretado="lembrete: sem hora")
                return
            if alvo_dt <= agora:
                C.finish(REPORT, "⚠️ Esse horário já passou.", resposta="direta", interpretado="lembrete: passado")
                return
            msg = texto_do_lembrete(texto, remover) or ("Timer terminou!" if re.search(r"\btimer|cronometro\b", n) else "Lembrete!")
            rid = d["prox"]
            d["prox"] += 1
            d["itens"].append({"id": rid, "texto": msg[:200], "quando": alvo_dt.timestamp(), "criado": time.time(), "estado": "ativo"})
            salvar(d)
            ok = iniciar_worker(rid)
            corpo = f"⏰ Combinado! Aviso #{rid}: “{msg}”\n   {fmt_quando(alvo_dt.timestamp(), agora)}" + ("" if ok else "\n⚠️ Não consegui iniciar o processo em segundo plano.")
            meta = {"acao": "criar", "id": rid}
    except Exception as exc:  # noqa: BLE001
        C.fail(f"Erro nos lembretes: {exc}", REPORT)
        return
    C.finish(REPORT, corpo, resposta="direta", interpretado=f"lembrete/{meta['acao']}: {bruto[:90]}", **meta)


if __name__ == "__main__":
    main()
