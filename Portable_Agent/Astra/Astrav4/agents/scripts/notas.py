"""Agente notas: anotações rápidas com busca por semelhança.

  anote que a senha do wifi da sala é azul42   | guarde: reunião com a Ana quinta
  o que anotei sobre wifi | procure nas notas ana | minhas notas | últimas notas
  apague a nota 3 | edite a nota 2: novo texto
Arquivo: notas/notas.json (texto puro, editável). Não guarde senhas importantes aqui.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
REPORT = "notas/relatorio.txt"
ARQ = Path("notas/notas.json")
_STOP = {"o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das", "que", "e", "em", "no", "na", "sobre", "para", "pra", "com", "eu", "meu", "minha", "foi", "era", "qual", "quais", "quando", "onde", "como", "procure", "procurar", "busque", "buscar", "ache", "achar", "pesquise", "nas", "anotei", "guardei", "salvei", "nota", "notas", "tenho", "sobre", "disse", "falei", "me", "lembra", "lembre"}


def carregar() -> dict:
    try:
        d = json.loads(ARQ.read_text(encoding="utf-8"))
        d.setdefault("notas", [])
        d.setdefault("prox", 1)
        return d
    except Exception:
        return {"notas": [], "prox": 1}


def salvar(d: dict) -> None:
    ARQ.parent.mkdir(parents=True, exist_ok=True)
    tmp = ARQ.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(ARQ)


def pontuar(nota: dict, termos: list[str]) -> float:
    alvo = C.norm(nota["texto"] + " " + " ".join(nota.get("tags", [])))
    palavras = alvo.split()
    total = 0.0
    for t in termos:
        if t in alvo:
            total += 1
        else:
            melhor = max((difflib.SequenceMatcher(None, t, w).ratio() for w in palavras), default=0)
            total += melhor if melhor >= 0.8 and len(t) > 3 else 0
    return total / len(termos) if termos else 0


def decidir(n: str) -> str:
    if re.search(r"\b(?:apag\w+|remov\w+|delet\w+|exclu\w+|esquec\w+)\b.*\bnota\b|\bapague a nota\b", n):
        return "apagar"
    if re.search(r"\b(?:edit\w+|atualiz\w+|corrij\w+|mud\w+)\b.*\bnota\b", n):
        return "editar"
    if re.search(r"\b(?:o que|oq|qual|quais|quando|onde|como)\b.*\b(?:anotei|guardei|salvei|nota|notas|disse)\b|\b(?:procur\w+|busc\w+|ach\w+|pesquis\w+)\b.*\bnotas?\b|\bnas notas\b", n):
        return "buscar"
    if re.search(r"\b(?:minhas notas|ultimas notas|todas as notas|liste? (?:as )?notas|mostre (?:as )?notas|notas recentes|^notas$)\b", n):
        return "listar"
    if re.search(r"\b(?:anot\w+|guard\w+|salv\w+|registr\w+|nota:|lembre-se|lembre se|memoriz\w+)\b", n):
        return "adicionar"
    return "buscar"


def main() -> None:
    bruto = C.contexto()
    texto = C.clean_command(bruto, {"notas", "nota"}) or bruto
    n = C.norm(bruto)
    d = carregar()
    acao = decidir(n)
    try:
        if acao == "adicionar":
            conteudo = re.sub(r"^(?:anot\w+|guard\w+|salv\w+|registr\w+|memoriz\w+|lembre-?\s*se)\s*(?:isso|ai|aí|uma nota|a nota|nota)?\s*(?:que|:|-)?\s*(?:de que\s+)?", "", texto.strip(), flags=re.I).strip(" :-")
            if len(conteudo) < 2:
                corpo = "⚠️ O que devo anotar? Ex.: “anote que a reunião é quinta às 10h”."
            else:
                conteudo = conteudo[:1500]
                nota = {"id": d["prox"], "texto": conteudo[:1].upper() + conteudo[1:], "criada": datetime.now().isoformat(timespec="seconds")}
                d["prox"] += 1
                d["notas"].append(nota)
                salvar(d)
                corpo = f"📝 Anotado (#{nota['id']}): {nota['texto']}"
        elif acao == "buscar":
            termos = [t for t in re.findall(r"[a-z0-9]+", n) if t not in _STOP and len(t) > 1]
            pont = sorted(((pontuar(x, termos), x) for x in d["notas"]), key=lambda p: (-p[0], -p[1]["id"]))
            achadas = [x for s, x in pont if s >= 0.6][:6]
            if not termos:
                corpo = "⚠️ Sobre o que? Ex.: “o que anotei sobre wifi”."
            elif not achadas:
                corpo = f"🔎 Nenhuma nota parecida com “{' '.join(termos)}”."
            else:
                corpo = f"🔎 {len(achadas)} nota(s):\n" + "\n".join(f"  #{x['id']} ({x['criada'][:10]})  {x['texto']}" for x in achadas)
        elif acao == "listar":
            ult = d["notas"][-10:][::-1]
            corpo = (f"📝 Últimas {len(ult)} de {len(d['notas'])} notas:\n" + "\n".join(f"  #{x['id']} ({x['criada'][:10]})  {x['texto']}" for x in ult)) if ult else "📝 Você ainda não tem notas."
        elif acao == "apagar":
            m = re.search(r"#?(\d+)", n)
            if not m:
                corpo = "⚠️ Qual nota? Diga o número: “apague a nota 3”."
            else:
                antes = len(d["notas"])
                alvo = next((x for x in d["notas"] if x["id"] == int(m.group(1))), None)
                d["notas"] = [x for x in d["notas"] if x["id"] != int(m.group(1))]
                if len(d["notas"]) < antes:
                    salvar(d)
                    corpo = f"🗑️ Apaguei a nota #{alvo['id']}: {alvo['texto']}"
                else:
                    corpo = f"⚠️ Não existe a nota #{m.group(1)}."
        else:  # editar
            m = re.search(r"nota\s*#?(\d+)\s*[:\-]?\s*(.+)$", texto, re.I)
            alvo = next((x for x in d["notas"] if m and x["id"] == int(m.group(1))), None)
            if not m or not alvo:
                corpo = "⚠️ Use: “edite a nota 2: novo texto”."
            else:
                alvo["texto"] = m.group(2).strip()[:1500]
                salvar(d)
                corpo = f"✏️ Nota #{alvo['id']} atualizada: {alvo['texto']}"
    except Exception as exc:  # noqa: BLE001
        C.fail(f"Erro nas notas: {exc}", REPORT)
        return
    C.finish(REPORT, corpo, resposta="direta", acao=acao, interpretado=f"notas/{acao}: {bruto[:90]}")


if __name__ == "__main__":
    main()
