"""Agente tarefas: listas de tarefas e de compras (pendentes, prazo, prioridade).

  adicione tarefa pagar a luz até sexta
  anote comprar leite, ovos e pão na lista de compras
  minhas tarefas | o que falta na lista de compras | tarefas de hoje
  concluí a tarefa 2 | risque leite da lista de compras
  apague a tarefa 3 | limpe as concluídas
  [urgente] marca prioridade alta. Para compromissos com HORA, use o calendário.
Arquivo: tarefas/listas.json (listas com qualquer nome: "lista de viagem").
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402
import _astra_datas as D  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "tarefas/relatorio.txt"
ARQ = Path("tarefas/listas.json")
LISTA_PADRAO = "tarefas"
_SINONIMOS = {"compras": "compras", "mercado": "compras", "supermercado": "compras", "feira": "compras", "padaria": "compras"}


# ------------------------------------------------------------------ dados

def carregar() -> dict:
    try:
        d = json.loads(ARQ.read_text(encoding="utf-8"))
        d.setdefault("listas", {})
        d.setdefault("prox_id", 1)
        return d
    except Exception:
        return {"listas": {LISTA_PADRAO: []}, "prox_id": 1}


def salvar(d: dict) -> None:
    ARQ.parent.mkdir(parents=True, exist_ok=True)
    tmp = ARQ.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(ARQ)


def nome_da_lista(texto: str, d: dict) -> tuple[str, str]:
    """Descobre a lista citada. Devolve (nome, texto_sem_a_citacao)."""
    n = C.norm(texto)
    m = re.search(r"\b(?:na|da|de|a|para a|pra|em|nas?)\s+lista\s+(?:de\s+|do\s+|da\s+)?([a-z0-9çãõáéíóúâêô]+(?:\s+[a-z0-9]+)?)", n)
    bruto = None
    if m:
        bruto = m.group(1).strip()
        # "lista de compras e " — corta ao encontrar conectivos
        bruto = re.split(r"\s+(?:e|para|ate|ate|com|amanha|hoje)\b", bruto)[0]
    elif re.search(r"\blista\s+(?:de\s+)?(compras|mercado|supermercado|feira)\b|\b(compras|mercado)\b", n):
        bruto = re.search(r"(compras|mercado|supermercado|feira)", n).group(1)
    if not bruto:
        return LISTA_PADRAO, texto
    chave = _SINONIMOS.get(bruto, bruto)
    existentes = {C.norm(k): k for k in d["listas"]}
    if chave in existentes:
        chave = existentes[chave]
    resto = re.sub(r"\b(?:na|da|de|a|para a|pra|em|nas?)?\s*lista\s+(?:de\s+|do\s+|da\s+)?" + re.escape(bruto.split()[0]) + r"(?:\s+\w+)?", " ", texto, flags=re.I, count=1) if m else texto
    return chave, resto


def achar(itens: list[dict], ref: str) -> list[dict]:
    m = re.search(r"#?\b(\d{1,4})\b", ref)
    if m:
        por_id = [i for i in itens if i["id"] == int(m.group(1))]
        if por_id:
            return por_id
    palavras = [p for p in C.norm(ref).split() if len(p) > 2 and p not in _RUIDO_REF]
    if not palavras:
        return []
    out = []
    for i in itens:
        t = C.norm(i["texto"])
        if all(p in t or any(difflib.SequenceMatcher(None, p, w).ratio() > 0.85 for w in t.split()) for p in palavras):
            out.append(i)
    return out


_RUIDO_REF = {"tarefa", "tarefas", "item", "itens", "lista", "numero", "concluida", "concluido", "feita", "feito", "como", "para", "pra", "que", "uma", "dos", "das", "ela", "ele", "isso", "pronta", "pronto", "completa", "completo"}


# ------------------------------------------------------------------ ações

def extrair_itens(texto: str) -> list[str]:
    t = re.sub(r"^(?:\s*(?:por favor|pf|astra)\b[, ]*)", "", texto, flags=re.I)
    t = re.sub(r"^(?:adicion\w*|anot\w*|cri(?:e|ar)|inclu\w*|coloc\w*|bot(?:a|e)|salv\w*|acrescent\w*|p(?:o|ô)e|lembr(?:e|ar)(?:-?me)?(?:\s+de)?|preciso|tenho\s+que|nova?)\s+", "", t, flags=re.I)
    t = re.sub(r"^(?:uma?\s+)?(?:nova\s+)?(?:tarefas?|itens?|item|lembretes?)\s*(?:de|:|-)?\s*", "", t, flags=re.I)
    t = re.sub(r"^(?:o\s+|a\s+|os\s+|as\s+)?(?:itens?\s+)?", "", t, flags=re.I).strip(" :-")
    return [p.strip(" .;") for p in re.split(r"\s*(?:,|;|\s+e\s+(?=[a-zà-ú]))\s*", t) if p.strip(" .;")]


def acao_adicionar(texto: str, d: dict, hoje: date) -> tuple[str, dict]:
    lista, resto = nome_da_lista(texto, d)
    prioridade = "alta" if re.search(r"\b(?:urgente|importante|prioridade alta|prioritari[oa])\b", resto, re.I) else "normal"
    resto = re.sub(r"\b(?:urgente|importante|prioridade alta|prioritari[oa])\b", "", resto, flags=re.I)
    it = D.interpretar(resto, datetime.combine(hoje, datetime.min.time()))
    prazo = it.data.isoformat() if it.data and not it.erro else None
    if it.data and not it.erro:
        resto = D.remover_spans(resto, it.spans)
    resto = re.sub(r"\b(?:at[eé]|para|pra|no\s+dia|prazo)\s*$", "", resto.strip(), flags=re.I)
    nomes = extrair_itens(resto)
    if prazo and len(nomes) > 1:  # prazo vale para cada item
        pass
    nomes = [x[:1].upper() + x[1:] for x in nomes if len(x) > 1]
    if not nomes:
        return "⚠️ O que devo anotar? Ex.: “adicione tarefa pagar a luz até sexta”.", {"ok": False}
    itens = d["listas"].setdefault(lista, [])
    novos = []
    for nome in nomes:
        if any(C.norm(i["texto"]) == C.norm(nome) and not i["feito"] for i in itens):
            continue  # já existe pendente
        novo = {"id": d["prox_id"], "texto": nome, "feito": False, "prazo": prazo, "prioridade": prioridade, "criado": datetime.now().isoformat(timespec="seconds")}
        d["prox_id"] += 1
        itens.append(novo)
        novos.append(novo)
    salvar(d)
    if not novos:
        return f"ℹ️ Já está na lista “{lista}”.", {"ok": True, "adicionados": 0}
    linhas = [f"✅ {len(novos)} item(ns) em “{lista}”:"]
    for n_ in novos:
        extra = (f"  📅 {D.descrever_data(date.fromisoformat(n_['prazo']), hoje)}" if n_["prazo"] else "") + ("  🔥" if prioridade == "alta" else "")
        linhas.append(f"  #{n_['id']} {n_['texto']}{extra}")
    return "\n".join(linhas), {"ok": True, "adicionados": len(novos), "lista": lista}


def _ordem(i: dict):
    return (i["feito"], i["prioridade"] != "alta", i["prazo"] or "9999", i["id"])


def acao_listar(texto: str, d: dict, hoje: date) -> tuple[str, dict]:
    n = C.norm(texto)
    lista, _ = nome_da_lista(texto, d)
    citou = bool(re.search(r"\blista\b|\bcompras\b|\bmercado\b", n)) or (lista != LISTA_PADRAO and C.norm(lista) in n.split())
    nomes = [lista] if citou and lista in d["listas"] else ([lista] if citou else list(d["listas"]))
    so_hoje = bool(re.search(r"\bhoje\b", n))
    inclui_feitas = bool(re.search(r"\b(?:todas|concluidas|feitas|completas|tudo)\b", n))
    linhas, total = [], 0
    for nome in nomes:
        itens = d["listas"].get(nome, [])
        sel = [i for i in itens if inclui_feitas or not i["feito"]]
        if so_hoje:
            sel = [i for i in sel if i["prazo"] and date.fromisoformat(i["prazo"]) <= hoje]
        if not sel and len(nomes) > 1:
            continue
        linhas.append(f"📋 {nome.capitalize()} ({len(sel)})")
        if not sel:
            linhas.append("   (vazia)")
        for i in sorted(sel, key=_ordem):
            marca = "☑" if i["feito"] else "☐"
            extra = ""
            if i["prazo"]:
                dp = date.fromisoformat(i["prazo"])
                extra = f"  ⚠️ atrasada ({dp:%d/%m})" if dp < hoje and not i["feito"] else f"  📅 {D.descrever_data(dp, hoje)}"
            linhas.append(f"   {marca} #{i['id']} {i['texto']}{extra}{'  🔥' if i['prioridade'] == 'alta' and not i['feito'] else ''}")
        linhas.append("")
        total += len(sel)
    if not linhas or (total == 0 and not citou):
        return "🎉 Nenhuma tarefa pendente!", {"ok": True, "quantidade": 0}
    return "\n".join(linhas).strip(), {"ok": True, "quantidade": total}


def _alvo_acao(texto: str, d: dict) -> tuple[str, list[dict], str]:
    lista, resto = nome_da_lista(texto, d)
    resto = re.sub(r"^\s*(?:conclu\w+|termin\w+|finaliz\w+|fiz|feito|marc\w+|risc\w+|ria?sque|apag\w+|remov\w+|delet\w+|exclu\w+|tire|tira|d[eê]\s+baixa)\b", "", resto, flags=re.I)
    resto = re.sub(r"\b(?:como\s+(?:feit[ao]|conclu[ií]d[ao]|pront[ao])|da\s+lista|de\s+lista)\b", " ", resto, flags=re.I).strip()
    candidatos = [(k, i) for k, itens in d["listas"].items() for i in itens]
    if lista in d["listas"] and (re.search(r"\blista\b|\bcompras\b", C.norm(texto))):
        candidatos = [(k, i) for k, i in candidatos if k == lista]
    achados = achar([i for _k, i in candidatos], resto)
    return lista, achados, resto


def acao_concluir(texto: str, d: dict, hoje: date) -> tuple[str, dict]:
    _l, achados, resto = _alvo_acao(texto, d)
    achados = [i for i in achados if not i["feito"]] or achados
    if not achados:
        return f"⚠️ Não achei essa tarefa (“{resto.strip() or texto}”). Veja a lista com “minhas tarefas”.", {"ok": True, "concluidas": 0}
    if len(achados) > 1 and not re.search(r"\b(?:todas|todos)\b", C.norm(texto)):
        return "🤔 Mais de uma combina — use o número:\n" + "\n".join(f"   #{i['id']} {i['texto']}" for i in achados[:8]), {"ok": True, "concluidas": 0}
    for i in achados:
        i["feito"], i["feito_em"] = True, datetime.now().isoformat(timespec="seconds")
    salvar(d)
    return "☑️ Concluída(s): " + ", ".join(f"#{i['id']} {i['texto']}" for i in achados), {"ok": True, "concluidas": len(achados)}


def acao_remover(texto: str, d: dict, hoje: date) -> tuple[str, dict]:
    n = C.norm(texto)
    if re.search(r"\b(?:limp\w+|apag\w+|remov\w+)\b.*\b(?:concluidas|feitas|completas|prontas)\b", n):
        lista, _ = nome_da_lista(texto, d)
        nomes = [lista] if lista in d["listas"] and re.search(r"\blista\b|compras", n) else list(d["listas"])
        total = 0
        for k in nomes:
            antes = len(d["listas"][k])
            d["listas"][k] = [i for i in d["listas"][k] if not i["feito"]]
            total += antes - len(d["listas"][k])
        salvar(d)
        return f"🧹 Removi {total} item(ns) concluído(s).", {"ok": True, "removidas": total}
    if re.search(r"\b(?:limp\w+|esvazi\w+|apag\w+)\b.*\b(?:toda|tudo|a)\b.*\blista\b", n) and not re.search(r"\d", n):
        lista, _ = nome_da_lista(texto, d)
        if lista in d["listas"]:
            qtd = len(d["listas"][lista])
            d["listas"][lista] = []
            salvar(d)
            return f"🧹 Lista “{lista}” esvaziada ({qtd} item/ns).", {"ok": True, "removidas": qtd}
    _l, achados, resto = _alvo_acao(texto, d)
    if not achados:
        return f"⚠️ Não achei essa tarefa (“{resto.strip() or texto}”).", {"ok": True, "removidas": 0}
    if len(achados) > 1 and not re.search(r"\b(?:todas|todos)\b", n):
        return "🤔 Mais de uma combina — use o número:\n" + "\n".join(f"   #{i['id']} {i['texto']}" for i in achados[:8]), {"ok": True, "removidas": 0}
    ids = {i["id"] for i in achados}
    for k in d["listas"]:
        d["listas"][k] = [i for i in d["listas"][k] if i["id"] not in ids]
    salvar(d)
    return "🗑️ Removida(s): " + ", ".join(f"#{i['id']} {i['texto']}" for i in achados), {"ok": True, "removidas": len(achados)}


def decidir(texto: str) -> str:
    n = C.norm(texto)
    if re.search(r"\b(?:concluir|conclui|concluida|concluido|terminei|terminar|finalizei|finalizar|fiz|risque|riscar|marque?\s+(?:como\s+)?(?:feit|conclu|pront)\w*|marcar\s+como\s+(?:feit|conclu|pront)\w*|de baixa|ja comprei|comprei)\b", n):
        return "concluir"
    if re.search(r"\b(?:apag\w+|remov\w+|delet\w+|exclu\w+|tire|tira|limp\w+|esvazi\w+)\b", n):
        return "remover"
    if re.search(r"\b(?:adicion\w+|anot\w+|cri(?:e|ar)|inclu\w+|coloc\w+|bota|salv\w+|acrescent\w+|preciso|tenho que|lembr\w+|nova tarefa|nov[ao] item)\b", n):
        return "adicionar"
    if re.search(r"\b(?:minhas?|meus?|quais|qual|o que|mostr\w+|list\w+|ver|veja|falta|faltando|pendentes?|tenho|hoje|todas|concluidas|feitas)\b|^(?:tarefas?|lista(?:s)?(?:\s+de\s+\w+)?|compras)$|\blista\b", n):
        return "listar"
    return "adicionar" if len(n.split()) >= 2 or re.search(r"\b(?:urgente|importante)\b", n) else "listar"


def main() -> None:
    bruto = C.contexto()
    texto = C.clean_command(bruto, {"tarefas", "tarefa", "lista"}) or bruto
    hoje = date.today()
    try:
        d = carregar()
        acao = decidir(C.clean_command(bruto) or bruto)
        corpo, meta = {"adicionar": acao_adicionar, "listar": acao_listar, "concluir": acao_concluir, "remover": acao_remover}[acao](texto, d, hoje)
    except Exception as exc:  # noqa: BLE001
        C.fail(f"Erro nas tarefas: {exc}", REPORT)
        return
    meta.pop("ok", None)
    C.finish(REPORT, corpo, resposta="direta", acao=acao, interpretado=f"tarefas/{acao}: {bruto[:100]}", **meta)


if __name__ == "__main__":
    main()
