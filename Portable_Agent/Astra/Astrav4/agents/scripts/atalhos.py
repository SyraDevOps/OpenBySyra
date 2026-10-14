"""Agente atalhos: abre sites/pastas por apelido ("abra cnn" → cnnbrasil.com.br)
e deixa o usuário criar, listar e remover os próprios atalhos.

  abra cnn                          → abre https://www.cnnbrasil.com.br/
  crie o atalho blog para https://meublog.com
  atalho trabalho: https://a.com, https://b.com   (abre vários de uma vez)
  atalho downloads → pasta (caminho local) também vale
  também chame zap de whatsapp     → adiciona apelido
  remova o atalho blog / liste meus atalhos
Os atalhos ficam em atalhos/atalhos.json (pode editar à mão).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "atalhos/relatorio.txt"

_URL_OU_CAMINHO = r"(https?://\S+|(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/\S*)?|~?[/\\][\w ./\\:~-]+|[A-Za-z]:[\\/][^\s,;]+)"


def _limpa(t: str) -> str:
    return t.strip(" \t.,;:!?\"'“”")


def decidir(texto: str) -> str:
    n = C.norm(texto)
    if re.search(r"\b(?:remov\w*|apag\w*|exclu\w*|delet\w*|tir[ae]|esquec\w*)\b", n) and re.search(r"\batalho", n):
        return "remover"
    if re.search(r"\b(?:list\w*|mostr\w*|quais|ver)\b.*\batalhos?\b|\bmeus atalhos\b|^atalhos?$", n):
        return "listar"
    if re.search(r"\b(?:cri\w*|adicion\w*|salv\w*|defin\w*|cadastr\w*|nov[oa]|configur\w*|guard\w*)\b.*\batalho\b", n) or (
        re.search(r"\batalho\b.*(?:\bpara\b|=|:|->|→)", n) and re.search(_URL_OU_CAMINHO, texto, re.I)
    ):
        return "criar"
    if re.search(r"\b(?:tamb[eé]m\s+(?:chame|chama|conhe[cç]a)|apelido|chame\s+\S+\s+de)\b", n):
        return "apelido"
    return "abrir"


def _alvos(texto: str) -> list[str]:
    achados = [_limpa(m.group(0)) for m in re.finditer(_URL_OU_CAMINHO, texto, re.I)]
    return [a for a in achados if a]


def acao_criar(texto: str, itens: dict) -> tuple[str, dict]:
    alvos = _alvos(texto)
    if not alvos:
        return "⚠️ Qual o endereço? Ex.: “crie o atalho blog para https://meublog.com”.", {"ok": False}
    # nome: palavra depois de "atalho" (ou antes de ":"/"=")
    resto = texto
    for a in alvos:
        resto = resto.replace(a, " ")
    m = re.search(r"\batalho\s+(?:chamado\s+|com\s+(?:o\s+)?nome\s+|de\s+nome\s+|para\s+(?:o\s+|a\s+)?)?[\"“']?([\w.\-À-ÿ ]{1,30}?)[\"”']?\s*(?:[:=→]|->|para|apontando|que abre|abre|como|$)", resto, re.I)
    nome = _limpa(m.group(1)) if m else ""
    if not nome:
        m2 = re.search(r"^\s*([\w.\-À-ÿ ]{1,30}?)\s*(?:[:=→]|->)", resto)
        nome = _limpa(m2.group(1)) if m2 else ""
    nome = re.sub(r"^(?:o|a|um|uma|novo|nova)\s+", "", nome, flags=re.I)
    nome_n = C.norm(nome)
    if not nome_n or nome_n in {"para", "novo", "nova", "chamado"}:
        return "⚠️ Qual nome dar ao atalho? Ex.: “crie o atalho blog para https://meublog.com”.", {"ok": False}
    validos, rejeitados = [], []
    for a in alvos:
        link = C.safe_url(a)
        if link:
            validos.append(link)
        elif re.match(r"^(?:~|/|[A-Za-z]:[\\/])", a):
            validos.append(a)
        else:
            rejeitados.append(a)
    if not validos:
        return f"⚠️ Não reconheci um link ou pasta válido em: {', '.join(rejeitados)}.", {"ok": False}
    existia = nome_n in {C.norm(k) for k in itens}
    chave = next((k for k in itens if C.norm(k) == nome_n), nome_n)
    antigo = itens.get(chave, {})
    itens[chave] = {"alvos": validos, "aliases": antigo.get("aliases", [])}
    C.save_shortcuts(itens)
    lista = "\n".join(f"   → {v}" for v in validos)
    return f"{'♻️ Atalho atualizado' if existia else '✅ Atalho criado'}: {chave}\n{lista}\nUse: “abra {chave}”.", {"ok": True, "criado": chave}


def acao_remover(texto: str, itens: dict) -> tuple[str, dict]:
    nome, cands = C.find_shortcut(re.sub(r"\batalhos?\b", " ", texto, flags=re.I), itens)
    if not nome:
        dica = f" Quis dizer: {', '.join(cands)}?" if cands else ""
        return f"⚠️ Não achei esse atalho.{dica}", {"ok": True, "removido": None}
    del itens[nome]
    C.save_shortcuts(itens)
    return f"🗑️ Atalho “{nome}” removido.", {"ok": True, "removido": nome}


def acao_apelido(texto: str, itens: dict) -> tuple[str, dict]:
    m = re.search(r"\b(?:tamb[eé]m\s+)?(?:chame|chama)\s+(?:o\s+|a\s+)?(.+?)\s+de\s+(.+)$", texto, re.I) or re.search(r"\bapelido\s+(.+?)\s+(?:para|de)\s+(.+)$", texto, re.I)
    if not m:
        return "⚠️ Diga assim: “chame zap de whatsapp” (apelido primeiro, atalho depois).", {"ok": False}
    apelido, alvo_txt = _limpa(m.group(1)), _limpa(m.group(2))
    nome, _ = C.find_shortcut(alvo_txt, itens)
    if not nome:
        # inverte: "chame whatsapp de zap"
        nome, _ = C.find_shortcut(apelido, itens)
        apelido = alvo_txt if nome else apelido
    if not nome:
        return "⚠️ Não encontrei o atalho de destino.", {"ok": False}
    al = itens[nome].setdefault("aliases", [])
    if C.norm(apelido) not in {C.norm(x) for x in al}:
        al.append(apelido)
    C.save_shortcuts(itens)
    return f"✅ Agora “{apelido}” também abre {nome}.", {"ok": True, "apelido": apelido, "atalho": nome}


def acao_listar(itens: dict) -> tuple[str, dict]:
    linhas = [f"🔖 {len(itens)} atalhos:"]
    for nome in sorted(itens):
        spec = itens[nome]
        al = f"  (também: {', '.join(spec['aliases'])})" if spec.get("aliases") else ""
        alvos = spec.get("alvos", [])
        linhas.append(f"  • {nome}{al}\n      {alvos[0] if alvos else '—'}" + (f"  (+{len(alvos) - 1})" if len(alvos) > 1 else ""))
    linhas.append("\nCrie os seus: “crie o atalho blog para https://meublog.com”.")
    return "\n".join(linhas), {"ok": True, "quantidade": len(itens)}


def acao_abrir(texto: str, itens: dict) -> tuple[str, dict]:
    # Link/domínio explícito no pedido? Abre direto (atalho também serve para "abra cnn.com").
    alvos_diretos = [a for a in _alvos(texto) if C.safe_url(a)]
    if alvos_diretos:
        ok = C.open_url(alvos_diretos[0])
        return (f"🌐 {'Abri' if ok else 'Não consegui abrir automaticamente; copie'}: {alvos_diretos[0]}", {"ok": True, "aberto": ok, "alvos": alvos_diretos[:1]})
    nome, cands = C.find_shortcut(texto, itens)
    if not nome:
        sug = f" Quis dizer: {', '.join(cands)}?" if cands else ""
        return (f"⚠️ Não tenho um atalho com esse nome.{sug}\nCrie com: “crie o atalho <nome> para <link>”. Veja os existentes: “liste meus atalhos”.", {"ok": True, "aberto": False, "encontrado": False})
    resultados = []
    for alvo in itens[nome].get("alvos", []):
        ok, mostrado = C.open_target(alvo)
        resultados.append((ok, mostrado))
    if not resultados:
        return f"⚠️ O atalho “{nome}” está sem destino.", {"ok": True, "aberto": False}
    linhas = [f"{'✅ Abri' if all(o for o, _ in resultados) else '⚠️ Tentei abrir'} “{nome}”:"]
    linhas += [f"   {'→' if o else '✗'} {l}" for o, l in resultados]
    if not all(o for o, _ in resultados):
        linhas.append("(se o navegador não abriu sozinho, use o link acima)")
    return "\n".join(linhas), {"ok": True, "aberto": any(o for o, _ in resultados), "atalho": nome, "alvos": [l for _, l in resultados]}


def main() -> None:
    bruto = C.contexto()
    texto = C.clean_command(bruto, {"atalho", "atalhos"}) or bruto
    try:
        itens = C.load_shortcuts()
        acao = decidir(bruto)
        funcs = {
            "criar": lambda: acao_criar(bruto, itens),
            "remover": lambda: acao_remover(bruto, itens),
            "apelido": lambda: acao_apelido(bruto, itens),
            "listar": lambda: acao_listar(itens),
            "abrir": lambda: acao_abrir(texto, itens),
        }
        corpo, meta = funcs[acao]()
    except Exception as exc:
        C.fail(f"Erro nos atalhos: {exc}", REPORT)
        return
    ok = meta.pop("ok", True)
    # ok=False = pedido incompleto: a resposta já orienta o usuário (não é falha de infraestrutura).
    C.finish(REPORT, corpo, resposta="direta", acao=acao, interpretado=f"atalhos/{acao}: {bruto[:100]}", **meta)


if __name__ == "__main__":
    main()
