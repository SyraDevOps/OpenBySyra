"""Agente gerador: senhas, PINs, sorteios, dados, moeda e UUID — com `secrets`.

  gere uma senha forte de 20 caracteres | senha sem símbolos | senha fácil de lembrar
  gere um PIN de 6 dígitos | gere um uuid
  sorteie um número de 1 a 100 | jogue um dado | role 2d6 | cara ou coroa
  sorteie entre ana, bia e caio | escolha entre pizza ou sushi
Usa o gerador criptográfico do sistema (não pseudoaleatório). A senha aparece
só aqui no relatório local; nada é enviado a lugar nenhum.
"""

from __future__ import annotations

import re
import secrets
import string
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
REPORT = "gerador/resultado.txt"
SIMBOLOS = "!@#$%&*?-_+=."
AMBIGUOS = set("O0oIl1|")
PALAVRAS = ("lua sol mar rio vento fogo terra nuvem pedra folha cafe pao leite mel uva lima manga gato lobo urso pato cavalo tigre " "trem barco carro praia monte campo ponte torre porta janela livro caneta mesa cadeira piano violao samba festa jogo sonho "
            "tempo ouro prata azul verde roxo laranja rosa cinza forte calmo veloz bravo doce amargo claro escuro").split()


def senha(tam: int, simbolos: bool, numeros: bool, maiusc: bool, ambiguos: bool) -> str:
    base = string.ascii_lowercase + (string.ascii_uppercase if maiusc else "") + (string.digits if numeros else "") + (SIMBOLOS if simbolos else "")
    if not ambiguos:
        base = "".join(c for c in base if c not in AMBIGUOS)
    grupos = [string.ascii_lowercase] + ([string.ascii_uppercase] if maiusc else []) + ([string.digits] if numeros else []) + ([SIMBOLOS] if simbolos else [])
    grupos = ["".join(c for c in g if ambiguos or c not in AMBIGUOS) for g in grupos]
    while True:
        s = "".join(secrets.choice(base) for _ in range(tam))
        if all(any(c in s for c in g) for g in grupos):  # garante um de cada tipo
            return s


def frase_senha(n_palavras: int = 4) -> str:
    return "-".join(secrets.choice(PALAVRAS) for _ in range(n_palavras)) + f"-{secrets.randbelow(900) + 100}"


def entropia(tam: int, universo: int) -> float:
    import math

    return tam * math.log2(universo)


def numero_de(n: str) -> int | None:
    m = re.search(r"\b(\d{1,3})\b", n)
    return int(m.group(1)) if m else None


def responder(texto: str) -> tuple[str, dict]:
    n = C.norm(texto)
    if re.search(r"\buuid\b|\bguid\b", n):
        return f"🆔 {uuid.uuid4()}", {"tipo": "uuid"}
    if re.search(r"\bpin\b", n):
        tam = int(re.search(r"\b(\d{1,2})\s*(?:digitos?|numeros?)\b", n).group(1)) if re.search(r"\b(\d{1,2})\s*(?:digitos?|numeros?)\b", n) else 4
        tam = max(3, min(tam, 12))
        return f"🔢 PIN: {''.join(secrets.choice(string.digits) for _ in range(tam))}", {"tipo": "pin"}
    if re.search(r"\bcara ou coroa\b|\bmoeda\b|\bcara\b.*\bcoroa\b", n):
        return f"🪙 {secrets.choice(['Cara', 'Coroa'])}!", {"tipo": "moeda"}
    m = re.search(r"\b(\d{1,2})\s*d\s*(\d{1,3})\b", n)
    if m or re.search(r"\b(?:dado|dados|rol(?:e|ar))\b", n):
        qtd, lados = (int(m.group(1)), int(m.group(2))) if m else (1, 6)
        ml = re.search(r"\bd(\d{1,3})\b", n)
        if not m and ml:
            lados = int(ml.group(1))
        qtd, lados = max(1, min(qtd, 20)), max(2, min(lados, 1000))
        rolls = [secrets.randbelow(lados) + 1 for _ in range(qtd)]
        return f"🎲 {qtd}d{lados}: " + (" + ".join(map(str, rolls)) + f" = {sum(rolls)}" if qtd > 1 else str(rolls[0])), {"tipo": "dado"}
    m = re.search(r"\b(?:sort\w+|escolh\w+|decid\w+)\b.*?\bentre\s+(.+)$", n)
    if m:
        opcoes = [o.strip(" .?!") for o in re.split(r"\s*(?:,|;|\bou\b|\be\b)\s*", re.sub(r"^\s*(?:os\s+|as\s+)?", "", m.group(1))) if o.strip(" .?!")]
        # preserva a grafia original das opções
        orig = re.search(r"\bentre\s+(.+)$", texto, re.I)
        if orig:
            ops2 = [o.strip(" .?!") for o in re.split(r"\s*(?:,|;|\bou\b|\be\b)\s*", orig.group(1)) if o.strip(" .?!")]
            if len(ops2) == len(opcoes):
                opcoes = ops2
        if len(opcoes) >= 2:
            return f"🎯 Sorteado: {secrets.choice(opcoes)}   (entre {len(opcoes)}: {', '.join(opcoes)})", {"tipo": "escolha"}
    m = re.search(r"\b(?:entre|de)\s+(-?\d+)\s*(?:a|e|ate|ao)\s*(-?\d+)\b", n)
    if m or re.search(r"\bsort\w+\b.*\bnumero\b|\bnumero aleatorio\b", n):
        lo, hi = (int(m.group(1)), int(m.group(2))) if m else (1, 100)
        lo, hi = min(lo, hi), max(lo, hi)
        return f"🎰 Número sorteado ({lo}–{hi}): {lo + secrets.randbelow(hi - lo + 1)}", {"tipo": "numero"}
    if re.search(r"\bsenha\b|\bpassword\b|\bpassphrase\b", n) or True:
        if re.search(r"\b(?:facil de lembrar|memoravel|frase|palavras|passphrase)\b", n):
            f = frase_senha()
            return f"🔐 Senha fácil de lembrar: {f}\n   (~{entropia(4, len(PALAVRAS)):.0f} bits de entropia nas palavras + número)", {"tipo": "frase"}
        tam = numero_de(n) or 16
        tam = max(8, min(tam, 128))
        simb = not re.search(r"\bsem\s+(?:simbolos?|especiais|caracteres especiais)\b|\bso\s+(?:letras|alfanumerica)\b|\balfanumerica\b", n)
        num_ = not re.search(r"\bsem\s+numeros?\b|\bso\s+letras\b", n)
        maius = not re.search(r"\bsem\s+maiusculas?\b|\bminusculas?\b", n)
        amb = not re.search(r"\bsem\s+ambiguos?\b|\blegivel\b", n)
        s = senha(tam, simb, num_, maius, amb)
        universo = len(set(string.ascii_lowercase + (string.ascii_uppercase if maius else "") + (string.digits if num_ else "") + (SIMBOLOS if simb else "")))
        forca = "muito forte" if entropia(tam, universo) >= 90 else "forte" if entropia(tam, universo) >= 60 else "média"
        return f"🔐 Senha ({tam} caracteres, {forca}, ~{entropia(tam, universo):.0f} bits):\n   {s}\nGuarde num gerenciador de senhas; não reutilize em outros sites.", {"tipo": "senha", "tamanho": tam}
    return "", {}


def main() -> None:
    bruto = C.contexto()
    try:
        corpo, meta = responder(C.clean_command(bruto) or bruto)
    except Exception as exc:  # noqa: BLE001
        C.fail(f"Erro no gerador: {exc}", REPORT)
        return
    C.finish(REPORT, corpo, resposta="direta", interpretado=f"gerador/{meta.get('tipo')}", **{k: v for k, v in meta.items() if k != "tipo"})


if __name__ == "__main__":
    main()
