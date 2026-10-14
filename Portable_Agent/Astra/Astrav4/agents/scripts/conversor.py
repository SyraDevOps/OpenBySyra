"""Agente conversor: unidades, moedas e contas.

  converta 10 km para milhas        | quantos metros tem 3 pés
  100 fahrenheit em celsius         | 2 GB em MB
  50 dólares em reais               | 100 eur para brl   (cotação ao vivo)
  quanto é 15% de 230               | 230 mais 10%
  raiz quadrada de 144              | (15+3)*4/2 | 2^10
Moedas: AwesomeAPI (reserva Frankfurter). Contas: avaliador seguro (sem eval).
"""

from __future__ import annotations

import ast
import math
import operator
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "conversor/resultado.txt"

# unidade -> (grupo, fator para a base do grupo)
U: dict[str, tuple[str, float]] = {}


def _reg(grupo: str, base_fator: float, *nomes: str) -> None:
    for n in nomes:
        U[C.norm(n)] = (grupo, base_fator)


# comprimento (base: metro)
_reg("comprimento", 0.001, "mm", "milimetro", "milimetros")
_reg("comprimento", 0.01, "cm", "centimetro", "centimetros")
_reg("comprimento", 1, "m", "metro", "metros")
_reg("comprimento", 1000, "km", "quilometro", "quilometros", "kilometro", "kilometros")
_reg("comprimento", 0.0254, "pol", "polegada", "polegadas", "in", "inch", "inches")
_reg("comprimento", 0.3048, "pe", "pes", "ft", "foot", "feet")
_reg("comprimento", 0.9144, "jarda", "jardas", "yd")
_reg("comprimento", 1609.344, "milha", "milhas", "mi", "mile", "miles")
_reg("comprimento", 1852, "milha nautica", "milhas nauticas")
# massa (base: grama)
_reg("massa", 0.001, "mg", "miligrama", "miligramas")
_reg("massa", 1, "g", "grama", "gramas")
_reg("massa", 1000, "kg", "quilo", "quilos", "quilograma", "quilogramas", "kilo", "kilos")
_reg("massa", 1_000_000, "t", "tonelada", "toneladas")
_reg("massa", 453.59237, "lb", "libra", "libras", "lbs")
_reg("massa", 28.349523, "oz", "onca", "oncas")
# volume (base: litro)
_reg("volume", 0.001, "ml", "mililitro", "mililitros")
_reg("volume", 1, "l", "litro", "litros", "lt")
_reg("volume", 1000, "m3", "metro cubico", "metros cubicos")
_reg("volume", 3.785411784, "galao", "galoes", "gal")
_reg("volume", 0.24, "xicara", "xicaras")
_reg("volume", 0.015, "colher de sopa", "colheres de sopa")
_reg("volume", 0.005, "colher de cha", "colheres de cha")
# dados (base: byte)
_reg("dados", 1, "b", "byte", "bytes")
_reg("dados", 1024, "kb", "kilobyte", "kilobytes")
_reg("dados", 1024**2, "mb", "megabyte", "megabytes")
_reg("dados", 1024**3, "gb", "gigabyte", "gigabytes")
_reg("dados", 1024**4, "tb", "terabyte", "terabytes")
# velocidade (base: m/s)
_reg("velocidade", 1, "m/s", "metros por segundo")
_reg("velocidade", 1 / 3.6, "km/h", "kmh", "quilometros por hora", "km por hora")
_reg("velocidade", 0.44704, "mph", "milhas por hora")
_reg("velocidade", 0.514444, "no", "nos", "knot", "knots")
# área (base: m²)
_reg("area", 1, "m2", "metro quadrado", "metros quadrados")
_reg("area", 1_000_000, "km2", "quilometro quadrado", "quilometros quadrados")
_reg("area", 10_000, "ha", "hectare", "hectares")
_reg("area", 4046.8564, "acre", "acres")
_reg("area", 0.092903, "ft2", "pe quadrado", "pes quadrados")
# tempo (base: segundo)
_reg("tempo", 1, "s", "seg", "segundo", "segundos")
_reg("tempo", 60, "min", "minuto", "minutos")
_reg("tempo", 3600, "h", "hora", "horas")
_reg("tempo", 86400, "dia", "dias")
_reg("tempo", 604800, "semana", "semanas")
_reg("tempo", 2_592_000, "mes", "meses")
_reg("tempo", 31_536_000, "ano", "anos")
# temperatura (tratada à parte)
TEMP = {"c": "c", "celsius": "c", "graus celsius": "c", "°c": "c", "f": "f", "fahrenheit": "f", "graus fahrenheit": "f", "°f": "f", "k": "k", "kelvin": "k"}

MOEDAS = {
    "usd": "USD", "dolar": "USD", "dolares": "USD", "dolar americano": "USD", "dolares americanos": "USD", "us$": "USD",
    "eur": "EUR", "euro": "EUR", "euros": "EUR", "€": "EUR",
    "brl": "BRL", "real": "BRL", "reais": "BRL", "r$": "BRL",
    "gbp": "GBP", "libra esterlina": "GBP", "libras esterlinas": "GBP",
    "jpy": "JPY", "iene": "JPY", "ienes": "JPY", "cad": "CAD", "dolar canadense": "CAD", "aud": "AUD", "dolar australiano": "AUD",
    "chf": "CHF", "franco suico": "CHF", "ars": "ARS", "peso argentino": "ARS", "pesos argentinos": "ARS", "clp": "CLP", "peso chileno": "CLP",
    "cny": "CNY", "yuan": "CNY", "mxn": "MXN", "peso mexicano": "MXN", "uyu": "UYU", "pyg": "PYG", "guarani": "PYG",
}

_NUM = r"(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)"


def num(s: str) -> float:
    s = s.strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", s):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    return float(s)


def fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e12 or abs(v) < 1e-4:
        return f"{v:.6g}"
    s = f"{v:,.6f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.2f}".rstrip("0").rstrip(".")
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# ------------------------------------------------------------------ matemática

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "raiz": math.sqrt, "abs": abs, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log10, "ln": math.log, "round": round}


def _avaliar(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _avaliar(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _avaliar(node.left), _avaliar(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 1000 or abs(a) > 1e9):
            raise ValueError("potência grande demais")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_avaliar(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and len(node.args) == 1:
        return _FUNCS[node.func.id](_avaliar(node.args[0]))
    if isinstance(node, ast.Name) and node.id in ("pi", "e"):
        return math.pi if node.id == "pi" else math.e
    raise ValueError("expressão não permitida")


def calcular(texto: str) -> tuple[str, float] | None:
    n = C.norm(texto)
    n = n.replace("×", "*").replace("÷", "/")
    m = re.search(rf"{_NUM}\s*%\s*de\s*{_NUM}", n)
    if m:
        a, b = num(m.group(1)), num(m.group(2))
        return f"{fmt(a)}% de {fmt(b)}", a / 100 * b
    m = re.search(rf"{_NUM}\s*(mais|menos|\+|-)\s*{_NUM}\s*%", n)
    if m:
        base, pct = num(m.group(1)), num(m.group(3))
        v = base * (1 + pct / 100) if m.group(2) in ("mais", "+") else base * (1 - pct / 100)
        return f"{fmt(base)} {'+' if m.group(2) in ('mais', '+') else '−'} {fmt(pct)}%", v
    m = re.search(rf"quanto\s+(?:e|eh)\s+{_NUM}\s*(?:por\s*cento|%)\s*(?:de|sobre)\s*{_NUM}", n)
    if m:
        a, b = num(m.group(1)), num(m.group(2))
        return f"{fmt(a)}% de {fmt(b)}", a / 100 * b
    m = re.search(rf"{_NUM}\s*(?:e|eh|sao)\s+quantos?\s*(?:por\s*cento|%)\s*de\s*{_NUM}", n)
    if m:
        a, b = num(m.group(1)), num(m.group(2))
        return f"{fmt(a)} é quanto % de {fmt(b)}", a / b * 100
    m = re.search(rf"raiz\s*(?:quadrada\s*)?de\s*{_NUM}", n)
    if m:
        return f"√{fmt(num(m.group(1)))}", math.sqrt(num(m.group(1)))
    m = re.search(rf"{_NUM}\s*ao\s*quadrado", n)
    if m:
        return f"{fmt(num(m.group(1)))}²", num(m.group(1)) ** 2
    m = re.search(rf"{_NUM}\s*ao\s*cubo", n)
    if m:
        return f"{fmt(num(m.group(1)))}³", num(m.group(1)) ** 3
    # expressão aritmética direta
    expr = re.sub(r"\b(?:quanto|quantos|quanta|e|eh|igual|resultado|calcule|calcula|calcular|conta|faca|a|de|o|qual|me|diga|vale|sera|da)\b", " ", n)
    expr = re.sub(r"\bmais\b", "+", expr)
    expr = re.sub(r"\bmenos\b", "-", expr)
    expr = re.sub(r"\b(?:vezes|multiplicado por)\b|(?<=[\d)])\s*x\s*(?=[\d(])", "*", expr)
    expr = re.sub(r"\b(?:dividido por|dividir por|sobre)\b", "/", expr)
    expr = expr.replace("^", "**").replace("?", " ").replace("=", " ")
    if re.fullmatch(r"[\d\s.,+\-*/()%]*\d[\d\s.,+\-*/()%]*", expr) and re.search(r"[+\-*/]", expr):
        py = re.sub(r"\d{1,3}(?:\.\d{3})+(?!\d)", lambda mm: mm.group().replace(".", ""), expr)
        py = re.sub(r"(\d)\s*,\s*(\d)", r"\1.\2", py)
        py = re.sub(r"(?<=\d)\s*%", "/100", py)
        try:
            return expr.strip(), float(_avaliar(ast.parse(py.strip(), mode="eval")))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            return None
    return None


# ------------------------------------------------------------------ unidades

def _achar_unidade(txt: str) -> str | None:
    t = C.norm(txt).strip(" .?!")
    t = re.sub(r"^(?:de|em|para|pra|a|as|os|o)\s+", "", t)
    if t in MOEDAS:
        return "moeda:" + MOEDAS[t]
    if t in TEMP:
        return "temp:" + TEMP[t]
    if t in U:
        return "un:" + t
    t2 = t.replace("graus ", "").replace("grau ", "")
    if t2 in TEMP:
        return "temp:" + TEMP[t2]
    return None


def _temp(v: float, de: str, para: str) -> float:
    c = v if de == "c" else (v - 32) * 5 / 9 if de == "f" else v - 273.15
    return c if para == "c" else c * 9 / 5 + 32 if para == "f" else c + 273.15


def extrair_conversao(texto: str) -> tuple[float, str, str] | None:
    n = C.norm(texto).replace("°", " ").replace("º", " ")
    n = re.sub(r"\b(?:converta|converter|convertendo|quanto\s+(?:e|eh|vale|da|seria|fica)|quantos?|quantas?|qual\s+(?:e|eh)\s+o\s+valor\s+de|me\s+diga|calcule|em\s+quanto\s+fica)\b", " ", n)
    n = re.sub(r"\b(?:tem|tem em|equivale|equivalem|corresponde|valem)\b", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    unidades = sorted([*U, *TEMP, *MOEDAS], key=len, reverse=True)
    uni_re = "|".join(re.escape(u) for u in unidades)
    padroes = [
        rf"{_NUM}\s*({uni_re})\s+(?:em|para|pra|to|->|=|sao|vira|viram|equivale a|equivalem a)\s+({uni_re})\b",
        rf"{_NUM}\s*({uni_re})\s*(?:em|para|pra)\s*({uni_re})\b",
    ]
    for pad in padroes:
        m = re.search(pad, n)
        if m:
            return num(m.group(1)), m.group(2), m.group(3)
    # "quantos metros tem 3 pés" → destino antes da quantidade
    m = re.search(rf"^({uni_re})\s+(?:(?:em|tem em|tem|ha em|ha)\s+)?{_NUM}\s*({uni_re})\b", n)
    if m:
        return num(m.group(2)), m.group(3), m.group(1)
    return None


def cotacao(de: str, para: str) -> tuple[float, str]:
    """Taxa de conversão de→para. AwesomeAPI primeiro, Frankfurter como reserva."""
    if de == para:
        return 1.0, "—"
    try:
        par = f"{de}-{para}"
        d = C.http_json(f"https://economia.awesomeapi.com.br/json/last/{par}", timeout=8, retries=1)
        reg = d.get(f"{de}{para}")
        if reg and reg.get("bid"):
            return float(reg["bid"]), "AwesomeAPI · " + str(reg.get("create_date", ""))
    except Exception:
        pass
    d = C.http_json(f"https://api.frankfurter.dev/v1/latest?base={de}&symbols={para}", timeout=8, retries=1)
    return float(d["rates"][para]), f"Frankfurter · {d.get('date', '')}"


def converter(valor: float, a: str, b: str) -> tuple[str, dict] | None:
    ua, ub = _achar_unidade(a), _achar_unidade(b)
    if not ua or not ub:
        return None
    ka, na = ua.split(":", 1)
    kb, nb = ub.split(":", 1)
    if ka != kb:
        return f"⚠️ Não dá para converter “{a}” em “{b}”: são tipos diferentes.", {"convertido": False}
    if ka == "temp":
        r = _temp(valor, na, nb)
        return f"🌡️ {fmt(valor)}°{na.upper()} = {fmt(r)}°{nb.upper()}", {"resultado": r}
    if ka == "moeda":
        taxa, fonte = cotacao(na, nb)
        r = valor * taxa
        return f"💱 {fmt(valor)} {na} = {fmt(r)} {nb}\n   1 {na} = {fmt(taxa)} {nb}  ({fonte})", {"resultado": r, "taxa": taxa}
    ga, fa = U[na]
    gb, fb = U[nb]
    if ga != gb:
        return f"⚠️ Não dá para converter {ga} em {gb} (“{a}” → “{b}”).", {"convertido": False}
    r = valor * fa / fb
    return f"📐 {fmt(valor)} {a.strip()} = {fmt(r)} {b.strip()}", {"resultado": r}


def main() -> None:
    bruto = C.contexto()
    texto = C.clean_command(bruto, {"converter", "conversor", "calculadora"}) or bruto
    try:
        conv = extrair_conversao(texto)
        if conv:
            res = converter(*conv)
            if res:
                msg, meta = res
                C.finish(REPORT, msg, resposta="direta", interpretado=f"conversor: {conv[0]} {conv[1]} → {conv[2]}", **meta)
                return
        calc = calcular(texto)
        if calc:
            C.finish(REPORT, f"🧮 {calc[0]} = {fmt(calc[1])}", resposta="direta", resultado=calc[1], interpretado=f"conta: {calc[0]}")
            return
    except (C.HttpError, OSError, KeyError, ValueError) as exc:
        C.fail(f"Não consegui obter a cotação agora ({exc}).", REPORT)
        return
    C.finish(REPORT, "⚠️ Não entendi a conversão/conta. Exemplos: “10 km em milhas”, “50 dólares em reais”, “15% de 230”, “(15+3)*4”.", resposta="direta", interpretado="conversor: não entendido")


if __name__ == "__main__":
    main()
