"""Agente grafico_carteira (modo "lancar"): gera um gráfico da carteira e o abre.

Lê carteira/carteira.csv (colunas: ativo,valor). Se não existir, cria um exemplo
para você editar. Escreve carteira/grafico.html (SVG puro, sem dependências) e o
abre no navegador padrão. Roda em segundo plano: a Astra só avisa que foi executado.
"""

from __future__ import annotations

import csv
import math
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
PASTA = Path("carteira")
CSV = PASTA / "carteira.csv"
HTML = PASTA / "grafico.html"
EXEMPLO = "ativo,valor\nRenda fixa,12000\nAções,8000\nFIIs,5000\nCripto,2500\nCaixa,1500\n"
CORES = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


def ler() -> list[tuple[str, float]]:
    if not CSV.exists():
        C.write_text(CSV, EXEMPLO)
    itens: list[tuple[str, float]] = []
    with CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                nome = (row.get("ativo") or "").strip()
                v = float((row.get("valor") or "0").replace(".", "").replace(",", ".")) if "," in (row.get("valor") or "") else float(row.get("valor") or 0)
            except ValueError:
                continue
            if nome and v > 0:
                itens.append((nome, v))
    return itens


def brl(v: float) -> str:
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pizza(itens, total) -> str:
    cx = cy = 130
    r = 120
    ang = -math.pi / 2
    out = []
    for i, (nome, v) in enumerate(itens):
        fr = v / total
        if fr >= 0.9999:
            out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{CORES[i % 10]}"/>')
            continue
        a2 = ang + fr * 2 * math.pi
        x1, y1 = cx + r * math.cos(ang), cy + r * math.sin(ang)
        x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
        big = 1 if fr > 0.5 else 0
        out.append(f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} A{r},{r} 0 {big} 1 {x2:.1f},{y2:.1f} Z" fill="{CORES[i % 10]}"><title>{escape(nome)}: {brl(v)} ({fr:.1%})</title></path>')
        ang = a2
    return f'<svg viewBox="0 0 260 260" width="260" height="260">{"".join(out)}</svg>'


def legenda(itens, total) -> str:
    linhas = []
    for i, (nome, v) in enumerate(itens):
        fr = v / total
        linhas.append(
            f'<div class="l"><span class="d" style="background:{CORES[i % 10]}"></span><b>{escape(nome)}</b>'
            f'<span class="v">{brl(v)} · {fr:.1%}</span><div class="b"><i style="width:{fr * 100:.1f}%;background:{CORES[i % 10]}"></i></div></div>'
        )
    return "".join(linhas)


def main() -> None:
    C.contexto()  # consome o stdin
    try:
        itens = ler()
    except OSError as exc:
        C.fail(f"Não consegui ler {CSV}: {exc}", "carteira/resultado.txt")
        return
    if not itens:
        C.fail(f"{CSV} está vazio ou sem valores válidos (colunas: ativo,valor).", "carteira/resultado.txt")
        return
    itens.sort(key=lambda x: -x[1])
    total = sum(v for _, v in itens)
    html = f"""<!doctype html><meta charset="utf-8"><title>Carteira</title>
<style>body{{font:16px system-ui;max-width:760px;margin:30px auto;padding:0 16px;background:#fafafa;color:#222}}
.w{{display:flex;gap:28px;flex-wrap:wrap;align-items:center}}.l{{margin:8px 0;min-width:300px}}.d{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:8px}}
.v{{float:right;color:#555}}.b{{height:6px;background:#e5e5e5;border-radius:3px;margin-top:4px;clear:both}}.b i{{display:block;height:6px;border-radius:3px}}
@media(prefers-color-scheme:dark){{body{{background:#161616;color:#eee}}.b{{background:#333}}.v{{color:#aaa}}}}</style>
<h1>Carteira · {brl(total)}</h1><div class="w">{pizza(itens, total)}<div>{legenda(itens, total)}</div></div>
<p style="color:#888;font-size:13px">Fonte: {escape(str(CSV))} — edite o CSV e execute o agente de novo.</p>"""
    C.write_text(HTML, html)
    aberto = C.open_path(HTML.resolve())
    C.finish("carteira/resultado.txt", f"Gráfico da carteira ({len(itens)} ativos, {brl(total)}) em {HTML}. Aberto: {aberto}.", interpretado="grafico_carteira")


if __name__ == "__main__":
    main()
