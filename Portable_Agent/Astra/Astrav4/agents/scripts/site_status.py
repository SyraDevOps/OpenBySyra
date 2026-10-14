"""Agente site_status: o site está no ar? quanto demora? (DNS, TLS, HTTP)

  o github está fora do ar? | status do site google.com | ping uol.com.br
Mostra DNS, conexão, código HTTP e tempo de resposta. Só http(s) públicos.
"""

from __future__ import annotations

import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
REPORT = "rede/status.txt"


def alvo(texto: str) -> str | None:
    m = re.search(r"(https?://[^\s]+|(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/\S*)?)", texto, re.I)
    if m:
        return C.safe_url(m.group(1).rstrip(".,;:!?)"))
    nome, _ = C.find_shortcut(texto)
    if nome:
        a = C.load_shortcuts()[nome].get("alvos", [])
        return C.safe_url(a[0]) if a else None
    return None


def main() -> None:
    url = alvo(C.contexto())
    if not url:
        C.finish(REPORT, "⚠️ Qual site? Ex.: “o github está fora do ar?”, “status do google.com”.", resposta="direta", interpretado="site_status: sem alvo")
        return
    host = re.sub(r"^https?://", "", url).split("/")[0].split(":")[0]
    if not C.is_public_host(host):
        C.finish(REPORT, "⛔ Não consegui resolver esse endereço (sem internet, site inexistente ou rede interna — só verifico sites públicos).", resposta="direta", interpretado="site_status: bloqueado")
        return
    linhas = [f"🌐 {host}"]
    t0 = time.time()
    try:
        ip = socket.gethostbyname(host)
        linhas.append(f"   DNS: {ip} ({(time.time() - t0) * 1000:.0f} ms)")
    except OSError:
        C.finish(REPORT, f"🌐 {host}\n   ❌ DNS não resolve (site inexistente ou sem internet).", resposta="direta", no_ar=False, interpretado=f"site_status: {host} dns")
        return
    ctx = C.ssl_context()
    req = urllib.request.Request(url, headers={"User-Agent": C.DEFAULT_UA})
    t1 = time.time()
    codigo, erro = None, None
    try:
        with urllib.request.urlopen(req, timeout=12, context=ctx) as r:
            codigo = r.status
            r.read(2048)
    except urllib.error.HTTPError as e:
        codigo = e.code
    except Exception as e:  # noqa: BLE001
        erro = str(e)[:120]
    ms = (time.time() - t1) * 1000
    if erro:
        linhas.append(f"   ❌ Sem resposta: {erro}")
        no_ar = False
    else:
        no_ar = codigo is not None and codigo < 500
        linhas.append(f"   {'✅ NO AR' if no_ar else '⚠️ com erro no servidor'} — HTTP {codigo} em {ms:.0f} ms" + ("  (lento)" if ms > 2000 else ""))
        if codigo in (401, 403, 429):
            linhas.append("   (o site respondeu, mas limita acesso automático: provavelmente está no ar)")
    C.finish(REPORT, "\n".join(linhas), resposta="direta", no_ar=no_ar, interpretado=f"site_status: {host}")


if __name__ == "__main__":
    main()
