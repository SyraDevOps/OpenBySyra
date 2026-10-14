"""Agente dispositivo: informações do computador (sistema, memória, discos, bateria, rede).

  info do computador | quanto de memória livre | espaço nos discos
  estou com internet? | qual meu ip | onde estou (localização aproximada por IP)
  bateria | há quanto tempo o pc está ligado | processos que mais usam memória
IP público/localização só são consultados se você pedir (ou pedir “completo”).
Funciona sem psutil (usa o que o sistema oferece); com psutil traz mais detalhes.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astra_common as C  # noqa: E402

C.setup_utf8()
C.enable_user_site()
REPORT = "dispositivo/relatorio.txt"
try:
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None


def gb(b: float) -> str:
    return f"{b / 1024 ** 3:.1f} GB"


def barra(pct: float, n: int = 20) -> str:
    cheio = int(round(pct / 100 * n))
    return "█" * cheio + "░" * (n - cheio)


def duracao(seg: float) -> str:
    d, r = divmod(int(seg), 86400)
    h, r = divmod(r, 3600)
    m = r // 60
    return (f"{d}d " if d else "") + f"{h}h {m:02d}min"


def secao_sistema() -> list[str]:
    linhas = ["💻 SISTEMA", f"   Computador: {socket.gethostname()}", f"   SO: {platform.system()} {platform.release()} ({platform.machine()})"]
    ver = platform.version()
    if ver and len(ver) < 80:
        linhas.append(f"   Versão: {ver}")
    cpu = platform.processor() or ""
    if not cpu and Path("/proc/cpuinfo").exists():
        m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text(errors="replace"))
        cpu = m.group(1) if m else ""
    nucleos = os.cpu_count()
    linhas.append(f"   CPU: {cpu or 'desconhecida'} · {nucleos} núcleos lógicos" + (f" · uso agora {psutil.cpu_percent(interval=0.4):.0f}%" if psutil else ""))
    linhas.append(f"   Python: {platform.python_version()}")
    boot = None
    if psutil:
        boot = time.time() - psutil.boot_time()
    elif Path("/proc/uptime").exists():
        boot = float(Path("/proc/uptime").read_text().split()[0])
    elif os.name == "nt":
        try:
            import ctypes

            boot = ctypes.windll.kernel32.GetTickCount64() / 1000  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            boot = None
    if boot:
        linhas.append(f"   Ligado há: {duracao(boot)}")
    return linhas


def secao_memoria() -> list[str]:
    total = usado = None
    if psutil:
        m = psutil.virtual_memory()
        total, usado = m.total, m.total - m.available
    elif Path("/proc/meminfo").exists():
        info = {k: int(v.split()[0]) * 1024 for k, v in (l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l)}
        total, usado = info.get("MemTotal"), info.get("MemTotal", 0) - info.get("MemAvailable", 0)
    elif os.name == "nt":
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tot", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]

            ms = MS()
            ms.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))  # type: ignore[attr-defined]
            total, usado = ms.tot, ms.tot - ms.avail
        except Exception:  # noqa: BLE001
            pass
    if not total:
        return ["🧠 MEMÓRIA", "   indisponível neste sistema"]
    pct = usado / total * 100
    return ["🧠 MEMÓRIA RAM", f"   {barra(pct)} {pct:.0f}%", f"   Em uso {gb(usado)} de {gb(total)} · livre {gb(total - usado)}"]


def secao_disco() -> list[str]:
    linhas = ["💾 DISCOS"]
    pontos: list[str] = []
    if psutil:
        for p in psutil.disk_partitions(all=False):
            if "cdrom" in p.opts or not p.fstype or p.mountpoint.startswith(("/snap", "/boot", "/sys", "/proc")):
                continue
            pontos.append(p.mountpoint)
    if not pontos:
        pontos = [str(Path.home().anchor or "/")]
    vistos = set()
    for mp in pontos[:8]:
        try:
            u = shutil.disk_usage(mp)
        except OSError:
            continue
        if (u.total, u.used) in vistos or u.total < 2 * 1024 ** 3:
            continue
        vistos.add((u.total, u.used))
        pct = u.used / u.total * 100
        linhas.append(f"   {mp:<10} {barra(pct, 16)} {pct:.0f}%  livre {gb(u.free)} de {gb(u.total)}")
    return linhas


def secao_bateria() -> list[str]:
    if psutil and hasattr(psutil, "sensors_battery"):
        try:
            b = psutil.sensors_battery()
        except Exception:  # noqa: BLE001
            b = None
        if b:
            extra = "🔌 carregando" if b.power_plugged else (f"restam ≈ {duracao(b.secsleft)}" if b.secsleft and b.secsleft > 0 else "na bateria")
            return ["🔋 BATERIA", f"   {b.percent:.0f}% · {extra}"]
    return ["🔋 BATERIA", "   sem bateria detectada (desktop) ou psutil ausente"]


def ip_local() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect(("10.255.255.255", 1))  # não envia nada; só escolhe a interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:  # noqa: BLE001
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:  # noqa: BLE001
            return "desconhecido"


def online() -> bool:
    for host in (("1.1.1.1", 443), ("8.8.8.8", 53)):
        try:
            socket.create_connection(host, timeout=2).close()
            return True
        except OSError:
            continue
    return False


def secao_rede(publico: bool) -> list[str]:
    linhas = ["🌐 REDE", f"   Internet: {'✅ conectado' if online() else '❌ sem conexão'}", f"   IP local: {ip_local()}"]
    if publico:
        try:
            d = C.http_json("https://ipwho.is/", timeout=7, retries=1)
            if d.get("success"):
                linhas += [f"   IP público: {d.get('ip')}", f"   Localização aproximada: {d.get('city')}, {d.get('region')} — {d.get('country')}", f"   Provedor: {(d.get('connection') or {}).get('isp', '—')}"]
            else:
                linhas.append("   IP público: não consegui obter")
        except Exception:  # noqa: BLE001
            linhas.append("   IP público: não consegui obter (sem internet?)")
    return linhas


def secao_processos() -> list[str]:
    if not psutil:
        return ["⚙️ PROCESSOS", "   precisa do psutil: python -m pip install psutil"]
    procs = []
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            procs.append((p.info["memory_info"].rss, p.info["name"] or "?"))
        except Exception:  # noqa: BLE001
            continue
    por_nome: dict[str, int] = {}
    for rss, nome in procs:
        por_nome[nome] = por_nome.get(nome, 0) + rss
    top = sorted(por_nome.items(), key=lambda kv: -kv[1])[:8]
    return ["⚙️ MAIORES USOS DE MEMÓRIA"] + [f"   {nome:<28} {rss / 1024 ** 2:>7.0f} MB" for nome, rss in top]


def main() -> None:
    n = C.norm(C.contexto())
    quer = {
        "sistema": bool(re.search(r"\b(?:sistema|so|windows|linux|mac|cpu|processador|computador|pc|maquina|ligado|uptime|hostname|nome do)\b", n)),
        "memoria": bool(re.search(r"\b(?:memoria|ram)\b", n)),
        "disco": bool(re.search(r"\b(?:disco|discos|armazenamento|espaco|hd|ssd)\b", n)),
        "bateria": bool(re.search(r"\b(?:bateria|carga|carregando)\b", n)),
        "rede": bool(re.search(r"\b(?:internet|rede|wifi|conexao|conectado|online|ip|dns)\b", n)),
        "publico": bool(re.search(r"\b(?:ip publico|meu ip|qual (?:e )?meu ip|localizacao|onde estou|cidade|provedor|isp|completo|tudo|geolocalizacao)\b", n)),
        "processos": bool(re.search(r"\b(?:processos?|consumindo|usando mais|pesados|travando|lento)\b", n)),
    }
    completo = bool(re.search(r"\b(?:completo|completa|tudo|geral|resumo|informacoes|info|dados)\b", n)) or not any(v for k, v in quer.items() if k != "publico")
    quer["publico"] = quer["publico"] and (quer["publico"] or completo)
    ordem = [("sistema", secao_sistema), ("memoria", secao_memoria), ("disco", secao_disco), ("bateria", secao_bateria), ("rede", lambda: secao_rede(quer["publico"] or completo and bool(re.search(r"\b(?:completo|tudo)\b", n)))), ("processos", secao_processos)]
    blocos = []
    try:
        for chave, fn in ordem:
            if quer[chave] or (completo and chave != "processos") or (chave == "rede" and quer["publico"]):
                blocos.append("\n".join(fn()))
    except Exception as exc:  # noqa: BLE001
        C.fail(f"Erro ao coletar dados do dispositivo: {exc}", REPORT)
        return
    if not psutil and completo:
        blocos.append("ℹ️ Para mais detalhes (CPU em tempo real, bateria, processos): python -m pip install psutil")
    C.finish(REPORT, "\n\n".join(blocos), resposta="direta", interpretado=f"dispositivo: {','.join(k for k, v in quer.items() if v) or 'geral'}")


if __name__ == "__main__":
    main()
