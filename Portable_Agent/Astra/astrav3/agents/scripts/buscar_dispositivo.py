import sys
import json
import socket
import platform
import urllib.request
import ssl
import contextlib
from pathlib import Path

# Adiciona o diretório de pacotes do usuário do Windows no path caso o psutil esteja lá
with contextlib.suppress(Exception):
    import site
    sys.path.append(str(Path.home() / "AppData" / "Roaming" / "Python" / "Python311" / "site-packages"))
    sys.path.append(str(Path.home() / "AppData" / "Roaming" / "Python" / "Python310" / "site-packages"))

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

ssl_context = ssl._create_unverified_context()

def verificar_conexao_internet() -> bool:
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except Exception:
        return False

def obter_ip_local() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def obter_dados_publicos_ip() -> dict:
    endpoints = [
        "https://ipapi.co/json/",
        "https://ipwho.is/"
    ]
    for url in endpoints:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Astra-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=4, context=ssl_context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "ip": data.get("ip") or data.get("query"),
                    "cidade": data.get("city") or "Desconhecida",
                    "estado": data.get("region") or data.get("region_code") or "",
                    "pais": data.get("country_name") or data.get("country") or "",
                    "provedor": data.get("org") or data.get("connection", {}).get("isp") or "Desconhecido"
                }
        except Exception:
            continue
    return {"ip": "Indisponível", "cidade": "Indisponível", "estado": "", "pais": "", "provedor": "Indisponível"}

def formatar_bytes(b):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if b < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"

def main():
    arquivo_saida = Path("dispositivo/relatorio.txt")
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)

    conectado = verificar_conexao_internet()
    status_rede = "Conectado à Internet 🟢" if conectado else "Desconectado / Sem Acesso à Internet 🔴"
    
    hostname = socket.gethostname()
    ip_local = obter_ip_local()
    
    os_nome = platform.system()
    os_versao = platform.release()
    arquitetura = platform.machine()

    geo = {"ip": "Offline", "cidade": "-", "estado": "-", "pais": "-", "provedor": "-"}
    if conectado:
        geo = obter_dados_publicos_ip()

    ram_str = "Módulo psutil indisponível"
    disco_str = "Módulo psutil indisponível"
    
    if HAS_PSUTIL:
        try:
            mem = psutil.virtual_memory()
            ram_str = (
                f"Total: {formatar_bytes(mem.total)} | "
                f"Usada: {formatar_bytes(mem.used)} ({mem.percent}%) | "
                f"Disponível: {formatar_bytes(mem.available)}"
            )
        except Exception as e:
            ram_str = f"Erro ao ler RAM: {e}"
            
        try:
            disk = psutil.disk_usage('/')
            disco_str = (
                f"Total: {formatar_bytes(disk.total)} | "
                f"Usado: {formatar_bytes(disk.used)} ({disk.percent}%) | "
                f"Livre: {formatar_bytes(disk.free)}"
            )
        except Exception as e:
            disco_str = f"Erro ao ler disco: {e}"

    relatorio = (
        f"💻 RELATÓRIO DE STATUS DO DISPOSITIVO E REDE\n"
        f"--------------------------------------------------\n"
        f"🌐 STATUS DA REDE:\n"
        f"• Conectividade: {status_rede}\n"
        f"• IP Local (Rede Interna): {ip_local}\n"
        f"• IP Público (Externo): {geo.get('ip')}\n"
        f"• Localização aproximada: {geo.get('cidade')}, {geo.get('estado')} - {geo.get('pais')}\n"
        f"• Provedor (ISP): {geo.get('provedor')}\n\n"
        f"🖥️ DADOS DO COMPUTADOR:\n"
        f"• Nome do Host: {hostname}\n"
        f"• Sistema Operacional: {os_nome} {os_versao} ({arquitetura})\n"
        f"• Processador: {platform.processor() or 'Padrão'}\n\n"
        f"📊 HARDWARE E RECURSOS:\n"
        f"• Memória RAM: {ram_str}\n"
        f"• Disco Principal (/): {disco_str}\n"
    )

    arquivo_saida.write_text(relatorio, encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "hostname": hostname,
        "conectado": conectado,
        "ip_local": ip_local,
        "ip_publico": geo.get('ip'),
        "cidade": geo.get('cidade'),
        "status": "sucesso"
    }, ensure_ascii=True))

if __name__ == "__main__":
    main()