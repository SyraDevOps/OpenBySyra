import os
import sys
import platform
import subprocess
import json
import urllib.request
import zipfile
import glob
import http.client
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Dependências visuais
try:
    from colorama import Fore, Style, init
    from tqdm import tqdm
    init(autoreset=True)
except ImportError:
    print("❌ Dependências ausentes! Execute no terminal:")
    print("   pip install tqdm colorama")
    sys.exit(1)

# --- CONFIGURAÇÃO ---
MODEL_NAME = "model.gguf"
INSTALL_DIR = os.path.abspath("llama")
REPO_API_LATEST = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
REPO_API_TAG = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{}"

# Portas
PROXY_PORT = 8080      # Porta que os clientes (ex: Kael) continuam usando
BACKEND_PORT = 8081    # Porta interna do llama-server
RECIVE_PORT = 2719     # API que expõe a última mensagem do modelo
LOCAL_HOST = "127.0.0.1"

# Arquivos do agente (baixados sempre que não existirem)
ADDITIONAL_DOWNLOADS = [
    {
        "url": "https://raw.githubusercontent.com/SyraDevOps/OpenBySyra/main/Portable_Agent/Gen.py",
        "dest": "Sd.py"  # Gen.py é salvo como Sd.py
    },
    {
        "url": "https://raw.githubusercontent.com/SyraDevOps/OpenBySyra/main/Portable_Agent/Kael/Kael_withAPIs.py",
        "dest": os.path.join("Kael", "Kael_withAPIs.py")
    },
]

# Modelos disponíveis (só são oferecidos se não existir model.gguf)
MODEL_OPTIONS = [
    {
        "label": "SmolLM3-3B-Base Q3_K_M (mais leve)", 
        "url": "https://huggingface.co/Syradevops/Selene-Kim-THk_05-3B-Instruct_gguf/resolve/main/SmolLM3-3B-Base.Q3_K_M.gguf",
    },
    {
        "label": "SmolLM3-3B-Base Q4_K_M (melhor qualidade)",
        "url": "https://huggingface.co/Syradevops/Selene-Kim-THk_05-3B-Instruct_gguf/resolve/main/SmolLM3-3B-Base.Q4_K_M.gguf",
    },
    {
        "label": "Selene-Rebel 4B Instruct Q3_K_M",
        "url": "https://huggingface.co/Syradevops/Selene-Rebel-08-4B-Instruct_GGUF/resolve/main/Selene-Rebel-4B-Instruct.Q3_K_M.gguf",
    },
]

# Palavras que indicam GPU INTEGRADA
INTEGRATED_KEYWORDS = [
    "intel(r) uhd", "intel(r) hd", "intel(r) iris",
    "uhd graphics", "hd graphics", "iris xe", "iris plus",
    "radeon(tm) graphics", "radeon graphics", "vega",
    "amd radeon graphics", "microsoft basic display",
    "microsoft remote display", "vmware", "virtualbox",
]

# --- ESTADO COMPARTILHADO: ÚLTIMA MENSAGEM DO MODELO ---
_last_message = {"text": "", "lock": threading.Lock()}


_last_question = {"text": "", "lock": threading.Lock()}


def set_last_message(text):
    with _last_message["lock"]:
        _last_message["text"] = text


def get_last_message():
    with _last_message["lock"]:
        return _last_message["text"]


def set_last_question(text):
    with _last_question["lock"]:
        _last_question["text"] = text


def get_last_question():
    with _last_question["lock"]:
        return _last_question["text"]


# --- HELPERS VISUAIS ---
def print_banner():
    banner = f"""{Fore.CYAN}
   _____ __   ______  ___        ____  _____
  / ___/\\ \\ / / __ \\/   |      / __ \\/ ___/
  \\__ \\  \\ V / /_/ / /| |     / / / /\\__ \\ 
 ___/ /   | / _, _/ ___ |    / /_/ /___/ / 
/____/    |/_/ |_/_/  |_|    \\____//____/  
{Fore.MAGENTA}      ⚡ AUTOMATED RUNTIME & BACKEND DETECTOR ⚡{Style.RESET_ALL}
    """
    print(banner)


def log_info(msg):
    print(f"{Fore.BLUE}ℹ️  {msg}{Style.RESET_ALL}")


def log_success(msg):
    print(f"{Fore.GREEN}✅ {msg}{Style.RESET_ALL}")


def log_warn(msg):
    print(f"{Fore.YELLOW}⚠️  {msg}{Style.RESET_ALL}")


def log_error(msg):
    print(f"{Fore.RED}❌ {msg}{Style.RESET_ALL}")


# --- DOWNLOADER COM TQDM ---
def download_file(url, dest_path):
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    filename = os.path.basename(dest_path)
    tmp_path = dest_path + ".part"  # evita deixar arquivo incompleto com o nome final
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            total_size = int(resp.info().get("Content-Length", 0))
            with open(tmp_path, "wb") as f, tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                colour="cyan",
                desc=f"{Fore.CYAN}📥 {filename[:25]:<25}{Style.RESET_ALL}"
            ) as pbar:
                while True:
                    chunk = resp.read(16384)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        os.replace(tmp_path, dest_path)
    except Exception as e:
        log_error(f"Falha ao baixar {filename}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        sys.exit(1)


# --- DETECÇÃO DO SISTEMA E HARDWARE ---
def detect_os_arch():
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        os_tag = "win"
        arch = "x64" if machine in ["amd64", "x86_64"] else "arm64" if "arm" in machine else "x64"
    elif system == "linux":
        os_tag = "ubuntu"
        arch = "x64" if machine in ["x86_64", "amd64"] else "arm64"
    elif system == "darwin":
        os_tag = "macos"
        arch = "arm64" if machine == "arm64" else "x64"
    else:
        log_error(f"SO não suportado: {system}")
        sys.exit(1)
    return os_tag, arch


def list_video_controllers():
    system = platform.system().lower()
    try:
        if system == "windows":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=20
            )
            if result.returncode == 0:
                return [l.strip() for l in result.stdout.splitlines() if l.strip()]
        elif system == "linux":
            result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return [l for l in result.stdout.splitlines() if "vga" in l.lower() or "3d controller" in l.lower()]
    except Exception:
        pass
    return []


def detect_backend(os_tag):
    if "--cpu" in sys.argv:
        log_info("CPU forçado via argumento --cpu.")
        return "cpu"
    if "--vulkan" in sys.argv:
        log_info("Vulkan forçado via argumento --vulkan.")
        return "vulkan"

    gpus = list_video_controllers()
    if not gpus:
        log_info("Nenhuma GPU detectada. Usando CPU.")
        return "cpu"

    print(f"{Fore.MAGENTA}🎮 Placa(s) de vídeo detectada(s):{Style.RESET_ALL}")
    for g in gpus:
        print(f"   {Fore.LIGHTBLACK_EX}→ {Fore.WHITE}{g}{Style.RESET_ALL}")

    all_integrated = all(
        any(k in g.lower() for k in INTEGRATED_KEYWORDS) for g in gpus
    )

    if all_integrated:
        log_info("GPU integrada detectada. Usando CPU (use --vulkan para forçar).")
        return "cpu"

    if any("nvidia" in g.lower() for g in gpus):
        try:
            subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True, timeout=10)
            log_success("NVIDIA com suporte CUDA detectada.")
            return "cuda"
        except Exception:
            log_warn("NVIDIA detectada, mas nvidia-smi falhou. Fallback para Vulkan.")

    log_success("GPU dedicada detectada, usando Vulkan.")
    return "vulkan"


# --- OBTENÇÃO DA RELEASE DO LLAMA ---
def get_nightly_tag():
    log_info("Buscando release mais recente do llama.cpp...")
    try:
        req = urllib.request.Request(REPO_API_LATEST, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        log_error(f"Erro na API GitHub (latest): {e}")
        sys.exit(1)

    for asset in data.get("assets", []):
        if asset.get("name") == "nightly-tag.txt":
            try:
                sub_req = urllib.request.Request(asset["browser_download_url"], headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(sub_req, timeout=30) as f:
                    tag = f.read().decode().strip()
                log_success(f"Versão identificada: {Fore.YELLOW}{tag}{Style.RESET_ALL}")
                return tag
            except Exception as e:
                log_error(f"Erro ao baixar nightly-tag.txt: {e}")
                sys.exit(1)

    log_error("nightly-tag.txt não encontrado.")
    sys.exit(1)


def get_download_url(tag, os_tag, arch, backend):
    log_info(f"Buscando binário para [{os_tag}-{arch} | {backend}]...")
    try:
        req = urllib.request.Request(REPO_API_TAG.format(tag), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        log_error(f"Erro na API ({tag}): {e}")
        sys.exit(1)

    assets = data.get("assets", [])
    wanted = [os_tag, arch]
    if backend != "cpu":
        wanted.append(backend)
    elif os_tag == "win":
        wanted.append("cpu")

    for a in assets:
        name = a.get("name", "").lower()
        if not name.endswith(".zip") or "cudart" in name:
            continue
        if all(p in name for p in wanted):
            return a["browser_download_url"], a["name"]

    log_error(f"Nenhum binário compatível para os filtros: {wanted}")
    sys.exit(1)


def clean_old_backends():
    patterns = ["ggml-vulkan*.dll", "ggml-cuda*.dll", "ggml-sycl*.dll",
                "ggml-hip*.dll", "ggml-opencl*.dll"]
    removed = 0
    for pat in patterns:
        for f in glob.glob(os.path.join(INSTALL_DIR, pat)):
            try:
                os.remove(f)
                removed += 1
            except Exception:
                pass
    if removed:
        log_info(f"{removed} biblioteca(s) antigas limpas.")


def setup_llama(url, filename):
    os.makedirs(INSTALL_DIR, exist_ok=True)
    zip_dest = os.path.join(INSTALL_DIR, filename)

    download_file(url, zip_dest)
    clean_old_backends()

    log_info("Extraindo binários do llama.cpp...")
    try:
        with zipfile.ZipFile(zip_dest, "r") as z:
            members = z.infolist()
            with tqdm(total=len(members), desc=f"{Fore.CYAN}📦 Extraindo{Style.RESET_ALL}", colour="green") as pbar:
                for member in members:
                    z.extract(member, INSTALL_DIR)
                    pbar.update(1)
        log_success("Extração do Llama concluída com sucesso.")
    except zipfile.BadZipFile:
        log_error("Arquivo ZIP corrompido.")
        sys.exit(1)
    finally:
        if os.path.exists(zip_dest):
            os.remove(zip_dest)


def find_server_executable():
    exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    for root, _, files in os.walk(INSTALL_DIR):
        if exe in files:
            return os.path.join(root, exe)
    return None


# --- MODELO: ESCOLHA E DOWNLOAD ---
def choose_and_download_model():
    """Se model.gguf já existe, não faz nada. Senão, pergunta qual baixar."""
    if os.path.exists(MODEL_NAME):
        log_info(f"'{MODEL_NAME}' já existe. Nenhum modelo será baixado.")
        return

    print(f"\n{Fore.LIGHTBLACK_EX}{'─'*60}{Style.RESET_ALL}")
    log_warn(f"'{MODEL_NAME}' não encontrado. Escolha um modelo para baixar:\n")
    for i, opt in enumerate(MODEL_OPTIONS, start=1):
        print(f"   {Fore.CYAN}[{i}]{Style.RESET_ALL} {opt['label']}")
    print()

    while True:
        try:
            choice = input(f"{Fore.MAGENTA}👉 Digite o número do modelo (1-{len(MODEL_OPTIONS)}): {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            log_error("Seleção cancelada.")
            sys.exit(1)
        if choice.isdigit() and 1 <= int(choice) <= len(MODEL_OPTIONS):
            selected = MODEL_OPTIONS[int(choice) - 1]
            break
        log_warn("Opção inválida, tente novamente.")

    log_info(f"Baixando: {Fore.YELLOW}{selected['label']}{Style.RESET_ALL}")
    download_file(selected["url"], MODEL_NAME)  # já salva direto como model.gguf
    log_success(f"Modelo salvo como {Fore.GREEN}{MODEL_NAME}{Style.RESET_ALL}")


# --- ARQUIVOS DO AGENTE ---
def process_additional_downloads():
    print(f"\n{Fore.LIGHTBLACK_EX}{'─'*60}{Style.RESET_ALL}")
    log_info("Verificando arquivos do agente...")

    for item in ADDITIONAL_DOWNLOADS:
        dest = item["dest"]
        if os.path.exists(dest):
            log_info(f"Arquivo já existe: {Fore.YELLOW}{dest}{Style.RESET_ALL} (Download ignorado)")
        else:
            download_file(item["url"], dest)
            log_success(f"Baixado: {Fore.YELLOW}{dest}{Style.RESET_ALL}")


# --- CAPTURA DA ÚLTIMA MENSAGEM DO MODELO ---
def _text_from_parts(value):
    """Aceita string ou lista de blocos ({'text': ...}) e devolve texto."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = []
        for b in value:
            if isinstance(b, dict):
                t = b.get("text")
                if isinstance(t, str):
                    out.append(t)
            elif isinstance(b, str):
                out.append(b)
        return "".join(out)
    return ""


def _pieces_from_obj(j):
    """Devolve (conteudo, raciocinio) de um objeto JSON em qualquer formato conhecido."""
    if not isinstance(j, dict):
        return "", ""

    # OpenAI / llama.cpp (/v1/chat/completions, /v1/completions)
    choices = j.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0] if isinstance(choices[0], dict) else {}
        d = c0.get("delta") or c0.get("message") or {}
        content = _text_from_parts(d.get("content")) or _text_from_parts(c0.get("text"))
        reasoning = _text_from_parts(d.get("reasoning_content")) or _text_from_parts(d.get("reasoning"))
        return content, reasoning

    t = j.get("type")
    # OpenAI Responses API (streaming)
    if t == "response.output_text.delta":
        return _text_from_parts(j.get("delta")), ""
    # Anthropic (streaming)
    if t == "content_block_delta":
        return _text_from_parts((j.get("delta") or {}).get("text")), ""

    # OpenAI Responses API (nao-streaming)
    if isinstance(j.get("output"), list):
        out = []
        for item in j["output"]:
            if isinstance(item, dict):
                out.append(_text_from_parts(item.get("content")))
        return "".join(out), ""

    # llama.cpp nativo (/completion) e Anthropic (nao-streaming)
    if "content" in j:
        return _text_from_parts(j.get("content")), ""

    # Estilo Ollama (/api/chat)
    if isinstance(j.get("message"), dict):
        m = j["message"]
        return _text_from_parts(m.get("content")), _text_from_parts(m.get("thinking"))

    return "", ""


def extract_message(raw_bytes, path=""):
    """Extrai o texto gerado pelo modelo (resposta normal, SSE ou NDJSON)."""
    text = raw_bytes.decode("utf-8", errors="ignore")
    objs = []

    if "data:" in text:
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                objs.append(json.loads(payload))
            except Exception:
                continue
    else:
        try:
            objs.append(json.loads(text.strip()))
        except Exception:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        objs.append(json.loads(line))
                    except Exception:
                        continue

    content_parts, reasoning_parts = [], []
    for o in objs:
        c, r = _pieces_from_obj(o)
        if c:
            content_parts.append(c)
        if r:
            reasoning_parts.append(r)

    msg = "".join(content_parts) or "".join(reasoning_parts)
    if msg.strip():
        set_last_message(msg)
        log_info(f"[recive] {path} -> mensagem capturada ({len(msg)} caracteres).")
    elif objs:
        # Resposta sem texto de modelo (ex: /tokenize); só avisa em rotas de geração
        if any(p in path for p in ("completion", "chat", "messages", "responses", "generate")):
            log_warn(f"[recive] {path} -> resposta sem texto reconhecível: {text[:200]!r}")


def extract_question(body_bytes, path=""):
    """Extrai a última pergunta do usuário do corpo da requisição."""
    if not body_bytes:
        return
    try:
        j = json.loads(body_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        return
    if not isinstance(j, dict):
        return

    question = ""
    items = j.get("messages")
    if not isinstance(items, list):
        items = j.get("input") if isinstance(j.get("input"), list) else None

    if items:
        for m in reversed(items):
            if isinstance(m, dict) and m.get("role") == "user":
                question = _text_from_parts(m.get("content"))
                if question.strip():
                    break
    else:
        for key in ("prompt", "input"):
            v = j.get(key)
            if isinstance(v, str):
                question = v
            elif isinstance(v, list) and v and isinstance(v[0], str):
                question = v[-1]
            if question.strip():
                break

    if question.strip():
        set_last_question(question)
        log_info(f"[recive] {path} -> pergunta capturada ({len(question)} caracteres).")


class ProxyHandler(BaseHTTPRequestHandler):
    """Proxy transparente 8080 -> llama-server (8081) que registra a resposta do modelo."""
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _proxy(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        if self.command == "POST" and body:
            extract_question(body, self.path)
        skip = {"host", "connection", "content-length", "accept-encoding", "transfer-encoding"}
        headers = {k: v for k, v in self.headers.items() if k.lower() not in skip}
        headers["Accept-Encoding"] = "identity"

        conn = None
        try:
            conn = http.client.HTTPConnection(LOCAL_HOST, BACKEND_PORT, timeout=900)
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as e:
            if conn:
                conn.close()
            self.send_error(502, f"llama-server indisponivel: {e}")
            return

        if self.command == "POST":
            log_info(f"[proxy] POST {self.path} -> {resp.status}")

        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()

        # Captura toda resposta POST 200 (qualquer rota de geração); o parser ignora o que não é texto
        capture = self.command == "POST" and resp.status == 200
        captured = bytearray()
        try:
            while True:
                chunk = resp.read1(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                if capture:
                    captured.extend(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

        if capture and captured:
            extract_message(bytes(captured), self.path)

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_PATCH = do_HEAD = _proxy


class ReciveHandler(BaseHTTPRequestHandler):
    """API na porta 2719: GET /recive devolve a última mensagem enviada pelo modelo."""
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        route = self.path.split("?")[0].rstrip("/")
        if route in ("/recive", "/receive"):
            data = {"question": get_last_question(), "message": get_last_message()}
        elif route == "/question":
            data = {"question": get_last_question()}
        elif route == "/message":
            data = {"message": get_last_message()}
        else:
            self.send_error(404, "Rota nao encontrada. Use /recive, /question ou /message")
            return
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_http_server(port, handler, name):
    try:
        server = ThreadingHTTPServer((LOCAL_HOST, port), handler)
    except OSError as e:
        log_error(f"Não foi possível abrir {name} na porta {port}: {e}")
        sys.exit(1)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# --- FLUXO PRINCIPAL ---
def main():
    print_banner()

    os_tag, arch = detect_os_arch()
    backend = detect_backend(os_tag)

    print(f"\n{Fore.LIGHTBLACK_EX}{'─'*60}{Style.RESET_ALL}")
    print(f"🖥️  {Fore.WHITE}Plataforma:{Style.RESET_ALL} {Fore.CYAN}{os_tag}-{arch}{Style.RESET_ALL} | "
          f"{Fore.WHITE}Backend:{Style.RESET_ALL} {Fore.YELLOW}{backend.upper()}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTBLACK_EX}{'─'*60}{Style.RESET_ALL}\n")

    # 1. Verifica/Instala llama.cpp
    server = find_server_executable()
    if server and os.path.exists(server):
        log_info(f"Instalação do Llama encontrada: {Fore.YELLOW}{server}{Style.RESET_ALL} (Download ignorado)")
    else:
        tag = get_nightly_tag()
        url, filename = get_download_url(tag, os_tag, arch, backend)
        setup_llama(url, filename)
        server = find_server_executable()

    if not server:
        log_error("Executável 'llama-server' não encontrado.")
        sys.exit(1)

    # 2. Modelo: usa model.gguf se existir, senão pergunta qual baixar
    choose_and_download_model()

    # 3. Arquivos do agente (Gen.py -> Sd.py, Kael)
    process_additional_downloads()

    # 4. Validação do modelo
    model_path = os.path.abspath(MODEL_NAME)
    if not os.path.exists(model_path):
        log_error(f"Arquivo essencial '{MODEL_NAME}' não encontrado para inicializar o servidor.")
        sys.exit(1)

    log_success(f"Executável: {Fore.CYAN}{server}{Style.RESET_ALL}")
    log_success(f"Modelo: {Fore.CYAN}{model_path}{Style.RESET_ALL}")

    if backend == "vulkan":
        os.environ["GGML_VK_PREFER_HOST_MEMORY"] = "1"
        os.environ["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] = "1"
        log_info("Ajustes de memória Vulkan aplicados.")

    # 5. APIs auxiliares (proxy 8080 + /recive na 2719)
    start_http_server(PROXY_PORT, ProxyHandler, "proxy do modelo")
    start_http_server(RECIVE_PORT, ReciveHandler, "API recive")
    log_success(f"API do modelo: {Fore.CYAN}http://{LOCAL_HOST}:{PROXY_PORT}{Style.RESET_ALL}")
    log_success(f"Última mensagem: {Fore.CYAN}http://{LOCAL_HOST}:{RECIVE_PORT}/recive{Style.RESET_ALL}")

    # 6. Iniciar llama-server (porta interna)
    cmd = [server, "-m", model_path, "--host", LOCAL_HOST, "--port", str(BACKEND_PORT)]

    print(f"\n{Fore.LIGHTBLACK_EX}{'─'*60}{Style.RESET_ALL}")
    print(f"🚀 {Fore.GREEN}Inicializando Servidor SYRA OS...{Style.RESET_ALL}")
    print(f"Comando: {Fore.LIGHTBLACK_EX}{subprocess.list2cmdline(cmd)}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTBLACK_EX}{'─'*60}{Style.RESET_ALL}\n")

    # llama-server roda em janela separada (Windows) ou em segundo plano com log em arquivo
    proc = None
    log_file = None
    try:
        kwargs = {"cwd": os.path.dirname(server)}
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            log_info("llama-server aberto em uma janela separada.")
        else:
            log_path = os.path.join(INSTALL_DIR, "llama-server.log")
            log_file = open(log_path, "w", encoding="utf-8")
            kwargs["stdout"] = log_file
            kwargs["stderr"] = subprocess.STDOUT
            log_info(f"llama-server em segundo plano. Log: {log_path}")
        proc = subprocess.Popen(cmd, **kwargs)

        print(f"\n{Fore.GREEN}✅ SYRA OS pronto.{Style.RESET_ALL}")
        print(f"   Use {Fore.CYAN}http://{LOCAL_HOST}:{PROXY_PORT}{Style.RESET_ALL} para falar com o modelo "
              f"(NÃO use a porta {BACKEND_PORT}, ela não passa pela API).")
        print(f"   Última mensagem: {Fore.CYAN}http://{LOCAL_HOST}:{RECIVE_PORT}/recive{Style.RESET_ALL}")
        print(f"   {Fore.LIGHTBLACK_EX}Ctrl+C para encerrar tudo.{Style.RESET_ALL}\n")

        while proc.poll() is None:
            time.sleep(1)

        log_error(f"llama-server encerrou com código {proc.returncode}.")
        if backend == "vulkan":
            log_warn("Dica: tente rodar com '--cpu' para contornar problemas na GPU.")
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}🛑 Encerrando SYRA OS...{Style.RESET_ALL}")
    except FileNotFoundError:
        log_error("Falha ao inicializar o executável.")
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
