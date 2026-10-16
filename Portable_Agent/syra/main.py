#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SYRA — main.py
Escuta o microfone o tempo todo, transcreve cada frase (modelo rápido) e,
ao ouvir a wake word ("selene" ou termos parecidos), toca um "plin", grava
o pedido (Whisper "small"), envia ao modelo GGUF local (model.gguf) e mostra
a resposta em texto e em voz alta (Piper TTS).

Arquivos na mesma pasta:  main.py, voice_io.py, model.gguf  (Gen.py fica pra depois)

    pip install colorama tqdm sounddevice faster-whisper piper-tts llama-cpp-python
Opcional: pip install silero-vad-notorch rapidfuzz

Variáveis úteis (opcionais):
    ASTRA_WAKE_WORD=selene          palavra de ativação
    ASTRA_WHISPER_WAKE_MODEL=tiny   modelo rápido das frases soltas
    ASTRA_WHISPER_MODEL=small       modelo do comando
    ASTRA_MIC_DEVICE=<n>            índice do microfone
    ASTRA_TTS_VOICE=<voz piper>     voz da resposta falada
    SYRA_GGUF=<caminho>             outro arquivo .gguf (padrão: ./model.gguf)
    SYRA_CTX=4096                   tamanho do contexto
    SYRA_GPU_LAYERS=0               camadas na GPU (-1 = todas)
    SYRA_MAX_TOKENS=512             limite de tokens por resposta
    SYRA_SD_LOGS=1                  0 desliga os logs no cartão SD
    SYRA_SD_PORT=<porta>            força a porta do Arduino (ex.: COM5)
    SYRA_SD_BAUD=115200             baudrate do Arduino

Modelo de linguagem (escolha ao iniciar):
    python main.py                                   usa o model.gguf local
    python main.py --google gemini-2.5-flash-lite --key SUA_CHAVE
                                                     usa a API do Google Gemini
  A chave também pode vir da variável GEMINI_API_KEY (ou GOOGLE_API_KEY),
  o que evita deixá-la visível no histórico/lista de processos do sistema.

Prompt: prompts/prompt.json é criado na primeira execução (se não existir) e
relido a cada conversa — edite nome, system, regras, temperature e max_turnos
com o programa aberto e a mudança vale na próxima resposta.

Logs: cada interação vira um txt curto em /LOGS/AAMMDD/HHMMSS.TXT no cartão SD
(gravado pelo Gen.py), com timestamp, a fala do usuário (U:) e a resposta (S:).
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# A wake word precisa ser definida ANTES de importar o voice_io,
# que lê as configurações no momento do import.
os.environ.setdefault("ASTRA_WAKE_WORD", "selene")
os.environ.setdefault("ASTRA_WHISPER_MODEL", "small")
os.environ.setdefault("ASTRA_WHISPER_WAKE_MODEL", "tiny")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════
#  DEPENDÊNCIAS
# ══════════════════════════════════════════════════════════════════

try:
    from colorama import init as _cinit, Fore, Style
    _cinit(autoreset=False)
except ImportError:
    print("Falta o colorama.  →  pip install colorama tqdm")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Falta o tqdm.  →  pip install colorama tqdm")
    sys.exit(1)

try:
    import voice_io as vio
except ImportError as exc:
    print(f"Não encontrei o voice_io.py na mesma pasta: {exc}")
    sys.exit(1)

try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

# Gen.py: gerenciador do cartão SD (usado aqui só para gravar os logs).
try:
    import Gen as gen
    HAS_GEN = True
except ImportError:
    try:
        import gen  # nome em minúsculas
        HAS_GEN = True
    except ImportError:
        gen = None
        HAS_GEN = False

ROOT = Path(__file__).resolve().parent
GGUF_PATH = Path(os.getenv("SYRA_GGUF", str(ROOT / "model.gguf")))
N_CTX = int(os.getenv("SYRA_CTX", "4096"))
N_GPU_LAYERS = int(os.getenv("SYRA_GPU_LAYERS", "0"))
MAX_TOKENS = int(os.getenv("SYRA_MAX_TOKENS", "512"))
MAX_TURNOS = 6  # pares pergunta/resposta mantidos na conversa

# Logs no SD (um txt por interação, curto, com timestamp)
SD_LOGS = os.getenv("SYRA_SD_LOGS", "1") == "1"
SD_PORT = os.getenv("SYRA_SD_PORT", "").strip() or None
SD_BAUD = int(os.getenv("SYRA_SD_BAUD", "115200"))
SD_LOG_DIR = "/LOGS"          # pasta raiz dos logs no cartão
LOG_MAX_USER = 300            # caracteres máx. da fala do usuário
LOG_MAX_RESP = 600            # caracteres máx. da resposta do modelo

GOOGLE_BASE = os.getenv(
    "SYRA_GOOGLE_BASE", "https://generativelanguage.googleapis.com/v1beta"
).rstrip("/")

# Pasta de prompts: prompts/prompt.json é criado se não existir e relido a
# cada interação — dá pra editar com o programa rodando.
PROMPTS_DIR = ROOT / "prompts"
PROMPT_FILE = PROMPTS_DIR / "prompt.json"

PROMPT_PADRAO = {
    "nome": "Syra",
    "system": "Você é a Syra, uma assistente de voz.",
    "regras": [
        "Responda sempre em português do Brasil.",
        "Seja clara e curta (idealmente até 3 frases), pois a resposta será lida em voz alta.",
        "Não use markdown, listas nem emojis.",
    ],
    "temperature": 0.7,
    "max_turnos": MAX_TURNOS,
}

# ══════════════════════════════════════════════════════════════════
#  VISUAL
# ══════════════════════════════════════════════════════════════════

R = Style.RESET_ALL
B = Style.BRIGHT


def _c256(n):
    return f"\033[38;5;{n}m"


# Roxo choque (degradê do violeta ao magenta-púrpura)
ROXO = [_c256(n) for n in (93, 129, 129, 165, 165, 171)]
ROXO_FORTE = _c256(129)
ROXO_CLARO = _c256(177)
CINZA = _c256(244)

SYRA_ART = [
    " ███████╗██╗   ██╗██████╗  █████╗ ",
    " ██╔════╝╚██╗ ██╔╝██╔══██╗██╔══██╗",
    " ███████╗ ╚████╔╝ ██████╔╝███████║",
    " ╚════██║  ╚██╔╝  ██╔══██╗██╔══██║",
    " ███████║   ██║   ██║  ██║██║  ██║",
    " ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝",
]

LARGURA = 66


def limpar_tela():
    os.system("cls" if os.name == "nt" else "clear")


def desenhar_logo():
    print()
    for linha, cor in zip(SYRA_ART, ROXO):
        pad = max(0, (LARGURA - len(linha)) // 2)
        print(" " * pad + cor + B + linha + R)
    print()
    print(ROXO_CLARO + "  " + "═" * 62 + R)
    print()


def info(msg):
    print(f"{ROXO_CLARO}{B}  ▸ {R}{Fore.WHITE}{msg}{R}")


def ok(msg):
    print(f"{Fore.GREEN}{B}  ✓ {msg}{R}")


def aviso(msg):
    print(f"{Fore.YELLOW}{B}  ⚠ {msg}{R}")


def erro(msg):
    print(f"{Fore.RED}{B}  ✗ {msg}{R}")


# ══════════════════════════════════════════════════════════════════
#  VERIFICAÇÃO / DOWNLOAD DOS MODELOS
# ══════════════════════════════════════════════════════════════════

def verificar_dependencias(usar_google=False):
    faltando = []
    if not vio.HAS_SOUNDDEVICE:
        faltando.append("sounddevice")
    if not vio.HAS_FASTER_WHISPER:
        faltando.append("faster-whisper")
    if faltando:
        erro("Módulos ausentes: " + ", ".join(faltando))
        print(f"{Fore.CYAN}      pip install {' '.join(faltando)}{R}")
        if not vio.HAS_SOUNDDEVICE:
            print(f"{CINZA}      (Linux: sudo apt install libportaudio2){R}")
        return False
    ok("Dependências de áudio e Whisper encontradas.")

    if vio.HAS_SILERO_VAD:
        ok("VAD neural (Silero) disponível.")
    else:
        aviso("silero-vad-notorch ausente — usando VAD por energia "
              "(pip install silero-vad-notorch para mais precisão).")

    if not vio.HAS_PIPER:
        aviso("piper-tts ausente — as respostas não serão faladas "
              "(pip install piper-tts).")
    if not HAS_LLAMA and not usar_google:
        aviso("llama-cpp-python ausente — sem modelo GGUF "
              "(pip install llama-cpp-python).")
    return True


def detectar_gguf():
    if GGUF_PATH.is_file():
        tam = GGUF_PATH.stat().st_size / (1024 ** 3)
        ok(f"Modelo GGUF detectado: {GGUF_PATH.name}  ({tam:.2f} GB)")
        return True
    aviso(f"Não achei {GGUF_PATH.name} em {GGUF_PATH.parent} — "
          "seguindo só com transcrição.")
    return False


def preparar_modelos(usar_gguf, usar_tts):
    """Baixa (se preciso) e carrega Whisper, VAD, voz Piper e GGUF."""
    wake_t = vio.WhisperTranscriber(model_size=vio.WHISPER_WAKE_MODEL_SIZE)
    main_t = vio.WhisperTranscriber(model_size=vio.WHISPER_MODEL_SIZE)
    tts = vio.PiperTTS() if usar_tts else None
    estado = {}

    etapas = [
        (f"Whisper '{vio.WHISPER_WAKE_MODEL_SIZE}' (frases)", wake_t.ensure_loaded),
        (f"Whisper '{vio.WHISPER_MODEL_SIZE}' (comando)", main_t.ensure_loaded),
        ("VAD (detector de fala)",
         lambda: estado.__setitem__("vad", vio.create_vad())),
    ]

    if tts is not None:
        etapas.append((f"Voz '{vio.TTS_VOICE}'",
                       lambda: tts._get_voice(tts.voice_name)))

    if usar_gguf:
        def _carregar_llm():
            estado["llm"] = Llama(
                model_path=str(GGUF_PATH),
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
                verbose=False,
            )
        etapas.append(("Modelo GGUF (model.gguf)", _carregar_llm))

    print()
    with tqdm(total=len(etapas), ncols=70, colour="magenta",
              bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt}") as barra:
        for nome, fn in etapas:
            barra.set_description(f"{nome[:28]:<28}")
            try:
                fn()
            except Exception as exc:
                barra.close()
                if nome.startswith("Voz"):
                    aviso(f"Voz indisponível ({exc}) — seguindo sem fala.")
                    tts = None
                    continue
                if nome.startswith("Modelo GGUF"):
                    aviso(f"Não consegui carregar o GGUF: {exc}")
                    estado["llm"] = None
                    continue
                erro(f"Falha em {nome}: {exc}")
                return None
            barra.update(1)
    print()
    ok(f"Modelos prontos (Whisper em {vio.MODELS_DIR}).")
    return wake_t, main_t, estado["vad"], estado.get("llm"), tts


def verificar_microfone(vad):
    info("Testando o microfone (fique em silêncio por ~1s)...")
    mic = vio.check_microphone()
    if mic.hard_fail:
        erro(mic.error)
        dispositivos = vio.list_input_devices()
        if dispositivos:
            print(f"{CINZA}      Dispositivos disponíveis (use ASTRA_MIC_DEVICE=<n>):{R}")
            for d in dispositivos:
                print(f"{CINZA}        [{d['index']}] {d['name']}{R}")
        return False
    vad.calibrate(mic.rms)
    ok(f"Microfone OK: {mic.device_name or 'padrão'}  ({mic.dbfs:.1f} dBFS)")
    if mic.quiet:
        aviso(mic.error)
    return True


# ══════════════════════════════════════════════════════════════════
#  MODELO + VOZ
# ══════════════════════════════════════════════════════════════════

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ══════════════════════════════════════════════════════════════════
#  PROMPT (prompts/prompt.json)
# ══════════════════════════════════════════════════════════════════
# Campos (todos opcionais; o que faltar usa o padrão):
#   nome         nome do assistente (informativo)
#   system       texto do prompt de sistema (string ou lista de linhas)
#   regras       lista de regras, anexadas ao prompt de sistema
#   temperature  criatividade, 0 a 2
#   max_turnos   quantos pares pergunta/resposta recentes o modelo vê

class PromptConfig:
    def __init__(self, caminho=PROMPT_FILE):
        self.caminho = Path(caminho)
        self._mtime = None
        self._cfg = dict(PROMPT_PADRAO)
        self.avisos = []

    # -- arquivo ----------------------------------------------------
    def garantir(self):
        """Cria a pasta e o prompt.json se não existirem. Devolve True se criou."""
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        if self.caminho.is_file():
            return False
        with open(self.caminho, "w", encoding="utf-8") as f:
            json.dump(PROMPT_PADRAO, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return True

    def _validar(self, dados):
        """Aplica os campos válidos por cima do padrão; anota avisos."""
        cfg = dict(PROMPT_PADRAO)
        avisos = []
        if not isinstance(dados, dict):
            return cfg, ["o JSON precisa ser um objeto { ... }"]

        if "nome" in dados:
            if isinstance(dados["nome"], str) and dados["nome"].strip():
                cfg["nome"] = dados["nome"].strip()
            else:
                avisos.append("'nome' inválido")

        if "system" in dados:
            s = dados["system"]
            if isinstance(s, list) and all(isinstance(x, str) for x in s):
                s = "\n".join(s)
            if isinstance(s, str) and s.strip():
                cfg["system"] = s.strip()
            else:
                avisos.append("'system' deve ser texto (ou lista de textos)")

        if "regras" in dados:
            r = dados["regras"]
            if isinstance(r, list) and all(isinstance(x, str) for x in r):
                cfg["regras"] = [x.strip() for x in r if x.strip()]
            else:
                avisos.append("'regras' deve ser uma lista de textos")

        if "temperature" in dados:
            t = dados["temperature"]
            if isinstance(t, (int, float)) and not isinstance(t, bool) and 0 <= t <= 2:
                cfg["temperature"] = float(t)
            else:
                avisos.append("'temperature' deve ser um número entre 0 e 2")

        if "max_turnos" in dados:
            n = dados["max_turnos"]
            if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 50:
                cfg["max_turnos"] = n
            else:
                avisos.append("'max_turnos' deve ser um inteiro entre 1 e 50")

        return cfg, avisos

    def recarregar(self):
        """Lê o arquivo. Se estiver quebrado, mantém o padrão e NÃO o sobrescreve."""
        try:
            with open(self.caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)
        except FileNotFoundError:
            self._cfg, self.avisos = dict(PROMPT_PADRAO), ["arquivo não encontrado"]
        except (ValueError, OSError) as exc:
            self._cfg = dict(PROMPT_PADRAO)
            self.avisos = [f"JSON inválido ({exc}) — usando o prompt padrão"]
        else:
            self._cfg, self.avisos = self._validar(dados)
        try:
            self._mtime = self.caminho.stat().st_mtime_ns
        except OSError:
            self._mtime = None

    def atual(self):
        """Config em vigor; relê o arquivo se ele mudou. Devolve (cfg, mudou)."""
        try:
            mtime = self.caminho.stat().st_mtime_ns
        except OSError:
            mtime = None
        mudou = mtime != self._mtime
        if mudou:
            self.recarregar()
        return self._cfg, mudou

    # -- uso --------------------------------------------------------
    @staticmethod
    def texto_sistema(cfg):
        partes = [cfg["system"]]
        if cfg["regras"]:
            partes.append("Regras:\n" + "\n".join(f"- {r}" for r in cfg["regras"]))
        return "\n\n".join(partes)


PROMPT = PromptConfig()


def preparar_prompts():
    """Na partida: cria/verifica prompts/prompt.json e informa o resultado."""
    try:
        criado = PROMPT.garantir()
    except OSError as exc:
        aviso(f"Não consegui criar {PROMPT_FILE}: {exc} — usando o prompt padrão.")
        return
    PROMPT.recarregar()
    rel = PROMPT_FILE.relative_to(ROOT)
    if criado:
        ok(f"Pasta de prompts criada: {rel} (edite à vontade, é relido a cada conversa).")
    else:
        ok(f"Prompt carregado: {rel}  (assistente: {PROMPT._cfg['nome']})")
    for a in PROMPT.avisos:
        aviso(f"prompt.json: {a}")


class GGUFBackend:
    """Modelo local (model.gguf) via llama-cpp-python."""

    def __init__(self, llm, nome="model.gguf"):
        self.llm = llm
        self.nome = nome

    def stream(self, mensagens, temperature=0.7):
        for chunk in self.llm.create_chat_completion(
            messages=mensagens,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            stream=True,
        ):
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta


class GoogleBackend:
    """API do Google Gemini (REST + streaming SSE), sem dependências extras."""

    def __init__(self, modelo, chave):
        modelo = modelo.strip()
        if modelo.startswith("models/"):
            modelo = modelo[len("models/"):]
        self.modelo = modelo
        self.chave = chave
        self.nome = modelo
        self._modelo_url = urllib.parse.quote(modelo, safe="-._")

    def _abrir(self, url, dados=None):
        corpo = None if dados is None else json.dumps(dados).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=corpo,
            method="POST" if corpo is not None else "GET",
            # A chave vai no cabeçalho (não na URL) para não aparecer em logs.
            headers={"x-goog-api-key": self.chave,
                     "Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._msg_erro(exc)) from None
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"sem conexão com a API do Google ({exc.reason})") from None

    @staticmethod
    def _msg_erro(exc):
        try:
            msg = json.loads(exc.read().decode("utf-8", "ignore"))["error"]["message"]
        except Exception:
            msg = exc.reason or "erro desconhecido"
        return f"Google respondeu {exc.code}: {msg}"

    def verificar(self):
        """Confere se a chave e o nome do modelo são válidos (levanta RuntimeError)."""
        with self._abrir(f"{GOOGLE_BASE}/models/{self._modelo_url}") as resp:
            resp.read()

    def stream(self, mensagens, temperature=0.7):
        sistema = [m["content"] for m in mensagens if m["role"] == "system"]
        conteudos = [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in mensagens if m["role"] != "system"
        ]
        corpo = {"contents": conteudos,
                 "generationConfig": {"temperature": temperature}}
        if sistema:
            corpo["systemInstruction"] = {
                "parts": [{"text": "\n".join(sistema)}]}

        url = (f"{GOOGLE_BASE}/models/{self._modelo_url}"
               ":streamGenerateContent?alt=sse")
        with self._abrir(url, corpo) as resp:
            for bruto in resp:
                linha = bruto.decode("utf-8", errors="ignore").strip()
                if not linha.startswith("data:"):
                    continue
                try:
                    obj = json.loads(linha[5:].strip())
                except ValueError:
                    continue
                bloqueio = (obj.get("promptFeedback") or {}).get("blockReason")
                if bloqueio:
                    raise RuntimeError(f"pedido bloqueado pelo Google ({bloqueio})")
                for cand in obj.get("candidates", []):
                    for parte in (cand.get("content") or {}).get("parts", []):
                        texto = parte.get("text")
                        if texto and not parte.get("thought"):
                            yield texto


# ══════════════════════════════════════════════════════════════════
#  LOGS NO CARTÃO SD (via Gen.py)
# ══════════════════════════════════════════════════════════════════
# Estrutura:  /LOGS/AAMMDD/HHMMSS.TXT   (nomes 8.3, compatíveis com SD.h)
# Cada arquivo guarda UMA interação (usuário + modelo), em poucas linhas,
# para consultar depois sem precisar carregar conversas longas no contexto.

_dirs_sd_criados = set()
_ultimo_nome_log = [""]   # AAMMDDHHMMSS do último arquivo gravado


def _compactar(texto, limite):
    """Uma linha só, sem espaços repetidos, cortada no limite."""
    texto = re.sub(r"\s+", " ", texto or "").strip()
    if len(texto) > limite:
        texto = texto[:limite - 3].rstrip() + "..."
    return texto


def _quieto(fn, *args):
    """Roda uma função do Gen.py sem deixar ela poluir o terminal."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*args)


def conectar_sd():
    """Procura o Arduino/SD pelo Gen.py. Devolve a serial ou None."""
    if not SD_LOGS:
        info("Logs no SD desativados (SYRA_SD_LOGS=0).")
        return None
    if not HAS_GEN:
        aviso("Gen.py não encontrado na pasta — sem logs no SD.")
        return None
    if not getattr(gen, "_SERIAL_OK", False):
        aviso("pyserial ausente — sem logs no SD (pip install pyserial).")
        return None

    info("Procurando o módulo SD para gravar os logs...")
    try:
        ser = gen.encontrar_arduino(SD_PORT, SD_BAUD)
    except Exception as exc:
        aviso(f"Erro ao procurar o SD: {exc}")
        return None
    if ser is None:
        aviso("Módulo SD não detectado — seguindo sem logs.")
        return None
    ok("Logs serão gravados em " + SD_LOG_DIR + "/AAMMDD/HHMMSS.TXT")
    return ser


def _garantir_pasta(ser, caminho):
    if caminho in _dirs_sd_criados:
        return
    # Se já existir, o MKDIR pode "falhar" — não tem problema.
    _quieto(gen.cmd_mkdir, ser, caminho)
    _dirs_sd_criados.add(caminho)


def gravar_log_sd(ser, pergunta, resposta, modelo=""):
    """Grava UMA interação como um txt curto no SD. Nunca derruba o app."""
    if ser is None:
        return False
    from datetime import datetime, timedelta
    agora = datetime.now()
    nome_dt = agora
    # Dois logs no mesmo segundo não podem ter o mesmo nome (sobrescreveria):
    # o timestamp dentro do txt continua o real, só o nome avança 1s.
    while f"{nome_dt:%y%m%d%H%M%S}" <= _ultimo_nome_log[0]:
        nome_dt += timedelta(seconds=1)
    _ultimo_nome_log[0] = f"{nome_dt:%y%m%d%H%M%S}"
    pasta = f"{SD_LOG_DIR}/{nome_dt:%y%m%d}"
    arquivo = f"{pasta}/{nome_dt:%H%M%S}.TXT"

    conteudo = (
        f"{agora:%Y-%m-%d %H:%M:%S}"
        + (f" [{_compactar(modelo, 40)}]" if modelo else "")
        + "\n"
        f"U:{_compactar(pergunta, LOG_MAX_USER)}\n"
        f"S: {_compactar(resposta, LOG_MAX_RESP)}\n"
    )

    try:
        _garantir_pasta(ser, SD_LOG_DIR)
        _garantir_pasta(ser, pasta)
        gravado = _quieto(gen.cmd_create, ser, arquivo, conteudo)
    except Exception as exc:
        aviso(f"Falha ao gravar log no SD: {exc}")
        return False

    if gravado:
        print(f"{CINZA}  💾 log salvo: {arquivo}{R}")
        return True
    aviso(f"Não consegui salvar o log em {arquivo}.")
    return False


def responder(backend, tts, historico, pergunta, ser_sd=None):
    """Envia o pedido ao modelo (GGUF ou Google), mostra a resposta em
    streaming, fala em voz alta e grava o log no SD."""
    # prompts/prompt.json é relido se foi editado desde a última conversa.
    cfg, mudou = PROMPT.atual()
    if mudou:
        info("prompt.json atualizado — usando a nova versão.")
        for a in PROMPT.avisos:
            aviso(f"prompt.json: {a}")

    historico.append({"role": "user", "content": pergunta})
    mensagens = [{"role": "system", "content": PromptConfig.texto_sistema(cfg)}]
    mensagens += historico[-cfg["max_turnos"] * 2:]

    print(f"\n{ROXO_FORTE}{B}  ◆ {cfg['nome']}: {R}", end="", flush=True)
    partes = []
    try:
        for delta in backend.stream(mensagens, cfg["temperature"]):
            partes.append(delta)
            print(Fore.WHITE + delta + R, end="", flush=True)
    except Exception as exc:
        print()
        erro(f"Erro no modelo: {exc}")
        historico.pop()  # descarta a pergunta que não teve resposta
        return
    print("\n")

    resposta = _THINK_RE.sub("", "".join(partes)).strip()
    if not resposta:
        aviso("O modelo não devolveu texto.")
        historico.pop()
        return

    historico.append({"role": "assistant", "content": resposta})

    if tts is not None:
        info("Falando a resposta...")
        tts.speak(resposta)

    gravar_log_sd(ser_sd, pergunta, resposta, backend.nome)


# ══════════════════════════════════════════════════════════════════
#  LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def on_transcript(texto, is_command):
    if is_command:
        print(f"{ROXO_FORTE}{B}  ✎ Pedido (small): {R}{Fore.WHITE}{B}{texto}{R}")
    else:
        print(f"{CINZA}  … ouvi: {texto}{R}")


def on_wake():
    print(f"\n{ROXO_FORTE}{B}  ★ '{vio.WAKE_WORD}' reconhecida! "
          f"Pode falar...{R}")


def loop_escuta(wake_t, main_t, vad, backend, tts, ser_sd=None):
    voz = vio.VoiceLoop(
        transcriber=main_t,
        wake_transcriber=wake_t,
        wake_word=vio.WAKE_WORD,
        vad=vad,
        on_transcript=on_transcript,
        on_wake_detected=on_wake,
    )
    historico = []

    print()
    print(ROXO_CLARO + B + "  " + "═" * 62 + R)
    print(f"{ROXO_CLARO}{B}  Escutando... diga '{vio.WAKE_WORD}' para chamar. "
          f"Ctrl+C para sair.{R}")
    print(ROXO_CLARO + B + "  " + "═" * 62 + R)
    print()

    while True:
        comando = voz.listen_for_command()
        if comando is None:
            continue  # sem wake word: segue escutando
        if not comando:
            aviso("Não entendi o pedido.")
            continue
        if backend is not None:
            responder(backend, tts, historico, comando, ser_sd)
        print()


def ler_argumentos():
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="SYRA — assistente de voz (wake word, Whisper, "
                    "modelo local GGUF ou Google Gemini, Piper TTS).",
    )
    ap.add_argument("--google", metavar="MODELO",
                    help="usa a API do Google Gemini em vez do model.gguf "
                         "(ex.: gemini-2.5-flash-lite)")
    ap.add_argument("--key", metavar="CHAVE",
                    help="chave da API do Google (ou defina GEMINI_API_KEY / "
                         "GOOGLE_API_KEY)")
    return ap.parse_args()


def main():
    args = ler_argumentos()

    limpar_tela()
    desenhar_logo()

    usar_google = bool(args.google)
    chave = (args.key or os.getenv("GEMINI_API_KEY")
             or os.getenv("GOOGLE_API_KEY") or "").strip()

    if usar_google and not chave:
        erro("--google precisa de uma chave: use --key SUA_CHAVE "
             "ou defina GEMINI_API_KEY.")
        return
    if args.key and not usar_google:
        aviso("--key ignorada: sem --google o modelo usado é o model.gguf.")

    if not verificar_dependencias(usar_google):
        return

    preparar_prompts()

    backend = None
    if usar_google:
        backend = GoogleBackend(args.google, chave)
        info(f"Verificando o modelo Google '{backend.modelo}'...")
        try:
            backend.verificar()
        except RuntimeError as exc:
            erro(str(exc))
            return
        ok(f"Google Gemini pronto: {backend.modelo}")

    usar_gguf = (not usar_google) and HAS_LLAMA and detectar_gguf()
    usar_tts = vio.HAS_PIPER

    preparados = preparar_modelos(usar_gguf, usar_tts)
    if preparados is None:
        return
    wake_t, main_t, vad, llm, tts = preparados

    if backend is None and llm is not None:
        backend = GGUFBackend(llm)

    if not verificar_microfone(vad):
        return

    # Só vale conectar ao SD se houver modelo respondendo (nada a logar sem ele).
    ser_sd = conectar_sd() if backend is not None else None

    try:
        loop_escuta(wake_t, main_t, vad, backend, tts, ser_sd)
    finally:
        if ser_sd is not None:
            try:
                ser_sd.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{ROXO_CLARO}  Até logo!{R}\n")
