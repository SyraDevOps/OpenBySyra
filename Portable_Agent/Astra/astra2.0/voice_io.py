"""
voice_io.py — Entrada e saída de voz da Astra: microfone sempre ligado, wake
word, detecção de fim de fala (VAD), transcrição local com Whisper e
resposta falada em voz alta com Piper TTS.

Fluxo de entrada (ver `VoiceLoop.listen_for_command`):
  1. Grava o microfone em pequenos frames o tempo todo (ASTRA_VOICE_MODE=1).
  2. Um VAD decide quando começa e quando termina uma fala.
  3. Essa fala é transcrita com um modelo Whisper pequeno e rápido (padrão
     "tiny", ASTRA_WHISPER_WAKE_MODEL) só para checar se contém a wake word
     — não precisa da precisão do modelo grande, só velocidade, já que roda
     em TODA fala captada. Usa um "initial_prompt" com a própria wake word
     para ajudar o Whisper a acertar quando o áudio é ambíguo.
  4. Se a wake word ("astra", por padrão) for encontrada (comparação
     tolerante a erro de transcrição — ver find_wake_word), mostra uma
     confirmação na tela e toca um "plin", e SÓ ENTÃO grava uma fala nova,
     dedicada ao pedido — transcrita com o modelo Whisper principal (padrão
     "small", mais preciso), que pode ser travado num idioma ou deixado
     "geral"/multilíngue (ASTRA_WHISPER_LANGUAGE).
  5. O texto reconhecido volta para quem chamou (main.py), que segue o
     pipeline normal (roteamento, ReAct, resposta).

Fluxo de saída (ver `PiperTTS.speak`):
  6. A resposta da Astra é limpa de markdown e falada em voz alta com uma
     das 4 vozes Piper baixadas (2 masculinas pt-BR, 2 femininas — sem voz
     feminina oficial em português, então usamos inglês para essas, como
     combinado). A voz ativa pode ser trocada a qualquer momento com
     `/voz <nome ou número>`.

Este módulo não depende de main.py e pode ser testado sozinho:
    python voice_io.py --selftest   # verifica tudo (mic, Whisper, Piper)
    python voice_io.py --setup      # baixa Whisper + as 4 vozes Piper

Dependências (não fazem parte dos requisitos "obrigatórios" do main.py;
o app cai para digitação/texto se elas não estiverem instaladas):
    pip install sounddevice faster-whisper piper-tts
Opcional, mas recomendado — VAD neural (mais preciso que o padrão por
energia em ambiente ruidoso), via ONNX Runtime puro, sem torch:
    pip install silero-vad-notorch
No Linux também é preciso a lib nativa do PortAudio:
    sudo apt install libportaudio2
No Windows/macOS o wheel do sounddevice já traz o PortAudio embutido.
O piper-tts já embute o espeak-ng-data que usa para fonemizar o texto —
não precisa instalar espeak-ng à parte.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import sounddevice as sd

    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    HAS_SOUNDDEVICE = False

try:
    from faster_whisper import WhisperModel

    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

try:
    from piper import PiperVoice
    from piper.download_voices import download_voice as _piper_download_voice

    HAS_PIPER = True
except ImportError:
    HAS_PIPER = False

try:
    from silero_vad_notorch import load_silero_vad as _load_silero_vad

    HAS_SILERO_VAD = True
except ImportError:
    HAS_SILERO_VAD = False

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


ROOT = Path(__file__).resolve().parent

# ============================================================
# CONFIG
# ============================================================

VOICE_MODE = os.getenv("ASTRA_VOICE_MODE", "1") == "1"
WAKE_WORD = os.getenv("ASTRA_WAKE_WORD", "astra").strip().casefold() or "astra"

WHISPER_MODEL_SIZE = os.getenv("ASTRA_WHISPER_MODEL", "small")
WHISPER_COMPUTE_TYPE = os.getenv("ASTRA_WHISPER_COMPUTE", "int8")
WHISPER_DEVICE = os.getenv("ASTRA_WHISPER_DEVICE", "cpu")
# Modelo separado, bem menor, usado só pra checar se a wake word foi dita
# (fase 1). Não precisa da precisão do modelo principal — só precisa ser
# rápido, já que roda em toda fala captada, o tempo todo. Usar o mesmo
# modelo "small" nas duas fases foi o principal motivo de lentidão
# percebida na detecção.
WHISPER_WAKE_MODEL_SIZE = os.getenv("ASTRA_WHISPER_WAKE_MODEL", "tiny")
# "pt" (ou outro código) trava o idioma — mais rápido e preciso para quem só
# fala uma língua. "auto"/vazio deixa o Whisper "geral": ele detecta o
# idioma a cada fala (útil pra quem alterna português/inglês).
WHISPER_LANGUAGE = os.getenv("ASTRA_WHISPER_LANGUAGE", "pt").strip().casefold()

# Pasta própria para o(s) modelo(s) do Whisper — pedido explícito do usuário
# em vez de usar o cache global (~/.cache) escondido do huggingface_hub.
MODELS_DIR = Path(os.getenv("ASTRA_MODELS_DIR", str(ROOT / "models" / "whisper")))

SAMPLE_RATE = 16000  # o Whisper espera 16 kHz mono
FRAME_MS = 32
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 512 amostras por frame (16kHz) — exigido pelo Silero VAD

MIC_DEVICE_ENV = os.getenv("ASTRA_MIC_DEVICE", "").strip()
MIC_DEVICE = int(MIC_DEVICE_ENV) if MIC_DEVICE_ENV.lstrip("-").isdigit() else None

# Quanto tempo de silêncio contínuo, após já ter havido fala, encerra a
# gravação ("até ele parar de falar").
SILENCE_HANG_MS = int(os.getenv("ASTRA_VOICE_SILENCE_MS", "900"))
MAX_UTTERANCE_S = float(os.getenv("ASTRA_VOICE_MAX_UTTERANCE_S", "20"))
MIN_SPEECH_MS = int(os.getenv("ASTRA_VOICE_MIN_SPEECH_MS", "200"))

# Piso absoluto de energia (RMS, escala 0..1) abaixo do qual consideramos que
# não há áudio real chegando (mic mudo, driver errado, dispositivo errado).
# Pisos de energia (RMS, escala 0..1) para avaliar o teste de microfone.
# Calibrados a partir de um caso real: uma sala silenciosa comum mediu
# -65 dBFS (rms 0.00056) com o microfone funcionando perfeitamente — o
# antigo piso único de 0.0008 (-62 dBFS) classificava isso como "mudo".
# Silêncio digital de verdade (mic mudo/desconectado) fica bem mais baixo,
# tipicamente abaixo de -80/-85 dBFS.
SILENT_MIC_RMS = float(os.getenv("ASTRA_VOICE_SILENT_RMS", "0.00012"))  # ~-78 dBFS: bloqueia só isto
QUIET_MIC_RMS = float(os.getenv("ASTRA_VOICE_QUIET_RMS", "0.0008"))     # ~-62 dBFS: só avisa, não bloqueia

# "auto" usa o Silero VAD (ONNX, mais preciso em ambiente ruidoso) se
# silero-vad-notorch estiver instalado; senão cai pro VAD por energia, que
# não tem dependência nenhuma. "energy" e "silero" forçam um dos dois.
VAD_BACKEND = os.getenv("ASTRA_VAD_BACKEND", "auto").strip().casefold()
VAD_THRESHOLD = float(os.getenv("ASTRA_VAD_THRESHOLD", "0.5"))

# Áudio mantido ANTES de o VAD disparar. Sem isso, o começo da palavra
# ("As-") é cortado e o Whisper ouve "Strá"/"Astrê".
PRE_ROLL_MS = int(os.getenv("ASTRA_VOICE_PRE_ROLL_MS", "320"))

# Wake word: >= WAKE_THRESHOLD aceita direto; entre WAKE_BORDERLINE e
# WAKE_THRESHOLD pede uma segunda opinião ao modelo principal.
WAKE_THRESHOLD = float(os.getenv("ASTRA_WAKE_THRESHOLD", "0.75"))
WAKE_BORDERLINE = float(os.getenv("ASTRA_WAKE_BORDERLINE", "0.60"))


def normalize_text(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(c)
    ).strip()


_PUNCT_STRIP = ".,!?;:\"'()[]{}…-"


def _lcs_length(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _indel_ratio(a: str, b: str) -> float:
    """Mesma métrica de rapidfuzz.fuzz.ratio (2*LCS/(len a+len b)), em Python puro."""
    total = len(a) + len(b)
    return 0.0 if total == 0 else 2.0 * _lcs_length(a, b) / total


def _word_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if HAS_RAPIDFUZZ:
        return _rapidfuzz_fuzz.ratio(a, b) / 100.0
    return _indel_ratio(a, b)  # idêntico ao rapidfuzz.fuzz.ratio, sem dependência


# ============================================================
# MICROFONE — CAPTURA E VERIFICAÇÃO
# ============================================================

@dataclass
class MicCheckResult:
    ok: bool
    rms: float
    dbfs: float
    device_name: str = ""
    error: str = ""
    # True só quando o dispositivo nem abre (sem hardware, sem permissão do
    # SO etc.) — nesse caso não tem sentido tentar o modo de voz. Um nível
    # de áudio baixo (quiet: True) é tratado como aviso, não bloqueio: uma
    # sala silenciosa de verdade também mede baixo nos 1.2s do teste, e o
    # teste real de que o mic funciona é se o VAD pega sua fala depois.
    hard_fail: bool = False
    quiet: bool = False


def list_input_devices() -> list[dict[str, Any]]:
    """Lista dispositivos de entrada de áudio disponíveis (para diagnóstico)."""
    if not HAS_SOUNDDEVICE:
        return []
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    return [
        {"index": i, "name": d.get("name", f"device {i}"), "channels": d.get("max_input_channels", 0)}
        for i, d in enumerate(devices)
        if d.get("max_input_channels", 0) > 0
    ]


def check_microphone(duration: float = 1.2) -> MicCheckResult:
    """
    Grava um trecho curto e mede o nível de áudio real recebido.
    É a garantia pedida de que "o mic está pegando": mede o ruído ambiente
    para calibrar o VAD e avisar se o nível estiver perto do silêncio
    digital de verdade (mudo/desconectado) — mas não trata "sala
    silenciosa" como falha, já que 1.2s sem ninguém falando naturalmente
    mede baixo mesmo com o microfone funcionando perfeitamente.
    """
    if not HAS_SOUNDDEVICE:
        return MicCheckResult(ok=False, rms=0.0, dbfs=-120.0, error="sounddevice não está instalado.", hard_fail=True)

    try:
        device_name = ""
        if MIC_DEVICE is not None:
            device_name = str(sd.query_devices(MIC_DEVICE).get("name", MIC_DEVICE))
        else:
            default_idx = sd.default.device[0]
            if default_idx is not None and default_idx >= 0:
                device_name = str(sd.query_devices(default_idx).get("name", "padrão"))

        frames = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=MIC_DEVICE,
        )
        sd.wait()
    except Exception as exc:
        return MicCheckResult(
            ok=False,
            rms=0.0,
            dbfs=-120.0,
            error=f"Não consegui abrir o microfone: {exc}",
            hard_fail=True,
        )

    audio = frames.reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    dbfs = 20 * np.log10(max(rms, 1e-9))

    # Só bloqueia mesmo quando o nível está perto do silêncio digital
    # (mudo/desconectado de verdade) — não quando é só uma sala quieta.
    hard_fail = rms < SILENT_MIC_RMS
    quiet = not hard_fail and rms < QUIET_MIC_RMS

    if hard_fail:
        error = (
            "O microfone respondeu, mas o nível de áudio captado é "
            "praticamente silêncio digital — sinal de que está mudo, no "
            "dispositivo errado, ou sem permissão do sistema operacional "
            "(ASTRA_MIC_DEVICE seleciona outro dispositivo)."
        )
    elif quiet:
        error = (
            "Nível de ruído ambiente baixo — normal em sala silenciosa, mas "
            "se a Astra não responder quando você falar, tente falar mais "
            "perto do microfone ou aumentar o ganho de entrada no sistema."
        )
    else:
        error = ""

    return MicCheckResult(
        ok=not hard_fail,
        rms=rms,
        dbfs=dbfs,
        device_name=device_name,
        error=error,
        hard_fail=hard_fail,
        quiet=quiet,
    )


# ============================================================
# VAD (DETECÇÃO DE FALA/SILÊNCIO) — SEM DEPENDÊNCIA OBRIGATÓRIA
# ============================================================

class EnergyVAD:
    """
    VAD por energia (RMS por frame) com piso de ruído calibrado no início.
    Não depende de webrtcvad; funciona em qualquer ambiente. Menos preciso
    que um VAD estatístico, mas suficiente para decidir "começou a falar" /
    "parou de falar" com debounce nos dois sentidos.
    """

    def __init__(self, noise_floor: float = 0.0015, factor: float = 3.0):
        self.threshold = max(noise_floor * factor, SILENT_MIC_RMS * 2)

    def calibrate(self, ambient_rms: float, factor: float = 3.0) -> None:
        self.threshold = max(ambient_rms * factor, SILENT_MIC_RMS * 2)

    def is_speech(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        return rms >= self.threshold

    def reset(self) -> None:
        pass  # sem estado entre falas; nada a resetar


class SileroVAD:
    """
    VAD neural (Silero, via ONNX Runtime puro — pacote `silero-vad-notorch`,
    sem depender de torch). Bem mais preciso que o VAD por energia em
    ambiente ruidoso ou com fala baixa, ao custo de uma dependência a mais
    (~15 MB, onnxruntime) e um pouco mais de CPU por frame.

    O modelo é stateful e espera frames sequenciais de exatamente
    FRAME_SAMPLES amostras a 16kHz — exatamente como record_until_silence já
    lê o microfone, então não precisa de nenhum ajuste ali.
    """

    def __init__(self, threshold: float = VAD_THRESHOLD):
        if not HAS_SILERO_VAD:
            raise RuntimeError(
                "silero-vad-notorch não está instalado. Rode: pip install silero-vad-notorch"
            )
        self._model = _load_silero_vad(onnx=True)
        self.threshold = threshold

    def calibrate(self, ambient_rms: float, factor: float = 3.0) -> None:
        # O Silero já foi treinado em milhares de horas de áudio com ruído
        # de fundo variado; ao contrário do EnergyVAD, não precisa recalibrar
        # por um piso de ruído medido na hora. Mantido só para ter a mesma
        # interface do EnergyVAD (main.py chama calibrate() sem saber qual é).
        pass

    def is_speech(self, frame: np.ndarray) -> bool:
        if frame.shape[0] != FRAME_SAMPLES:
            return False  # frame incompleto (borda de stream): trata como silêncio
        try:
            prob = float(self._model(frame, SAMPLE_RATE).item())
        except Exception:
            return False
        return prob >= self.threshold

    def reset(self) -> None:
        # Evita que o fim de uma fala "vaze" estado para o início da próxima
        # quando a mesma instância é reaproveitada em várias gravações.
        reset_fn = getattr(self._model, "reset_states", None)
        if callable(reset_fn):
            reset_fn()


def create_vad(backend: str = VAD_BACKEND):
    """
    Fábrica do VAD ativo: 'auto' prefere o Silero (ONNX) quando disponível e
    cai pro VAD por energia sem avisar erro (é só uma dependência opcional a
    mais); 'silero'/'energy' forçam explicitamente um dos dois.
    """
    backend = (backend or "auto").strip().casefold()

    if backend == "energy":
        return EnergyVAD()

    if backend == "silero":
        return SileroVAD()  # deixa o RuntimeError subir se não estiver instalado

    # auto
    if HAS_SILERO_VAD:
        try:
            return SileroVAD()
        except Exception:
            pass
    return EnergyVAD()


def record_until_silence(
    vad,
    *,
    on_listening: Callable[[], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> np.ndarray | None:
    """
    Fica com o microfone aberto e devolve o áudio de UMA fala completa:
    espera a fala começar (sem limite de tempo — "sempre escutando"),
    grava, e para quando detecta SILENCE_HANG_MS de silêncio contínuo depois
    de já ter havido fala (ou ao bater o teto de segurança MAX_UTTERANCE_S).

    Devolve None se a captura falhar, se a fala for curta demais, ou se
    cancel_event for acionado (ex.: o usuário digitou algo) — nesse caso
    a gravação é abandonada na hora.
    """
    if not HAS_SOUNDDEVICE:
        raise RuntimeError("sounddevice não está instalado.")

    reset_fn = getattr(vad, "reset", None)
    if callable(reset_fn):
        reset_fn()

    hang_frames = max(1, SILENCE_HANG_MS // FRAME_MS)
    min_speech_frames = max(1, MIN_SPEECH_MS // FRAME_MS)
    max_frames = int(MAX_UTTERANCE_S * 1000 // FRAME_MS)

    pre_roll: deque = deque(maxlen=max(1, PRE_ROLL_MS // FRAME_MS))
    buffer: list[np.ndarray] = []
    speech_frames = 0
    silence_run = 0
    started = False
    notified = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        device=MIC_DEVICE,
    ) as stream:
        total_frames = 0
        while total_frames < max_frames:
            if cancel_event is not None and cancel_event.is_set():
                return None
            frame, _overflow = stream.read(FRAME_SAMPLES)
            frame = frame.reshape(-1)
            total_frames += 1

            speaking = vad.is_speech(frame)

            if not started:
                if speaking:
                    started = True
                    buffer.extend(pre_roll)
                    buffer.append(frame)
                    speech_frames += 1
                    if on_listening and not notified:
                        on_listening()
                        notified = True
                else:
                    pre_roll.append(frame)
                continue

            buffer.append(frame)
            if speaking:
                speech_frames += 1
                silence_run = 0
            else:
                silence_run += 1
                if silence_run >= hang_frames:
                    break

    if not buffer or speech_frames < min_speech_frames:
        return None

    return np.concatenate(buffer).astype(np.float32)


# ============================================================
# CHIME ("PLIN") — CONFIRMAÇÃO SONORA DE WAKE WORD
# ============================================================

def _tone(freq: float, duration: float, sample_rate: int = SAMPLE_RATE, amplitude: float = 0.35) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    fade = max(1, int(sample_rate * 0.01))
    envelope = np.ones_like(wave)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return (wave * envelope).astype(np.float32)


def play_chime() -> None:
    """Toca um 'plin' de dois tons curtos confirmando que a wake word foi reconhecida."""
    if not HAS_SOUNDDEVICE:
        print("🔔 (plin)")
        return
    try:
        chime = np.concatenate([_tone(880.0, 0.1, amplitude=0.55), _tone(1318.5, 0.14, amplitude=0.55)])
        sd.play(chime, samplerate=SAMPLE_RATE)
        sd.wait()
    except Exception:
        # Sem saída de áudio disponível (ex.: servidor headless): degrada
        # para um aviso visual, sem quebrar o fluxo de escuta.
        print("🔔 (plin)")


# ============================================================
# TRANSCRIÇÃO (WHISPER LOCAL)
# ============================================================

class WhisperTranscriber:
    """
    Encapsula o modelo Whisper local (faster-whisper). Carrega uma única vez
    (lazy) e guarda os pesos em MODELS_DIR — pasta própria do projeto, não no
    cache global do usuário — para o download ficar visível e portátil.
    """

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        download_root: Path = MODELS_DIR,
        language: str = WHISPER_LANGUAGE,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = Path(download_root)
        self.language = language  # mutável: "/idioma" pode trocar em runtime
        self._model: Any = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not HAS_FASTER_WHISPER:
            raise RuntimeError(
                "faster-whisper não está instalado. Rode: pip install faster-whisper"
            )
        self.download_root.mkdir(parents=True, exist_ok=True)
        already_cached = any(self.download_root.rglob("*.bin")) or any(
            self.download_root.rglob("model.bin")
        )
        if not already_cached:
            print(
                f"⏬ Baixando o modelo Whisper '{self.model_size}' pela primeira vez "
                f"em {self.download_root} (pode levar alguns minutos)..."
            )
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.download_root),
            )
        except Exception as exc:
            raise RuntimeError(
                "Não foi possível carregar/baixar o modelo Whisper "
                f"'{self.model_size}': {exc}\n"
                "Se o erro menciona rede/huggingface.co, baixe com internet "
                "disponível uma vez (rode: python voice_io.py --setup) ou "
                "copie manualmente os arquivos do modelo para "
                f"{self.download_root}."
            ) from exc

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        beam_size: int = 1,
    ) -> str:
        self.ensure_loaded()
        effective_language = self.language if language is None else language
        # "auto"/vazio = Whisper "geral": deixa o modelo detectar o idioma
        # da fala em vez de travar num só.
        lang = None if effective_language in ("", "auto") else effective_language
        kwargs: dict[str, Any] = dict(
            language=lang,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            vad_filter=False,  # já fizemos VAD antes de chegar aqui
            condition_on_previous_text=False,
        )
        if hotwords:
            kwargs["hotwords"] = hotwords
        try:
            segments, _info = self._model.transcribe(audio, **kwargs)
        except TypeError:
            kwargs.pop("hotwords", None)  # faster-whisper antigo sem hotwords
            segments, _info = self._model.transcribe(audio, **kwargs)
        return " ".join(seg.text.strip() for seg in segments).strip()


# ============================================================
# SAÍDA DE VOZ (PIPER TTS)
# ============================================================
# 4 vozes oficiais do catálogo rhasspy/piper-voices: 2 masculinas em pt-BR
# (não existe voz feminina oficial em português no Piper — nem pt-BR nem
# pt-PT — então, como combinado, as 2 femininas ficam em inglês). Trocar de
# voz é só apontar outro nome deste dicionário; nenhuma delas é obrigatória
# fixa no código, dá pra escolher via ASTRA_TTS_VOICE ou o comando /voz.

PIPER_VOICES: dict[str, dict[str, str]] = {
    "pt_BR-faber-medium": {
        "gender": "masculina",
        "language": "pt-BR",
        "label": "Faber (PT-BR, masculina)",
    },
    "pt_BR-cadu-medium": {
        "gender": "masculina",
        "language": "pt-BR",
        "label": "Cadu (PT-BR, masculina)",
    },
    "en_US-amy-medium": {
        "gender": "feminina",
        "language": "en-US",
        "label": "Amy (EN-US, feminina — sem opção feminina oficial em PT)",
    },
    "en_GB-alba-medium": {
        "gender": "feminina",
        "language": "en-GB",
        "label": "Alba (EN-GB, feminina — sem opção feminina oficial em PT)",
    },
}
DEFAULT_PIPER_VOICE = "pt_BR-faber-medium"

TTS_MODE = os.getenv("ASTRA_TTS_MODE", "1") == "1"
TTS_VOICE = os.getenv("ASTRA_TTS_VOICE", DEFAULT_PIPER_VOICE).strip()
if TTS_VOICE not in PIPER_VOICES:
    TTS_VOICE = DEFAULT_PIPER_VOICE

# > 1.0 fala mais rápido, < 1.0 mais devagar (Piper usa length_scale
# invertido: length_scale = 1 / velocidade).
TTS_SPEED = float(os.getenv("ASTRA_TTS_SPEED", "1.0"))
TTS_MAX_CHARS = int(os.getenv("ASTRA_TTS_MAX_CHARS", "1200"))

PIPER_MODELS_DIR = Path(os.getenv("ASTRA_PIPER_MODELS_DIR", str(ROOT / "models" / "piper")))

_MD_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_BOLD_ITALIC = re.compile(r"(\*\*\*|\*\*|\*|___|__|_)(.*?)\1")
_MD_HEADER = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
_URL_RE = re.compile(r"https?://\S+")
_MULTI_WS = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    """
    Prepara a resposta em texto para ser falada: tira marcações de markdown
    que soam estranho lidas em voz alta (**negrito**, `código`, # títulos,
    marcadores de lista) e encurta blocos de código para uma menção curta,
    em vez de tentar "ler" código linha a linha.
    """
    text = _MD_CODE_BLOCK.sub(" (trecho de código, veja o texto na tela) ", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_BOLD_ITALIC.sub(r"\2", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _URL_RE.sub(" link ", text)
    text = _MULTI_WS.sub(" ", text).strip()
    return text


class PiperTTS:
    """
    Fala texto em voz alta com Piper (voz neural local, roda em CPU). Mantém
    as vozes já carregadas em memória (são leves, ~15-30M parâmetros) para
    trocar de voz na hora sem recarregar do disco a cada frase.
    """

    def __init__(self, voice_name: str = TTS_VOICE, models_dir: Path = PIPER_MODELS_DIR):
        self.voice_name = voice_name if voice_name in PIPER_VOICES else DEFAULT_PIPER_VOICE
        self.models_dir = Path(models_dir)
        self._voices: dict[str, Any] = {}
        self.speed = TTS_SPEED

    @staticmethod
    def list_voices() -> list[tuple[str, dict[str, str]]]:
        return list(PIPER_VOICES.items())

    def _voice_path(self, name: str) -> Path:
        return self.models_dir / f"{name}.onnx"

    def ensure_voice_downloaded(self, name: str) -> Path:
        if name not in PIPER_VOICES:
            raise ValueError(f"Voz Piper desconhecida: {name!r}. Use uma de {list(PIPER_VOICES)}.")

        onnx_path = self._voice_path(name)
        if onnx_path.is_file():
            return onnx_path

        if not HAS_PIPER:
            raise RuntimeError("piper-tts não está instalado. Rode: pip install piper-tts")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        print(f"⏬ Baixando a voz Piper '{name}' pela primeira vez em {self.models_dir}...")
        try:
            _piper_download_voice(name, self.models_dir)
        except Exception as exc:
            raise RuntimeError(
                f"Não foi possível baixar a voz Piper '{name}': {exc}\n"
                "Se o erro menciona rede/huggingface.co, garanta internet "
                "disponível uma vez (rode: python voice_io.py --setup) ou "
                f"copie manualmente {name}.onnx e {name}.onnx.json para "
                f"{self.models_dir}."
            ) from exc

        if not onnx_path.is_file():
            raise RuntimeError(f"Download da voz '{name}' terminou, mas {onnx_path} não existe.")
        return onnx_path

    def _get_voice(self, name: str):
        if name in self._voices:
            return self._voices[name]
        if not HAS_PIPER:
            raise RuntimeError("piper-tts não está instalado. Rode: pip install piper-tts")
        onnx_path = self.ensure_voice_downloaded(name)
        voice = PiperVoice.load(onnx_path)
        self._voices[name] = voice
        return voice

    def set_voice(self, name: str) -> None:
        if name not in PIPER_VOICES:
            raise ValueError(f"Voz Piper desconhecida: {name!r}.")
        self.voice_name = name
        self._get_voice(name)  # carrega/baixa já na troca, não na próxima fala

    def synthesize(self, text: str, voice_name: str | None = None) -> tuple[np.ndarray, int]:
        """Devolve (áudio float32 mono, sample_rate) sem tocar nada."""
        name = voice_name or self.voice_name
        voice = self._get_voice(name)

        length_scale = 1.0 / self.speed if self.speed > 0 else 1.0
        syn_config = None
        try:
            from piper.config import SynthesisConfig

            syn_config = SynthesisConfig(length_scale=length_scale)
        except Exception:
            pass  # fala no tempo padrão se a versão instalada não expuser isso

        chunks = list(voice.synthesize(text, syn_config=syn_config))
        if not chunks:
            return np.zeros(0, dtype=np.float32), 22050
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        return audio, chunks[0].sample_rate

    def speak(
        self,
        text: str,
        voice_name: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        """Fala o texto em voz alta (bloqueia até terminar ou até
        cancel_event ser acionado, que corta o áudio na hora). Nunca derruba
        o programa: se algo falhar, avisa, devolve False e segue — a resposta
        já foi mostrada em texto. Devolve True se terminou de falar inteira."""
        clean = clean_for_speech(text)
        if not clean:
            return False
        if len(clean) > TTS_MAX_CHARS:
            clean = clean[:TTS_MAX_CHARS].rsplit(" ", 1)[0] + "... resposta completa está no texto acima."

        if cancel_event is not None and cancel_event.is_set():
            return False

        try:
            audio, sample_rate = self.synthesize(clean, voice_name)
        except Exception as exc:
            print(f"⚠ Não consegui gerar a fala: {exc}")
            return False

        if audio.size == 0 or not HAS_SOUNDDEVICE:
            return False
        if cancel_event is not None and cancel_event.is_set():
            return False  # cortado enquanto sintetizava

        try:
            sd.play(audio, samplerate=sample_rate)
            duration = audio.size / float(sample_rate)
            deadline = time.monotonic() + duration + 1.0
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    sd.stop()
                    return False
                try:
                    if not sd.get_stream().active:
                        break
                except Exception:
                    break  # sem stream consultável: trata como terminado
                time.sleep(0.03)
            else:
                sd.stop()
        except Exception as exc:
            print(f"⚠ Não consegui tocar a fala: {exc}")
            return False

        return True


# ============================================================
# WAKE WORD
# ============================================================

def best_wake_score(transcript: str, wake_word: str = WAKE_WORD) -> float:
    """Maior similaridade (0..1) entre qualquer palavra (ou par de palavras
    coladas, "as"+"tra") da transcrição e a wake word."""
    words = [w.strip(_PUNCT_STRIP) for w in normalize_text(transcript).split()]
    words = [w for w in words if w]
    wake_norm = normalize_text(wake_word)
    best = 0.0
    for i, word in enumerate(words):
        best = max(best, _word_similarity(word, wake_norm))
        if i + 1 < len(words):
            best = max(best, _word_similarity(word + words[i + 1], wake_norm))
    return best


def find_wake_word(transcript: str, wake_word: str = WAKE_WORD, threshold: float = WAKE_THRESHOLD) -> int | None:
    """
    Devolve o índice (em palavras) onde a wake word foi encontrada, tolerando
    erros do Whisper, ou None. threshold=0.75 foi calibrado com transcrições
    reais: "astrê", "astr", "asfra", "astro" ficam em 0.80–0.89; palavras
    comuns sem relação ("outra", "agora", "extra", "estrada") em 0.55–0.67.
    """
    words = [w.strip(_PUNCT_STRIP) for w in normalize_text(transcript).split()]
    wake_norm = normalize_text(wake_word)

    for i, word in enumerate(words):
        if _word_similarity(word, wake_norm) >= threshold:
            return i
    for i in range(len(words) - 1):
        if _word_similarity(words[i] + words[i + 1], wake_norm) >= threshold:
            return i
    return None


# ============================================================
# LOOP DE VOZ (usado pelo main.py)
# ============================================================

@dataclass
class VoiceLoop:
    transcriber: WhisperTranscriber = field(default_factory=WhisperTranscriber)
    # Modelo separado e menor (padrão "tiny") só para a fase 1 (checar se a
    # wake word foi dita). Rodar o mesmo modelo grande nas duas fases era o
    # principal motivo da detecção parecer lenta.
    wake_transcriber: WhisperTranscriber = field(
        default_factory=lambda: WhisperTranscriber(model_size=WHISPER_WAKE_MODEL_SIZE)
    )
    wake_word: str = WAKE_WORD
    vad: Any = field(default_factory=create_vad)
    # Chamado para TODA fala transcrita (achando a wake word ou não) — serve
    # pra mostrar ao usuário, numa cor discreta, que o microfone está
    # realmente ouvindo, sem que isso vire um comando. is_command=True só no
    # texto que de fato vai ser enviado ao modelo.
    on_transcript: Callable[[str, bool], None] | None = None
    # Chamado assim que a wake word é reconhecida, ANTES do plim — dá uma
    # confirmação visível na hora (texto na tela), que não depende do
    # dispositivo de áudio de saída estar configurado certo. O plim é o
    # reforço sonoro; isto aqui é a garantia de que o usuário SEMPRE vê que
    # foi reconhecido, mesmo se o áudio de saída falhar silenciosamente.
    on_wake_detected: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        # Um "Astra." curto no início ajuda o Whisper a "ouvir" a wake word
        # certa quando o áudio é ambíguo, em vez de transcrever algo parecido
        # (ex.: "astrê", "asfra") — ataca a causa do problema, não só o
        # limiar de comparação.
        self.wake_prompt = f"{self.wake_word.capitalize()}."

    def calibrate(self) -> MicCheckResult:
        result = check_microphone()
        if result.ok:
            self.vad.calibrate(result.rms)
        return result

    def detect_wake_word_once(self, cancel_event: threading.Event | None = None) -> bool | None:
        """Escuta UMA fala e diz se continha a wake word.
        True = sim; False = fala sem wake word; None = nada captado/cancelado.
        Usa hotwords + beam 3 (mais preciso que o beam 1 antigo). Casos
        duvidosos (score entre WAKE_BORDERLINE e WAKE_THRESHOLD) recebem uma
        segunda opinião do modelo principal."""
        audio = record_until_silence(self.vad, cancel_event=cancel_event)
        if audio is None:
            return None
        if cancel_event is not None and cancel_event.is_set():
            return None

        transcript = self.wake_transcriber.transcribe(
            audio, initial_prompt=self.wake_prompt, hotwords=self.wake_word, beam_size=3
        )
        if not transcript:
            return None
        if self.on_transcript:
            self.on_transcript(transcript, False)

        if find_wake_word(transcript, self.wake_word) is not None:
            return True

        score = best_wake_score(transcript, self.wake_word)
        if score >= WAKE_BORDERLINE and not (cancel_event is not None and cancel_event.is_set()):
            second = self.transcriber.transcribe(
                audio, initial_prompt=self.wake_prompt, hotwords=self.wake_word, beam_size=3
            )
            if second and find_wake_word(second, self.wake_word) is not None:
                return True
        return False

    def record_command(self, cancel_event: threading.Event | None = None) -> str:
        """Grava uma fala NOVA (o pedido) e a transcreve com o modelo
        principal. Devolve "" se nada foi entendido/cancelado."""
        follow_up = record_until_silence(self.vad, cancel_event=cancel_event)
        if follow_up is None:
            return ""
        command = self.transcriber.transcribe(follow_up, beam_size=3)
        if command and self.on_transcript:
            self.on_transcript(command, True)
        return command

    def listen_for_command(self, cancel_event: threading.Event | None = None) -> str | None:
        """
        Um ciclo do "sempre escutando": (1) escuta uma fala e checa a wake
        word — se não tem, devolve None e quem chama escuta de novo; (2) SÓ
        com a wake word, avisa, toca o plim e grava uma fala NOVA, que é o
        comando. O texto da fala com a wake word nunca vira comando.
        Devolve None se cancelado (ex.: usuário digitou) ou sem wake word.
        """
        hit = self.detect_wake_word_once(cancel_event)
        if not hit:
            return None
        if self.on_wake_detected:
            self.on_wake_detected()
        play_chime()
        return self.record_command(cancel_event)


# ============================================================
# AUTOTESTE / SETUP (uso: `python voice_io.py --selftest` ou `--setup`)
# ============================================================

def run_selftest(verbose: bool = True, download_all_voices: bool = False) -> bool:
    ok = True

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("== Autoteste de voz da Astra ==")

    log(f"sounddevice instalado: {HAS_SOUNDDEVICE}")
    log(f"faster-whisper instalado: {HAS_FASTER_WHISPER}")
    log(f"silero-vad-notorch instalado (opcional): {HAS_SILERO_VAD}")
    try:
        vad_probe = create_vad()
        log(f"VAD ativo: {type(vad_probe).__name__} (ASTRA_VAD_BACKEND={VAD_BACKEND!r})")
    except Exception as exc:
        log(f"✗ Falha ao inicializar o VAD: {exc}")
        ok = False
    log(f"piper-tts instalado: {HAS_PIPER}")
    log(f"rapidfuzz instalado (opcional): {HAS_RAPIDFUZZ}")
    if not HAS_SOUNDDEVICE:
        log("✗ Instale com: pip install sounddevice (e, no Linux, sudo apt install libportaudio2)")
        ok = False
    if not HAS_FASTER_WHISPER:
        log("✗ Instale com: pip install faster-whisper")
        ok = False
    if not HAS_PIPER:
        log("✗ Instale com: pip install piper-tts")
        ok = False

    if HAS_SOUNDDEVICE:
        devices = list_input_devices()
        log(f"Dispositivos de entrada encontrados: {len(devices)}")
        for d in devices:
            log(f"  [{d['index']}] {d['name']} ({d['channels']} canal(is))")
        if not devices:
            log("✗ Nenhum dispositivo de entrada de áudio encontrado.")
            ok = False

    if HAS_SOUNDDEVICE and list_input_devices():
        mic = check_microphone()
        log(f"Verificação de microfone: rms={mic.rms:.5f} ({mic.dbfs:.1f} dBFS) device={mic.device_name!r}")
        if mic.hard_fail:
            log(f"✗ {mic.error}")
            ok = False
        elif mic.quiet:
            log(f"⚠ {mic.error}")
            log("  (isso é um aviso, não um erro — sala silenciosa mede baixo mesmo com o mic OK)")
        else:
            log("✓ Microfone está captando áudio.")

    if HAS_FASTER_WHISPER:
        try:
            transcriber = WhisperTranscriber()
            transcriber.ensure_loaded()
            lang_desc = "geral (auto-detecção)" if WHISPER_LANGUAGE in ("", "auto") else WHISPER_LANGUAGE
            log(f"✓ Modelo Whisper '{WHISPER_MODEL_SIZE}' (comando) carregado de {MODELS_DIR} (idioma: {lang_desc}).")
        except Exception as exc:
            log(f"✗ Falha ao carregar o Whisper de comando: {exc}")
            ok = False

        try:
            wake_transcriber = WhisperTranscriber(model_size=WHISPER_WAKE_MODEL_SIZE)
            wake_transcriber.ensure_loaded()
            log(f"✓ Modelo Whisper '{WHISPER_WAKE_MODEL_SIZE}' (detecção da wake word, mais rápido) carregado.")
        except Exception as exc:
            log(f"✗ Falha ao carregar o Whisper de detecção: {exc}")
            ok = False

    if HAS_PIPER:
        voices_to_check = list(PIPER_VOICES) if download_all_voices else [TTS_VOICE]
        tts = PiperTTS()
        for name in voices_to_check:
            try:
                tts.ensure_voice_downloaded(name)
                log(f"✓ Voz Piper '{name}' ({PIPER_VOICES[name]['label']}) disponível em {PIPER_MODELS_DIR}.")
            except Exception as exc:
                log(f"✗ Falha ao baixar a voz Piper '{name}': {exc}")
                ok = False
        try:
            if tts.speak("Astra pronta para conversar.", voice_name=TTS_VOICE):
                log(f"✓ Teste de fala com a voz '{TTS_VOICE}' (ouviu a frase de teste?).")
            else:
                log(f"✗ Teste de fala com a voz '{TTS_VOICE}' não tocou nada (veja avisos acima).")
                ok = False
        except Exception as exc:
            log(f"✗ Falha no teste de fala: {exc}")
            ok = False

    if HAS_SOUNDDEVICE:
        try:
            play_chime()
            log("✓ Chime de confirmação tocado (ouviu o 'plin'?).")
        except Exception as exc:
            log(f"✗ Falha ao tocar o chime: {exc}")

    log("== Resultado: " + ("OK" if ok else "PROBLEMAS ENCONTRADOS (veja acima)") + " ==")
    return ok


def main() -> None:
    args = sys.argv[1:]
    if "--setup" in args or "--selftest" in args:
        success_flag = run_selftest(download_all_voices="--setup" in args)
        sys.exit(0 if success_flag else 1)
    print(__doc__)


if __name__ == "__main__":
    main()
