#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEN — Gerenciador de arquivos via Arduino + módulo SD
Interface ASCII colorida (colorama) + barras de progresso (tqdm) + CLI (argparse).

Mudanças desta versão:
  • Upload/Download em BLOCOS com handshake (ACK por chunk) → sem overflow
  • Parâmetro -b/--baud configurável
  • --tree faz recursão de verdade no lado do Python
  • tqdm mostra progresso real (por bloco enviado/recebido)
  • Aviso quando o nome não cabe no formato 8.3 (SD.h clássica)

Uso:
    Sd.py                                    -> mostra a arte + menu
    Sd.py -l [PASTA]                         -> lista arquivos
    Sd.py --tree [PASTA]                     -> lista recursiva (arvore real)
    Sd.py --cat NOME                         -> mostra o conteudo
    Sd.py --info                             -> info do cartao
    Sd.py -c NOME -t "conteudo"              -> cria txt
    Sd.py -c NOME -j '{"a":1}'               -> cria json
    Sd.py -c NOME -f ./local.py              -> cria a partir de arquivo local
    Sd.py -c NOME                            -> le conteudo do stdin
    Sd.py -u ./arquivo.py [-d /GEN]          -> upload
    Sd.py -g NOME [-o local.txt]             -> download (cat se sem -o)
    Sd.py --copy ORIG DEST                   -> copia
    Sd.py --delete NOME                      -> apaga
    Sd.py --mkdir /GEN/sub                   -> cria pasta
    Sd.py -p COM5 -b 9600 ...                -> forca porta e baudrate
"""

import os
import sys
import time
import argparse
import base64
import json as _json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════
#  DEPENDÊNCIAS
# ══════════════════════════════════════════════════════════════════

_faltando = []

try:
    from colorama import init as _cinit, Fore, Back, Style
    _cinit(autoreset=True)
except ImportError:
    _faltando.append("colorama")

    class _Vazio:
        def __getattr__(self, _n): return ""
    Fore = Back = Style = _Vazio()

try:
    from tqdm import tqdm
except ImportError:
    _faltando.append("tqdm")

    class _BarraFalsa:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, *a): pass
        def set_description(self, *a): pass
        def set_postfix(self, *a, **k): pass
        def close(self): pass

    def tqdm(iteravel=None, **_kw):
        return _BarraFalsa() if iteravel is None else iteravel

try:
    import serial
    import serial.tools.list_ports
    _SERIAL_OK = True
except ImportError:
    _faltando.append("pyserial")
    _SERIAL_OK = False
    serial = None


# ══════════════════════════════════════════════════════════════════
#  CONSTANTES / ARTE
# ══════════════════════════════════════════════════════════════════

BAUDRATE_PADRAO = 115200
LARGURA         = 66

# Tamanho útil do "chunk" em bytes *antes* do base64.
# 32 bytes → ~44 chars base64 + prefixo "C " + "\n" = ~48 bytes.
# Cabe folgado no buffer de 64 bytes do Uno/Nano.
CHUNK_BYTES = 32
# Caracteres base64 por linha enviada (múltiplo de 4 p/ não quebrar trios)
CHUNK_B64_CHARS = 48
TIMEOUT_ACK = 5.0

BOLINHAS = [
    "      ___           ___      ",
    "     /   \\         /   \\     ",
    "    |  O  |-------|  O  |    ",
    "     \\___/         \\___/     ",
]

GEN_ART = [
    "  ██████╗ ███████╗███╗   ██╗ ",
    " ██╔════╝ ██╔════╝████╗  ██║ ",
    " ██║  ███╗█████╗  ██╔██╗ ██║ ",
    " ██║   ██║██╔══╝  ██║╚██╗██║ ",
    " ╚██████╔╝███████╗██║ ╚████║ ",
    "  ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ",
]

SETA = [
    "              ▼              ",
    "             ▼▼▼             ",
    "            ▼▼▼▼▼            ",
    "           ▼▼▼▼▼▼▼           ",
    "          ▼▼▼▼▼▼▼▼▼          ",
]

CORES_GEN = [Fore.RED, Fore.YELLOW, Fore.GREEN,
             Fore.CYAN, Fore.BLUE, Fore.MAGENTA]


# ══════════════════════════════════════════════════════════════════
#  DESENHO
# ══════════════════════════════════════════════════════════════════

def limpar_tela():
    os.system("cls" if os.name == "nt" else "clear")


def _emitir(puro, pintado=None):
    if pintado is None:
        pintado = puro
    pad = max(0, (LARGURA - len(puro)) // 2)
    print(" " * pad + pintado)


def _padronizar(bloco, alvo):
    saida = []
    for linha in bloco:
        falta = alvo - len(linha)
        if falta > 0:
            esq = falta // 2
            linha = " " * esq + linha + " " * (falta - esq)
        saida.append(linha)
    return saida


def desenhar_logo():
    Y = Fore.YELLOW + Style.BRIGHT
    W = Fore.WHITE  + Style.BRIGHT
    C = Fore.CYAN   + Style.BRIGHT
    R = Style.RESET_ALL

    bol = _padronizar(BOLINHAS, 29)
    gen = _padronizar(GEN_ART , 29)
    set_ = _padronizar(SETA,    29)

    print()
    _emitir(bol[0], C + bol[0] + R)
    _emitir(bol[1], C + bol[1] + R)
    _emitir(bol[2], "    " + Y + "|  O  |" + W + "-------" + Y + "|  O  |" + R)
    _emitir(bol[3], C + bol[3] + R)
    print()

    for linha, cor in zip(gen, CORES_GEN):
        _emitir(linha, cor + Style.BRIGHT + linha + R)
    print()

    for linha in set_:
        _emitir(linha, Y + linha + R)
    print()


def mostrar_menu():
    B = Style.BRIGHT
    R = Style.RESET_ALL
    print(Fore.CYAN + B + "  " + "═" * 62)
    print(Fore.CYAN + B + "  GEN — Gerenciador de arquivos do módulo SD (Arduino)")
    print(Fore.CYAN + B + "  " + "═" * 62)
    print()

    def linha(cmd, desc, cor=Fore.GREEN):
        print(f"   {cor}{B}{cmd:<44}{R}{Fore.WHITE}{desc}")

    print(Fore.YELLOW + B + "  ▸ LISTAGEM")
    linha("Sd.py -l [PASTA]", "lista arquivos da pasta")
    linha("Sd.py --tree [PASTA]", "lista recursiva (arvore real)")
    linha("Sd.py --cat NOME", "mostra o conteudo de um arquivo")
    linha("Sd.py --info", "informacoes do cartao SD")
    print()

    print(Fore.YELLOW + B + "  ▸ CRIACAO")
    linha('Sd.py -c "modulo.txt" -t "ola mundo"', "cria txt com texto")
    linha("Sd.py -c \"dados.json\" -j '{\"a\":1}'", "cria arquivo JSON")
    linha("Sd.py -c \"script.py\" -f ./script.py", "cria a partir de arquivo local")
    linha('Sd.py -c "modulo.txt"', "le conteudo do stdin")
    print()

    print(Fore.YELLOW + B + "  ▸ TRANSFERENCIA")
    linha("Sd.py -u ./arquivo.py [-d /GEN]", "upload de arquivo local")
    linha("Sd.py -g \"modulo.txt\" [-o local.txt]", "download para o PC")
    linha("Sd.py --copy \"a.txt\" \"b.txt\"", "copia no cartao")
    linha("Sd.py --delete \"a.txt\"", "apaga arquivo")
    linha("Sd.py --mkdir \"/GEN/sub\"", "cria pasta")
    print()

    print(Fore.YELLOW + B + "  ▸ OPCIONAL")
    linha("Sd.py -p COM5 ...", "forca a porta serial")
    linha("Sd.py -b 9600 ...", "muda o baudrate (default: 115200)")
    linha("Sd.py -h", "mostra esta ajuda")
    print()
    print(Fore.CYAN + B + "  " + "═" * 62)
    print()


def mostrar_avisos_dependencia():
    if not _faltando:
        return
    print()
    print(Fore.RED + Style.BRIGHT + "  " + "═" * 62)
    print(Fore.RED + Style.BRIGHT + "  ⚠  MÓDULOS PYTHON AUSENTES")
    print(Fore.RED + Style.BRIGHT + "  " + "═" * 62)
    for mod in _faltando:
        print(Fore.YELLOW + Style.BRIGHT + f"   • {mod}")
    print(Fore.WHITE + "  " + "-" * 62)
    print(Fore.CYAN + "  Instale com:")
    print(Fore.GREEN + Style.BRIGHT + f"      pip install {' '.join(_faltando)}")
    print(Fore.RED + Style.BRIGHT + "  " + "═" * 62)
    print()


def aviso_sd_ausente():
    print()
    print(Fore.RED + Style.BRIGHT + "  " + "═" * 62)
    print(Fore.RED + Style.BRIGHT + "  ⚠  MÓDULO SD NÃO DETECTADO")
    print(Fore.RED + Style.BRIGHT + "  " + "═" * 62)
    print(Fore.YELLOW + "  Verifique:")
    print(Fore.YELLOW + "   • O Arduino está conectado pela USB?")
    print(Fore.YELLOW + "   • O módulo SD está ligado corretamente?")
    print(Fore.YELLOW + "   • O driver CH340 / CP210x está instalado?")
    print(Fore.YELLOW + "   • O cartão SD está inserido e formatado em FAT32?")
    print(Fore.RED + Style.BRIGHT + "  " + "═" * 62)
    print()


def aviso_8_3(nome):
    """Avisa se o nome não cabe no formato 8.3 (SD.h clássica)."""
    base = os.path.basename(nome)
    if "." in base:
        nome_sem_ext, ext = base.rsplit(".", 1)
    else:
        nome_sem_ext, ext = base, ""

    if len(nome_sem_ext) > 8 or len(ext) > 3:
        print(Fore.YELLOW + Style.BRIGHT +
              f"  ⚠ Aviso: '{base}' excede o padrão 8.3 "
              f"({len(nome_sem_ext)}.{len(ext)}).")
        print(Fore.YELLOW +
              "    Se o firmware usa a SD.h clássica, isso pode falhar.\n"
              "    Bibliotecas como SdFat.h aceitam nomes longos.\n")


# ══════════════════════════════════════════════════════════════════
#  SERIAL — CONEXÃO
# ══════════════════════════════════════════════════════════════════

def _tentar_conectar(porta, baud):
    print(Fore.CYAN + Style.BRIGHT +
          f"\n  → Testando {porta.device}  ({porta.description})  @ {baud}")
    try:
        ser = serial.Serial(port=porta.device, baudrate=baud, timeout=0.2)
    except serial.SerialException as e:
        print(Fore.RED + f"    ✗ Não foi possível abrir: {e}")
        return None

    # Auto-reset: abre a porta reinicia o Arduino -> espera o boot.
    for _ in tqdm(range(30), desc="    boot", ncols=70, leave=False,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        time.sleep(0.1)

    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    for comando in (b"LIST\n", b"HELP\n"):
        try:
            ser.write(comando)
        except Exception:
            break
        inicio = time.time()
        while time.time() - inicio < 4:
            if not ser.in_waiting:
                time.sleep(0.02)
                continue
            linha = ser.readline().decode("utf-8", errors="ignore").strip()
            if not linha:
                continue
            print(Fore.WHITE + f"    SD: {linha}")
            if linha in ("OK", "END") or linha.startswith("COMMANDS"):
                print(Fore.GREEN + Style.BRIGHT +
                      f"\n  ✓ Módulo SD detectado em {porta.device}!")
                return ser
    ser.close()
    return None


def encontrar_arduino(porta_forcada=None, baud=BAUDRATE_PADRAO):
    print(Fore.CYAN + Style.BRIGHT + "\n  Procurando módulo SD (Arduino)...\n")
    portas = list(serial.tools.list_ports.comports())
    if not portas:
        return None

    if porta_forcada:
        for p in portas:
            if p.device.lower() == porta_forcada.lower():
                return _tentar_conectar(p, baud)
        print(Fore.RED + f"  Porta {porta_forcada} não encontrada.")
        return None

    candidatas = []
    for porta in tqdm(portas, desc="  Escaneando portas USB", ncols=70,
                      leave=False,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        desc = (porta.description or "").lower()
        if any(t in desc for t in
               ("arduino", "uno", "ch340", "cp210", "usb serial")):
            candidatas.append(porta)

    if not candidatas:
        return None

    for porta in candidatas:
        ser = _tentar_conectar(porta, baud)
        if ser is not None:
            return ser
    return None


# ══════════════════════════════════════════════════════════════════
#  PROTOCOLO
# ══════════════════════════════════════════════════════════════════

def _ler_ate(ser, timeout, marcas):
    linhas = []
    inicio = time.time()
    while time.time() - inicio < timeout:
        if ser.in_waiting:
            linha = ser.readline().decode("utf-8", errors="ignore").strip()
            if not linha:
                continue
            linhas.append(linha)
            if linha in marcas:
                return linhas
            if linha.startswith("ERROR"):
                return linhas
        else:
            time.sleep(0.02)
    return linhas


def _enviar_blocos(ser, dados_bytes, descricao):
    """
    Envia 'dados_bytes' em chunks base64 com handshake ACK por bloco.
    O Arduino responde 'ACK' a cada chunk recebido, garantindo que
    nunca enchemos o buffer serial de 64 bytes do Uno/Nano.
    """
    b64 = base64.b64encode(dados_bytes).decode("ascii")
    total = len(b64) if b64 else 1

    with tqdm(total=total, desc=descricao, ncols=70, leave=False,
              unit="ch", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as p:
        pos = 0
        if not b64:
            # arquivo vazio -> só manda o terminador depois
            return

        while pos < len(b64):
            pedaco = b64[pos:pos + CHUNK_B64_CHARS]
            ser.write(f"C {pedaco}\n".encode("ascii"))

            inicio = time.time()
            ack = False
            while time.time() - inicio < TIMEOUT_ACK:
                if ser.in_waiting:
                    linha = ser.readline().decode("ascii",
                                                  errors="ignore").strip()
                    if linha == "ACK":
                        ack = True
                        break
                    if linha.startswith("ERROR"):
                        raise IOError(linha)
                else:
                    time.sleep(0.005)

            if not ack:
                raise IOError("Timeout esperando ACK do Arduino.")

            pos += len(pedaco)
            p.update(len(pedaco))


def _receber_blocos(ser, descricao):
    """
    Recebe chunks base64 do Arduino até 'EOF', respondendo 'ACK' a cada um.
    Retorna bytes decodificados.
    """
    pedacos_b64 = []
    inicio = time.time()
    total_estimado = None

    with tqdm(desc=descricao, ncols=70, leave=False, unit="ch",
              bar_format="{l_bar}{bar}| {n_fmt} blocos") as p:
        while True:
            # segura conexão "viva" mas com timeout global generoso
            if time.time() - inicio > 60:
                raise IOError("Timeout global no download.")

            if not ser.in_waiting:
                time.sleep(0.01)
                continue

            linha = ser.readline().decode("ascii", errors="ignore").strip()

            if not linha:
                continue

            if linha == "EOF":
                break
            if linha.startswith("ERROR"):
                raise IOError(linha)
            if linha.startswith("C "):
                pedacos_b64.append(linha[2:])
                p.update(1)
                ser.write(b"ACK\n")
            # linhas inesperadas: ignora silenciosamente

    b64 = "".join(pedacos_b64)
    return base64.b64decode(b64) if b64 else b""


# ─── LIST (raw) ────────────────────────────────────────────────────
def _list_raw(ser, caminho="/"):
    """Devolve lista de tuplas (tipo, nome, tam) — tipo = 'F' ou 'D'."""
    ser.reset_input_buffer()
    ser.write(f"LIST {caminho}\n".encode("utf-8"))
    linhas = _ler_ate(ser, timeout=8, marcas=("END",))
    if not linhas or linhas[-1] != "END":
        return None  # timeout

    itens = []
    for l in linhas:
        if l == "END" or l.startswith("ERROR"):
            continue
        partes = l.split()
        if partes[0] == "F" and len(partes) >= 3:
            itens.append(("F", partes[1], partes[2]))
        elif partes[0] == "D" and len(partes) >= 2:
            itens.append(("D", partes[1], "0"))
        else:
            # formato simples: só o nome — assume arquivo
            itens.append(("F", l, "?"))
    return itens


def cmd_list(ser, caminho="/"):
    itens = _list_raw(ser, caminho)
    if itens is None:
        print(Fore.RED + "  ✗ Timeout ao listar.")
        return []

    if not itens:
        print(Fore.YELLOW + "  (vazio)")
        return []

    print(Fore.CYAN + Style.BRIGHT +
          f"\n  📁 {caminho}   —   {len(itens)} item(ns)\n")
    for tipo, nome, tam in itens:
        if tipo == "D":
            print(Fore.BLUE + Style.BRIGHT + f"   📁 {nome}/")
        else:
            print(Fore.WHITE + f"   📄 {nome:<30} " +
                  Fore.YELLOW + f"{tam} bytes")
    print()
    return itens


# ─── TREE (recursivo de verdade, no lado do Python) ────────────────
def cmd_tree(ser, caminho="/", prefixo="", profundidade_max=8):
    """
    Recursão real feita no Python: pede LIST de cada diretório encontrado
    e monta a árvore visualmente aqui no PC.
    """
    if profundidade_max <= 0:
        return

    itens = _list_raw(ser, caminho)
    if itens is None:
        print(prefixo + Fore.RED + "└── <timeout>")
        return

    # ordena: pastas primeiro, depois arquivos
    itens.sort(key=lambda t: (0 if t[0] == "D" else 1, t[1].lower()))

    for i, (tipo, nome, tam) in enumerate(itens):
        ultimo = (i == len(itens) - 1)
        conector = "└── " if ultimo else "├── "
        base_pref = prefixo + conector

        if tipo == "D":
            print(base_pref + Fore.BLUE + Style.BRIGHT + f"📁 {nome}/"
                  + Style.RESET_ALL)
            novo_caminho = f"{caminho.rstrip('/')}/{nome}"
            novo_prefixo = prefixo + ("    " if ultimo else "│   ")
            cmd_tree(ser, novo_caminho, novo_prefixo, profundidade_max - 1)
        else:
            print(base_pref + Fore.WHITE + f"📄 {nome}  "
                  + Fore.YELLOW + f"({tam} bytes)" + Style.RESET_ALL)


def cmd_tree_root(ser, caminho="/"):
    print(Fore.CYAN + Style.BRIGHT + f"\n  🌳 {caminho}\n")
    cmd_tree(ser, caminho, "", profundidade_max=8)
    print()


# ─── INFO ──────────────────────────────────────────────────────────
def cmd_info(ser):
    ser.reset_input_buffer()
    ser.write(b"INFO\n")
    linhas = _ler_ate(ser, timeout=8, marcas=("END",))
    if not linhas:
        print(Fore.RED + "  ✗ Sem resposta a INFO.")
        return
    print(Fore.CYAN + Style.BRIGHT + "\n  ℹ  Informações do cartão:\n")
    for l in linhas:
        if l == "END":
            continue
        print(Fore.WHITE + f"   {l}")
    print()


# ─── CREATE (chunked) ──────────────────────────────────────────────
def cmd_create(ser, nome, conteudo):
    aviso_8_3(nome)

    if isinstance(conteudo, str):
        dados = conteudo.encode("utf-8")
        modo = "texto"
    else:
        dados = bytes(conteudo)
        modo = "binário"

    print(Fore.CYAN + Style.BRIGHT +
          f"\n  → Criando {nome}   ({len(dados)} bytes, {modo}) ...")

    ser.reset_input_buffer()
    ser.write(f"WRITE {nome}\n".encode("utf-8"))

    linhas = _ler_ate(ser, timeout=5, marcas=("READY_WRITE",))
    if "READY_WRITE" not in linhas:
        print(Fore.RED + f"  ✗ Falha: {linhas}")
        return False

    try:
        _enviar_blocos(ser, dados, "  ↑ enviando")
        # terminador
        ser.write(b"EOF\n")
        linhas = _ler_ate(ser, timeout=15, marcas=("OK",))
        if "OK" in linhas:
            print(Fore.GREEN + Style.BRIGHT +
                  f"  ✓ {nome} criado ({len(dados)} bytes).")
            return True
        print(Fore.RED + f"  ✗ Falha ao finalizar: {linhas}")
        return False
    except IOError as e:
        print(Fore.RED + f"  ✗ {e}")
        return False


# ─── READ (chunked) ────────────────────────────────────────────────
def cmd_read(ser, nome):
    ser.reset_input_buffer()
    ser.write(f"READ {nome}\n".encode("utf-8"))

    linhas = _ler_ate(ser, timeout=5, marcas=("READY_READ",))
    if "READY_READ" not in linhas:
        print(Fore.RED + f"  ✗ Falha: {linhas}")
        return None

    try:
        return _receber_blocos(ser, "  ↓ recebendo")
    except IOError as e:
        print(Fore.RED + f"  ✗ {e}")
        return None


def cmd_cat(ser, nome):
    dados = cmd_read(ser, nome)
    if dados is None:
        return False
    print(Fore.CYAN + Style.BRIGHT + f"\n  ── {nome} ──")
    try:
        print(Fore.WHITE + dados.decode("utf-8"))
    except UnicodeDecodeError:
        print(Fore.YELLOW + dados.decode("latin-1", errors="replace"))
    print(Fore.CYAN + Style.BRIGHT + f"\n  ── fim ({len(dados)} bytes) ──\n")
    return True


# ─── UPLOAD ────────────────────────────────────────────────────────
def cmd_upload(ser, local, destino=None):
    if not os.path.isfile(local):
        print(Fore.RED + f"  ✗ Arquivo local não encontrado: {local}")
        return False

    nome = os.path.basename(local)
    remoto = f"{destino.rstrip('/')}/{nome}" if destino else nome

    with open(local, "rb") as f:
        dados = f.read()

    print(Fore.CYAN + Style.BRIGHT +
          f"\n  ↑ Upload: {local}  →  {remoto}   ({len(dados)} bytes)")
    return cmd_create(ser, remoto, dados)


# ─── DOWNLOAD ──────────────────────────────────────────────────────
def cmd_download(ser, remoto, local=None):
    dados = cmd_read(ser, remoto)
    if dados is None:
        return False

    if local is None:
        try:
            print(Fore.WHITE + dados.decode("utf-8"))
        except UnicodeDecodeError:
            print(Fore.YELLOW + dados.decode("latin-1", errors="replace"))
        return True

    with open(local, "wb") as f:
        f.write(dados)

    print(Fore.GREEN + Style.BRIGHT +
          f"  ✓ Baixado: {remoto}  →  {local}  ({len(dados)} bytes)")
    return True


# ─── COPY ──────────────────────────────────────────────────────────
def cmd_copy(ser, orig, dest):
    print(Fore.CYAN + Style.BRIGHT + f"\n  → Copiando {orig} → {dest} ...")
    ser.reset_input_buffer()
    ser.write(f"COPY {orig} {dest}\n".encode("utf-8"))
    linhas = _ler_ate(ser, timeout=10, marcas=("OK",))
    if "OK" in linhas:
        print(Fore.GREEN + Style.BRIGHT + "  ✓ Copiado.")
        return True
    print(Fore.RED + f"  ✗ Falha: {linhas}")
    return False


# ─── DELETE ────────────────────────────────────────────────────────
def cmd_delete(ser, nome):
    print(Fore.CYAN + Style.BRIGHT + f"\n  → Apagando {nome} ...")
    ser.reset_input_buffer()
    ser.write(f"DELETE {nome}\n".encode("utf-8"))
    linhas = _ler_ate(ser, timeout=8, marcas=("OK",))
    if "OK" in linhas:
        print(Fore.GREEN + Style.BRIGHT + "  ✓ Apagado.")
        return True
    print(Fore.RED + f"  ✗ Falha: {linhas}")
    return False


# ─── MKDIR ─────────────────────────────────────────────────────────
def cmd_mkdir(ser, nome):
    print(Fore.CYAN + Style.BRIGHT + f"\n  → Criando pasta {nome} ...")
    ser.reset_input_buffer()
    ser.write(f"MKDIR {nome}\n".encode("utf-8"))
    linhas = _ler_ate(ser, timeout=8, marcas=("OK",))
    if "OK" in linhas:
        print(Fore.GREEN + Style.BRIGHT + "  ✓ Pasta criada.")
        return True
    print(Fore.RED + f"  ✗ Falha: {linhas}")
    return False


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def build_parser():
    ap = argparse.ArgumentParser(
        prog="Sd.py",
        description="GEN — Gerenciador de arquivos do módulo SD via Arduino.",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-h", "--help", action="store_true")

    ap.add_argument("-p", "--port", metavar="PORTA",
                    help="força a porta serial (ex: COM5 ou /dev/ttyUSB0)")
    ap.add_argument("-b", "--baud", type=int, default=BAUDRATE_PADRAO,
                    metavar="BAUD",
                    help=f"baudrate (default: {BAUDRATE_PADRAO})")

    # listagem
    ap.add_argument("-l", "--list", nargs="?", const="/", metavar="PASTA",
                    help="lista arquivos da pasta (padrão: /)")
    ap.add_argument("--tree", nargs="?", const="/", metavar="PASTA",
                    help="lista recursiva (árvore real)")
    ap.add_argument("--cat", metavar="NOME",
                    help="mostra o conteúdo de um arquivo")
    ap.add_argument("--info", action="store_true",
                    help="informações do cartão SD")

    # criação
    ap.add_argument("-c", "--create", metavar="NOME",
                    help="cria um arquivo")
    ap.add_argument("-t", "--text", metavar="TEXTO",
                    help="conteúdo textual (com -c)")
    ap.add_argument("-j", "--json", metavar="JSON",
                    help="conteúdo JSON (com -c)")
    ap.add_argument("-f", "--file", metavar="LOCAL",
                    help="arquivo local de origem (com -c)")

    # transferência
    ap.add_argument("-u", "--upload", metavar="LOCAL",
                    help="envia arquivo local")
    ap.add_argument("-d", "--dest", metavar="REMOTO",
                    help="destino remoto (upload)")
    ap.add_argument("-g", "--get", "--download", dest="download",
                    metavar="REMOTO", help="baixa arquivo remoto")
    ap.add_argument("-o", "--output", metavar="LOCAL",
                    help="caminho local de saída (download)")

    # utilitários
    ap.add_argument("--copy", nargs=2, metavar=("ORIG", "DEST"),
                    help="copia arquivo no cartão")
    ap.add_argument("--delete", metavar="NOME", help="apaga arquivo")
    ap.add_argument("--mkdir",  metavar="NOME", help="cria pasta")
    return ap


def _cmd_ajuda():
    limpar_tela()
    desenhar_logo()
    mostrar_menu()


def main():
    # sem argumentos → só a arte e o menu
    if len(sys.argv) == 1:
        _cmd_ajuda()
        return

    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        _cmd_ajuda()
        return

    mostrar_avisos_dependencia()

    if not _SERIAL_OK:
        aviso_sd_ausente()
        print(Fore.RED + Style.BRIGHT +
              "  O módulo 'pyserial' não está instalado.\n"
              "  → pip install pyserial\n")
        return

    args = build_parser().parse_args()

    if args.create and args.text is not None and args.json is not None:
        print(Fore.RED + "  ✗ Use apenas -t OU -j para o conteúdo.")
        return

    ser = encontrar_arduino(args.port, args.baud)
    if ser is None:
        aviso_sd_ausente()
        return

    try:
        # ─── LIST ───────────────────────────────────────────────
        if args.list is not None:
            cmd_list(ser, args.list)
            return

        if args.tree is not None:
            cmd_tree_root(ser, args.tree)
            return

        if args.info:
            cmd_info(ser)
            return

        if args.cat:
            cmd_cat(ser, args.cat)
            return

        # ─── CREATE ─────────────────────────────────────────────
        if args.create:
            nome = args.create

            if args.text is not None:
                conteudo = args.text

            elif args.json is not None:
                try:
                    obj = _json.loads(args.json)
                except _json.JSONDecodeError as e:
                    print(Fore.RED + f"  ✗ JSON inválido: {e}")
                    return
                conteudo = _json.dumps(obj, indent=2, ensure_ascii=False)

            elif args.file:
                if not os.path.isfile(args.file):
                    print(Fore.RED +
                          f"  ✗ Arquivo local não encontrado: {args.file}")
                    return
                with open(args.file, "rb") as fp:
                    conteudo = fp.read()

            else:
                print(Fore.CYAN +
                      "  Digite o conteúdo e finalize com Ctrl+Z (Win) "
                      "ou Ctrl+D (Linux/Mac):")
                conteudo = sys.stdin.read().rstrip("\n")

            cmd_create(ser, nome, conteudo)
            return

        # ─── UPLOAD ─────────────────────────────────────────────
        if args.upload:
            cmd_upload(ser, args.upload, args.dest)
            return

        # ─── DOWNLOAD ───────────────────────────────────────────
        if args.download:
            cmd_download(ser, args.download, args.output)
            return

        # ─── COPY ───────────────────────────────────────────────
        if args.copy:
            cmd_copy(ser, args.copy[0], args.copy[1])
            return

        # ─── DELETE ─────────────────────────────────────────────
        if args.delete:
            cmd_delete(ser, args.delete)
            return

        # ─── MKDIR ──────────────────────────────────────────────
        if args.mkdir:
            cmd_mkdir(ser, args.mkdir)
            return

        print(Fore.YELLOW +
              "  Nenhum comando reconhecido. Use -h para ajuda.")

    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\n  Interrompido pelo usuário.\n")