#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_interativo.py — Facilitador para o Sd.py (GEN + Arduino + módulo SD)

Coloque este arquivo na MESMA pasta do Sd.py.

Uso interativo (menu):
    python gen_interativo.py
    python gen_interativo.py -p COM5 -b 115200

Uso como biblioteca em outros scripts:
    from gen_interativo import GenSD

    with GenSD(porta="COM5") as sd:
        sd.listar("/GEN")
        sd.criar("nota.txt", "ola mundo")
        print(sd.ler("nota.txt"))

Observação: o firmware atual grava apenas UMA linha por arquivo
e usa nomes no formato 8.3 (se usar a SD.h clássica).
"""

import os
import sys
import argparse

try:
    import Sd  # o seu Sd.py, na mesma pasta
except ImportError as e:
    print("Não encontrei o Sd.py na mesma pasta deste arquivo.")
    print(f"Detalhe: {e}")
    sys.exit(1)

Fore, Style = Sd.Fore, Sd.Style


# ══════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPAL (usável como biblioteca)
# ══════════════════════════════════════════════════════════════════

class GenSD:
    """Mantém a conexão aberta e expõe operações simples."""

    def __init__(self, porta=None, baud=Sd.BAUDRATE_PADRAO):
        self.porta = porta
        self.baud = baud
        self.ser = None

    # ─── conexão ───────────────────────────────────────────────
    @staticmethod
    def portas_disponiveis():
        """Retorna lista de (device, descricao) de todas as portas seriais."""
        if not Sd._SERIAL_OK:
            return []
        return [(p.device, p.description or "")
                for p in Sd.serial.tools.list_ports.comports()]

    @property
    def conectado(self):
        return self.ser is not None and getattr(self.ser, "is_open", False)

    def conectar(self):
        """Detecta o Arduino/módulo SD (ou usa a porta forçada)."""
        if not Sd._SERIAL_OK:
            print(Fore.RED + "  ✗ pyserial não instalado: pip install pyserial")
            return False
        if self.conectado:
            return True
        self.ser = Sd.encontrar_arduino(self.porta, self.baud)
        if self.ser is None:
            Sd.aviso_sd_ausente()
            return False
        return True

    def desconectar(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def __enter__(self):
        self.conectar()
        return self

    def __exit__(self, *a):
        self.desconectar()

    def _ok(self):
        if not self.conectado:
            print(Fore.RED + "  ✗ Não conectado. Use a opção de conectar primeiro.")
            return False
        return True

    # ─── operações ─────────────────────────────────────────────
    def listar(self, pasta="/GEN"):
        return Sd.cmd_list(self.ser, pasta) if self._ok() else []

    def arvore(self, pasta="/GEN"):
        if self._ok():
            Sd.cmd_tree_root(self.ser, pasta)

    def ler(self, nome):
        """Retorna o conteúdo (str) ou None em caso de erro."""
        if not self._ok():
            return None
        dados = Sd.cmd_read(self.ser, nome)
        if dados is None:
            return None
        return dados.decode("utf-8", errors="replace")

    def mostrar(self, nome):
        return Sd.cmd_cat(self.ser, nome) if self._ok() else False

    def criar(self, nome, texto):
        return Sd.cmd_create(self.ser, nome, texto) if self._ok() else False

    def enviar(self, local, destino=None):
        return Sd.cmd_upload(self.ser, local, destino) if self._ok() else False

    def baixar(self, remoto, local=None):
        return Sd.cmd_download(self.ser, remoto, local) if self._ok() else False

    def copiar(self, orig, dest):
        return Sd.cmd_copy(self.ser, orig, dest) if self._ok() else False

    def apagar(self, nome):
        return Sd.cmd_delete(self.ser, nome) if self._ok() else False

    def criar_pasta(self, nome):
        return Sd.cmd_mkdir(self.ser, nome) if self._ok() else False


# ══════════════════════════════════════════════════════════════════
#  MENU INTERATIVO
# ══════════════════════════════════════════════════════════════════

MENU = [
    ("1",  "Detectar portas seriais"),
    ("2",  "Conectar ao módulo SD"),
    ("3",  "Listar diretório"),
    ("4",  "Árvore (recursiva)"),
    ("5",  "Ler arquivo"),
    ("6",  "Criar arquivo (texto, uma linha)"),
    ("7",  "Criar arquivo a partir de arquivo local"),
    ("8",  "Baixar arquivo para o PC"),
    ("9",  "Copiar arquivo no cartão"),
    ("10", "Apagar arquivo"),
    ("11", "Criar pasta"),
    ("0",  "Sair"),
]


def _entrada(msg, padrao=None):
    sufixo = f" [{padrao}]" if padrao else ""
    try:
        valor = input(Fore.CYAN + f"  {msg}{sufixo}: " + Style.RESET_ALL).strip()
    except EOFError:
        return padrao or ""
    return valor or (padrao or "")


def _mostrar_menu(sd):
    estado = (Fore.GREEN + "conectado" if sd.conectado
              else Fore.RED + "desconectado")
    print()
    print(Fore.CYAN + Style.BRIGHT + "  ═" * 31)
    print(Fore.CYAN + Style.BRIGHT + "  GEN — menu interativo   " + estado)
    print(Fore.CYAN + Style.BRIGHT + "  ═" * 31)
    for k, d in MENU:
        print(f"   {Fore.YELLOW}{Style.BRIGHT}{k:>2}{Style.RESET_ALL}  {d}")
    print()


def _acao_detectar(sd):
    portas = sd.portas_disponiveis()
    if not portas:
        print(Fore.RED + "  ✗ Nenhuma porta serial encontrada.")
        return
    print(Fore.CYAN + Style.BRIGHT + "\n  Portas encontradas:")
    for dev, desc in portas:
        print(f"   • {Fore.GREEN}{dev:<12}{Style.RESET_ALL} {desc}")


def _acao_conectar(sd):
    portas = sd.portas_disponiveis()
    if portas:
        _acao_detectar(sd)
    porta = _entrada("Porta (Enter = auto-detectar)", "")
    sd.desconectar()
    sd.porta = porta or None
    sd.conectar()


def _acao_listar(sd):
    sd.listar(_entrada("Pasta", "/GEN"))


def _acao_arvore(sd):
    sd.arvore(_entrada("Pasta", "/GEN"))


def _acao_ler(sd):
    nome = _entrada("Nome do arquivo")
    if nome:
        sd.mostrar(nome)


def _acao_criar(sd):
    nome = _entrada("Nome do arquivo (ex: nota.txt)")
    if not nome:
        return
    texto = _entrada("Conteúdo (uma linha)")
    sd.criar(nome, texto)


def _acao_criar_de_arquivo(sd):
    local = _entrada("Caminho do arquivo local")
    if not local:
        return
    if not os.path.isfile(local):
        print(Fore.RED + f"  ✗ Arquivo não encontrado: {local}")
        return
    destino = _entrada("Pasta de destino no cartão (Enter = raiz do GEN)", "")
    sd.enviar(local, destino or None)


def _acao_baixar(sd):
    remoto = _entrada("Arquivo no cartão")
    if not remoto:
        return
    local = _entrada("Salvar como (Enter = só mostrar na tela)", "")
    sd.baixar(remoto, local or None)


def _acao_copiar(sd):
    orig = _entrada("Origem")
    dest = _entrada("Destino")
    if orig and dest:
        sd.copiar(orig, dest)


def _acao_apagar(sd):
    nome = _entrada("Arquivo a apagar")
    if not nome:
        return
    if _entrada(f"Confirma apagar '{nome}'? (s/N)", "n").lower() == "s":
        sd.apagar(nome)


def _acao_mkdir(sd):
    nome = _entrada("Nome da pasta (ex: /GEN/sub)")
    if nome:
        sd.criar_pasta(nome)


ACOES = {
    "1": _acao_detectar,
    "2": _acao_conectar,
    "3": _acao_listar,
    "4": _acao_arvore,
    "5": _acao_ler,
    "6": _acao_criar,
    "7": _acao_criar_de_arquivo,
    "8": _acao_baixar,
    "9": _acao_copiar,
    "10": _acao_apagar,
    "11": _acao_mkdir,
}


def menu_interativo(porta=None, baud=Sd.BAUDRATE_PADRAO):
    Sd.mostrar_avisos_dependencia()
    if not Sd._SERIAL_OK:
        print(Fore.RED + "  Instale o pyserial: pip install pyserial")
        return

    sd = GenSD(porta, baud)
    # tenta conectar automaticamente ao abrir
    sd.conectar()

    try:
        while True:
            _mostrar_menu(sd)
            escolha = _entrada("Opção")
            if escolha == "0":
                break
            acao = ACOES.get(escolha)
            if acao is None:
                print(Fore.YELLOW + "  Opção inválida.")
                continue
            # as ações 1 e 2 não exigem conexão prévia
            if escolha not in ("1", "2") and not sd.conectado:
                print(Fore.YELLOW + "  Conectando antes de continuar...")
                if not sd.conectar():
                    continue
            try:
                acao(sd)
            except Exception as e:
                print(Fore.RED + f"  ✗ Erro: {e}")
    finally:
        sd.desconectar()
        print(Fore.YELLOW + "\n  Conexão encerrada.\n")


def main():
    ap = argparse.ArgumentParser(description="Menu interativo para o Sd.py")
    ap.add_argument("-p", "--port", help="porta serial (ex: COM5, /dev/ttyUSB0)")
    ap.add_argument("-b", "--baud", type=int, default=Sd.BAUDRATE_PADRAO,
                    help=f"baudrate (default: {Sd.BAUDRATE_PADRAO})")
    args = ap.parse_args()
    menu_interativo(args.port, args.baud)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\n  Interrompido pelo usuário.\n")
