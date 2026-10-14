"""Processo em segundo plano do agente lembrete: espera até a hora e avisa.
Uso interno: lembrete_worker.py <pasta_estado> <id>"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def avisar(msg: str) -> None:
    titulo = "Astra — lembrete"
    try:
        if os.name == "nt":
            import ctypes

            try:
                import winsound

                for f in (880, 1175):
                    winsound.Beep(f, 180)
            except Exception:
                pass
            ctypes.windll.user32.MessageBoxW(0, msg, titulo, 0x00040000 | 0x00010000 | 0x40)  # topmost + foreground + info
        elif sys.platform == "darwin":
            subprocess.run(["osascript", "-e", f'display notification {json.dumps(msg)} with title {json.dumps(titulo)} sound name "Glass"'], timeout=10)
        else:
            for cmd in (["notify-send", "-u", "critical", titulo, msg], ["zenity", "--info", "--title", titulo, "--text", msg]):
                try:
                    subprocess.run(cmd, timeout=3600 if cmd[0] == "zenity" else 10)
                    break
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
    except Exception:
        pass
    print("\a", end="", flush=True)


def main() -> None:
    pasta, rid = Path(sys.argv[1]), sys.argv[2]
    arq = pasta / "ativos.json"
    while True:
        try:
            dados = json.loads(arq.read_text(encoding="utf-8"))
        except Exception:
            return
        item = next((x for x in dados["itens"] if str(x["id"]) == rid), None)
        if not item or item.get("estado") != "ativo":
            return  # cancelado
        falta = item["quando"] - time.time()
        if falta <= 0:
            break
        time.sleep(min(falta, 5))
    avisar(item["texto"])
    try:
        dados = json.loads(arq.read_text(encoding="utf-8"))
        for x in dados["itens"]:
            if str(x["id"]) == rid and x.get("estado") == "ativo":
                x["estado"] = "disparado"
        arq.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
