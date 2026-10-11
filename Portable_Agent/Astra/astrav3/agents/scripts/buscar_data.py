import sys
import json
from datetime import datetime
from pathlib import Path

# Garante suporte a UTF-8 no Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

def main():
    # Pega a data e hora atual do sistema
    agora = datetime.now()
    
    # Formatações úteis em português do Brasil
    dias_semana = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
    dia_semana = dias_semana[agora.weekday()]
    
    data_formatada = agora.strftime("%d/%m/%Y")
    hora_formatada = agora.strftime("%H:%M:%S")
    
    # Texto amigável que será salvo e lido pela Astra/Kael
    relatorio = (
        f"📅 Informações de Data e Hora Atuais:\n"
        f"• Dia da semana: {dia_semana}\n"
        f"• Data: {data_formatada}\n"
        f"• Hora exata: {hora_formatada}\n"
    )

    # Salva na pasta de trabalho do agente
    pasta = Path("data")
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo_saida = pasta / "resultado.txt"
    arquivo_saida.write_text(relatorio, encoding="utf-8")

    # Retorna o JSON de sucesso padrão do sandbox
    print(json.dumps({
        "ok": True,
        "data": data_formatada,
        "hora": hora_formatada,
        "dia_semana": dia_semana,
        "status": "sucesso"
    }, ensure_ascii=True))

if __name__ == "__main__":
    main()