import sys
import json
import re
import urllib.request
import urllib.parse
import webbrowser
from pathlib import Path

def limpar_termo_busca(texto_bruto: str) -> str:
    """Remove palavras-comando comuns da Astra de forma inteligente, preservando o assunto."""
    termo = str(texto_bruto).strip()
    
    # Se a IA passou a string inteira de comando, removemos verbos de ação comuns
    padroes_remover = [
        r"\b(rode|o|a|os|as|um|uma|agente|youtube|pesquisa|pesquise|pesquisar|abra|abrir|procure|procurar|mostre|mostrar|vídeo|video|videos|vídeos|resultado|resultados|sobre|de|do|da|das|dos|com|e)\b"
    ]
    
    termo_limpo = termo
    for padrao in padroes_remover:
        termo_limpo = re.sub(padrao, "", termo_limpo, flags=re.IGNORECASE)
    
    # Limpa espaços duplos e pontuações excedentes
    termo_limpo = re.sub(r"\s+", " ", termo_limpo).strip(" .,:;!?\t\r\n")
    
    # Se sobrou algo útil, usa; caso contrário, usa o texto original tratado ou fallback seguro
    if len(termo_limpo) > 1:
        return termo_limpo
    
    # Fallback caso a limpeza remova tudo por engano
    return termo.strip() or "música relaxante"

def obter_id_primeiro_video(termo_busca: str) -> str | None:
    """Busca no YouTube e extrai o ID do primeiro vídeo usando múltiplas estratégias de regex."""
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(termo_busca)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Cookie": "CONSENT=YES+cb.20210328-04-p0.en+FX+435; SOCS=CAI"
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")
        
        # Estratégia 1: Padrão JSON interno do YouTube
        matches = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
        if not matches:
            # Estratégia 2: Links de watch tradicionais no HTML
            matches = re.findall(r'/watch\?v=([a-zA-Z0-9_-]{11})', html)
        if not matches:
            # Estratégia 3: Links encurtados ou embeds
            matches = re.findall(r'youtu\.be/([a-zA-Z0-9_-]{11})', html)
            
        if matches:
            # Retorna o primeiro ID único válido encontrado
            for m in matches:
                if len(m) == 11:
                    return m
    except Exception:
        pass
    return None

def main():
    try:
        raw_in = sys.stdin.read().strip()
        data = json.loads(raw_in) if raw_in else {}
    except Exception:
        data = {}

    contexto_bruto = str(data.get("contexto", "")).strip()
    termo = limpar_termo_busca(contexto_bruto)

    # Busca o ID do 1º vídeo
    video_id = obter_id_primeiro_video(termo)

    if video_id:
        url_final = f"https://www.youtube.com/watch?v={video_id}"
        msg = f"Vídeo sobre '{termo}' encontrado com sucesso!\nLink direto: {url_final}"
        status = "sucesso"
    else:
        # Fallback de segurança absoluto se a extração falhar
        url_final = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(termo)}"
        msg = f"Não foi possível extrair o link direto do primeiro vídeo, mas a busca está pronta.\nLink de resultados: {url_final}"
        status = "fallback_busca"

    # Tentativa segura de abrir o navegador (captura exceções caso rode em servidores headless/Docker)
    navegador_aberto = False
    try:
        navegador_aberto = webbrowser.open(url_final)
    except Exception:
        navegador_aberto = False

    if not navegador_aberto:
        msg += "\n(Nota: o navegador gráfico não pôde ser aberto automaticamente neste ambiente, use o link acima)."

    # Grava a saída para o agente ler no passo seguinte
    arquivo = Path("youtube/resultado.txt")
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(msg, encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "termo": termo,
        "video_id": video_id,
        "url": url_final,
        "navegador_aberto": navegador_aberto,
        "status": status
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()