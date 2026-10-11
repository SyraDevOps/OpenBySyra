import sys
import json
import re
import ssl
import urllib.request
import urllib.parse
import webbrowser
import os
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

ssl_context = ssl._create_unverified_context()
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def extrair_termo_puro(frase: str) -> str:
    """Isola cirurgicamente o objeto da busca, eliminando comandos e preposições sobras."""
    texto = str(frase).strip()
    
    # Remove qualquer variação de comando da Astra
    padroes_remover = [
        r"\b(use|o|a|os|as|um|uma|agent|agente|imagens|imagem|foto|fotos|busque|buscar|pesquise|pesquisar|por|de|do|da|sobre|me|mostre|mostrar|e)\b"
    ]
    
    termo_limpo = texto
    for p in padroes_remover:
        termo_limpo = re.sub(p, "", termo_limpo, flags=re.IGNORECASE)
        
    # Limpa espaços duplos e pontuações
    termo_limpo = re.sub(r"\s+", " ", termo_limpo).strip(" .,:;!?\t\r\n")
    return termo_limpo if len(termo_limpo) > 1 else "jaguatirica"

def baixar_primeira_imagem(termo: str, pasta_destino: Path) -> tuple[str | None, str | None]:
    """Tenta baixar a imagem via DuckDuckGo."""
    try:
        url_vqd = f"https://duckduckgo.com/?q={urllib.parse.quote(termo)}"
        req_vqd = urllib.request.Request(url_vqd, headers=HEADERS)
        with urllib.request.urlopen(req_vqd, timeout=5, context=ssl_context) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            m_vqd = re.search(r'vqd=["\']([\d-]+)["\']', html)
            if not m_vqd:
                return None, None
            vqd = m_vqd.group(1)

        url_api = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={urllib.parse.quote(termo)}&vqd={vqd}"
        req_api = urllib.request.Request(url_api, headers=HEADERS)
        with urllib.request.urlopen(req_api, timeout=5, context=ssl_context) as resp_api:
            data = json.loads(resp_api.read().decode("utf-8"))
            results = data.get("results", [])
            
            for item in results[:2]:
                img_url = item.get("image")
                if not img_url:
                    continue
                try:
                    req_img = urllib.request.Request(img_url, headers=HEADERS)
                    with urllib.request.urlopen(req_img, timeout=4, context=ssl_context) as img_resp:
                        img_data = img_resp.read()
                        if len(img_data) < 4000:
                            continue
                            
                        ext = ".jpg"
                        if ".png" in img_url.lower():
                            ext = ".png"
                        elif ".webp" in img_url.lower():
                            ext = ".webp"
                            
                        nome_arquivo = pasta_destino / f"download_{termo.replace(' ', '_')}{ext}"
                        nome_arquivo.write_bytes(img_data)
                        return str(nome_arquivo), img_url
                except Exception:
                    continue
    except Exception:
        pass
    return None, None

def abrir_no_pc(caminho_ou_url: str, eh_arquivo: bool = True):
    """Abre de forma forçada e garantida no sistema operacional ou navegador."""
    try:
        if eh_arquivo:
            path_abs = os.path.abspath(caminho_ou_url)
            if sys.platform == "win32":
                os.startfile(path_abs)
            elif sys.platform == "darwin":
                os.system(f"open '{path_abs}'")
            else:
                os.system(f"xdg-open '{path_abs}'")
        else:
            webbrowser.open(caminho_ou_url)
        return True
    except Exception:
        return False

def main():
    try:
        raw_bytes = sys.stdin.buffer.read()
        data = json.loads(raw_bytes.decode("utf-8", errors="replace")) if raw_bytes else {}
    except Exception:
        data = {}

    contexto = str(data.get("contexto", ""))
    termo = extrair_termo_puro(contexto)

    pasta_imagens = Path("imagens")
    pasta_imagens.mkdir(parents=True, exist_ok=True)
    arquivo_relatorio = pasta_imagens / "resultado.txt"

    caminho_local, url_original = baixar_primeira_imagem(termo, pasta_imagens)

    if caminho_local:
        abrir_no_pc(caminho_local, eh_arquivo=True)
        msg = f"Baixei a imagem de '{termo}' para o seu computador e a abri no visualizador padrão!\nArquivo: {caminho_local}"
        status = "sucesso_download"
    else:
        # Fallback robusto: Abre o Google Imagens diretamente no navegador padrão do usuário
        url_busca = f"https://images.google.com/search?q={urllib.parse.quote(termo)}&tbm=isch"
        abrir_no_pc(url_busca, eh_arquivo=False)
        msg = f"Abri o navegador com os resultados de busca de imagens para '{termo}'.\nLink: {url_busca}"
        status = "sucesso_navegador"

    arquivo_relatorio.write_text(msg, encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "termo": termo,
        "status": status
    }, ensure_ascii=True))

if __name__ == "__main__":
    main()