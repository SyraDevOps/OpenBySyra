import sys
import json
import re
import ssl
import urllib.request
import urllib.parse
from xml.etree import ElementTree as ET
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

# Fontes RSS de tecnologia e notícias gerais em português e inglês
RSS_FONTES = {
    "tecnoblog": "https://tecnoblog.net/feed/",
    "canaltech": "https://feeds.feedburner.com/canaltechblog",
    "google_news": "https://news.google.com/rss?hl=pt-BR&gl=BR&ceid=BR:pt-415"
}

def limpar_termo(texto: str) -> str:
    termo = str(texto).strip()
    padroes = [
        r"\b(use|o|a|os|as|um|uma|agent|agente|noticia|notícia|noticias|notícias|sobre|de|do|da|das|dos|pesquise|pesquisar|busque|buscar|me|mostre|mostrar|fale|falar|e)\b"
    ]
    for p in padroes:
        termo = re.sub(p, "", termo, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", termo).strip(" .,:;!?\t\r\n")

def buscar_rss(url_feed: str, limite: int = 5) -> list:
    noticias = []
    try:
        req = urllib.request.Request(url_feed, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=6, context=ssl_context) as resp:
            xml_data = resp.read()
            
        root = ET.fromstring(xml_data)
        # Compatibilidade com RSS (channel/item) e Atom (feed/entry)
        items = root.findall(".//item") or root.findall("{http://www.w3.org/2005/Atom}entry")
        
        for item in items[:limite]:
            # Padrão RSS
            titulo = item.find("title")
            link = item.find("link")
            pub_date = item.find("pubDate")
            
            # Padrão Atom se necessário
            if titulo is None:
                titulo = item.find("{http://www.w3.org/2005/Atom}title")
            if link is None:
                link_elem = item.find("{http://www.w3.org/2005/Atom}link")
                link_url = link_elem.attrib.get("href") if link_elem is not None else ""
            else:
                link_url = link.text.strip() if link.text else ""

            titulo_txt = titulo.text.strip() if titulo is not None and titulo.text else "Sem título"
            data_txt = pub_date.text.strip() if pub_date is not None and pub_date.text else ""
            
            # Limpa tags HTML acidentais nos títulos
            titulo_txt = re.sub(r"<.*?>", "", titulo_txt)

            if titulo_txt and link_url:
                noticias.append({
                    "titulo": titulo_txt,
                    "link": link_url,
                    "data": data_txt[:16] # Corta para exibir data/hora legível
                })
    except Exception:
        pass
    return noticias

def main():
    try:
        raw_bytes = sys.stdin.buffer.read()
        data = json.loads(raw_bytes.decode("utf-8", errors="replace")) if raw_bytes else {}
    except Exception:
        data = {}

    contexto = str(data.get("contexto", ""))
    termo_busca = limpar_termo(contexto).casefold()

    arquivo_saida = Path("noticias/relatorio.txt")
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)

    # Escolhe a fonte com base no contexto (ex: se pediu IA ou tecnologia, foca em tech; senão, Google News geral)
    todas_noticias = []
    
    if any(k in termo_busca for k in ["tecnologia", "tech", "ia", "inteligencia", "celular", "software"]):
        fontes_usadas = [RSS_FONTES["tecnoblog"], RSS_FONTES["canaltech"]]
    else:
        fontes_usadas = [RSS_FONTES["google_news"]]

    for url in fontes_usadas:
        todas_noticias.extend(buscar_rss(url, limite=4))

    # Se o usuário digitou um termo específico (ex: "notícias sobre o bitcoin" ou "notícias de sao paulo"), filtra os resultados
    if termo_busca and len(termo_busca) > 2:
        filtradas = [n for n in todas_noticias if termo_busca in n["titulo"].casefold()]
        if filtradas:
            todas_noticias = filtradas

    if not todas_noticias:
        # Fallback se a filtragem restrita zerasse os itens: puxa do Google News geral
        todas_noticias = buscar_rss(RSS_FONTES["google_news"], limite=6)

    # Montagem do relatório
    relatorio = "📰 ÚLTIMAS NOTÍCIAS RECOLHIDAS DA INTERNET:\n"
    relatorio += "--------------------------------------------------\n"

    for idx, n in enumerate(todas_noticias[:6], 1):
        relatorio += f"{idx}. {n['titulo']}\n"
        if n['data']:
            relatorio +=   f"   📅 {n['data']}\n"
        relatorio +=   f"   🔗 {n['link']}\n\n"

    arquivo_saida.write_text(relatorio, encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "total_noticias": len(todas_noticias),
        "status": "sucesso"
    }, ensure_ascii=True))

if __name__ == "__main__":
    main()