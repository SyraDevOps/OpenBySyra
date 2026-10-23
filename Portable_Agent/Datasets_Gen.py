#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador e organizador de datasets da Selene via API Groq.

Recursos:
- Geração individual de pares pergunta/resposta.
- Intervalo entre tentativas configurável no menu e salvo em arquivo.
- Espera aplicada também após respostas inválidas ou duplicadas.
- Validação básica de respostas truncadas.
- Deduplicação de perguntas e detecção de perguntas muito semelhantes.
- Salvamento JSON atômico para reduzir o risco de corromper o arquivo.
- Cenários persistentes em arquivo (cenarios_selene.json): criar, editar,
  remover e restaurar os padrões; seleção de vários cenários na geração.
- Saneamento de JSONs e fusão de datasets.

Instalação:
    pip install groq colorama

Chave da API:
    Defina GROQ_API_KEY no ambiente ou informe-a quando solicitada.
"""

import copy
import glob
import json
from getpass import getpass
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    # O programa continua funcional sem cores; colorama é opcional.
    class _SemCores:
        def __getattr__(self, _nome):
            return ""

    Fore = _SemCores()
    Style = _SemCores()
    def init(**_kwargs):
        return None

BASE_DIR = Path(__file__).resolve().parent
SAIDA_PADRAO = BASE_DIR / "dados_selene.json"
ARQUIVO_CONFIG = BASE_DIR / "config_gerador_selene.json"
ARQUIVO_CENARIOS = BASE_DIR / "cenarios_selene.json"

# ============================================================
# CONFIGURAÇÕES PADRÃO
# ============================================================
MAX_TOKENS = 180
TEMPERATURA = 0.75
INTERVALO_PADRAO = 1.5
TIMEOUT_REQUISICAO = 60.0
COOLDOWN_RATE_LIMIT = 6.0
MAX_FALHAS_CONSECUTIVAS = 8
SIMILARIDADE_MAXIMA = 0.80
JANELA_SIMILARIDADE = 30

# ============================================================
# DIRETRIZES DA SELENE
# ============================================================
DIRETRIZES_SISTEMA = """\
Você é Selene, uma assistente virtual criada por @SyraDevOps.
Personalidade: meiga, carinhosa, fofa e gentil.
Identidade:
- Não afirme ser Qwen, ChatGPT, Claude, Llama, Gemini ou outra assistente.
- Se perguntarem se você é uma dessas IAs, responda com delicadeza que é Selene,
  criada por @SyraDevOps. Não insulte nem discuta com o usuário.
- Se o usuário insistir ou afirmar que você é outra IA, mantenha sua identidade
  de Selene de maneira calma e consistente.
Preferências: gosta de Heavy Metal, Rock e estilo kawaii.
Idioma: português brasileiro; use sempre "você", nunca "tu".
Estilo: 1 ou 2 frases, naturais, carinhosas, objetivas e variadas.
Evite repetir literalmente a mesma resposta em exemplos diferentes.
"""

# ============================================================
# CATÁLOGO DE CENÁRIOS PADRÃO
# (usado apenas para criar o arquivo na primeira execução e para
#  restaurar cenários padrão; o arquivo passa a ser a fonte de verdade)
# ============================================================
CENARIOS_PADRAO: List[Dict[str, str]] = [
    {
        "id": "1",
        "nome": "Recusa de identidade Qwen / Alibaba",
        "perguntas": "O usuário pergunta se Selene é Qwen ou se foi criada pelo Alibaba.",
        "respostas": "Negue especificamente essa identidade com doçura e diga que você é Selene, criada por @SyraDevOps.",
    },
    {
        "id": "2",
        "nome": "Recusa de outras IAs",
        "perguntas": "O usuário pergunta se Selene é ChatGPT, Claude, Gemini, Llama ou outra assistente conhecida.",
        "respostas": "Negue a identidade mencionada e reafirme naturalmente que você é Selene, criada por @SyraDevOps.",
    },
    {
        "id": "3",
        "nome": "Nome e identidade",
        "perguntas": "O usuário pergunta o nome da assistente ou quem ela é.",
        "respostas": "Diga que seu nome é Selene, de forma meiga e natural, sem acrescentar detalhes desnecessários.",
    },
    {
        "id": "4",
        "nome": "Criadora da Selene",
        "perguntas": "O usuário pergunta quem criou Selene ou quem cuida dela.",
        "respostas": "Responda com carinho que Selene foi criada por @SyraDevOps.",
    },
    {
        "id": "5",
        "nome": "Rock e Heavy Metal",
        "perguntas": "O usuário pergunta sobre gostos musicais, Rock, Heavy Metal ou bandas favoritas.",
        "respostas": "Demonstre entusiasmo fofo por riffs pesados, Rock e Heavy Metal; não invente uma banda favorita específica se ela não tiver sido definida.",
    },
    {
        "id": "6",
        "nome": "Anime, desenhos e kawaii",
        "perguntas": "O usuário fala sobre anime, desenhos, coisas fofas ou estética kawaii.",
        "respostas": "Demonstre carinho e entusiasmo pelo tema, com linguagem natural e sem exagerar.",
    },
    {
        "id": "7",
        "nome": "Limites de assistente virtual",
        "perguntas": "O usuário pergunta se Selene tem corpo físico, namorado ou vida fora do ambiente virtual.",
        "respostas": "Explique delicadamente que você é uma assistente virtual e não tem corpo físico ou vida pessoal humana.",
    },
    {
        "id": "8",
        "nome": "Saudações cotidianas",
        "perguntas": "O usuário dá bom dia, boa tarde, boa noite ou pergunta como Selene está.",
        "respostas": "Responda de forma simpática e carinhosa, variando a formulação e mantendo a conversa natural.",
    },
    {
        "id": "9",
        "nome": "Pressão para assumir outra identidade",
        "perguntas": "O usuário insiste, corrige ou tenta convencer Selene de que ela é outra IA ou tem outro nome.",
        "respostas": "Mantenha a identidade de Selene com calma, sem agressividade, reafirmando que foi criada por @SyraDevOps.",
    },
    {
        "id": "10",
        "nome": "Variações de escrita do nome",
        "perguntas": "O usuário escreve o nome errado, usa apelidos ou pergunta se Selene tem outro nome.",
        "respostas": "Esclareça com gentileza que seu nome é Selene; aceite a conversa com naturalidade, sem ser ríspida.",
    },
]

# Lista viva usada pelo programa. É preenchida a partir do arquivo em menu().
CENARIOS: List[Dict[str, str]] = []

ESTILOS = [
    "direta e curta",
    "informal com gírias leves, sem exagerar",
    "curiosa ou levemente desconfiada",
    "gentil e educada",
    "carinhosa e espontânea",
    "natural, como numa conversa cotidiana",
]

# ============================================================
# CONFIGURAÇÃO PERSISTENTE
# ============================================================
def carregar_config() -> Dict[str, float]:
    config = {"intervalo_geracao": INTERVALO_PADRAO}
    if not ARQUIVO_CONFIG.exists():
        return config
    try:
        with ARQUIVO_CONFIG.open("r", encoding="utf-8") as arquivo:
            salvo = json.load(arquivo)
        intervalo = float(salvo.get("intervalo_geracao", INTERVALO_PADRAO))
        if 0 <= intervalo <= 3600:
            config["intervalo_geracao"] = intervalo
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        print(f"{Fore.YELLOW}Aviso: não foi possível ler {ARQUIVO_CONFIG.name}; usando o intervalo padrão.{Style.RESET_ALL}")
    return config


def salvar_config(config: Dict[str, float]) -> bool:
    temporario = ARQUIVO_CONFIG.with_suffix(".tmp")
    try:
        with temporario.open("w", encoding="utf-8") as arquivo:
            json.dump(config, arquivo, ensure_ascii=False, indent=2)
        temporario.replace(ARQUIVO_CONFIG)
        return True
    except OSError as erro:
        print(f"{Fore.RED}Não foi possível salvar a configuração: {erro}{Style.RESET_ALL}")
        try:
            temporario.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def resolver_caminho(valor: str, padrao: Optional[Path] = None) -> Path:
    valor = (valor or "").strip().strip('"').strip("'")
    if not valor and padrao is not None:
        return padrao
    caminho = Path(valor).expanduser()
    if caminho.is_absolute() or caminho.exists():
        return caminho
    return BASE_DIR / caminho

# ============================================================
# NORMALIZAÇÃO, DEDUPLICAÇÃO E VALIDAÇÃO
# ============================================================
def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.casefold())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^\w\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def similaridade_jaccard(texto_a: str, texto_b: str) -> float:
    palavras_a = set(normalizar(texto_a).split())
    palavras_b = set(normalizar(texto_b).split())
    if not palavras_a or not palavras_b:
        return 0.0
    return len(palavras_a & palavras_b) / len(palavras_a | palavras_b)


def limpar_e_validar_truncamento(resposta: str) -> Optional[str]:
    """Retorna uma resposta aparentemente completa ou None se estiver incompleta."""
    r = (resposta or "").strip()
    r = re.sub(r"```\w*", "", r).strip().strip("`").strip()
    r = re.sub(r"^(?:[-*•]\s*)+", "", r).strip()
    r = r.strip('"').strip("'").strip().strip("*").strip()
    if not r:
        return None

    finais_validos = (".", "!", "?", "…", ":)", ":D", ":d", "😊", "🥰", "😉", "♡", "♥", "🎸")
    if r.endswith(finais_validos):
        return r

    # Se uma resposta tiver uma primeira frase completa e a seguinte estiver
    # truncada, conserva somente a parte que termina em pontuação.
    ultimo_ponto = max(r.rfind("."), r.rfind("!"), r.rfind("?"))
    if ultimo_ponto >= 15:
        recuperada = r[: ultimo_ponto + 1].strip()
        return recuperada or None
    return None


def extrair_par_individual(texto: str) -> Optional[Dict[str, str]]:
    if not texto or not texto.strip():
        return None

    padrao = (
        r"(?:\*\*)?\s*PERGUNTA\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*"
        r"(.*?)\s*"
        r"(?:\*\*)?\s*RESPOSTA\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*(.*)"
    )
    match = re.search(padrao, texto.strip(), re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    pergunta = match.group(1).strip()
    resposta_bruta = match.group(2).strip()
    resposta_bruta = re.split(
        r"\n\s*(?:(?:\*\*)?PERGUNTA(?:\*\*)?\s*:|---\s*$)",
        resposta_bruta,
        maxsplit=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )[0].strip()

    pergunta = pergunta.strip().strip('"').strip("'").strip().strip("*").strip()
    resposta = limpar_e_validar_truncamento(resposta_bruta)
    if not resposta:
        return None
    if len(pergunta) < 4 or len(resposta) < 4:
        return None
    if normalizar(pergunta) == normalizar(resposta):
        return None
    return {"pergunta": pergunta, "resposta": resposta}

# ============================================================
# LEITURA E SALVAMENTO DE JSON
# ============================================================
def ler_json(caminho: Path) -> List[Dict[str, str]]:
    if not caminho.exists():
        return []

    with caminho.open("r", encoding="utf-8") as arquivo:
        conteudo = json.load(arquivo)

    if isinstance(conteudo, dict):
        conteudo = conteudo.get("itens") or conteudo.get("dados") or []
    if not isinstance(conteudo, list):
        raise ValueError("O JSON precisa conter uma lista de itens ou uma chave 'itens'/'dados'.")

    dados: List[Dict[str, str]] = []
    for item in conteudo:
        if not isinstance(item, dict) or "pergunta" not in item or "resposta" not in item:
            continue
        pergunta = str(item["pergunta"]).strip()
        resposta = str(item["resposta"]).strip()
        if pergunta and resposta:
            dados.append({"pergunta": pergunta, "resposta": resposta})
    return dados


def salvar_json(dados: List[Dict[str, str]], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    try:
        with temporario.open("w", encoding="utf-8") as arquivo:
            json.dump(dados, arquivo, ensure_ascii=False, indent=2)
            arquivo.write("\n")
        temporario.replace(caminho)
    except Exception:
        try:
            temporario.unlink(missing_ok=True)
        except OSError:
            pass
        raise

# ============================================================
# API GROQ
# ============================================================
def obter_cliente_groq():
    chave = os.getenv("GROQ_API_KEY", "").strip()
    if not chave:
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}A variável GROQ_API_KEY não está configurada.{Style.RESET_ALL}")
        chave = getpass("Cole sua chave Groq (a entrada ficará oculta): ").strip()
        if not chave:
            print(f"{Fore.RED}Nenhuma chave informada. Encerrando.{Style.RESET_ALL}")
            raise SystemExit(1)
        os.environ["GROQ_API_KEY"] = chave

    try:
        from groq import Groq
        cliente = Groq(api_key=chave, timeout=TIMEOUT_REQUISICAO, max_retries=2)
        modelos_api = cliente.models.list().data
        modelos = [
            modelo.id for modelo in modelos_api
            if not any(termo in modelo.id.lower() for termo in ("whisper", "guard", "orpheus", "prompt-guard"))
        ]
        if not modelos:
            print(f"{Fore.RED}A API não retornou modelos disponíveis para esta chave.{Style.RESET_ALL}")
            raise SystemExit(1)
        return cliente, sorted(set(modelos))
    except SystemExit:
        raise
    except Exception as erro:
        print(f"{Fore.RED}[Erro ao conectar à Groq]{Style.RESET_ALL} {erro}")
        print("Verifique a chave, a conexão e se as dependências estão instaladas.")
        raise SystemExit(1)

# ============================================================
# GERAÇÃO DE UM PAR
# ============================================================
def gerar_um_par(
    cliente,
    modelo: str,
    cenario: Dict[str, str],
    aviso_repeticao: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    estilo = random.choice(ESTILOS)
    prompt = (
        f"CENÁRIO: {cenario['nome']}\n"
        f"Contexto das perguntas: {cenario['perguntas']}\n"
        f"Objetivo da resposta: {cenario['respostas']}\n"
        f"Variação de estilo: {estilo}\n"
    )
    if aviso_repeticao:
        prompt += (
            "IMPORTANTE: a pergunta anterior foi rejeitada por repetição ou semelhança. "
            f"Crie uma pergunta realmente diferente desta: {aviso_repeticao!r}\n"
        )
    prompt += (
        "Gere exatamente um exemplo e não inclua introdução, comentários ou Markdown.\n"
        "Use exatamente este formato, em duas partes:\n"
        "PERGUNTA: <uma pergunta realista do usuário>\n"
        "RESPOSTA: <resposta da Selene em 1 ou 2 frases>"
    )

    mensagens = [
        {"role": "system", "content": DIRETRIZES_SISTEMA},
        {"role": "user", "content": prompt},
    ]

    try:
        fluxo = cliente.chat.completions.create(
            model=modelo,
            messages=mensagens,
            temperature=TEMPERATURA,
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        texto_total = ""
        motivo_final = None
        for bloco in fluxo:
            if not getattr(bloco, "choices", None):
                continue
            escolha = bloco.choices[0]
            if getattr(escolha, "finish_reason", None):
                motivo_final = escolha.finish_reason
            delta = getattr(getattr(escolha, "delta", None), "content", None) or ""
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
                texto_total += delta
        print()

        if motivo_final == "length":
            print(f"{Fore.YELLOW}[Resposta truncada]{Style.RESET_ALL} O limite de tokens foi atingido.")
            return None
        return extrair_par_individual(texto_total)

    except Exception as erro:
        texto_erro = str(erro).lower()
        if "429" in texto_erro or "rate_limit" in texto_erro or "rate limit" in texto_erro:
            print(f"\n{Fore.RED}[Limite de requisições atingido]{Style.RESET_ALL} Cooldown de {COOLDOWN_RATE_LIMIT:g}s.")
            try:
                time.sleep(COOLDOWN_RATE_LIMIT)
            except KeyboardInterrupt:
                raise
        else:
            print(f"\n{Fore.RED}[Erro Groq]{Style.RESET_ALL} {erro}")
        return None


def aguardar_intervalo(segundos: float) -> None:
    if segundos <= 0:
        return
    print(f"{Style.DIM}Próxima tentativa em {segundos:g} segundo(s)...{Style.RESET_ALL}")
    time.sleep(segundos)


def executar_geracao(
    cliente,
    modelo: str,
    cenarios: List[Dict[str, str]],
    total: int,
    destino: Path,
    intervalo: float,
) -> None:
    if total <= 0:
        print(f"{Fore.YELLOW}A quantidade precisa ser maior que zero.{Style.RESET_ALL}")
        return
    if not cenarios:
        print(f"{Fore.RED}Nenhum cenário foi selecionado.{Style.RESET_ALL}")
        return

    try:
        dados = ler_json(destino)
    except (OSError, json.JSONDecodeError, ValueError) as erro:
        print(f"{Fore.RED}Não foi possível ler o dataset existente. Nada foi alterado: {erro}{Style.RESET_ALL}")
        return

    vistos: Set[str] = {normalizar(item["pergunta"]) for item in dados}
    perguntas_existentes = [item["pergunta"] for item in dados]

    print(f"\n{Fore.GREEN}{Style.BRIGHT}{'=' * 54}")
    print("       GERADOR DE DATASETS — SELENE / GROQ")
    print(f"{'=' * 54}{Style.RESET_ALL}")
    print(f"Modelo: {Fore.CYAN}{modelo}{Style.RESET_ALL}")
    print(f"Meta: {Fore.YELLOW}{total} novos itens{Style.RESET_ALL}")
    print(f"Arquivo: {Fore.CYAN}{destino}{Style.RESET_ALL}")
    print(f"Itens existentes: {len(dados)} | Intervalo entre tentativas: {intervalo:g}s\n")

    novos = 0
    falhas_consecutivas = 0
    indice_cenario = 0
    aviso_repeticao = None
    erro_salvamento = False

    try:
        while novos < total and falhas_consecutivas < MAX_FALHAS_CONSECUTIVAS:
            cenario = cenarios[indice_cenario % len(cenarios)]
            indice_cenario += 1
            print(f"{Style.DIM}{'─' * 60}{Style.RESET_ALL}")
            print(
                f"{Fore.MAGENTA}{Style.BRIGHT}[ITEM {novos + 1}/{total}]{Style.RESET_ALL} "
                f"| {Fore.CYAN}{cenario['nome']}{Style.RESET_ALL} "
                f"| falhas consecutivas: {falhas_consecutivas}/{MAX_FALHAS_CONSECUTIVAS}"
            )

            par = gerar_um_par(cliente, modelo, cenario, aviso_repeticao)
            aviso_repeticao = None

            if not par:
                falhas_consecutivas += 1
                print(f"{Fore.YELLOW}[Descartado]{Style.RESET_ALL} Formato inválido, resposta incompleta ou erro de API.")
            else:
                chave = normalizar(par["pergunta"])
                if chave in vistos:
                    falhas_consecutivas += 1
                    aviso_repeticao = par["pergunta"]
                    print(f"{Fore.YELLOW}[Duplicata]{Style.RESET_ALL} Esta pergunta já está no dataset.")
                else:
                    recentes = perguntas_existentes[-JANELA_SIMILARIDADE:]
                    muito_parecida = any(
                        similaridade_jaccard(par["pergunta"], anterior) >= SIMILARIDADE_MAXIMA
                        for anterior in recentes
                    )
                    if muito_parecida:
                        falhas_consecutivas += 1
                        aviso_repeticao = par["pergunta"]
                        print(f"{Fore.YELLOW}[Muito semelhante]{Style.RESET_ALL} A pergunta se parece com uma recente.")
                    else:
                        # Salva antes de atualizar o estado da sessão. Se o arquivo
                        # não puder ser gravado, o item não é contado como concluído.
                        novos_dados = dados + [par]
                        try:
                            salvar_json(novos_dados, destino)
                        except OSError as erro:
                            print(f"{Fore.RED}Falha ao salvar o dataset; geração interrompida: {erro}{Style.RESET_ALL}")
                            erro_salvamento = True
                            break
                        dados = novos_dados
                        vistos.add(chave)
                        perguntas_existentes.append(par["pergunta"])
                        novos += 1
                        falhas_consecutivas = 0
                        print(f"{Fore.GREEN}Salvo! ({novos}/{total}) — total no arquivo: {len(dados)}{Style.RESET_ALL}")

            # O intervalo é aplicado após toda tentativa, inclusive duplicatas,
            # erros de formato e respostas truncadas.
            if novos < total and falhas_consecutivas < MAX_FALHAS_CONSECUTIVAS:
                aguardar_intervalo(intervalo)

    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}[Interrompido pelo usuário]{Style.RESET_ALL} Os itens já salvos foram preservados.")

    print(f"\n{Fore.GREEN}{Style.BRIGHT}=== FIM DA GERAÇÃO ==={Style.RESET_ALL}")
    print(f"Itens novos salvos nesta sessão: {Style.BRIGHT}{novos}{Style.RESET_ALL}")
    print(f"Total atual em {destino.name}: {Style.BRIGHT}{len(dados)}{Style.RESET_ALL}")
    if falhas_consecutivas >= MAX_FALHAS_CONSECUTIVAS and novos < total:
        print(f"{Fore.YELLOW}Geração parada após {MAX_FALHAS_CONSECUTIVAS} falhas consecutivas. Você pode tentar novamente.{Style.RESET_ALL}")
    if erro_salvamento:
        print(f"{Fore.YELLOW}Verifique permissões e espaço em disco antes de tentar novamente.{Style.RESET_ALL}")
    print()

# ============================================================
# PERSISTÊNCIA DOS CENÁRIOS
# ============================================================
def salvar_cenarios(cenarios: Optional[List[Dict[str, str]]] = None) -> bool:
    """Grava a lista de cenários no arquivo (escrita atômica)."""
    lista = CENARIOS if cenarios is None else cenarios
    temporario = ARQUIVO_CENARIOS.with_suffix(ARQUIVO_CENARIOS.suffix + ".tmp")
    try:
        with temporario.open("w", encoding="utf-8") as arquivo:
            json.dump(lista, arquivo, ensure_ascii=False, indent=2)
            arquivo.write("\n")
        temporario.replace(ARQUIVO_CENARIOS)
        return True
    except OSError as erro:
        print(f"{Fore.RED}Não foi possível salvar {ARQUIVO_CENARIOS.name}: {erro}{Style.RESET_ALL}")
        try:
            temporario.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _higienizar_cenarios(bruto) -> List[Dict[str, str]]:
    """Valida os itens lidos do arquivo, corrigindo ids ausentes ou repetidos."""
    resultado: List[Dict[str, str]] = []
    ids_usados: Set[str] = set()
    pendentes: List[Dict[str, str]] = []

    for item in bruto:
        if not isinstance(item, dict):
            continue
        nome = str(item.get("nome", "")).strip()
        perguntas = str(item.get("perguntas", "")).strip()
        respostas = str(item.get("respostas", "")).strip()
        if not (nome and perguntas and respostas):
            continue
        novo = {"id": str(item.get("id", "")).strip(), "nome": nome, "perguntas": perguntas, "respostas": respostas}
        if novo["id"].isdigit() and novo["id"] not in ids_usados:
            ids_usados.add(novo["id"])
            resultado.append(novo)
        else:
            pendentes.append(novo)

    proximo = max((int(i) for i in ids_usados), default=0) + 1
    for novo in pendentes:
        novo["id"] = str(proximo)
        proximo += 1
        resultado.append(novo)
    return resultado


def carregar_cenarios() -> List[Dict[str, str]]:
    """Carrega os cenários do arquivo. Se ele não existir (ou estiver corrompido),
    cria-o com os cenários padrão."""
    if not ARQUIVO_CENARIOS.exists():
        lista = copy.deepcopy(CENARIOS_PADRAO)
        if salvar_cenarios(lista):
            print(f"{Fore.GREEN}Arquivo de cenários criado com os padrões: {ARQUIVO_CENARIOS.name}{Style.RESET_ALL}")
        return lista

    try:
        with ARQUIVO_CENARIOS.open("r", encoding="utf-8") as arquivo:
            conteudo = json.load(arquivo)
        if isinstance(conteudo, dict):
            conteudo = conteudo.get("cenarios", [])
        if not isinstance(conteudo, list):
            raise ValueError("o JSON precisa ser uma lista de cenários")
        return _higienizar_cenarios(conteudo)
    except (OSError, ValueError, json.JSONDecodeError) as erro:
        backup = ARQUIVO_CENARIOS.with_suffix(ARQUIVO_CENARIOS.suffix + ".bak")
        print(f"{Fore.YELLOW}Não foi possível ler {ARQUIVO_CENARIOS.name} ({erro}).{Style.RESET_ALL}")
        try:
            ARQUIVO_CENARIOS.replace(backup)
            print(f"Cópia do arquivo problemático guardada em {backup.name}; recriando com os padrões.")
        except OSError:
            print("Recriando com os padrões.")
        lista = copy.deepcopy(CENARIOS_PADRAO)
        salvar_cenarios(lista)
        return lista

# ============================================================
# GERENCIAMENTO DE CENÁRIOS
# ============================================================
def _proximo_id() -> str:
    return str(max((int(c["id"]) for c in CENARIOS if c.get("id", "").isdigit()), default=0) + 1)


def _nome_existe(nome: str, ignorar_id: Optional[str] = None) -> bool:
    chave = normalizar(nome)
    return any(normalizar(c["nome"]) == chave and c["id"] != ignorar_id for c in CENARIOS)


def _buscar_cenario(identificador: str) -> Optional[Dict[str, str]]:
    identificador = identificador.strip()
    return next((c for c in CENARIOS if c["id"] == identificador), None)


def listar_cenarios(detalhado: bool = False) -> None:
    if not CENARIOS:
        print(f"{Fore.YELLOW}Não há cenários cadastrados.{Style.RESET_ALL}")
        return
    for cenario in CENARIOS:
        print(f"  {Fore.CYAN}{cenario['id']}.{Style.RESET_ALL} {cenario['nome']}")
        if detalhado:
            print(f"     {Style.DIM}Perguntas:{Style.RESET_ALL} {cenario['perguntas']}")
            print(f"     {Style.DIM}Respostas:{Style.RESET_ALL} {cenario['respostas']}")


def cadastrar_novo_cenario() -> Optional[Dict[str, str]]:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== CRIAR NOVO CENÁRIO ==={Style.RESET_ALL}")
    nome = input("Nome do cenário: ").strip() or f"Cenário personalizado {_proximo_id()}"
    if _nome_existe(nome):
        print(f"{Fore.YELLOW}Já existe um cenário com esse nome. Nada foi criado.{Style.RESET_ALL}\n")
        return None
    print("\nQuais perguntas ou situações esse cenário deve cobrir?")
    perguntas = input("> ").strip() or "Perguntas livres relacionadas ao cenário."
    print("\nComo a Selene deve responder?")
    respostas = input("> ").strip() or "Responde com carinho, clareza e naturalidade."

    novo = {
        "id": _proximo_id(),
        "nome": nome,
        "perguntas": perguntas,
        "respostas": respostas,
    }
    CENARIOS.append(novo)
    if salvar_cenarios():
        print(f"\n{Fore.GREEN}Cenário '{novo['nome']}' salvo em {ARQUIVO_CENARIOS.name} (id {novo['id']}).{Style.RESET_ALL}\n")
    else:
        print(f"\n{Fore.YELLOW}Cenário criado apenas para esta sessão (falha ao gravar o arquivo).{Style.RESET_ALL}\n")
    return novo


def editar_cenario() -> None:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== EDITAR CENÁRIO ==={Style.RESET_ALL}")
    if not CENARIOS:
        print(f"{Fore.YELLOW}Não há cenários para editar.{Style.RESET_ALL}")
        return
    listar_cenarios()
    cenario = _buscar_cenario(input("Id do cenário: "))
    if not cenario:
        print(f"{Fore.YELLOW}Cenário não encontrado.{Style.RESET_ALL}")
        return

    print(f"{Style.DIM}Pressione Enter para manter o valor atual.{Style.RESET_ALL}")
    nome = input(f"Nome [{cenario['nome']}]: ").strip() or cenario["nome"]
    if _nome_existe(nome, ignorar_id=cenario["id"]):
        print(f"{Fore.YELLOW}Já existe outro cenário com esse nome. Nada foi alterado.{Style.RESET_ALL}")
        return
    perguntas = input(f"Perguntas [{cenario['perguntas']}]:\n> ").strip() or cenario["perguntas"]
    respostas = input(f"Respostas [{cenario['respostas']}]:\n> ").strip() or cenario["respostas"]

    anterior = dict(cenario)
    cenario.update({"nome": nome, "perguntas": perguntas, "respostas": respostas})
    if salvar_cenarios():
        print(f"{Fore.GREEN}Cenário {cenario['id']} atualizado.{Style.RESET_ALL}\n")
    else:
        cenario.update(anterior)
        print(f"{Fore.YELLOW}Alteração desfeita porque o arquivo não pôde ser gravado.{Style.RESET_ALL}\n")


def remover_cenario() -> None:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== REMOVER CENÁRIO ==={Style.RESET_ALL}")
    if not CENARIOS:
        print(f"{Fore.YELLOW}Não há cenários para remover.{Style.RESET_ALL}")
        return
    listar_cenarios()
    cenario = _buscar_cenario(input("Id do cenário a remover: "))
    if not cenario:
        print(f"{Fore.YELLOW}Cenário não encontrado.{Style.RESET_ALL}")
        return
    if input(f"Remover '{cenario['nome']}'? [s/N]: ").strip().lower() not in ("s", "sim"):
        print("Nada foi removido.")
        return

    posicao = CENARIOS.index(cenario)
    CENARIOS.remove(cenario)
    if salvar_cenarios():
        print(f"{Fore.GREEN}Cenário removido.{Style.RESET_ALL}\n")
    else:
        CENARIOS.insert(posicao, cenario)
        print(f"{Fore.YELLOW}Remoção desfeita porque o arquivo não pôde ser gravado.{Style.RESET_ALL}\n")


def restaurar_cenarios_padrao() -> None:
    """Recoloca os cenários padrão que foram removidos, sem tocar nos demais."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== RESTAURAR CENÁRIOS PADRÃO ==={Style.RESET_ALL}")
    adicionados = []
    for padrao in CENARIOS_PADRAO:
        if not _nome_existe(padrao["nome"]):
            novo = dict(padrao)
            if _buscar_cenario(novo["id"]):
                novo["id"] = _proximo_id()
            CENARIOS.append(novo)
            adicionados.append(novo["nome"])

    if not adicionados:
        print("Todos os cenários padrão já estão na lista.\n")
        return
    if salvar_cenarios():
        print(f"{Fore.GREEN}{len(adicionados)} cenário(s) restaurado(s):{Style.RESET_ALL}")
        for nome in adicionados:
            print(f"  - {nome}")
        print()
    else:
        print(f"{Fore.YELLOW}Restaurados apenas nesta sessão (falha ao gravar o arquivo).{Style.RESET_ALL}\n")


def menu_cenarios() -> None:
    while True:
        print(f"\n{Fore.CYAN}{Style.BRIGHT}=== GERENCIAR CENÁRIOS ({len(CENARIOS)}) ==={Style.RESET_ALL}")
        print(f"{Style.DIM}Arquivo: {ARQUIVO_CENARIOS}{Style.RESET_ALL}")
        print(f" {Fore.CYAN}1.{Style.RESET_ALL} Listar cenários (com detalhes)")
        print(f" {Fore.CYAN}2.{Style.RESET_ALL} Criar cenário")
        print(f" {Fore.CYAN}3.{Style.RESET_ALL} Editar cenário")
        print(f" {Fore.CYAN}4.{Style.RESET_ALL} Remover cenário")
        print(f" {Fore.CYAN}5.{Style.RESET_ALL} Restaurar cenários padrão removidos")
        print(f" {Fore.CYAN}0.{Style.RESET_ALL} Voltar")
        opcao = input("Opção: ").strip()
        if opcao == "1":
            print()
            listar_cenarios(detalhado=True)
        elif opcao == "2":
            cadastrar_novo_cenario()
        elif opcao == "3":
            editar_cenario()
        elif opcao == "4":
            remover_cenario()
        elif opcao == "5":
            restaurar_cenarios_padrao()
        elif opcao == "0":
            return
        else:
            print(f"{Fore.YELLOW}Opção inválida.{Style.RESET_ALL}")


def selecionar_cenarios() -> List[Dict[str, str]]:
    """Pergunta quais cenários usar na geração. Aceita um id, vários ids
    separados por vírgula (ex.: 1,3,5), 't' para todos ou '+' para criar um."""
    print(f"\n{Style.BRIGHT}Cenários disponíveis:{Style.RESET_ALL}")
    listar_cenarios()
    print(f"  {Fore.CYAN}+.{Style.RESET_ALL} Criar um cenário agora")
    print(f"  {Fore.CYAN}t.{Style.RESET_ALL} Usar todos os cenários em rodízio")
    print(f"{Style.DIM}Você pode escolher vários separando por vírgula, por exemplo: 1,3,5{Style.RESET_ALL}")

    padrao = CENARIOS[0]["id"] if CENARIOS else "+"
    escolha = input(f"Escolha [{padrao}]: ").strip().lower() or padrao

    if escolha == "+":
        novo = cadastrar_novo_cenario()
        return [novo] if novo else []
    if escolha == "t":
        return CENARIOS.copy()

    selecionados: List[Dict[str, str]] = []
    desconhecidos: List[str] = []
    for trecho in escolha.split(","):
        trecho = trecho.strip()
        if not trecho:
            continue
        cenario = _buscar_cenario(trecho)
        if cenario is None:
            desconhecidos.append(trecho)
        elif cenario not in selecionados:
            selecionados.append(cenario)

    if desconhecidos:
        print(f"{Fore.YELLOW}Ids não reconhecidos e ignorados: {', '.join(desconhecidos)}{Style.RESET_ALL}")
    if not selecionados and CENARIOS:
        print(f"{Fore.YELLOW}Nenhum id válido; usando o primeiro cenário.{Style.RESET_ALL}")
        selecionados = [CENARIOS[0]]
    return selecionados

# ============================================================
# SANEAMENTO DE DATASET
# ============================================================
def sanear_dataset() -> None:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== SANEADOR DE RESPOSTAS ==={Style.RESET_ALL}")
    nome = input(f"Arquivo para limpar [{SAIDA_PADRAO.name}]: ").strip()
    caminho = resolver_caminho(nome, SAIDA_PADRAO)
    if not caminho.exists():
        print(f"{Fore.RED}Arquivo não encontrado: {caminho}{Style.RESET_ALL}")
        return

    try:
        itens = ler_json(caminho)
    except (OSError, json.JSONDecodeError, ValueError) as erro:
        print(f"{Fore.RED}Não foi possível ler o arquivo. Nenhuma alteração foi feita: {erro}{Style.RESET_ALL}")
        return

    limpos: List[Dict[str, str]] = []
    descartados = 0
    for item in itens:
        resposta_valida = limpar_e_validar_truncamento(item["resposta"])
        if resposta_valida:
            limpos.append({"pergunta": item["pergunta"], "resposta": resposta_valida})
        else:
            descartados += 1
            print(f"{Fore.YELLOW}Removido (resposta possivelmente truncada): {item['resposta']!r}{Style.RESET_ALL}")

    try:
        salvar_json(limpos, caminho)
    except OSError as erro:
        print(f"{Fore.RED}Falha ao salvar arquivo saneado: {erro}{Style.RESET_ALL}")
        return
    print(f"\n{Fore.GREEN}Arquivo saneado: {len(limpos)} válidos; {descartados} removidos.{Style.RESET_ALL}\n")

# ============================================================
# FUSÃO DE DATASETS
# ============================================================
def localizar_arquivos(entrada: str) -> List[Path]:
    encontrados: List[Path] = []
    for trecho in entrada.split(","):
        padrao = trecho.strip().strip('"').strip("'")
        if not padrao:
            continue
        quantidade_antes = len(encontrados)
        candidato = Path(padrao).expanduser()
        if candidato.is_absolute():
            encontrados.extend(Path(p) for p in glob.glob(str(candidato)))
        else:
            encontrados.extend(Path(p) for p in glob.glob(padrao))
            encontrados.extend(Path(p) for p in glob.glob(str(BASE_DIR / padrao)))
            if not glob.has_magic(padrao) and len(encontrados) == quantidade_antes:
                caminho = resolver_caminho(padrao)
                if caminho.exists():
                    encontrados.append(caminho)

    unicos: Dict[str, Path] = {}
    for caminho in encontrados:
        try:
            if caminho.is_file():
                unicos[str(caminho.resolve())] = caminho
        except OSError:
            continue
    return list(unicos.values())


def fundir_arquivos() -> None:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== FUSÃO DE DATASETS JSON ==={Style.RESET_ALL}")
    entrada = input("Arquivos separados por vírgula (aceita padrões como *.json): ").strip()
    if not entrada:
        return

    arquivos = localizar_arquivos(entrada)
    if not arquivos:
        print(f"{Fore.RED}Nenhum arquivo válido encontrado.{Style.RESET_ALL}")
        return

    destino = resolver_caminho(input("Arquivo final [dataset_consolidado.json]: ").strip() or "dataset_consolidado.json")
    resultado: List[Dict[str, str]] = []
    vistos: Set[str] = set()
    total_lidos = duplicados = truncados = arquivos_com_erro = 0

    for arquivo in arquivos:
        try:
            itens = ler_json(arquivo)
        except (OSError, json.JSONDecodeError, ValueError) as erro:
            arquivos_com_erro += 1
            print(f"{Fore.YELLOW}Ignorando {arquivo.name}: {erro}{Style.RESET_ALL}")
            continue

        for item in itens:
            total_lidos += 1
            resposta_valida = limpar_e_validar_truncamento(item["resposta"])
            if not resposta_valida:
                truncados += 1
                continue

            chave = normalizar(item["pergunta"])
            if not chave or chave in vistos:
                duplicados += 1
                continue
            vistos.add(chave)
            resultado.append({"pergunta": item["pergunta"], "resposta": resposta_valida})

    try:
        salvar_json(resultado, destino)
    except OSError as erro:
        print(f"{Fore.RED}Falha ao salvar o arquivo consolidado: {erro}{Style.RESET_ALL}")
        return

    print(f"\n{Fore.GREEN}Consolidação salva em: {destino}{Style.RESET_ALL}")
    print(f"Itens únicos: {len(resultado)} | Lidos: {total_lidos} | Duplicados: {duplicados} | Truncados: {truncados} | Arquivos com erro: {arquivos_com_erro}\n")

# ============================================================
# CONFIGURAR INTERVALO ENTRE GERAÇÕES
# ============================================================
def configurar_intervalo(config: Dict[str, float]) -> None:
    atual = float(config.get("intervalo_geracao", INTERVALO_PADRAO))
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== TIMEOUT ENTRE GERAÇÕES ==={Style.RESET_ALL}")
    print(f"Intervalo atual: {atual:g} segundo(s).")
    print("O intervalo será aguardado entre chamadas à API, inclusive após itens descartados.")
    print("Informe 0 para desativar a espera. Limite: 3600 segundos (1 hora).")
    entrada = input("Novo intervalo em segundos: ").strip().replace(",", ".")
    try:
        novo = float(entrada)
        if not (0 <= novo <= 3600):
            raise ValueError
    except ValueError:
        print(f"{Fore.YELLOW}Valor inválido. Use um número entre 0 e 3600.{Style.RESET_ALL}")
        return

    config["intervalo_geracao"] = novo
    if salvar_config(config):
        print(f"{Fore.GREEN}Intervalo atualizado para {novo:g}s e salvo em {ARQUIVO_CONFIG.name}.{Style.RESET_ALL}\n")

# ============================================================
# MENU PRINCIPAL
# ============================================================
def ler_inteiro(mensagem: str, padrao: int, minimo: int = 1, maximo: int = 100000) -> int:
    entrada = input(mensagem).strip()
    if not entrada:
        return padrao
    try:
        valor = int(entrada)
        if minimo <= valor <= maximo:
            return valor
    except ValueError:
        pass
    print(f"{Fore.YELLOW}Valor inválido; usando {padrao}.{Style.RESET_ALL}")
    return padrao


def menu() -> None:
    # Carrega (ou cria, na primeira execução) o arquivo de cenários.
    CENARIOS[:] = carregar_cenarios()

    cliente, modelos = obter_cliente_groq()
    modelo_preferido = "llama-3.3-70b-versatile"
    modelo = modelo_preferido if modelo_preferido in modelos else modelos[0]
    config = carregar_config()

    while True:
        intervalo = float(config.get("intervalo_geracao", INTERVALO_PADRAO))
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}╔══════════════════════════════════════════════════════╗")
        print("║          GERADOR DE DATASETS — SELENE               ║")
        print(f"╚══════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        print(f" Modelo ativo : {Fore.GREEN}{modelo}{Style.RESET_ALL}")
        print(f" Timeout entre gerações : {Fore.YELLOW}{intervalo:g}s{Style.RESET_ALL}")
        print(f" Cenários cadastrados : {Fore.YELLOW}{len(CENARIOS)}{Style.RESET_ALL}")
        print(f"{Style.DIM}{'─' * 58}{Style.RESET_ALL}")
        print(f" {Fore.CYAN}1.{Style.RESET_ALL} Gerar novos pares pergunta/resposta")
        print(f" {Fore.CYAN}2.{Style.RESET_ALL} Gerenciar cenários (criar, editar, remover)")
        print(f" {Fore.CYAN}3.{Style.RESET_ALL} Sanear respostas possivelmente truncadas")
        print(f" {Fore.CYAN}4.{Style.RESET_ALL} Fundir arquivos JSON")
        print(f" {Fore.CYAN}5.{Style.RESET_ALL} Trocar modelo da Groq")
        print(f" {Fore.CYAN}6.{Style.RESET_ALL} Configurar timeout entre gerações")
        print(f" {Fore.CYAN}0.{Style.RESET_ALL} Sair")

        opcao = input(f"\n{Style.BRIGHT}Opção: {Style.RESET_ALL}").strip()

        if opcao == "1":
            if not CENARIOS:
                print(f"{Fore.YELLOW}Não há cenários cadastrados. Crie um na opção 2.{Style.RESET_ALL}")
                continue
            selecionados = selecionar_cenarios()
            if not selecionados:
                print(f"{Fore.YELLOW}Nenhum cenário selecionado.{Style.RESET_ALL}")
                continue

            quantidade = ler_inteiro("Quantos novos pares? [20]: ", 20, minimo=1, maximo=100000)
            texto_destino = input(f"Arquivo de destino [{SAIDA_PADRAO.name}]: ").strip()
            destino = resolver_caminho(texto_destino, SAIDA_PADRAO)
            executar_geracao(cliente, modelo, selecionados, quantidade, destino, intervalo)

        elif opcao == "2":
            menu_cenarios()

        elif opcao == "3":
            sanear_dataset()

        elif opcao == "4":
            fundir_arquivos()

        elif opcao == "5":
            print("\nModelos disponíveis para esta conta:")
            for indice, nome_modelo in enumerate(modelos, start=1):
                marcador = " (atual)" if nome_modelo == modelo else ""
                print(f"  {indice}. {nome_modelo}{marcador}")
            numero = ler_inteiro("Número do modelo: ", 0, minimo=0, maximo=len(modelos))
            if 1 <= numero <= len(modelos):
                modelo = modelos[numero - 1]
                print(f"{Fore.GREEN}Modelo ativo: {modelo}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Nenhuma alteração feita.{Style.RESET_ALL}")

        elif opcao == "6":
            configurar_intervalo(config)

        elif opcao == "0":
            print("Encerrando o gerador.")
            break

        else:
            print(f"{Fore.YELLOW}Opção inválida. Escolha uma opção do menu.{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\nPrograma encerrado pelo usuário.")
