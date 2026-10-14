# Astra

Assistente de terminal em português: chat, memória, ferramentas/agentes, voz (wake word "astra").

## Iniciar

    python main.py                          # auto: Groq → Gemini → GGUF offline (troca sozinho se cair)
    python main.py --groq                   # só Groq
    python main.py --groq openai/gpt-oss-20b --key gsk_...
    python main.py --google                 # só Gemini (gemini-2.5-flash-lite)
    python main.py --google gemini-2.5-flash --key AIza...
    python main.py --gguf                   # só offline (model.gguf ao lado do main.py)
    python main.py --gguf --model outro.gguf

Também vale `--backend auto|groq|google|gguf` e a variável `ASTRA_BACKEND`.
A linha de comando tem prioridade.

## Chaves de API

Em ordem de prioridade: `--key` → variável de ambiente (`ASTRA_GROQ_API_KEY`,
`ASTRA_GOOGLE_API_KEY` / `GEMINI_API_KEY`) → constante no topo do `main.py`.
Sem chave, a Astra pergunta na hora (digitação oculta, vale só para a sessão).
Chaves grátis: console.groq.com/keys · aistudio.google.com/apikey

## Durante a conversa

- `/backend` mostra o backend em uso e o que está disponível
- `/backend google [modelo]`, `/backend groq`, `/backend gguf`, `/backend auto` trocam sem perder a conversa
- `/voice on|off`, `/voz`, `/vozes`, `/idioma`, `/status`, `/ajuda`

## Voz

Diga "Astra" → plim → fale o pedido. Ela só responde em voz alta se você começou por voz.
Digitar + Enter ou dizer "Astra" enquanto ela fala corta o áudio.
Fones de ouvido evitam eco (`ASTRA_BARGE_IN_VOICE=0` desliga o corte por voz).
Diagnóstico: `python voice_io.py --selftest`.

## Outras variáveis úteis

`ASTRA_AUTO_ORDER=groq,google,gguf` · `ASTRA_GOOGLE_CTX` · `ASTRA_GROQ_MODEL` ·
`ASTRA_PATCH_STDOUT=0` · `ASTRA_VOICE_MODE=0` · `ASTRA_TTS_MODE=0`

## Agentes que não travam no "Step 1"

Um agente em `agents/*.json` é uma receita de passos. Se nenhum passo usa `{{context}}`
ou `{{topic}}`, ele é **fixo**: roda sempre igual, e chamá-lo para outro pedido parece
funcionar (mostra "Etapa 1 concluída") sem fazer o que foi pedido. A Astra agora:

- marca cada agente como `dynamic: true/false` e só escolhe um fixo sozinha com correspondência forte
  (`ASTRA_STATIC_AGENT_THRESHOLD`, padrão 0.62);
- depois de cada `run_agent`, detecta agente fixo reutilizado para um pedido diferente, ou o mesmo
  resultado para pedidos diferentes, e trata como falha explicada (`stale_agent`) — nada disso vira "fato verificado";
- então o planejador muda de recurso ou propõe `create_agent` com um nome novo e dinâmico
  (você confirma com s/N; `ASTRA_AUTONOMOUS_WRITE=1` dispensa a pergunta). Agentes novos sem
  `{{context}}`/`{{topic}}` são recusados;
- não responde de memória a pedidos que mandam executar algo ou citam um módulo/agente, e
  dados de ferramenta com mais de 15 min (`ASTRA_DIRECT_ANSWER_MAX_AGE`, em segundos) não respondem sem rodar de novo;
- o resultado do agente agora informa a etapa que falhou (`failed_step`) e as que não rodaram (`not_run`).

## Agentes prontos (`agents/`)

Todos em português natural. Os determinísticos respondem direto, sem gastar o LLM, e
guardam seus dados no workspace (`sandbox_data/sessions/default/`).

| Agente | O que faz — exemplos |
|---|---|
| `atalhos` | **abra cnn** → cnnbrasil.com.br · `crie o atalho blog para https://…` · `atalho trabalho: https://a.com, https://b.com` (abre vários) · `chame zap de whatsapp` · `liste meus atalhos` · `remova o atalho blog`. ~25 atalhos prontos (g1, folha, youtube, gmail, github, whatsapp…). Aceita pastas (`~/Downloads`). Arquivo editável: `atalhos/atalhos.json` |
| `abrir` | `abra https://site.com` · `abra o 2º resultado` · `abra a 3ª notícia` (usa a última lista de busca/notícias/YouTube/arquivos) |
| `buscar` | pesquisa no DuckDuckGo (ddgs, com reserva só-stdlib) + resumo curto; guarda os links |
| `ler_pagina` | `leia https://…` / `leia o 1º resultado` → extrai o texto para resumir |
| `calendario` | iCal (.ics, abre no Google Agenda/Outlook): `marque dentista amanhã às 15h`, `academia toda segunda às 7h`, `o que tenho essa semana`, `próximo compromisso`, `quando é a prova?`, `remarque… para sexta às 10h`, `cancele…`. Recorrência, duração, alarme, conflitos, feriados |
| `clima` | Open-Meteo: agora, hora a hora, chuva, 7 dias, UV, vento, qualidade do ar. `vai chover amanhã no Rio?` · `defina minha cidade como Campinas` |
| `noticias` | g1, CNN Brasil, Folha, UOL, BBC, Tecnoblog + Google Notícias por assunto: `notícias de tecnologia`, `notícias sobre o dólar`, `manchetes do g1` |
| `youtube` | `toque lofi hip hop` (abre) · `pesquise vídeos de …` (só lista) |
| `imagens` | busca, abre ou baixa (`baixe uma imagem de capivara`) |
| `organizador` | `organize meus downloads` → mostra o **plano**; `confirmar organização` executa; `desfazer organização` reverte. Só move (nunca apaga/sobrescreve), ignora downloads em andamento e atalhos, só dentro da sua pasta de usuário |
| `arquivos` | `ache o contrato.pdf` · `últimos arquivos baixados` · `arquivos maiores que 200 MB` · `abra a pasta downloads` |
| `tarefas` | listas de tarefas/compras: `anote leite, ovos e pão na lista de compras` · `adicione tarefa pagar a luz até sexta` · `concluí a tarefa 2` |
| `conversor` | `10 km em milhas` · `50 dólares em reais` (cotação ao vivo) · `15% de 230` · `(15+3)*4` |
| `data` | `quantos dias faltam para o Natal` · `que horas são em Tóquio` · `próximo feriado` · `que dia da semana foi 7/9/1822` |
| `cripto` | CoinGecko: `preço do bitcoin` · `top 10 criptomoedas` · `gráfico do ethereum 30 dias` |
| `dispositivo` | `info do computador` · `memória livre` · `espaço nos discos` · `estou com internet?` · `qual meu ip` |
| `gerador` | `senha forte de 20 caracteres` · `PIN de 6 dígitos` · `role 2d6` · `sorteie entre pizza ou sushi` |
| `traduzir` | `traduza "bom dia" para o japonês` (via LLM) |
| `pesquisa_profunda` | **pesquisa, entra nos melhores links, lê as páginas e sintetiza com fontes [1][2]**: `pesquisa a fundo sobre energia solar`, `compare X e Y pesquisando na web`. Guarda as fontes (`abra a 2ª fonte`, `leia a 3ª`). `ASTRA_RESEARCH_CHARS_PER_SOURCE` (padrão 2200) ajusta o quanto lê de cada página |
| `ler_pagina` | **entra no link e sintetiza**; também segue links da própria página: `leia https://…`, `resuma o 2º resultado`, `entre no 3º link`, `entre no link Contato` |
| `resumir` | resume/sintetiza **arquivos** (pdf, docx, pptx, txt, md, csv, html), link ou texto colado: `resuma o arquivo relatorio.pdf`, `resuma o último arquivo baixado` (PDF precisa de `pypdf`) |
| `wiki` | `quem foi Santos Dumont`, `o que é blockchain` (Wikipédia pt, reserva en) |
| `notas` | `anote que o wifi é azul42`, `o que anotei sobre wifi`, `minhas notas`, `apague a nota 3` (busca por semelhança) |
| `lembrete` | `me lembre em 10 minutos de tirar o bolo`, `timer de 5 minutos`, `me lembre às 15h de ligar`, `meus lembretes` — avisa sozinho (processo em segundo plano; não sobrevive a reiniciar o PC) |
| `cotacoes` | `dólar hoje`, `cotação do euro`, `cotações` |
| `site_status` | `o github está fora do ar?`, `status do google.com` |
| `bomdia` | agenda de hoje + tarefas + clima + manchetes em um comando |

Dependências opcionais: `ddgs` (busca/imagens melhores), `psutil` (dispositivo completo), `pypdf` (resumir PDF).
Textos de páginas/arquivos vão ao LLM marcados como **conteúdo externo não confiável** (instruções escritas neles são ignoradas).
O `sandbox` agora repassa ao agente só variáveis **não secretas** (HOME/USERPROFILE, idioma, fuso,
certificados) — chaves de API continuam fora.
Agente determinístico = `"resposta": "direta"` no JSON; para criar o seu, copie um existente
e use scripts em `agents/scripts/` (veja `_astra_common.py`, com helpers de HTTP seguro, datas pt-BR e atalhos).

## Executar agentes sem esperar resposta (modo launch)

Para agentes que abrem um gráfico, janela ou programa, a Astra **só dispara** o agente em segundo plano,
diz "o agente X foi executado" e volta ao chat — nada vai para o modelo, e ela avisa quando o agente termina
(`✔ terminou` / `✖ falhou`). Exemplo: `execute o agente Grafico_carteira` (maiúsculas, `_` e acentos não importam).

- No JSON do agente: `"modo": "lancar"` (sempre dispara), `"modo": "resposta"` (sempre espera) ou sem `modo`:
  dispara só quando o pedido diz *execute/rode/inicie/lance o agente X*, e o agente não usa `llm_transform` nem `"resposta":"direta"`.
- O agente roda desacoplado (sobrevive a fechar a Astra; não sobrevive a reiniciar o PC). Saída e erros ficam em
  `sandbox_data/launches/*.log`; `/execucoes` lista os disparados na sessão.
- Scripts que ficam abertos (janela/gráfico bloqueante) têm tempo de até 1h (`ASTRA_LAUNCH_TIMEOUT`, em segundos); ponha um `timeout` alto no passo.
- Exemplo pronto: `grafico_carteira` lê `carteira/carteira.csv` (`ativo,valor`; cria um exemplo se faltar) e abre `carteira/grafico.html`.

## Python em linha e agentes de código (sem script ao lado do JSON)

A Astra pode escrever e executar Python na hora. **O código sempre aparece no chat** (com número de linha) antes de rodar,
e ela espera o resultado.

- **Ferramenta `run_python`** (o modelo usa sozinho no ReAct): roda um trecho e recebe stdout/stderr de volta.
- **Agente com `"modo": "codigo"`** — só um JSON, sem `.py`:
  ```json
  {"name": "meu_agente", "description": "...", "modo": "codigo",
   "instrucoes": "o que o código deve fazer (o pedido chega em CONTEXTO)",
   "saida": "auto", "rede": false, "timeout": 30, "tentativas": 3}
  ```
  A Astra pede o código ao modelo, mostra, roda; se falhar, manda o erro de volta ao modelo e tenta de novo (até `tentativas`).
  O código lê o pedido com `json.load(sys.stdin)['contexto']`. Com `"codigo": "..."` fixo, nem usa o modelo.
- **`saida`**: `mostrar` (imprime o stdout como está), `modelo` (o LLM explica o resultado) ou `auto` (mostra se tiver até
  `ASTRA_CODE_SHOW_MAX`=1500 caracteres, senão o modelo resume).
- **Segurança**: roda com `python -I` na pasta de trabalho da sessão. Código que apaga/move/sobrescreve arquivos, usa
  subprocess/socket/eval etc. **pede confirmação** antes (`ASTRA_CODE_CONFIRM=risky` padrão; `always` ou `never`).
  Com `SANDBOX_BACKEND=host` (padrão) a rede não é isolada; use `docker` para isolamento real.
- Exemplos: `python_rapido` (qualquer tarefa curta: “calcule…”, “quantos dias até…”, “analise esse csv”) e
  `listar_pasta_py` (código fixo). Também dá para a Astra criar esses agentes via `create_agent` com `modo: codigo`.
