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
