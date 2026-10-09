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
