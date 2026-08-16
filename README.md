# OpenBySyra

Coleção open source da **SyraDevOps** para experimentar interfaces web, visão computacional, voz e hardware conectado.

Projetos simples de integrar, com demos locais e foco em transformar sinais do mundo real em experiências na web.

> Tecnologia com propósito.

## Índice

| Projeto | O que faz | Demo | Documentação |
|---|---|---|---|
| [SyraVisage](#syravisage) | Coordenadas faciais em tempo real | [Abrir](Scripts/demo-syravisage.html) | [Ler](mds/SyraVisage.md) |
| [SyraIris](#syrairis) | Rastreamento de íris e métricas de olhar | [Abrir](Scripts/demo-syrairis.html) | [Ler](mds/SyraIris.md) |
| [SyraVisionHands](#syravisionhands) | Coordenadas das mãos e distância de pinça | [Abrir](Scripts/demo-syravisionhands.html) | [Ler](mds/SyraVisionHands.md) |
| [SyraEcho](#syraecho) | Voz para texto em PT-BR e EN-US | [PT-BR](Scripts/demo-syraecho-pt.html) · [EN-US](Scripts/demo-syraecho-en.html) | [Ler](mds/SyraEcho.md) |
| [Open-Bridge-Syra](#open-bridge-syra) | Nó local Wi-Fi e mensagens para ESP8266 | — | [Ler](Firmware/Open-Bridge-Syra.md) |
| [SkyMonitor](#skymonitor) | Monitoramento de estrelas, meteoros e nuvens | — | [Ler](mds/SkyMonitor.md) |

## Instalação

### Opção 1 — baixar o projeto

Baixe o arquivo ZIP, extraia-o e abra a pasta `OpenBySyra` no seu editor.

### Opção 2 — Git

```bash
git clone https://github.com/SyraDevOps/OpenBySyra.git
cd OpenBySyra
```

Para as demos que usam câmera ou microfone, execute um servidor local. Por exemplo:

```bash
python -m http.server 8080
```

Depois acesse `http://localhost:8080/Scripts/demo-syravisage.html` no navegador.

### SkyMonitor

Instale as dependências do SkyMonitor antes de executá-lo:

```bash
pip install -r Scripts/requirements.txt
python Scripts/SkyMonitor.py
```

## Uso rápido

Inclua o script desejado e inicialize a biblioteca com os elementos de vídeo e canvas.

```html
<video id="camera" playsinline></video>
<canvas id="overlay"></canvas>
<script src="./Scripts/SyraVisage.js"></script>
<script>
  const visage = new SyraVisage({
    video: document.querySelector('#camera'),
    canvas: document.querySelector('#overlay'),
    onResults: ({ faces }) => console.log(faces)
  });
  visage.start();
</script>
```

As bibliotecas de visão computacional carregam o MediaPipe pela CDN. As demos exigem permissão explícita de câmera; o SyraEcho exige permissão de microfone.

## Projetos

### SyraVisage

Obtém pontos do rosto, coordenadas normalizadas/pixels, caixa delimitadora e atalhos para olhos, nariz e boca.

### SyraIris

Obtém pontos da íris e dos olhos, abertura estimada dos olhos e posição média do olhar.

### SyraVisionHands

Rastreia até duas mãos, retornando 21 landmarks por mão e a distância entre as pontas do polegar e do indicador.

### SyraEcho

Transforma voz em texto com suporte a `pt-BR` e `en-US`, palavras-chave e visualização de áudio.

### Open-Bridge-Syra

Firmware para ESP8266 com configuração Wi-Fi, mDNS, lista de dispositivos e ponte local de mensagens.

### SkyMonitor

Aplicação Python com interface Tkinter para observar estrelas, meteoros, nuvens e triangulações a partir de RTSP, webcam, vídeo ou imagens.

## Estrutura

```text
OpenBySyra/
├── Firmware/   # Arduino/ESP8266
├── Mkt/        # artes e legendas de divulgação
├── Scripts/    # bibliotecas JavaScript e demos
└── mds/        # documentação individual
```

## Créditos

Criado por [@SyraDevOps](https://github.com/SyraDevOps).

## Licença

Este projeto usa a [SyraDevOps Community Source License 1.0](LICENSE.md): a comunidade pode usar, estudar e compartilhar o código sem alterações. Modificações e redistribuição de versões modificadas não são permitidas.
