# SyraVisage

Biblioteca JavaScript da OpenBySyra para obter coordenadas de pontos faciais pela câmera, no navegador.

## Uso rápido

```html
<video id="camera" playsinline></video>
<canvas id="overlay"></canvas>
<script src="./SyraVisage.js"></script>
<script>
  const visage = new SyraVisage({
    video: document.querySelector('#camera'),
    canvas: document.querySelector('#overlay'),
    onResults: ({ faces }) => {
      if (faces[0]) console.log(faces[0].coordinates);
    }
  });
  visage.start();
</script>
```

## Retorno

Cada face inclui:

- `coordinates`: todos os pontos detectados, com `x`, `y`, `z`, `px` e `py`.
- `boundingBox`: área da face em coordenadas normalizadas e pixels.
- `landmarks`: atalhos para nariz, testa, queixo, olhos e cantos da boca.

`x` e `y` são coordenadas normalizadas (0 a 1). `px` e `py` são coordenadas em pixels da imagem. A biblioteca usa MediaPipe Face Mesh carregado via CDN, portanto a demonstração precisa de internet e permissão de câmera.

## Opções

| Opção | Padrão | Descrição |
|---|---:|---|
| `video` | obrigatório | elemento de vídeo da câmera |
| `canvas` | — | canvas opcional para prévia e malha |
| `mirror` | `true` | espelha a visualização e as coordenadas retornadas |
| `drawMesh` | `true` | desenha a malha facial |
| `maxFaces` | `1` | máximo de faces rastreadas |
| `onResults` | — | recebe faces e dimensões de cada quadro |
| `onError` | — | recebe falhas de câmera ou carregamento |

## Créditos

OpenBySyra — [@SyraDevOps](https://github.com/SyraDevOps).
