# SyraVisionHands

Biblioteca JavaScript para obter as 21 coordenadas de cada mão detectada e medir a distância entre a ponta do polegar e do indicador.

```js
const hands = new SyraVisionHands({
  video: document.querySelector('video'),
  canvas: document.querySelector('canvas'),
  onResults: ({ hands }) => console.log(hands[0].coordinates, hands[0].pinchDistance)
});
hands.start();
```

Cada mão retorna `coordinates`, `handedness`, `thumbTip`, `indexTip` e `pinchDistance` em pixels. Requer câmera, MediaPipe via CDN e permissão do navegador.

Créditos: [@SyraDevOps](https://github.com/SyraDevOps).
