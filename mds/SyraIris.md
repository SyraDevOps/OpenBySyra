# SyraIris

Biblioteca JavaScript para rastrear íris, contorno dos olhos e métricas simples de abertura e direção do olhar.

```js
const iris = new SyraIris({
  video: document.querySelector('video'),
  canvas: document.querySelector('canvas'),
  onResults: ({ leftIris, rightIris, metrics }) => console.log(leftIris, rightIris, metrics)
});
iris.start();
```

Retorna coordenadas normalizadas e pixels de cada ponto, além de `leftEyeOpening`, `rightEyeOpening`, `horizontalGaze` e `verticalGaze`. Usa MediaPipe Face Mesh via CDN e requer permissão de câmera.

Créditos: [@SyraDevOps](https://github.com/SyraDevOps).
