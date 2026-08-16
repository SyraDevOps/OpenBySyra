# SyraEcho

Biblioteca JavaScript da OpenBySyra que une reconhecimento de voz em português e inglês, com transcrição contínua, palavras-chave e visualização de áudio.

```js
const echo = new SyraEcho({
  lang: 'pt-BR',
  targetWords: ['olá', 'syra'],
  onResult: ({ text, matches }) => console.log(text, matches)
});
echo.start();
```

Use `lang: 'pt-BR'` ou `lang: 'en-US'`. As duas demonstrações estão em `Scripts`. O recurso depende da Web Speech API e de permissão para usar o microfone.

Créditos: [@SyraDevOps](https://github.com/SyraDevOps).
