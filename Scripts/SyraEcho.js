/* SyraEcho.js — reconhecimento de voz para a web | @SyraDevOps */
(function (global) {
  "use strict";
  class SyraEcho {
    constructor(options = {}) {
      this.lang = options.lang || "pt-BR";
      this.continuous = options.continuous !== false;
      this.interimResults = options.interimResults !== false;
      this.targetWords = options.targetWords || [];
      this.canvas = options.canvas || null;
      this.onResult = options.onResult || (() => {});
      this.onStatus = options.onStatus || (() => {});
      this.onError = options.onError || ((error) => console.error("SyraEcho:", error));
      const Recognition = global.SpeechRecognition || global.webkitSpeechRecognition;
      if (!Recognition) throw new Error("Reconhecimento de voz não é suportado neste navegador.");
      this.recognition = new Recognition();
      this.recognition.lang = this.lang;
      this.recognition.continuous = this.continuous;
      this.recognition.interimResults = this.interimResults;
      this.active = false;
      this.audioStream = null;
      this.audioContext = null;
      this.analyser = null;
      this.frame = null;
      this._bind();
    }
    _bind() {
      this.recognition.onstart = () => { this.active = true; this.onStatus("listening"); };
      this.recognition.onend = () => { this.active = false; this.onStatus("stopped"); this._stopVisual(); };
      this.recognition.onerror = (event) => { this.onError(event.error); this.onStatus("error"); };
      this.recognition.onresult = (event) => {
        let final = "", interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const text = event.results[i][0].transcript.trim();
          event.results[i].isFinal ? final += `${text} ` : interim += `${text} `;
        }
        const text = (final || interim).trim();
        const lower = text.toLocaleLowerCase(this.lang);
        const matches = this.targetWords.filter((word) => lower.includes(word.toLocaleLowerCase(this.lang)));
        this.onResult({ text, final: Boolean(final), confidence: event.results[event.results.length - 1][0].confidence, matches, event });
      };
    }
    async start() {
      try { await this._startVisual(); this.recognition.start(); return this; }
      catch (error) { this.onError(error); throw error; }
    }
    stop() { if (this.active) this.recognition.stop(); else this._stopVisual(); return this; }
    setLanguage(lang) { this.lang = lang; this.recognition.lang = lang; return this; }
    setTargetWords(words) { this.targetWords = Array.isArray(words) ? words : String(words).split(",").map((word) => word.trim()).filter(Boolean); return this; }
    async _startVisual() {
      if (!this.canvas || this.analyser) return;
      this.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.audioContext = new (global.AudioContext || global.webkitAudioContext)();
      this.analyser = this.audioContext.createAnalyser(); this.analyser.fftSize = 1024;
      this.audioContext.createMediaStreamSource(this.audioStream).connect(this.analyser);
      const ctx = this.canvas.getContext("2d"), data = new Uint8Array(this.analyser.frequencyBinCount);
      const draw = () => { if (!this.analyser) return; this.frame = requestAnimationFrame(draw); this.canvas.width = this.canvas.clientWidth * devicePixelRatio; this.canvas.height = this.canvas.clientHeight * devicePixelRatio; this.analyser.getByteTimeDomainData(data); ctx.fillStyle="#061d1c";ctx.fillRect(0,0,this.canvas.width,this.canvas.height);ctx.strokeStyle="#7cf5d7";ctx.lineWidth=3*devicePixelRatio;ctx.beginPath();data.forEach((v,i)=>{const x=i*this.canvas.width/(data.length-1),y=(v/255)*this.canvas.height;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke(); };
      draw();
    }
    _stopVisual() { cancelAnimationFrame(this.frame); if (this.audioStream) this.audioStream.getTracks().forEach((track) => track.stop()); if (this.audioContext) this.audioContext.close(); this.audioStream=this.audioContext=this.analyser=null; }
  }
  global.SyraEcho = SyraEcho;
})(window);
