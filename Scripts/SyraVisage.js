/*
 * SyraVisage.js — Face landmark coordinates for the web
 * OpenBySyra | Credits: @SyraDevOps
 * https://github.com/SyraDevOps
 */
(function (global) {
  "use strict";

  const MEDIAPIPE = "https://cdn.jsdelivr.net/npm/@mediapipe";
  const loaded = new Map();

  function loadScript(src) {
    if (loaded.has(src)) return loaded.get(src);
    const task = new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) {
        if (existing.dataset.syravisageLoaded === "true") return resolve();
        return existing.addEventListener("load", resolve, { once: true });
      }
      const script = document.createElement("script");
      script.src = src;
      script.crossOrigin = "anonymous";
      script.onload = () => { script.dataset.syravisageLoaded = "true"; resolve(); };
      script.onerror = () => reject(new Error(`Não foi possível carregar ${src}`));
      document.head.appendChild(script);
    });
    loaded.set(src, task);
    return task;
  }

  async function loadMediaPipe() {
    await Promise.all([
      loadScript(`${MEDIAPIPE}/drawing_utils/drawing_utils.js`),
      loadScript(`${MEDIAPIPE}/face_mesh/face_mesh.js`)
    ]);
  }

  function makePoint(point, width, height, mirror) {
    const x = mirror ? 1 - point.x : point.x;
    return {
      x,
      y: point.y,
      z: point.z,
      px: Math.round(x * width),
      py: Math.round(point.y * height)
    };
  }

  function makeFace(landmarks, width, height, mirror) {
    const coordinates = landmarks.map((point) => makePoint(point, width, height, mirror));
    const xs = coordinates.map((point) => point.x);
    const ys = coordinates.map((point) => point.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    return {
      coordinates,
      boundingBox: {
        x: minX, y: minY, width: maxX - minX, height: maxY - minY,
        px: Math.round(minX * width), py: Math.round(minY * height),
        pixelWidth: Math.round((maxX - minX) * width),
        pixelHeight: Math.round((maxY - minY) * height)
      },
      landmarks: {
        noseTip: coordinates[1],
        forehead: coordinates[10],
        chin: coordinates[152],
        leftEye: coordinates[33],
        rightEye: coordinates[263],
        mouthLeft: coordinates[61],
        mouthRight: coordinates[291]
      }
    };
  }

  class SyraVisage {
    constructor(options = {}) {
      if (!options.video) throw new Error("SyraVisage requer um elemento <video>.");
      this.video = options.video;
      this.canvas = options.canvas || null;
      this.context = this.canvas ? this.canvas.getContext("2d") : null;
      this.onResults = options.onResults || (() => {});
      this.onError = options.onError || ((error) => console.error("SyraVisage:", error));
      this.mirror = options.mirror !== false;
      this.drawMesh = options.drawMesh !== false;
      this.maxFaces = options.maxFaces || 1;
      this.refineLandmarks = options.refineLandmarks !== false;
      this.minDetectionConfidence = options.minDetectionConfidence || 0.5;
      this.minTrackingConfidence = options.minTrackingConfidence || 0.5;
      this.stream = null;
      this.faceMesh = null;
      this.running = false;
      this.processing = false;
      this.frameId = null;
    }

    async start() {
      try {
        await loadMediaPipe();
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false
        });
        this.video.srcObject = this.stream;
        this.video.playsInline = true;
        await this.video.play();
        this._resizeCanvas();
        this.faceMesh = new global.FaceMesh({
          locateFile: (file) => `${MEDIAPIPE}/face_mesh/${file}`
        });
        this.faceMesh.setOptions({
          maxNumFaces: this.maxFaces,
          refineLandmarks: this.refineLandmarks,
          minDetectionConfidence: this.minDetectionConfidence,
          minTrackingConfidence: this.minTrackingConfidence
        });
        this.faceMesh.onResults((results) => this._handleResults(results));
        this.running = true;
        this._loop();
        return this;
      } catch (error) {
        this.onError(error);
        throw error;
      }
    }

    stop() {
      this.running = false;
      if (this.frameId) cancelAnimationFrame(this.frameId);
      if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
      this.video.srcObject = null;
      return this;
    }

    _loop() {
      if (!this.running) return;
      this.frameId = requestAnimationFrame(async () => {
        if (!this.processing && this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
          this.processing = true;
          try { await this.faceMesh.send({ image: this.video }); }
          catch (error) { this.onError(error); }
          finally { this.processing = false; }
        }
        this._loop();
      });
    }

    _resizeCanvas() {
      if (!this.canvas) return;
      this.canvas.width = this.video.videoWidth || 1280;
      this.canvas.height = this.video.videoHeight || 720;
    }

    _handleResults(results) {
      const width = this.video.videoWidth || 1280;
      const height = this.video.videoHeight || 720;
      this._resizeCanvas();
      const rawFaces = results.multiFaceLandmarks || [];
      const faces = rawFaces.map((face) => makeFace(face, width, height, this.mirror));
      this._draw(results, rawFaces, width, height);
      this.onResults({ faces, faceCount: faces.length, width, height, timestamp: performance.now() });
    }

    _draw(results, rawFaces, width, height) {
      if (!this.context) return;
      const ctx = this.context;
      ctx.save();
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (this.mirror) { ctx.translate(width, 0); ctx.scale(-1, 1); }
      ctx.drawImage(results.image, 0, 0, width, height);
      if (this.drawMesh && global.drawConnectors) {
        rawFaces.forEach((face) => {
          global.drawConnectors(ctx, face, global.FACEMESH_TESSELATION, { color: "rgba(168, 85, 247, .38)", lineWidth: 1 });
          global.drawConnectors(ctx, face, global.FACEMESH_FACE_OVAL, { color: "#fb923c", lineWidth: 2 });
          global.drawConnectors(ctx, face, global.FACEMESH_LIPS, { color: "#fb923c", lineWidth: 2 });
          global.drawConnectors(ctx, face, global.FACEMESH_LEFT_EYE, { color: "#d8b4fe", lineWidth: 2 });
          global.drawConnectors(ctx, face, global.FACEMESH_RIGHT_EYE, { color: "#d8b4fe", lineWidth: 2 });
        });
      }
      ctx.restore();
    }
  }

  global.SyraVisage = SyraVisage;
})(window);
