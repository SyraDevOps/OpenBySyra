# SkyMonitor

Aplicação Python com interface gráfica para monitorar o céu a partir de uma câmera RTSP/IP, webcam local, arquivo de vídeo ou pasta de imagens.

O SkyMonitor detecta e acompanha estrelas, meteoros e nuvens, além de desenhar triangulações entre estrelas estáveis. Os resultados são salvos em uma pasta organizada para análise posterior.

## Requisitos

- Python 3.10 ou superior.
- Webcam, câmera IP/RTSP, vídeo ou conjunto de imagens.
- Windows, Linux ou macOS com suporte ao Tkinter.

## Instalação

Na pasta do projeto, instale as dependências:

```bash
pip install numpy opencv-python pillow scipy
```

O Pillow e o SciPy são opcionais, mas recomendados para a visualização e o mapeamento de constelações.

## Como usar

1. Abra um terminal na pasta `OpenBySyra/Scripts`.
2. Execute:

   ```bash
   python SkyMonitor.py
   ```

3. Escolha a fonte de entrada:

   - **Câmera RTSP/IP:** informe uma URL como `rtsp://usuario:senha@ip:554/caminho`.
   - **Câmera do dispositivo:** use `0` para a webcam principal, ou outro índice.
   - **Arquivo de vídeo:** clique no ícone de pasta e selecione um vídeo.
   - **Pasta de imagens:** selecione a pasta com imagens do céu.

4. Defina a pasta de saída, por exemplo `dataset_sky_monitor`.
5. Marque os elementos que deseja visualizar: estrelas, meteoros, nuvens e triangulações.
6. Clique em **INICIAR**. Use **PARAR** para interromper o processamento.

## Saídas geradas

Na pasta escolhida, o SkyMonitor cria:

```text
dataset_sky_monitor/
├── stars/identified/  # estrelas confirmadas
├── stars/grids/       # grades de recortes de estrelas
├── meteors/frames/    # frames com meteoros detectados
├── meteors/patches/   # recortes de meteoros
├── meteors/grids/     # grades de meteoros
├── sky_maps/          # mapas do céu
├── clouds/            # registros de nuvens
├── constellations/    # triangulações estáveis
└── filtered/          # frames filtrados
```

## Observações

- Para monitoramento noturno, use uma câmera fixa e reduza luzes diretas no campo de visão.
- As credenciais de RTSP devem ser inseridas apenas no seu ambiente local; não publique URLs com senha.
- O software usa OpenCV, Tkinter, NumPy e, quando disponível, SciPy para análise espacial.

## Créditos

OpenBySyra — [@SyraDevOps](https://github.com/SyraDevOps).
