# Open-Bridge-Syra

Firmware para ESP8266 que cria um nó local de configuração e comunicação da Syra.

## O que ele faz

- Cria a rede `Syra-Config` quando não há Wi-Fi configurado.
- Salva as credenciais de Wi-Fi na EEPROM.
- Disponibiliza uma interface web local e endereço mDNS.
- Registra até 10 dispositivos locais.
- Oferece a ponte de mensagens em memória pela rota `/bridge`.
- Mostra o estado do nó pela rota `/info`.

## Hardware e dependências

- Placa ESP8266 compatível (NodeMCU, Wemos D1 mini ou equivalente).
- Pacote ESP8266 instalado na Arduino IDE.
- Bibliotecas incluídas no pacote ESP8266: `ESP8266WiFi`, `ESP8266WebServer`, `ESP8266mDNS` e `EEPROM`.

## Como usar

1. Abra `Open-Bridge-Syra.ino` na Arduino IDE.
2. Selecione sua placa ESP8266 e a porta serial correta.
3. Faça o upload.
4. Na primeira inicialização, conecte-se à rede Wi-Fi `Syra-Config`.
5. Acesse `http://192.168.4.1` e informe sua rede Wi-Fi.
6. Depois de conectado, acesse o endereço exibido no Monitor Serial ou `http://Syra-Home-SYRA0156326.local` quando a rede suportar mDNS.

## Rotas disponíveis

| Rota | Método | Função |
|---|---|---|
| `/` | GET | Interface do nó / tela de configuração |
| `/info` | GET | Estado, IP, identificador e mDNS |
| `/dispositivos` | GET / POST / DELETE | Lista e gerencia dispositivos |
| `/bridge` | GET / POST | Recebe e entrega mensagens locais |
| `/disconnect` | GET | Volta ao modo de configuração |

## Créditos

Desenvolvido por [@SyraDevOps](https://github.com/SyraDevOps), como parte do **OpenBySyra**.
