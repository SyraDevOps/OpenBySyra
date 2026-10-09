#include <Arduino.h>

String comando = "";

#define LED_PIN LED_BUILTIN

// ==============================
// LED FEEDBACK (pisca 1s)
// ==============================
void blinkLED() {
  digitalWrite(LED_PIN, LOW);   // liga (ESP8266 invertido)
  delay(500);
  digitalWrite(LED_PIN, HIGH);  // desliga
  delay(500);
}

// ==============================
// ENVIO JSON
// ==============================
void sendJSON(String json) {
  blinkLED();
  Serial.println(json);
}

// ==============================
// SETUP
// ==============================
void setup() {

  pinMode(LED_PIN, OUTPUT);

  // LED ligado = sistema ativo
  digitalWrite(LED_PIN, LOW);

  Serial.begin(115200);
  delay(1500);

  sendJSON("{\"status\":\"ok\",\"message\":\"ESP8266 ONLINE\"}");
  sendJSON("{\"status\":\"ok\",\"message\":\"SYSTEM READY\"}");
}

// ==============================
// LOOP
// ==============================
void loop() {

  while (Serial.available()) {

    char c = Serial.read();

    if (c == '\n') {

      comando.trim();

      if (comando.length() > 0) {

        // LOG DO COMANDO
        Serial.print("{\"cmd\":\"");
        Serial.print(comando);
        Serial.println("\"}");

        // =====================
        // 1 - STATUS
        // =====================
        if (comando == "1") {
          sendJSON("{\"Status\":\"ok\",\"Key\":\"fvsdfvsrvsrbvsdfvsdf\", \"message\":\"System active\"}");
        }

        // =====================
        // 2 - DEVICE INFO
        // =====================
        else if (comando == "2") {
          sendJSON("{\"cmd\":2,\"device\":\"ESP8266\",\"board\":\"NodeMCU\",\"status\":\"running\"}");
        }

        // =====================
        // 3 - FIREBASE SERVICE ACCOUNT JSON
        // =====================
        else if (comando == "3") {

          String firebaseJSON =
            "{"
              "\"type\":\"service_account\","
              "\"project_id\":\"SEU_PROJECT_ID\","
              "\"private_key_id\":\"SEU_PRIVATE_KEY_ID\","
              "\"private_key\":\"-----BEGIN PRIVATE KEY-----\\nSEU_PRIVATE_KEY\\n-----END PRIVATE KEY-----\\n\","
              "\"client_email\":\"SEU_CLIENT_EMAIL\","
              "\"client_id\":\"SEU_CLIENT_ID\","
              "\"auth_uri\":\"https://accounts.google.com/o/oauth2/auth\","
              "\"token_uri\":\"https://oauth2.googleapis.com/token\","
              "\"auth_provider_x509_cert_url\":\"https://www.googleapis.com/oauth2/v1/certs\","
              "\"client_x509_cert_url\":\"SEU_CERT_URL\","
              "\"universe_domain\":\"googleapis.com\""
            "}";

          sendJSON(firebaseJSON);
        }

        // =====================
        // 4 - WIFI STATUS
        // =====================
        else if (comando == "4") {
          sendJSON("{\"cmd\":4,\"wifi\":\"connected\",\"ip\":\"192.168.0.25\"}");
        }

        // =====================
        // 5 - SYSTEM HEALTH
        // =====================
        else if (comando == "5") {
          sendJSON("{\"cmd\":5,\"system\":\"ok\",\"health\":100}");
        }

        // =====================
        // 6 - MEMORY
        // =====================
        else if (comando == "6") {
          sendJSON("{\"cmd\":6,\"free_memory\":\"38KB\"}");
        }

        // =====================
        // 7 - IP
        // =====================
        else if (comando == "7") {
          sendJSON("{\"cmd\":7,\"ip\":\"192.168.0.25\"}");
        }

        // =====================
        // 8 - RESTART
        // =====================
        else if (comando == "8") {
          sendJSON("{\"cmd\":8,\"message\":\"restarting...\"}");
          delay(500);
          ESP.restart();
        }

        // =====================
        // 9 - GPIO STATUS
        // =====================
        else if (comando == "9") {
          sendJSON("{\"cmd\":9,\"gpio\":\"ok\",\"pins\":\"stable\"}");
        }

        // =====================
        // 10 - PING
        // =====================
        else if (comando == "10") {
          sendJSON("{\"cmd\":10,\"ping\":\"pong\",\"status\":\"alive\"}");
        }

        // =====================
        // UNKNOWN COMMAND
        // =====================
        else {
          sendJSON("{\"error\":\"unknown_command\"}");
        }

        Serial.println("----------------");
      }

      comando = "";
    }

    else if (c != '\r') {
      comando += c;
    }
  }
}
