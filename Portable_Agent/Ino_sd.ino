#include <SPI.h>
#include <SD.h>

#define SD_CS 10
#define SERIAL_BAUD 115200

String command;

void setup() {
  Serial.begin(SERIAL_BAUD);

  while (!Serial) {
    ; // Aguarda Serial
  }

  Serial.println("INIT");

  if (!SD.begin(SD_CS)) {
    Serial.println("ERROR:SD");
    return;
  }

  Serial.println("SD:OK");

  // Cria a pasta GEN se ela não existir
  if (!SD.exists("/GEN")) {
    if (SD.mkdir("/GEN")) {
      Serial.println("GEN:CREATED");
    } else {
      Serial.println("ERROR:GEN");
      return;
    }
  } else {
    Serial.println("GEN:OK");
  }

  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    command = Serial.readStringUntil('\n');
    command.trim();

    if (command.length() > 0) {
      processCommand(command);
    }
  }
}

// ------------------------------------------------------------
// PROCESSADOR DE COMANDOS
// ------------------------------------------------------------

void processCommand(String cmd) {

  // LIST
  if (cmd.startsWith("LIST")) {
    String path = "/GEN";

    if (cmd.length() > 4) {
      path = cmd.substring(5);
      path.trim();

      if (!path.startsWith("/")) {
        path = "/" + path;
      }
    }

    listFiles(path);
    return;
  }

  // READ
  if (cmd.startsWith("READ ")) {
    String path = cmd.substring(5);
    path.trim();

    path = normalizePath(path);

    readFile(path);
    return;
  }

  // WRITE
  if (cmd.startsWith("WRITE ")) {
    String path = cmd.substring(6);
    path.trim();

    path = normalizePath(path);

    writeFile(path);
    return;
  }

  // APPEND
  if (cmd.startsWith("APPEND ")) {
    String path = cmd.substring(7);
    path.trim();

    path = normalizePath(path);

    appendFile(path);
    return;
  }

  // DELETE
  if (cmd.startsWith("DELETE ")) {
    String path = cmd.substring(7);
    path.trim();

    path = normalizePath(path);

    deleteFile(path);
    return;
  }

  // EXISTS
  if (cmd.startsWith("EXISTS ")) {
    String path = cmd.substring(7);
    path.trim();

    path = normalizePath(path);

    if (SD.exists(path)) {
      Serial.println("YES");
    } else {
      Serial.println("NO");
    }

    return;
  }

  // SIZE
  if (cmd.startsWith("SIZE ")) {
    String path = cmd.substring(5);
    path.trim();

    path = normalizePath(path);

    sizeFile(path);
    return;
  }

  // MKDIR
  if (cmd.startsWith("MKDIR ")) {
    String path = cmd.substring(6);
    path.trim();

    path = normalizePath(path);

    makeDirectory(path);
    return;
  }

  // HELP
  if (cmd == "HELP") {
    printHelp();
    return;
  }

  Serial.println("ERROR:UNKNOWN_COMMAND");
}


// ------------------------------------------------------------
// NORMALIZA CAMINHO
// ------------------------------------------------------------

String normalizePath(String path) {

  path.trim();

  if (!path.startsWith("/")) {
    path = "/" + path;
  }

  // Se não estiver dentro de GEN, coloca dentro
  if (!path.startsWith("/GEN")) {
    path = "/GEN" + path;
  }

  return path;
}


// ------------------------------------------------------------
// LIST
// ------------------------------------------------------------

void listFiles(String path) {

  File dir = SD.open(path);

  if (!dir) {
    Serial.println("ERROR:OPEN_DIR");
    return;
  }

  if (!dir.isDirectory()) {
    Serial.println("ERROR:NOT_DIR");
    dir.close();
    return;
  }

  Serial.println("OK");

  File entry;

  while (true) {

    entry = dir.openNextFile();

    if (!entry) {
      break;
    }

    Serial.print(entry.name());

    if (entry.isDirectory()) {
      Serial.println("/");
    } else {
      Serial.print("|");
      Serial.println(entry.size());
    }

    entry.close();
  }

  dir.close();

  Serial.println("END");
}


// ------------------------------------------------------------
// READ
// ------------------------------------------------------------

void readFile(String path) {

  File file = SD.open(path, FILE_READ);

  if (!file) {
    Serial.println("ERROR:FILE_NOT_FOUND");
    return;
  }

  Serial.println("OK");

  while (file.available()) {
    Serial.write(file.read());
  }

  file.close();

  Serial.println();
  Serial.println("END");
}


// ------------------------------------------------------------
// WRITE
// ------------------------------------------------------------

void writeFile(String path) {

  Serial.println("READY_WRITE");

  // Espera a próxima linha contendo o conteúdo
  while (!Serial.available()) {
    delay(1);
  }

  String data = Serial.readStringUntil('\n');

  File file = SD.open(path, FILE_WRITE);

  if (!file) {
    Serial.println("ERROR:OPEN_FILE");
    return;
  }

  // FILE_WRITE adiciona no final em algumas versões.
  // Para garantir sobrescrita, remove primeiro.
  file.close();

  SD.remove(path);

  file = SD.open(path, FILE_WRITE);

  if (!file) {
    Serial.println("ERROR:CREATE_FILE");
    return;
  }

  file.print(data);

  file.close();

  Serial.println("OK");
}


// ------------------------------------------------------------
// APPEND
// ------------------------------------------------------------

void appendFile(String path) {

  Serial.println("READY_APPEND");

  while (!Serial.available()) {
    delay(1);
  }

  String data = Serial.readStringUntil('\n');

  File file = SD.open(path, FILE_WRITE);

  if (!file) {
    Serial.println("ERROR:OPEN_FILE");
    return;
  }

  file.print(data);

  file.close();

  Serial.println("OK");
}


// ------------------------------------------------------------
// DELETE
// ------------------------------------------------------------

void deleteFile(String path) {

  if (!SD.exists(path)) {
    Serial.println("ERROR:NOT_FOUND");
    return;
  }

  if (SD.remove(path)) {
    Serial.println("OK");
  } else {
    Serial.println("ERROR:DELETE");
  }
}


// ------------------------------------------------------------
// SIZE
// ------------------------------------------------------------

void sizeFile(String path) {

  File file = SD.open(path);

  if (!file) {
    Serial.println("ERROR:NOT_FOUND");
    return;
  }

  Serial.println(file.size());

  file.close();
}


// ------------------------------------------------------------
// MKDIR
// ------------------------------------------------------------

void makeDirectory(String path) {

  if (SD.exists(path)) {
    Serial.println("EXISTS");
    return;
  }

  if (SD.mkdir(path)) {
    Serial.println("OK");
  } else {
    Serial.println("ERROR:MKDIR");
  }
}


// ------------------------------------------------------------
// HELP
// ------------------------------------------------------------

void printHelp() {

  Serial.println("COMMANDS:");

  Serial.println("LIST");
  Serial.println("LIST /GEN");

  Serial.println("READ arquivo.txt");

  Serial.println("WRITE arquivo.txt");
  Serial.println("APPEND arquivo.txt");

  Serial.println("DELETE arquivo.txt");

  Serial.println("EXISTS arquivo.txt");

  Serial.println("SIZE arquivo.txt");

  Serial.println("MKDIR pasta");

  Serial.println("HELP");

  Serial.println("END");
}
