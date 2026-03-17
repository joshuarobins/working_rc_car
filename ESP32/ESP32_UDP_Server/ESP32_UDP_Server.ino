#include <WiFi.h>
#include <WiFiUdp.h>
#include <L298N.h>


// Wifi Setup
const char* ssid = "JOSH-RC-CAR";
const char* password = "password";

// UDP Server Setup
WiFiUDP udp;
const int udpPort = 4210;

char incomingPacket[255];

// Pin definition
const unsigned int ENA = 6;
const unsigned int IN1 = 7;
const unsigned int IN2 = 17;

const unsigned int IN3 = 18;
const unsigned int IN4 = 11;
const unsigned int ENB = 13;

// Initialize both motors
L298N leftMotor(ENA, IN1, IN2);
L298N rightMotor(ENB, IN3, IN4);

int left_motor, right_motor;
int throttle, turn;


void setup() {
  Serial.begin(115200);
  
  //IPAddress local_IP(192,168,137,50);
  //IPAddress gateway(192,168,137,1);
  //IPAddress subnet(255,255,255,0);

  //WiFi.config(local_IP, gateway, subnet);
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");

  Serial.print("IP Adress: ");
  Serial.println(WiFi.localIP());
  
  udp.begin(udpPort);
  Serial.println("UDP server started");

}

void loop() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(incomingPacket, 255);
    if (len > 0) {
      incomingPacket[len] = 0; // null-terminate string
    }

    Serial.println(incomingPacket);
    
    // Parse "throttle,steering"
    char* comma = strchr(incomingPacket, ',');
    if (comma) {
      *comma = 0;
      throttle = atoi(incomingPacket);
      turn     = atoi(comma + 1);
    }

    // Mix to motor outputs
    left_motor  = throttle + turn;
    right_motor = throttle - turn;

    // Constrain to [-255, 255]
    left_motor  = constrain(left_motor,  -255, 255);
    right_motor = constrain(right_motor, -255, 255);

    // Left motor
    if (left_motor == 0) {
      leftMotor.stop();
    } else if (left_motor > 0) {
      leftMotor.forward();
      leftMotor.setSpeed(left_motor);
    } else {
      leftMotor.backward();
      leftMotor.setSpeed(-left_motor);
    }

    // Right motor
    if (right_motor == 0) {
      rightMotor.stop();
    } else if (right_motor >= 0) {
      rightMotor.forward();
      rightMotor.setSpeed(right_motor);
    } else {
      rightMotor.backward();
      rightMotor.setSpeed(-right_motor);
    }
    
  }
}


//Print some informations in Serial Monitor
void printSomeInfo()
{
  Serial.print("Right Motor is moving = ");
  Serial.print(rightMotor.isMoving() ? "YES" : "NO");
  Serial.print(" at speed = ");
  Serial.println(rightMotor.getSpeed());
  Serial.print("Left Motor is moving = ");
  Serial.print(leftMotor.isMoving() ? "YES" : "NO");
  Serial.print(" at speed = ");
  Serial.println(leftMotor.getSpeed());
}
