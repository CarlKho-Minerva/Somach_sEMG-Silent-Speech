// AlterEgo EMG Reader — N-Channel Adaptive
// Reads AD8232 sensors and prints labeled values over USB.
// Add or remove pins in CHANNEL_PINS[] to match your hardware.

const int CHANNEL_PINS[] = {34, 36};  // GPIO pins for each AD8232
// Examples:
//   1 channel:  {34}
//   2 channels: {34, 36}
//   3 channels: {34, 36, 35}
const int NUM_CHANNELS = sizeof(CHANNEL_PINS) / sizeof(CHANNEL_PINS[0]);

void setup() {
    Serial.begin(115200);           // USB communication speed
    analogReadResolution(12);       // 12-bit ADC (0-4095)
    analogSetAttenuation(ADC_11db); // Full voltage range
    Serial.println("READY");
}

void loop() {
    for (int i = 0; i < NUM_CHANNELS; i++) {
        if (i > 0) Serial.print(" ");
        Serial.print("CH");
        Serial.print(i + 1);
        Serial.print(":");
        Serial.print(analogRead(CHANNEL_PINS[i]));
    }
    Serial.println();

    delay(100);  // ~10 readings per second
}