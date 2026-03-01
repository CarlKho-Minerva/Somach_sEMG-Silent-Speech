// 3-high-speed-capture.ino
// N-Channel Adaptive EMG Capture for AlterEgo Pipeline
// Automatically adapts CSV output to the number of configured channels.

// ==========================================
// CHANNEL CONFIGURATION
// Add or remove pins to match your hardware.
// The entire pipeline (Python scripts 4-9) auto-adapts.
// ==========================================
const int CHANNEL_PINS[] = { 34, 36 };  // GPIO pins for each AD8232
// Examples:
//   1 channel (chin only):       {34}
//   2 channels (chin + throat):  {34, 36}
//   3 channels (chin + L + R):   {34, 36, 35}
const int NUM_CHANNELS = sizeof(CHANNEL_PINS) / sizeof(CHANNEL_PINS[0]);

const unsigned long SAMPLE_INTERVAL_MICROS = 4000;  // 250 Hz
unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < NUM_CHANNELS; i++) {
    pinMode(CHANNEL_PINS[i], INPUT);
  }
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  Serial.println("READY");
}

void loop() {
  unsigned long currentMicros = micros();
  if (currentMicros - lastSampleTime >= SAMPLE_INTERVAL_MICROS) {
    lastSampleTime = currentMicros;

    // CSV Format: Timestamp, Ch1, Ch2, ..., ChN
    Serial.print(millis());
    for (int i = 0; i < NUM_CHANNELS; i++) {
      Serial.print(",");
      Serial.print(analogRead(CHANNEL_PINS[i]));
    }
    Serial.println();
  }
}