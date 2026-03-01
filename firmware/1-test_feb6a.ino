// Single Channel Test with Diagnostics
void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
    
    Serial.println("=== AD8232 AMPLIFICATION TEST ===");
    Serial.println("Step 1: Should see stable ~1920");
    Serial.println("Step 2: Touch RA and LA wires - values should jump!");
    Serial.println("=====================================");
}

void loop() {
    
    int value = analogRead(34);
    
    // Calculate variability
    static int last10[10] = {2048,2048,2048,2048,2048,2048,2048,2048,2048,2048};
    static int idx = 0;
    
    last10[idx] = value;
    idx = (idx + 1) % 10;
    
    int minVal = 4095, maxVal = 0;
    for(int i = 0; i < 10; i++) {
        if(last10[i] < minVal) minVal = last10[i];
        if(last10[i] > maxVal) maxVal = last10[i];
    }
    int range = maxVal - minVal;
    
    Serial.print("Value: "); 
    Serial.print(value);
    Serial.print(" | Range: ");
    Serial.print(range);
    
    if(range < 10) {
        Serial.println(" [STABLE - touch wires to test!]");
    } else if(range < 50) {
        Serial.println(" [SLIGHT MOVEMENT]");
    } else if(range < 200) {
        Serial.println(" [GOOD MOVEMENT]");
    } else {
        Serial.println(" [STRONG SIGNAL! ✓]");
    }
    
    delay(10);
}