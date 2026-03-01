# Project Script: "Everything is an Interface"

## Prologue: The Momentum (Spring 2025)
*   **The High:** I was riding high.
    *   **Google AI Hackathon Winner:** Built *Padayon Ko* (Scholarship AI), won $8k.
    *   **The Internship:** AI Engineer at **Dell**, working on NVIDIA GB200 chips.
    *   **The Funding:** Secured **$25k Angel Investment** for my BCI startup "MindSet".
    *   **The Vanity:** My face was on a **Time Square Billboard**.
*   **The Reality Check:** I pitched "MindSet" (Telepathy) to Prof. Watson.
    *   **His Feedback:** "The science fiction vision of BCI is driven by a **magical belief**. You believe the marketing. This isn't magic."
    *   **The Pivot Advice:** "Never let your past self who didn't know as much hold your current self hostage."
    *   **The Result:** I scrapped "Telepathy". I needed to start smaller. I needed... a button.

## Act 1: The "Punch" Phase (Sept - Oct 2025)
*   **The "Silksong" Controller:** "I needed to control my computer without touching it."
    *   **Phase 1 (The Hack):** I strapped an iPhone to my chest. It was... gross.
    *   **Phase 2 (The Smartwatch):** I wrote a Wear OS app for my **Google Pixel Watch**.
    *   **The Math:** It wasn't just shaking. I used **World-Frame Transformation** (Linear Algebra) to cancel out gravity and wrist rotation.
    *   **The Result:** I played *Hollow Knight* by punching the air. It worked, but I looked ridiculous.
    *   **Realization:** "Motion is gross. I want to capture *intent*, not just measurement."

## Act 2: The "Muscle" Phase (Nov 2025)
*   **The Upgrade:** I bought a **$5 Heart Sensor (AD8232)** meant for ECG.
*   **The Catastrophe:** "The Floating Ground." My body became an antenna for 60Hz mains hum. I spent weeks debugging noise until I realized I needed a reference electrode on my *bone*.
*   **The Breakthrough:** Once grounded, I could detect a fist clench *milliseconds* before my hand actually moved.
*   **The Limit:** It was **Binary**. On/Off.
    *   **Problem:** To get fine control, I needed an array of 16 sensors. I had one.
    *   **The "Palm Pilot" Realization:** Prof. Watson told me: "Palm Pilot won because they didn't try to read handwriting. They taught humans a new alphabet (**Graffiti**)."
    *   **The Pivot:** "I don't need to read the user's mind. I need to **teach the user** how to speak to the machine."

## Side Quest: The Tools I Built to Survive (Dec 2025)
**"Don't chase butterflies. Build a garden."**
I realized I wasn't just building a device; I was building an ecosystem. This work won me the **Anthropic "Keep Learning" Award**.

### 1. OpenEMG: "The OS for Neural Data" (@`121225_OPENEMG`)
*   **The Problem:** Commercial EMG software sucks. It’s $20k and Windows-only.
*   **The Solution:** I built **OpenEMG**, a browser-based studio.
*   **Tech Stack:** React, TypeScript, **Web Serial API** (Driverless), standard HTML5 Canvas (60fps).
*   **The "Vibe Coding":** I used Gemini 3 Pro to architect the entire rendering engine.

### 2. AI Pomodoro Logger: "My Lab Partner" (@`02_Capstone-Work-Product`)
*   **The Problem:** I forget what I did 5 minutes ago (ADHD).
*   **The Solution:** A CLI tool that takes screenshots & audio recordings every 25 minutes.
*   **The AI:** It uses Gemini to summarize my work into a **Dev Diary**.
*   **Result:** It wrote **50,000 words** of documentation for me automatically.

### 3. The Portfolio: "Building in Public" (@`somach.vercel.app`)
*   **The Content:** **23 Technical Blog Posts** (War Stories, Failures, Pivots).
*   **The Aesthetic:** "Departure" Design System (Deep Void Black & Technical Orange).
*   **Why:** "If it looks like a scrappy science project, people ignore it. If it looks like **Stark Industries**, they listen."

### 4. The Hidden Utilities (The Glue)
*   **Playwright MCP (`omi-uber`):** A custom Model Context Protocol server that let AI book Ubers for me via voice.
*   **SensorStreamer (Android/WearOS):** The C# app I wrote to stream raw accelerometer data from my **Pixel Watch** to Unity.
*   **SilksongController (Unity):** The middleware that translated "Punch" signals into "Key Press 'X'" to play *Hollow Knight*.

## Act 3: The "AlterEgo" Dream (Jan 2026)
*   **The Goal:** Replicate MIT's $3,000 AlterEgo headset with my $15 hardware.
*   **The Insight:** "MIT used 7 sensors. I have 3. But I can use **'Internal Shouting'**."
    *   **Technique:** Deliberately tensing the tongue and throat without opening the mouth.
    *   **Result:** Signals were 5x stronger.
*   **The Pitch:** I presented **SOMACH** at **T-Hub Elevation Day 2026** in Hyderabad.
*   **The Setup:** A messy web of wires taped to my face. 3 Channels (Chin, Left Throat, Right Throat).
*   **The Crisis:** Experts looked at my sensor and said, *"True subvocalization is impossible with this noise floor."* And then... **Sensor #3 died.**

## Act 4: The Pivot (Feb 2026 - Present)
*   **The Disaster:** Feb 10, 2026. My third sensor broke. I only had 2 left.
    *   **The Realization:** "Left and Right throat signals are **90% collinear**. I don't need 3 sensors. I need **2**."
*   **The New Thesis:** The **LinguaMandibular Interface**.
    *   **Channel 1 (Chin):** Reads the Tongue (Articulation).
    *   **Channel 2 (Throat):** Reads the Voice (Phonation).
*   **The Innovation:** **Curriculum Learning**.
    *   I didn't teach the AI to read minds instantly.
    *   I taught it like a baby: "Speak Loud" -> "Mouth It" -> "Exaggerate It" -> "Think It".
*   **The Result:**
    *   **99.7% Accuracy** on Exaggerated Speech (The "Bridge").
    *   **95.0% Accuracy** on Silent Speech (The "Goal").
    *   **Note:** *This is far better than the 53% we started with.*
*   **The "Model War":** I asked Watson if I should build a Transformer.
    *   **His Advice:** "Just get it above zero. A dumb model that works today is better than a perfect model that works never."
    *   **The Validation:** My simple CNN beat a Transformer model because **Execution > Hype**.

---

## Achievements Unlocked
"This wasn't just a coding project. Along the way, I:"
1.  **Google AI for Impact Hackathon:** Won **$8,000**.
2.  **Y Combinator:** Selected for **AI Startup School** (Top 1% of 20,000).
3.  **Nature Scientific Data:** Contributed to the Inner Speech Dataset.
4.  **arXiv:** Published a first-author paper on Pareto-optimal EMG control.
5.  **MIT Media Lab:** Submitted my portfolio (with 20 seconds to spare).
6.  **Dell Technologies:** Built chaos engineering tools as an intern.

---

## Epilogue 0: The "Hard Part" (Jan 2026)
*   **The Final Meeting:** I showed the messy, wired-up, working prototype to Watson.
*   **The Verdict:** "You are one of the few students I'd send to a real ECE program. Because this requires doing some **actual electrical engineering**."
*   **The Value:** "If you post this online—'Hey, for $100 you can type silently'—people will listen. Because it's not a render. It's real."
*   **The Takeaway:** Ideation is cheap. Getting a $5 sensor to control a computer is engineering.

## Epilogue 1: "It's Not Magic, It's Physics"
"So, did I build a Universal Mind Reader? No.
I built a **Personal Cognitive Augmenter**.
*   It works for *me*, right *now*, because I calibrated it.
*   Just like FaceID works for *your* face.
*   I went from punching the air to silently commanding my computer.
*   I'm not Theranos. It actually works. And it cost $40."

## Epilogue 2: The Future (LifeOS)
"But the sensor is just the input.
My goal isn't just to detect words. It's to **offload my brain**.
Imagine:
1.  I'm washing dishes. I remember I need to buy milk.
2.  I mouth *'Task: Milk'*.
3.  My **LifeOS Agent** (running clearly on a server) catches that intent, checks my fridge inventory, and adds it to my Instacart.
Measurement -> Intent -> Agent -> Action.
That is the future of **AI Wearables**."

## The Blooper Reel (Real Engineering is Messy)
"Also, in case you think I'm a genius, here is what went wrong:
1.  **The Beard Issue:** I had to shave my neck 3 times a week because stubble creates static noise that looks like 'screaming'.
2.  **The 'Ed Sheeran' Incident:** I spent 2 weeks reading a paper called 'Thinking Out Loud' and thought it was about Ed Sheeran. It was about Neurology.
3.  **The Sweat Bug:** If I get nervous during a demo, I sweat. Sweat changes conductivity. Suddenly 'UP' becomes 'DOWN'. So... please don't make me nervous."

## The Conclusion
"I started this year punching the air. I ended it by building a **Silent Speech Interface** on a $5 chip.
I’m not Theranos. I’m not magic. I’m just an engineer who refused to give up."
