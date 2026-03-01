# Confirmed Electrode Wiring Mapping - Feb 1, 2026

**Context:** Based on multi-testing continuity results and previous "Debugging Gauntlet" logs.

src: https://www.youtube.com/shorts/_P9Kl3c9CXU

## The Wiring (Multimeter Result)
*   **🟡 YELLOW** = **Tip**
*   **🟢 GREEN** = **Ring**
*   **🔴 RED** = **Sleeve**

## The Functional Mapping (AD8232 Board)
| Wire Color | Function | AD8232 Pin | Placement Strategy |
| :--- | :--- | :--- | :--- |
| **🟡 YELLOW** | Signal (+) | **LA** | Primary Muscle Belly |
| **🟢 GREEN** | Signal (-) | **RA** | Secondary Muscle Belly |
| **🔴 RED** | **Reference** | **RL** | **Bony Landmark** (Elbow, Wrist, Mastoid) |

## Critical Configuration Note
*   **Red (Sleeve) MUST be Reference (RL).**
*   Placing Red on a signal pin causes "floating baseline" errors.
*   **Correct Usage:**
    *   **Yellow (Tip)** -> Signal Input 1
    *   **Green (Ring)** -> Signal Input 2
    *   **Red (Sleeve)** -> Reference (Ground)
