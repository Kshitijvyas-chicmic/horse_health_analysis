# Horse Pastern Axis (HPA) Estimation Methodology 🐎📐

This document outlines the technical methodology used to derive clinically accurate HPA measurements from AI-generated keypoints.

---

## 📍 1. Keypoint Definitions
The model identifies 4 critical anatomical landmarks:
*   **Pt 0 (Red)**: Pastern Top (Fetlock Joint)
*   **Pt 1 (Orange)**: Pastern Bottom (Coronary Band)
*   **Pt 2 (Green)**: Hoof Wall Top (Coronary Band Front)
*   **Pt 3 (Blue)**: Toe Tip (Ground Contact Point)

---

## 📐 2. Clinical Angle Calculation
Standard AI models calculate "Anatomical Angles" (relative to vertical). Clinical standards (used by farriers) require **Ground-Relative Angles**.

### Mathematical Conversion:
Given a vector $\vec{v} = (dx, dy)$ between two keypoints:
1.  **Image Angle** ($\theta_i$): Calculated relative to the vertical axis $(0, -1)$.
2.  **Ground Angle** ($\theta_g$): 
    $$\theta_{clinical} = 90^\circ - (|\theta_i| \pmod{180})$$
    *(This ensures the measurement matches physical inclinometer readings.)*

---

## 🔄 3. Directional Normalization (Flip-Invariance)
To ensure the HPA deviation is consistent whether the horse faces **Left** or **Right**, we implement **Safety Invariant Normalization**:
*   **Logic**: If the toe tip (Pt 3) is to the left of the pastern top (Pt 0), we mathematically mirror the keypoints across their horizontal center.
*   **Result**: The "Toe" is always effectively facing "Right" for calculation, eliminating directional bias.

---

## 🔭 4. Autonomous ROI Detection (Vanguard Multi-Scan)
The "In-the-Wild" problem (background clutter and varying distances) is solved via a 3-step ensemble:

1.  **Zone Scanning**: The AI independently analyzes 4 overlapping vertical zones:
    *   `Floor-Scan`: Bottom 60% (Focus on hoof)
    *   `Anatomy-Scan`: Middle 60% (Focus on pastern geometry)
    *   `Top-Anatomy`: Top 60% (Catching distal joints)
    *   `Global-Scan`: Full Image (General context)
2.  **Confidence Scoring**: Each zone generates an anatomical score:
    $$Score = (\mu_{conf} \times 10) - Penalty_{low\_conf}$$
3.  **Ensemble Selection**: The engine automatically selects the "Winner" (the zone with the clearest anatomical view) for final HPA calculation.

---

## ⚖️ 5. HPA Deviation Logic
The final health indicator is the **HPA Neutrality**:
*   **Parallel Axis**: $\text{Pastern Angle} \approx \text{Hoof Angle}$ ($\text{Dev} < 3^\circ$)
*   **Broken Axis (High)**: $\text{Hoof} > \text{Pastern}$ (Indicating upright/club hoof)
*   **Broken Axis (Low)**: $\text{Pastern} > \text{Hoof}$ (Indicating dropped/long-toe fetlock)

---

## 🏁 Technical Implementation
All logic is centralized in the Vanguard Inference Engine:
`mmpose/demo/inference_on_new_image.py`
