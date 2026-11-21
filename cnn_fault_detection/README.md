# SURYOKTI: CNN-Based Visual PV Fault Identification Module

The **CNN Fault Detection Module** provides automated visual inspection of solar panel surfaces using deep convolutional neural networks. It classifies physical surface degradation, soiling, and electrical defects into 6 distinct operational classes.

---

## 1. Directory Structure

```text
cnn_fault_detection/
├── README.md                                    # Module architecture and dataset reference (this file)
│
├── data/                                        # Dataset pointers, metadata manifests, and preprocessing
│   └── class_manifest.json                      # 6-class definitions, severities, and operational actions
│
├── models/                                      # Model architectures and weight references
│   └── cnn_architecture_reference.py           # PyTorch/TensorFlow CNN architecture definition stub
│
└── plots/                                       # Visual outputs (class distributions, confusion matrices)
```

---

## 2. External Vision Dataset Location

> [!IMPORTANT]
> The primary high-resolution image dataset resides externally to keep this code repository lightweight:
> **Primary Image Directory:** `C:\VITB\suryokti vision\`
>
> The repository scripts reference images directly from this external path. Do NOT duplicate gigabytes of image data into this repository.

---

## 3. The 6 Visual Fault Classes

| Class Name | Surface Description | Physical Mechanism | Primary Impact | Recommended Operational Action |
|---|---|---|---|---|
| `Clean` | Clean, undamaged PV panel surface | Optimal optical transmission | Maximum power generation | Routine monitoring; no action |
| `Dusty` | Uniform layer of accumulated airborne dust | Particle scattering & absorption | 5% - 25% gradual output drop | Scheduled water/brush cleaning |
| `Bird-drop` | Localized white/gray fecal deposits | Severe local shading; hotspot formation | Risk of permanent cell burning | High-priority targeted spot wash |
| `Electrical-damage` | Discoloration, burn marks, delamination | Solder failure, bypass diode failure | Cell/string open-circuit | Immediate electrical technician inspection |
| `Physical-Damage` | Cracked glass, shattered cells, frame bend | Hail impact, thermal stress, mechanical impact | Moisture ingress & shock hazard | Panel isolation & module replacement |
| `Snow-Covered` | Blanket of snow obscuring cells | Total reflection of incoming sunlight | Near 100% output reduction | Natural thaw or gentle mechanical clearing |

---

## 4. CNN Architecture Specification

* **Input Resolution:** $224 \times 224 \times 3$ (RGB)
* **Backbone Architecture:** Transfer learning backbone (e.g., EfficientNet-B0 or ResNet-50) pretrained on ImageNet.
* **Custom Classification Head:**
  - Global Average Pooling 2D
  - Dense Layer (256 units, ReLU activation, Dropout $p=0.4$)
  - Output Layer (6 units, Softmax activation)
* **Loss Function:** Categorical Cross-Entropy with label smoothing ($\alpha = 0.1$)
* **Optimizer:** AdamW with learning rate $1\times 10^{-4}$ and cosine decay

---

## 5. Integration with Maintenance Decision Engine

When telemetry anomaly detection identifies a `POTENTIAL_OPERATIONAL_ANOMALY` with negative power deficit:
1. Drone or camera imagery of the affected inverter array is routed to the CNN module.
2. The CNN predicts the surface condition class:
   - If classified as `Dusty` $\rightarrow$ triggers **Cleaning Work Order**.
   - If classified as `Bird-drop` $\rightarrow$ triggers **Spot Cleaning Work Order**.
   - If classified as `Electrical-damage` or `Physical-Damage` $\rightarrow$ triggers **Urgent Maintenance Dispatch**.
   - If classified as `Clean` $\rightarrow$ suggests an **Inverter / Balance-of-System (BOS) Electrical Issue**.

