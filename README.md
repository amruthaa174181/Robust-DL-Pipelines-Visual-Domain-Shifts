# Robust Deep Learning Pipelines under Visual Domain Shifts

An enterprise-grade research pipeline engineered in **PyTorch** utilizing a modified **ResNet-18** architecture. This repository benchmarks neural network degradation under severe distribution shifts (environmental data corruptions) and implements an adversarial data augmentation defense to bridge the resulting performance domain gap.

---

## 📊 Core Architecture & Engineering Lifecycle

The system is engineered as an end-to-end evaluation pipeline partitioned into four distinct operational phases:

[Phase 1: Input Data & Tensors] ──> [Phase 2: Deep Backbone (ResNet-18)] ──> [Phase 3: Domain Noise Insertion] ──> [Phase 4: Optimization & Mitigation]

### 1. Phase 1: Input Data Engineering & Quantization
* **Benchmarking Dataset:** CIFAR-10 (60,000 color images scaled across 10 balanced categories).
* **Tensor Formatting:** Raw 8-bit unsigned integer data arrays ($[0, 255]$) are scaled and cast into standard precision `torch.float32` 4D arrays formatted as: 
  $$\text{Batch Size} \times \text{Channels} \times \text{Height} \times \text{Width} \longrightarrow [64 \times 3 \times 32 \times 32]$$
* **Zero-Centered Normalization:** Elements are transformed via a linear transformation to map variances cleanly within a bounded range of $[-1.0, 1.0]$. This maintains zero-centered input variables to prevent gradient explosions in early convolutional layers.

### 2. Phase 2: Structural Neural Architecture Modifications
* **Feature Extraction Backbone:** State-of-the-art **ResNet-18 (Residual Network)** engine.
* **Mathematical Optimization:** Employs parallel **Skip Connections (Shortcuts)** defining the computational block state as:
  $$\text{Output} = F(x) + x$$
  This addition preserves error signaling by ensuring the gradient derivative ($\frac{\partial H}{\partial x} = \frac{\partial F(x)}{\partial x} + 1$) never shrinks to zero, completely eliminating the **Vanishing Gradient Problem** native to deep models.
* **Structural Adaptation:** The terminal Fully Connected (FC) Linear layer was dynamically refactored to override the standard 1000-class industrial output array, scaling down to output exactly 10 raw confidence classification logits.

### 3. Phase 3: Stress Testing & Covariate Domain Shifts
To simulate harsh real-world operating environments (e.g., severe fog, camera motion blur, low-light variations), evaluation data is subjected to a synthetic validation corruption pipeline:
* **Gaussian Blur Execution:** Acts as a low-pass filter by sliding a spatial kernel across the pixel matrix. This removes high-frequency details (textures and sharp edges), blinding standard convolutional filters.
* **Color Jittering:** Distorts the brightness and contrast matrices to evaluate model performance under severe ambient lighting shifts.

### 4. Phase 4: Optimization Engine & Adversarial Defenses
* **Loss Function:** Cross-Entropy Loss evaluates the distance between true distributions and predicted probabilities.
* **Weight Adjustment:** The **Adam Optimizer** applies adaptive learning rates based on tracking the first moment (momentum) and second moment (gradient variance).
* **Data Augmentation Defense:** Integrates randomized Gaussian blurs directly into the training loop. This prevents the model from over-relying on fragile surface textures and forces it to map robust geometric structures.

---

## 🧪 Experimental Benchmarks & Log Results

| Experiment Variant | Data Augmentation Defense | Clean Test Accuracy (Source Domain) | Corrupted Test Accuracy (Target Domain) | Quantified Domain Gap |
| :--- | :---: | :---: | :---: | :---: |
| **A: Baseline Model** | ❌ Disabled | **74.89%** | **36.12%** | **38.77% (Severe Crash)** |
| **B: Robust Model** |  Enabled | **73.26%** | **63.58%** | **9.68% (Minimized)** |

---

## 💻 Project Codebase Directory Structure

* `dataset.py`: Multi-domain data loaders handling clean data, domain corruption transformations, and training pipeline augmentations.
* `train.py`: Optimization suite encompassing the 4-step execution loop (Forward ──> Loss Check ──> Backward Pass ──> Weight Update).

## 🚀 How To Replicate This Project Locally

1. Clone the repository:
   git clone https://github.com/amruthaa174181/Robust-DL-Pipelines-Visual-Domain-Shifts.git
   cd Robust-DL-Pipelines-Visual-Domain-Shifts

2. Install core mathematical dependencies:
   pip install torch torchvision

3. Execute the experimental baseline and robust pipelines:
   python train.py

💻 Research Focus & Individual Contribution
    Developer: Amrutha Annepu

    Core Focus: High-performance deep learning model optimization, evaluation under distribution shifts, and structural adversarial defense exploration.