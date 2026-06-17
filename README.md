# Robust Deep Learning Pipelines under Visual Domain Shifts

An enterprise-grade research pipeline engineered in **PyTorch** utilizing a modified **ResNet-18** architecture. This repository benchmarks neural network degradation under severe distribution shifts (environmental data corruptions) and implements an adversarial data augmentation defense to bridge the resulting performance domain gap.

---

## 🛠️ Complete Technology Stack

### Core Frameworks and Libraries
* **Python:** The core programming language handling pipeline logic and array manipulations.
* **PyTorch (torch):** The primary deep learning framework utilized to manage neural network layers, track computational graphs, and execute backpropagation.
* **Torchvision (torchvision.transforms):** A specialized vision library used to construct the runtime image preprocessing pipeline and inject adversarial visual corruptions.
* **OpenCV (cv2):** An open-source computer vision library used to interface with webcam hardware, stream real-time camera feeds, and manage keyboard interrupts.
* **PIL (Pillow):** The Python Imaging Library, used for opening, format-checking, and managing image files.

### Development Tools and Version Control
* **VS Code:** The Integrated Development Environment (IDE) used to write scripts, debug environmental paths, and run terminal experiments.
* **Git and GitHub:** Used for tracking local codebase modifications and hosting the public cloud repository.

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

## 🧪 Experimental Benchmarks and Log Results

| Experiment Variant | Data Augmentation Defense | Clean Test Accuracy (Source Domain) | Corrupted Test Accuracy (Target Domain) | Quantified Domain Gap |
| :--- | :---: | :---: | :---: | :---: |
| **A: Baseline Model** | ❌ Disabled | **74.89%** | **36.12%** | **38.77% (Severe Crash)** |
| **B: Robust Model** |  Enabled | **73.26%** | **63.58%** | **9.68% (Minimized)** |

> 🔬 **Engineering Insight (Class Collapse Edge Case):** When data augmentation noise parameters are tuned too aggressively during optimization, the network encounters a common training trap. To lower its cross-entropy loss without learning structural definitions, it exploits uniform feature map activations and adjusts its internal linear biases to output a single dominant class (**DOG**) for every input. Moderating the kernel radius stabilizes gradient extraction and resolves the collapse.

---

## 💻 Project Codebase Directory Structure

* **`dataset.py`**: Multi-domain data loaders handling clean data, domain corruption transformations, and training pipeline augmentations.
* **`train.py`**: Optimization suite encompassing the 4-step execution loop (Forward -> Loss Check -> Backward Pass -> Weight Update) and automatic weight exportation.
* **`robust_resnet18.pt`**: The compiled binary weights file containing millions of optimized mathematical parameters generated by the training engine.
* **`test_live.py`**: An edge-deployment testing script utilizing OpenCV to pull live video frames from a webcam, inject an inline blur attack, and pipe the matrix into the saved weights file for real-time inference predictions.
* **`README.md`**: Clean markdown technical portfolio documentation.

---

## 🚀 How To Replicate This Project Locally

1. Clone the repository:
   git clone https://github.com/amruthaa174181/Robust-DL-Pipelines-Visual-Domain-Shifts.git
   cd Robust-DL-Pipelines-Visual-Domain-Shifts

2. Install core mathematical dependencies:
   pip install torch torchvision opencv-python pillow

3. Execute the experimental baseline and robust pipelines:
   python train.py

4. Execute the live real-time webcam test pipeline
   python test_live.py

💻 Research Focus & Individual Contribution
    Developer: Amrutha Annepu

    Core Focus: High-performance deep learning model optimization, evaluation under distribution shifts, and structural adversarial defense exploration.

***
