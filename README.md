# Model-based Robust Representations through Multimodal Fusion: Leveraging Intermodal and Intramodal Correlations

Minimizing unexpected disruptions in industrial environments is vital due to the steep costs tied to energy waste, idle labor, and resource inefficiencies. Traditional predictive maintenance techniques are useful for handling wear-and-tear over time, but they often fall short when it comes to spontaneous, unpredictable failures.

This repository presents a deep learning solution called the Multimodal Concept Fusion Architecture — a framework specifically built to handle diverse, noisy, and incomplete data from multiple sources. Instead of relying on a single unified representation, the model intelligently combines features from different data types by learning an abstract “fusion concept” that dynamically adjusts between shared and unique modality signals.

The architecture not only detects anomalies as part of its core functionality but also provides insight into the reasoning behind its predictions by attributing outcomes to specific data channels. This enhances interpretability and fosters trust in automated decision-making.

We validate our approach on three synthetic but industrially inspired multimodal datasets, each simulating realistic sensor failures using strategic data augmentations. The results highlight the method’s ability to remain robust and informative in the face of corrupted or incomplete inputs.

### 📦 Prerequisites

Before you begin, ensure your environment meets the following requirements:

* **Python** ≥ 3.6
* **PyTorch** ≥ 1.0 (CUDA support recommended for faster training)

We also recommend using a virtual environment (e.g., `venv` or `conda`) to avoid package conflicts.

## 🔧 Multimodal Autoencoder

The Multimodal Concept Fusion Autoencoder architecture is designed to handle multiple heterogeneous input modalities, encode them into latent spaces, aggregate these representations into a unified concept space, and then decode them back into the original modalities. The following figure illustrates the full flow of the Multimodal Concept Fusion Autoencoder:

<p align="center">
  <img src="https://github.com/Diyarino/concept-guided-multimodal-fusion/blob/6ba71050185c595592a58ab8fed63f5b153cca29/multimodal_autoencoder.png" alt="Multimodal Concept Fusion Autoencoder" width="600"/>
</p>Figure 1: Overview of the multimodal autoencoder architecture. Each modality is encoded separately, fused into a unified concept space, and decoded individually [1].




Below is a formal mathematical description of the framework:

---

#### **1. Encoder Stage**

Let there be $M$ input modalities $\{ \mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_M \}$, where each $\mathbf{x}_m \in \mathbb{R}^{d_m}$ corresponds to one modality.

Each modality $\mathbf{x}_m$ is passed through its own encoder $E_m: \mathbb{R}^{d_m} \rightarrow \mathbb{R}^{k_m}$, producing a latent representation:

$$
\mathbf{z}_m = E_m(\mathbf{x}_m), \quad \text{for } m = 1, \dots, M
$$

---

#### **2. Fusion Block (Concept-Level Aggregation)**

The latent vectors $\{ \mathbf{z}_1, \dots, \mathbf{z}_M \}$ are then processed by a **fusion function** $\mathcal{F}: \mathbb{R}^{k_1} \times \dots \times \mathbb{R}^{k_M} \rightarrow \mathbb{R}^{k_c}$, which computes a unified concept representation $\mathbf{z}_c$:

$$
\mathbf{z}_c = \mathcal{F}(\mathbf{z}_1, \dots, \mathbf{z}_M)
$$

This fusion block can be implemented using attention, contrastive regularization, or other learned mechanisms that adaptively weigh modality-specific features.

---

#### **3. Decoder Stage**

The unified concept vector $\mathbf{z}_c$ is then used to reconstruct each modality via its corresponding decoder $D_m: \mathbb{R}^{k_c} \rightarrow \mathbb{R}^{d_m}$:

$$
\hat{\mathbf{x}}_m = D_m(\mathbf{z}_c), \quad \text{for } m = 1, \dots, M
$$

The goal is to minimize the total reconstruction loss across all modalities:

$$
\mathcal{L} = \sum_{m=1}^{M} \ell(\mathbf{x}_m, \hat{\mathbf{x}}_m)
$$

where $\ell$ is typically the mean squared error (MSE) or another appropriate distance metric.


## 🧠 Concept-guided Multimodal Fusion

Our model uses a **concept-aware fusion module** that adaptively integrates features from different modalities using attention, dual fusion strategies, and residual refinement. This allows for robust and interpretable feature integration across noisy or partially missing inputs.

* **Concept Attention:**
  A learnable attention mechanism dynamically weighs modality contributions based on their contextual relevance. This enables the model to prioritize the most informative inputs during fusion, supporting interpretability and robustness.

* **Joint & Marginal Fusion:**
  We combine two fusion paths — *mean* (shared signal) and *max* (dominant signal) — using a soft weighting guided by the attention vector. This lets the model adaptively balance between cooperation and contrast across modalities:

$$\mathbf{f} = \mathbf{a} \odot \frac{\mathbf{z}_1 + \mathbf{z}_2}{2} + (1 - \mathbf{a}) \odot \max(\mathbf{z}_1, \mathbf{z}_2)$$

* **Residual Projection & Normalization:**
  To preserve modality-specific information, residual connections from the original encodings are added back into the fused representation and normalized. This enhances training stability and supports gradient flow, ensuring both shared and individual features are retained.

Together, these mechanisms enable the fusion block to adapt to varying input quality, enforce semantic alignment, and maintain robustness under challenging multimodal scenarios.


## 📊 Multimodal Robot Kinematic Datasets

This repository provides access to three multimodal robot movement datasets, each including at a minimum the **camera** and **kinematics** modalities. These datasets were used and described in detail in our accompanying research paper.

### 🔗 Datasets

1. **MuJoCo: UR5 Robot Motion** – [Link to Dataset 1](https://zenodo.org/records/14041622)
2. **ABB Studio: Single Robot Welding Station** – [Link to Dataset 2](https://zenodo.org/records/14041488)
3. **ABB Studio: Dual Robot Welding Station** – [Link to Dataset 3](https://zenodo.org/records/14041416)

Each dataset captures robot motion across various tasks and environments, providing synchronized data streams for machine learning and robotics research.

### 📄 Reference Paper

For detailed descriptions of the datasets, data collection procedures, and experimental use cases, please refer to our paper [2]:

**"Performance benchmarking of multimodal data-driven approaches in industrial settings"** – [Link to Paper](https://www.sciencedirect.com/science/article/pii/S266682702500074X?via%3Dihub)


## 📊 Results

We evaluated our approach on three synthetically constructed but industrially realistic multimodal datasets, each simulating various failure modes such as sensor dropouts, signal noise, and modality corruption.

work in progress...


### 🚨 Anomaly Detection via Concept Scores

Anomaly detection in our framework is performed using the **concept scores** derived from the unified latent representation. After fusing modality-specific embeddings into the shared concept vector $\mathbf{z}_c$, we compute a **concept score** that quantifies which aggregation function is used with the learned concept space. High concept scores indicate joint information, and low scores indicate marginal information. This allows the model to detect irregularities, enhancing robustness in real-world industrial scenarios. The failure injection is based on our pre-work [3].

![til](animation_single.gif)

![til](animation_multi.gif)

## References 

<a id="1">[1]</a> Altinses, D., & Schwung, A. (2023, October). Multimodal Synthetic Dataset Balancing: A Framework for Realistic and Balanced Training Data Generation in Industrial Settings. In IECON 2023-49th Annual Conference of the IEEE Industrial Electronics Society (pp. 1-7). IEEE.

<a id="2">[2]</a> Altinses, D., & Schwung, A. (2025, June). Performance benchmarking of multimodal data-driven approaches in industrial settings. In Machine Learning with Applications (pp. 1-7). Volume 21, 100691, ISSN 2666-8270.

<a id="3">[3]</a> Altinses, D., & Schwung, A. (2023, October). Multimodal Synthetic Dataset Balancing: A Framework for Realistic and Balanced Training Data Generation in Industrial Settings. In IECON 2023-49th Annual Conference of the IEEE Industrial Electronics Society (pp. 1-7). IEEE.



