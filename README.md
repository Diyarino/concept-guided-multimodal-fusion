# Model-based Robust Representations through Multimodal Fusion: Leveraging Intermodal and Intramodal Correlations

Minimizing unexpected disruptions in industrial environments is vital due to the steep costs tied to energy waste, idle labor, and resource inefficiencies. Traditional predictive maintenance techniques are useful for handling wear-and-tear over time, but they often fall short when it comes to spontaneous, unpredictable failures.

This repository presents a deep learning solution called the Multimodal Concept Fusion Architecture — a framework specifically built to handle diverse, noisy, and incomplete data from multiple sources. Instead of relying on a single unified representation, the model intelligently combines features from different data types by learning an abstract “fusion concept” that dynamically adjusts between shared and unique modality signals.

The architecture not only detects anomalies as part of its core functionality but also provides insight into the reasoning behind its predictions by attributing outcomes to specific data channels. This enhances interpretability and fosters trust in automated decision-making.

We validate our approach on three synthetic but industrially inspired multimodal datasets, each simulating realistic sensor failures using strategic data augmentations. The results highlight the method’s ability to remain robust and informative in the face of corrupted or incomplete inputs.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Multimodal Autoencoder

The Multimodal Concept Fusion Autoencoder architecture is designed to handle multiple heterogeneous input modalities, encode them into latent spaces, aggregate these representations into a unified concept space, and then decode them back into the original modalities. The following figure illustrates the full flow of the Multimodal Concept Fusion Autoencoder:

<p align="center">
  <img src="https://github.com/Diyarino/concept-guided-multimodal-fusion/blob/6ba71050185c595592a58ab8fed63f5b153cca29/multimodal_autoencoder.png" alt="Multimodal Concept Fusion Autoencoder" width="600"/>
</p>Figure 1: Overview of the multimodal autoencoder architecture. Each modality is encoded separately, fused into a unified concept space, and decoded individually[^1].




Below is a formal mathematical description of the framework:

---

1. Encoder Stage

Let there be  input modalities , where each  corresponds to one modality.

Each modality  is passed through its own encoder , producing a latent representation:

\mathbf{z}_m = E_m(\mathbf{x}_m), \quad \text{for } m = 1, \dots, M


---

2. Fusion Block (Concept-Level Aggregation)

The latent vectors  are then processed by a fusion function , which computes a unified concept representation :

\mathbf{z}_c = \mathcal{F}(\mathbf{z}_1, \dots, \mathbf{z}_M)

This fusion block can be implemented using attention, contrastive regularization, or other learned mechanisms that adaptively weigh modality-specific features.


---

3. Decoder Stage

The unified concept vector  is then used to reconstruct each modality via its corresponding decoder :

\hat{\mathbf{x}}_m = D_m(\mathbf{z}_c), \quad \text{for } m = 1, \dots, M

The goal is to minimize the total reconstruction loss across all modalities:

\mathcal{L}_{\text{recon}} = \sum_{m=1}^{M} \ell(\mathbf{x}_m, \hat{\mathbf{x}}_m)

where  is typically the mean squared error (MSE) or another appropriate distance metric.





## Concept-guided Multimodal Fusion

Classification of a Simple Finite State Machine based on trained Neural Logic Rule Layers.

## 02_Datasets

descrption and link and citation and images

<p align="center">
  <img src="https://user-images.githubusercontent.com/86289948/122931592-52dc5700-d36d-11eb-8a2f-eaba94ca60c3.PNG" alt="BGLP" width="600" height="420">
</p>

## Results

following figures show the elements and the full NLRL.

Basic blocks of NLRL       |  Full NLRL 
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/86289948/123260158-89dd7480-d4f5-11eb-9fec-dd9fdf44bdea.jpg" alt="NLRL_blocks_LI" width="450" height="250">   |  <img src="https://user-images.githubusercontent.com/86289948/123260300-b2656e80-d4f5-11eb-82b1-cde0057440ee.PNG" alt="FullNLRL" width="450" height="200">

## Anomaly detection


## References 

[^1] Altinses, D., & Schwung, A. (2023, October). Multimodal Synthetic Dataset Balancing: A Framework for Realistic and Balanced Training Data Generation in Industrial Settings. In IECON 2023-49th Annual Conference of the IEEE Industrial Electronics Society (pp. 1-7). IEEE.


