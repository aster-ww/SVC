# Getting started

**SVC (Spatially embedded Virtual Cell)** is a Vision Transformer (ViT)-based Spatial Virtual Cell model for deciphering subcellular spatial transcriptomic heterogeneity. 
SVC characterizes and predicts subcellular gene expression patterns within their native tissue context, enabling analyses across subcellular, cellular, and tissue scales.
It offers a new lens for interpreting subcellular ST data and yields a range of biological insights that are difficult to capture with approaches limited to single-cell resolution or specific analytic tasks. 

SVC introduces unique features in realizing AI-powered virtual cells (AIVC) in the subcellular regime:

- ***In silico* prediction at subcellular resolution**: SVC predicts subcellular spatial localization patterns for unmeasured genes in new datasets, and can be extended to predict perturbation-induced spatial redistribution, moving beyond conventional gene imputation and spatial reconstruction to unprecedented subcellular resolution.


- **Multi-scale and multi-modality modeling**: SVC jointly models subcellular spatial gene expression with tissue microenvironment context and paired cell morphology images, integrating multi-modal information across subcellular, cellular and tissue scales.


- **A unified, scalable ViT-based framework**: SVC adapts the powerful ViT architecture with self-attention mechanisms to capture complex multi-gene spatial dependencies within cells, producing a unified virtual representation shared across genes and cells. The framework scales to tissue-level datasets and supports scalable processing of hundreds to thousands of genes simultaneously.


- **Subcellularly informed cell and tissue-level analysis**: Leveraging subcellularly informed representations, SVC distinguishes cellular states and reveals tissue-level organization, offering a new perspective on tasks such as cell clustering and spatial domain detection.

---

## Installation

```bash
git clone https://github.com/aster-ww/SVC.git
cd SVC

---

Visit our [group website](https://xiangzhou.github.io/) for more statistical tools on analyzing genetics, genomics and transcriptomics data.