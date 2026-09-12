# Getting started

**SVC (Spatially embedded Virtual Cell)** is a Vision Transformer (ViT)-based spatial virtual cell model trained on subcellular spatial transcriptomics data that provide fine-grained molecular localization within each cell while retaining higher-level cellular and tissue context. It learns cell-specific gene dependencies to represent and predict subcellular transcript organization within native tissue context.

```{image} _static/input_data.png
:alt: Subcellular spatial transcriptomics data
:align: center
:width: 80%
```

---

## Model features

**Multi-modal and multi-scale integration.**
SVC integrates multi-modal and multi-scale information from five complementary inputs: registered gene images that capture gene subcellular spatial localization; prior gene-gene relationships derived from existing pretrained models; images of cell and nuclear morphologies; each cell's local spatial neighborhood; and optional cell type or state labels.

```{image} _static/data_input.png
:alt: The five inputs and the embeddings they are mapped to
:align: center
:width: 100%
```

**Five input and embedding types.**
The five inputs are mapped to five corresponding embeddings:

| Input | Embedding |
|---|---|
| Registered gene images | Gene-level localization embedding |
| Prior gene-gene relationships (Gene2vec) | Gene-level functional embedding |
| Cell and nuclear morphology images | Cell-level morphology embedding |
| Local spatial neighborhood | Cell-level neighbor embedding |
| Cell type or state labels (optional) | Cell-level identity embedding |

```{image} _static/svc_model.png
:alt: The SVC model
:align: center
:width: 70%
```

**Self-supervised masked image modeling.**
Each gene within each cell is represented by two gene-level embeddings, and each cell is represented by three types of cell-level embeddings; these representations are combined and fed into a Performer encoder block. SVC is trained using a self-supervised masked image modeling procedure, in which a random subset of gene expression images in each cell is masked. A decoder then reconstructs their spatial expression patterns by minimizing pixel- and cell-level loss functions.

---

## Applications

```{image} _static/applications.png
:alt: Applications of SVC across subcellular, cell and tissue levels
:align: center
:width: 70%
```

**Subcellular level**

- Prediction of fine-grained expression patterns for unmeasured genes within individual cells
- Spatial expression imputation across genes and cells at subcellular resolution
- Inference of subcellular gene-gene co-localization patterns
- Characterization of context-specific changes in subcellular organization across different
  cellular or spatial environments
- *In silico* prediction of perturbation-induced changes in subcellular spatial expression

**Cell and tissue level**

- Cell-level gene expression imputation
- Cell clustering
- Spatial domain detection

&emsp;&emsp;•<br>&emsp;&emsp;•<br>&emsp;&emsp;•

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aster-ww/SVC.git
cd SVC
```
### 2.  Create the conda environment
```bash
conda env create -f environment.yml
conda activate SVC
```
---

## Data

The processed datasets and the trained model checkpoints are on Zenodo:
<https://doi.org/10.5281/zenodo.22693727>

The tutorials read them through two environment variables:

`SVC_ROOT` — this repository, for `import svc` (default: `../../..`)

`SVC_DATA_ROOT` — the processed data, unpacked as `<SVC_DATA_ROOT>/<dataset>/` (default: `../../../data`)

`checkpoints.zip` from the same deposit unpacks into `../../../checkpoints`.

```bash
export SVC_ROOT=/path/to/SVC
export SVC_DATA_ROOT=/path/to/data
```

---

Visit our [group website](https://xiangzhou.github.io/) for more statistical tools on analyzing genetics, genomics and transcriptomics data.
