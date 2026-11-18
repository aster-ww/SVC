# Demo

`demo_seqFISHplus.ipynb` walks through SVC on the seqFISH+ test cells (batch 3, one field of view with 14 cells):

**raw transcripts → spatial registration → gene-specific spatial representations → masked prediction with SVC → observed vs. predicted localization**

The example is cell 9-3, with *Nid1* and *Hnrnpf* from Fig. 2c. The notebook trains nothing; it uses only the code of this repository:

- `svc/register.py` for registration and the gene maps;
- `seqfish/evaluate.py` for masked gene prediction with the released checkpoint;
- `seqfish/postprocess.py` and `svc/postprocess.py` to map predictions back into each cell.

`plotting.py` only draws the figures.

## Setup

As for the tutorials:

1. Create the environment from `environment.yml`.
2. From the Zenodo deposit linked in the main README, unzip `preprocessed_seqfish.zip` into `data/` and `checkpoints.zip` into `checkpoints/`, both at the repository root.
3. Run the notebook from this folder.

`SVC_DATA_ROOT` points to another data directory if needed.

`evaluate.py` predicts all 20 gene folds. The whole notebook runs in under a minute with a GPU and in about three minutes on a 48-thread CPU. Its outputs go to `checkpoints/`, as in the prediction tutorial, and `output/demo/` holds the reconstructed transcripts and the script logs.
