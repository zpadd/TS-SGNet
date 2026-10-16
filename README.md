# TS-SGNet

**TS-SGNet: A Transductive Semi-Supervised Superpixel Graph Network for Geospatial Classification under Label Scarcity and Class Imbalance**

This repository publishes the TS-SGNet core network definition and the primary model and optimisation settings reported in the manuscript. The training and evaluation entry point is provided separately as a runnable Python ZIP application (`reproduce.pyz`).

## Repository contents

- **[TS_SGNet.py](TS_SGNet.py)**: the core ACDR, multiscale region graph, attention fusion, and classification architecture. Its top-level settings follow the paper's main parameter table.
- **[reproduce.pyz](reproduce.pyz)**: runnable training, evaluation, and mapping pipeline. This archive requires `TS_SGNet.py` in the same directory.
- **[requirements.txt](requirements.txt)**: Python dependencies (install compatible PyTorch and PyTorch Geometric versions).
- **[DATA_FORMAT.md](DATA_FORMAT.md)** and **[example_data/](example_data/)**: data schemas and entirely fabricated samples.

## Execution

```bash
pip install -r requirements.txt
python reproduce.pyz
```

By default, the application reads the artificial samples in `example_data`. To use data that you are authorised to access, set `TSSGNET_DATA_DIR` to the directory containing the specified input filenames.

The ZIP application is a binary-format distribution, **not an encryption or access-control mechanism**. Its contents can be extracted using standard tools; the source of the full training pipeline is not separately published as a plain `.py` file in the current repository view.

## Data and code availability

The four original hydrocarbon exploration datasets (well records, spatial coordinates, labels, attribute rasters, and actual study-area/reserve boundaries) are confidential proprietary materials. They are neither public nor available upon request. The example files are wholly synthetic and preserve only the input schema. They do **not** reproduce the paper's experimental metrics or geographic patterns.

The readable core architecture source and an executable training workflow are available in this repository. A runnable archive together with synthetic data is **not** sufficient to independently reproduce the empirical results obtained from restricted datasets.

## Version history

GitHub's commit history may retain earlier versions of source files; replacing a file on `main` does not erase its historical content.
