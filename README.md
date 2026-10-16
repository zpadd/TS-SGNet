# TS-SGNet

**TS-SGNet: A Transductive Semi-Supervised Superpixel Graph Network for Geospatial Classification under Label Scarcity and Class Imbalance**

This repository provides the Python source code for TS-SGNet and synthetic examples of the expected input-file structure. The four original hydrocarbon exploration datasets are confidential proprietary data and are neither public nor available from the authors upon request.

## Method

TS-SGNet combines attribute-coordinate decoupled representation, multiscale superpixel region graph propagation, and EMA-based quality-quantity-balanced pseudo-label selection.

## Repository contents

- `TS_SGNet.py`: model architecture, training, evaluation, and spatial mapping.
- `requirements.txt`: Python package dependencies.
- `example_data/`: entirely synthetic illustrative data and boundaries.
- `validate_example_data.py`: checks synthetic data schemas.
- `DATA_FORMAT.md`: fields and input requirements.

## Data and code availability

**Code:** This repository provides the TS-SGNet Python source code.

**Data:** Original field observations, well locations, labels, geoscientific attributes, spatial grids, exploration boundaries, and reserve polygons are confidential. They cannot be publicly released or provided upon request. The examples in this repository are fabricated solely to demonstrate the input format. They are not anonymized records, transformed originals, or samples from any actual study area. The synthetic data cannot reproduce the numerical findings in the paper.

## Setup

Use Python 3.10 or newer and install mutually compatible PyTorch and PyTorch Geometric versions for your environment:

```bash
pip install -r requirements.txt
python validate_example_data.py
```

For a format-only demonstration, run `python TS_SGNet.py`. The script retains the original lengthy training configuration. A successful execution on artificial data does **not** validate the scientific results.

With data you are authorized to access, set `TSSGNET_DATA_DIR` to the folder containing the expected input filenames. **Do not commit restricted inputs or model outputs derived from them.**

Refer to [DATA_FORMAT.md](DATA_FORMAT.md) for the complete schema.

## Citation

TS-SGNet: A Transductive Semi-Supervised Superpixel Graph Network for Geospatial Classification under Label Scarcity and Class Imbalance. Publication details pending.
