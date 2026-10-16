# Liquid-Mamba Conformer

Official training/evaluation code of **Liquid-Mamba Conformer** (code name `DBConformer`), a dual-branch deep network for motor-imagery (MI) EEG decoding with an emphasis on post-stroke rehabilitation brain-computer interfaces.

The model extends DBConformer with four coupled designs:

- **Liquid-gated attention (LGA)** in the temporal branch, which replaces standard multi-head attention with input-dependent exponential time decay so the network can track the evolving ERD/ERS dynamics of a trial.
- **Continuous-time position embedding** (`ContinuousTimeEmbedder`) that encodes patch timestamps instead of fixed positions, so time is a physical input rather than an index.
- **Bidirectional channel Mamba (BiCMamba)** in the spatial branch, which scans the electrode sequence in both directions to model long-range inter-channel dependencies at linear cost.
- **Dynamic sparse channel attention (DSCA)** that re-weights electrodes ahead of the spatial branch, followed by a **closed-form continuous-time (CfC) fusion block** that merges the two branches with a continuous state update.

## Repository layout

```text
.
├── DBConformer_CO.py        # within-subject chronological-order protocol
├── DBConformer_CV.py        # within-subject chronological 5-fold CV
├── DBConformer_LOSO.py      # leave-one-subject-out protocol
├── models/                  # network implementations
│   ├── DBConformer.py       # Liquid-Mamba Conformer (ours)
│   ├── EEGNet.py, DeepConvNet.py, ShallowConvNet.py, ShallowConvNet_V2.py
│   ├── FBCNet.py, FBMSNet.py, EEGNeX.py, EEGWaveNet.py, ADFCNN.py
│   ├── Conformer.py, CTNet.py, MSCFormer.py, MSVTNet.py, SlimSeiz.py, TMSANet.py
│   └── IFNetV2.py, FC.py
└── utils/                   # data loading, splits, metrics, logging, EA, factories
```

## Requirements

Python 3.10 (the bytecode caches in this repo were produced with CPython 3.10) with:

```bash
pip install torch torchvision numpy scipy scikit-learn pandas einops timm \
            mne moabb learn2learn PyWavelets matplotlib torchsummary pillow
```

## Data preparation

Each dataset is a folder `./data/<DATASET>/` containing `X.npy` and `labels.npy` (shape `[trials, channels, samples]`). The in-house LU-EEGMI collection is split per paradigm session:

```text
data/
├── BNCI2014001/            # X.npy, labels.npy
├── BNCI2014002/
├── BNCI2014004/
├── EEG-ST/
└── LU-EEGMI/
    ├── pre/                # X.npy, labels.npy, meta.csv
    ├── post/
    ├── follow/
    ├── MI-SES/             # sensory electrical stimulation sessions
    └── MI-IES/             # motor electrical stimulation sessions
```

The public benchmarks (BNCI2014001/002/004/008/009, BNCI2015001/003/004) can be downloaded with MOABB. LU-EEGMI is the authors' own stroke-patient dataset (gait MI vs. idle, 22 sensorimotor electrodes, 5 s windows at 250 Hz) and is not redistributed in this repository.

## Quick start

The launchers select the dataset and backbone through environment variables. The first positional argument is the GPU index (or `cpu`):

```powershell
# Within-subject chronological split on LU-EEGMI pre-rehabilitation
$env:DBC_DATA = "LU-EEGMI-pre"
$env:DBC_GATE_FLAG = "1"   # complete model: liquid gate, CT-PE, DSCA, CfC all on
python DBConformer_CO.py 0

# Switch backbone (any name from models/)
$env:DBC_BACKBONE = "EEGNet"
python DBConformer_CO.py 0

# Leave-one-subject-out across the LU-EEGMI sessions
$env:DBC_DATA = "LU-EEGMI-post"
python DBConformer_LOSO.py 0
```

Environment hooks used by the scripts:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DBC_DATA` | `LU-EEGMI-pre` | dataset key from the table in the launcher |
| `DBC_BACKBONE` | `DBConformer` | backbone to train |
| `DBC_GATE_FLAG` | `0` | `1` enables the liquid gate |
| `DBC_POSEMB_FLAG` | `1` | `1` enables continuous-time position embedding |
| `DBC_CHN_ATTEN_FLAG` | `1` | `1` enables DSCA |
| `DBC_BRANCH` | `all` | `all`, `temporal`, or `spatial` branch only |
| `DBC_CFC_FLAG` | `1` | `1` enables the CfC fusion block |
| `DBC_DEPTH` | `2` | transformer / Mamba depth sweep hook |
| `DBC_PATCH_SIZE` | `125` | temporal patch size sweep hook |
| `DBC_BATCH_SIZE` | `32` | batch size sweep hook |
| `DBC_RESULT_DIR` | `./logs/` | where per-run logs, metrics CSV/JSON and probability dumps are written |

The three protocols are:

- **CO** (`DBConformer_CO.py`): within-subject chronological train/test split, no shuffling across time.
- **CV** (`DBConformer_CV.py`): within-subject chronological 5-fold cross-validation.
- **LOSO** (`DBConformer_LOSO.py`): leave-one-subject-out, the strictest cross-subject protocol.

Per-run artifacts land in `./logs/`: training logs, one metrics CSV per subject, a JSON with confusion matrices, and raw softmax probability dumps for offline recomputation of Macro-F1 / BCA / ROC.

## Vendored dependencies

`models/DBConformer.py` expects two internal packages at the repository root that are not redistributed here:

- `cmamba/` (provides `CMambaEncoder`), adapted from [CMamba](https://github.com/wanghq21/CMamba); it backs the spatial branch and is required.
- `liquid_gated_attention/` (provides `LiquidMixer`), from the Liquid-Gated-Attention implementation; if it is missing the model falls back to standard multi-head attention with a warning.

Copy both packages next to the launcher scripts before running `DBConformer`, i.e. the root should contain `cmamba/` and `liquid_gated_attention/` side by side with `models/` and `utils/`.

## Citation

If you use this code, please cite the corresponding paper (bibliographic details to be added upon publication).

