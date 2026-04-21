# ECG Signal Segmentation using 1D U-Net

This project implements a deep learning pipeline for ECG signal segmentation using a 1D U-Net architecture in PyTorch.

The model is designed to detect:
- QRS complexes
- Spikes (artifacts / peaks)

The pipeline supports multi-channel ECG signals (up to 12 channels) and includes training, evaluation, inference, and JSON markup generation.

## Features

- Multi-channel ECG processing
- Sliding window segmentation
- Signal normalization
- Threshold-based decoding
- Postprocessing of segments
- JSON output compatible with training markup
- Visualization of signal, probabilities, and predictions

## Dataset

The project uses custom ECG data in the following format:

data/
├── signals/
│   ├── 1.npy
│   ├── 2.npy
│   └── ...
├── markup/
│   ├── 1.json
│   ├── 2.json
│   └── ...

### Signal format
- .npy
- shape: [channels, samples]

### Markup format

{
  "Segments": [
    [
      {
        "Channel": 0,
        "Type": 0,
        "StartMark": 100,
        "EndMark": 150
      }
    ]
  ]
}

Where:
- Type = 0 → QRS
- Type = 1 → SPIKES


## Inference

Run prediction:
```bash
python -m scripts.predict_unlabeled
```
## Output

data/prediction_markin/<file_id>.json

## Configuration

Main parameters are defined in config.py
