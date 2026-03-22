# ECG Wave Segmentation using U-Net

This project implements a deep learning pipeline for **ECG waveform segmentation** (P wave, QRS complex, and T wave) using a **1D U-Net architecture** in PyTorch.

The model is trained on the LUDB dataset and performs semantic segmentation of ECG signals.

## Features

- ECG signal preprocessing
- Bandpass filtering (0.5–40 Hz)
- Sliding window segmentation
- 1D U-Net architecture
- Dice + CrossEntropy loss
- Evaluation using F1 score for P, QRS, and T waves

## Dataset

This project uses the **LUDB (Lobachevsky University Database)** ECG dataset.

Dataset link: https://physionet.org/content/ludb/1.0.1/

After downloading, place the dataset in the following directory: `data/ludb/data/`
