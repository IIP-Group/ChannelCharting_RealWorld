# Channel Charting in Real-World Coordinates

Channel charting using channel-state information (CSI) from the 5G physical uplink shared channel (PUSCH) with triplet loss and bilateration loss.

This repository implements channel charting methods that learn a low-dimensional representation (channel chart) of the radio environment from CSI without requiring ground-truth position labels. The implementation includes both triplet-based channel charting [1] and channel charting in real-world coordinates [2] for results published in [3] that are based on the CAEZ-5G-OUTDOOR dataset.

This codebase is a branch of the [channel charting in real-world coordinates](https://github.com/IIP-Group/ChannelCharting_RealWorld) repository, adapted for 5G CSI from the CAEZ-5G-OUTDOOR dataset.

## Overview

Channel charting is an unsupervised learning technique that maps CSI features to a low-dimensional representation (channel chart) that preserves local spatial proximity. Points that are close in physical space are mapped closely in the channel chart, enabling position estimation without ground-truth labels.

The system implements two methods:

- **Triplet-based Channel Charting [1]**: Uses a triplet loss to preserve local proximity. For each CSI sample (anchor), one sample close in time (positive) and one far in time (negative) form a triplet. The loss ensures the anchor-positive distance in the channel chart is smaller than the anchor-negative distance. This preserves local closeness but is not bound to real-world coordinates.

- **Channel Charting in Real-World Coordinates [2]**: Combines triplet loss with bilateration loss to ground the channel chart in real-world coordinates. The bilateration loss uses receive signal power estimates from CSI and O-RU positions to ensure samples with higher power at a particular O-RU are mapped closer to that O-RU's position.

The system uses:

- **CSI Features**: Approximate autocorrelation features in the delay domain. Full-spectrum CSI estimates of all 273 PRBs are squared, averaged over all three DMRS symbols, transformed to delay domain via IFFT, and truncated to the first 25 complex-valued taps. Real and imaginary parts are stacked to yield 50 real-valued features per O-RU antenna. Features from all O-RUs and antennas are aggregated and normalized to unit-norm.

- **Neural Network Architecture**: Fully-connected multi-layer perceptron with a linear output layer.

- **Training**: 
  - Triplet-based: 300 epochs, learning rate $10^{-3}$, batch size 100, one triplet per anchor
  - Real-world: 200 epochs, learning rate $10^{-3}$, batch size 256, two triplets per anchor
  - Both use Adam optimizer with learning rate decay (factor 0.1) after 200 epochs (triplet) or 50 epochs (real-world)
  - Bilateration loss uses a receive power margin of 13 dB

- **Dataset Split**: The CAEZ-5G-OUTDOOR (CAEZ stands for CSI Acquisition at ETH Zurich) dataset is randomly partitioned into training (80%) and testing (20%) samples. The last 500 samples, corresponding to a single connected sub-trajectory, are excluded from random partitioning and reserved for additional testing. For further information and to download the CSI dataset files (tar.zstd files), visit [https://caez.ethz.ch](https://caez.ethz.ch).

## Requirements

The code requires Python 3.x and the following packages:
- NumPy
- PyTorch
- Matplotlib
- SciPy
- tqdm

## Usage

### Step 1: Dataset Preparation

First, download the CAEZ-5G-OUTDOOR dataset from [https://caez.ethz.ch](https://caez.ethz.ch). The dataset files are provided as compressed tar.zstd archives containing CSI data from the PyAerial pipeline. Uncompress the dataset.

Then, preprocess the raw CSI data using the `gen_dataset.py` script from the [neural positioning](https://github.com/IIP-Group/neural-positioning). This script extracts and processes CSI features, loads WorldViz ground-truth position logs, and saves the processed data as `.npz` files:

```bash
python gen_dataset.py
```

**Note**: Configure the data path and feature extraction parameters in `gen_dataset.py` before running. For channel charting, you may need to adjust the feature extraction settings (e.g., autocorrelation features in delay domain) to match the requirements of the channel charting pipeline.

After preprocessing, configure the path to the processed `.npz` files in the channel charting training scripts.

### Step 2: Training

Train the channel charting network using one of the following scripts:

**Triplet-based channel charting:**
```bash
python train_5g_triplet.py
```

**Channel charting in real-world coordinates:**
```bash
python train_5g_real-world.py
```

**Configuration**: Edit the configuration in the respective training scripts to specify:
- Dataset filenames (training and validation sets)
- runID (used for naming the results directory)
- O-RU positions (for real-world channel charting)
- Training parameters (epochs, learning rate, batch size)

**Command Line Arguments**: The script takes two command line arguments to specify:
- Data path (path to the training and validation sets)
- Results path (path to the results directory) for saved models and results

Ensure the results directory and its subdirectories exist.

## File Structure

### Main Training Scripts
- `train_5g_triplet.py`: Training script for triplet-based channel charting
- `train_5g_real-world.py`: Training script for channel charting in real-world coordinates

### Supporting Files
- `test.py`: Testing and evaluation script
- `load_model.py`: Model loading utilities
- `plot_ap_powers.py`: Visualization of access point power estimates
- `utils/`: Utility modules including neural network models, loss functions, and helpers


## Version History

- **Version 0.1**: Implementation for CAEZ-5G-OUTDOOR experiments in [3]

## Citation

If you use this code (or parts of it), then you must cite references [2] and [3].

## References

[1] P. Ferrand, A. Decurninge, L. G. Ordonez, and M. Guillaud, "Triplet-based wireless channel charting: Architecture and experiments," *IEEE J. Sel. Areas Commun.*, vol. 39, no. 8, pp. 2361–2373, 2021.

[2] S. Taner, V. Palhares, and C. Studer, "Channel charting in real-world coordinates with distributed MIMO," *IEEE Trans. Wireless Commun.*, vol. 24, no. 9, pp. 7286–7300, 2025.

[3] R. Wiesmayr, F. Zumegen, S. Taner, C. Dick, and C. Studer, "CSI-based user positioning, channel charting, and device classification with an NVIDIA 5G testbed," in *Asilomar Conf. Signals, Syst., Comput.*, Oct. 2025, arXiv preprint https://arxiv.org/abs/2512.10809