# Decentralized-Privacy-Preserving-Federated-Learning-for-3D-Scan-Processing
## Project Structure

- **Local FL and experiments/**
  - `reconstruction.py`  
    Used for experimenting with the DLG algorithm, gradient compression, and noise injection on the CIFAR-10 dataset.

  - **CKKSTesting.py**  
  Experiments with noise accumulation under CKKS homomorphic encryption.

  - **UNet_segmentation_and_reconstruction.ipynb**  
  Jupyter notebook for running DLG experiments in combination with a UNet-based segmentation CNN.

  - **network.py**  
  Runs local federated learning via the OpenFL library to test various gradient-obfuscation methods and measure their impact on model accuracy.
- **NVIDIA Flare**
  - Main implementation of Federated Learning system
  - Uses NVIDIA Flare framework
  - Used in NVIDIA Jetson TX2 modules
