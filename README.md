# Decentralized-Privacy-Preserving-Federated-Learning-for-3D-Scan-Processing
## Project Structure

- **Local/**
  - `reconstruction.py`  
    Used for experimenting with the DLG algorithm, gradient compression, and noise injection on the CIFAR-10 dataset.

- **CKKSTesting.py**  
  Experiments with noise accumulation under CKKS homomorphic encryption.

- **UNet segmentation and reconstruction.py**  
  Jupyter notebook for running DLG experiments in combination with a UNet-based segmentation CNN.

- **network.py**  
  Runs local federated learning via the OpenFL library to test various gradient-obfuscation methods and measure their impact on model accuracy.
