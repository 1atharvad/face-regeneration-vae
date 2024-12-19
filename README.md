# Image Reconstruction using Variational AutoEncoder

## Overview
This project leverages Python version 3.11.5 to implement a Variational AutoEncoder (VAE) for image reconstruction tasks. The dataset used for training and evaluation is the `img_align_celeba` folder, which contains around 200,000 images of celebrities.

The dataset can be downloaded from this [link](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ).

## Features
- Implements a Variational AutoEncoder to learn compact latent representations of images.
- Reconstructs images from their latent representations.
- Utilizes the CelebA dataset for high-quality results.

## Getting Started
### Prerequisites
- Python 3.11.5
- Required libraries (install using the `requirements.txt` file):
```bash
pip install -r requirements.txt
```

### Downloading the Dataset
1. Access the dataset from the provided [Google Drive link](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ).
2. Extract the downloaded folder and place it in the root directory of the project.

### Running the Project
1. Clone this repository:
```bash
git clone https://github.com/1atharvad/face-regeneration-vae.git
cd face-regeneration-vae
```
2. Ensure the `img_align_celeba` dataset is correctly placed.
3. Train the model:
```bash
python train_model.py
```

## Project Structure
- `train_model.py`: Script for training the Variational AutoEncoder.
- `test_model.py`: Script for testing the model and generating reconstructed images.
- `models/`: Directory containing the VAE implementation.
- `img_align_celeba/`: Dataset folder (ensure it is added).

## Results
Sample reconstructions can be found in the `results/` directory after running the testing script.

## References
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

