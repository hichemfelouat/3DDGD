# 3DDGD: 3D Deepfake Generation and Detection Using 3D Face Meshes

This is the official PyTorch implementation of **3DDGD**.

**3DDGD** project focuses on enhancing security in 3D facial biometric systems by detecting deepfakes on 3D face meshes. Leveraging the unique advantages of 3D facial features over 2D, we created a dataset of real and fake 3D faces and developed advanced models, including mesh-based multi-layer perceptrons, self-attention mechanisms, and TabTransformer architectures, for accurate 3D deepfake detection. Our results demonstrate that 3D face meshes significantly improve the ability to distinguish real faces from deepfakes, paving the way for more secure biometric authentication and virtual interactions.

---

## 🔧 Getting Started

Follow the steps below to set up the environment and run inference.

### 🔄 Clone the Repository
```bash
git clone https://github.com/hichemfelouat/3DDGD 
cd 3DDGD
```
#### Create and activate the Conda environment:
```bash
conda create --name 3DDGD python=3.10
conda activate 3DDGD
```
#### Install PyTorch and CUDA dependencies:
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
#### Install additional Python dependencies:
```bash
pip install -r requirements.txt
```

### 🚀 Inference
```bash
python inference.py --input_data examples/3d_meshes --model_name Mesh_MLP_MHA --feature_type G0 --is_cropped 0
```
```bash
Arguments:
--input_data: Path to the input 3D mesh directory 
--model_name: Name of the model architecture to use 
--feature_type: Type of input features (e.g., G0, G1, etc.)
--is_cropped: Set to 1 if the input meshes are cropped, 0 otherwise
```
