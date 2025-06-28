# 3DDGD: 3D Deepfake Generation and Detection Using 3D Face Meshes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](3DDGD_Demo.ipynb)
[![View Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/document/11039631?source=authoralert)


<p align="center">
  <img src="3DDGD_Demo.gif" alt="3DDGD Demo" style="max-width: 50%; height: auto;">
</p>

---

<p align="justify">
  
This is the official PyTorch implementation of **3DDGD**, a project focused on enhancing the security of 3D facial biometric systems by detecting deepfakes in 3D face meshes.

Leveraging the unique geometric richness of 3D data over traditional 2D images, **3DDGD** introduces a dataset of real and fake 3D faces, along with advanced deep learning models, including mesh-based multi-layer perceptrons (MLPs), self-attention mechanisms, and TabTransformer architectures, to accurately distinguish authentic identities from forgeries. Our results demonstrate that 3D facial meshes offer a significant advantage in deepfake detection, supporting more secure biometric authentication and virtual identity systems.

</p>

---

### 🔧 Getting Started

Follow the instructions below to set up your environment and run inference or training.

#### 🔄 1- Clone the Repository
```bash
git clone https://github.com/hichemfelouat/3DDGD.git 
cd 3DDGD
```
##### 2- Create and activate the Conda environment:
```bash
conda create --name 3DDGD python=3.10
conda activate 3DDGD
```
#### 3- Install PyTorch and CUDA dependencies:
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
#### 4- Install additional Python dependencies:
```bash
pip install -r requirements.txt
```

---

### 🚀 Inference
To run inference using a pretrained model, follow these steps:

#### 📥 Step 1: Download Pretrained Weights

- Download the pretrained weights from the provided [link](https://drive.google.com/file/d/1lVYY5rzaP2FK4SHj38ror8f3LZNDIrU2/view?usp=sharing).
- Unzip the contents into the `weights/` folder.

#### ▶️ Step 2: Run Inference

Use the following command:

```bash
python inference.py --input_data examples --model_name Mesh_MLP_MHA --feature_type G0 --is_cropped 0
```
```bash
Arguments:
--input_data: Path to the input 3D mesh directory 
--model_name: Name of the model architecture to use 
--feature_type: Type of input features (e.g., G0, G1, etc.)
--is_cropped: Set to 1 if the input meshes are cropped, 0 otherwise
```

You can download a free .obj format example file [here](https://www.artec3d.com/3d-models/head).

---
### 🧠 Training

To train the models, follow these steps:

1. 🔧 **Prepare the training data**:  
   - Convert files to `.obj` format if needed.  
   - Perform feature extraction as required by your model.  
   - Create the necessary CSV files to organize the training data.

2. ⚙️ **Train the models**:  
   - Each model has its own training script located in its respective folder.  
   - Update the script with the correct file paths and adjust the hyperparameters as needed before training.

---

### 📄 License

This code and the associated models are available for **non-commercial scientific research purposes**.  

### Acknowledgements

We gratefully acknowledge the contributions of the following open-source projects that inspired or supported parts of this work:

- [Identifying Dense 3D Facial Landmarks by Utilizing 2D as an Intermediate](https://github.com/cse15-sip-interns/3d_face_landmark_identification.git)  
- [Laplacian2Mesh: Laplacian-Based Mesh Understanding](https://github.com/QiujieDong/Laplacian2Mesh.git)  
- [A Task-driven Network for Mesh Classification and Semantic Part Segmentation](https://github.com/QiujieDong/TaskDrivenNet2Mesh.git)

External functions are individually acknowledged in their respective files.

### 📚 Cite

If you find our work useful for your research, please consider citing the following papers :)
```bibtex
@article{felouat20253ddgd,
  title={3DDGD: 3D Deepfake Generation and Detection Using 3D Face Meshes},
  author={Felouat, Hichem and Nguyen, Huy H and Yamagishi, Junichi and Echizen, Isao},
  journal={IEEE Access},
  volume={13},
  pages={107429--107441},
  year={2025},
  publisher={IEEE}
}
```

### 📬 Contact

For questions or collaborations, feel free to contact:

📧 hichemfel@gmail.com  
📧 hichemfel@nii.ac.jp
