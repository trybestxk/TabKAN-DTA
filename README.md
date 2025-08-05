# TabKAN-DTA

## Disentangling Multimodal Contributions in Drug-Target Binding Affinity Prediction via Tabular Feature augmented Kolmogorov-Arnold Gating

## Overview

TabKAN-DTA is a novel multimodal deep learning framework for drug-target binding affinity prediction that integrates heterogeneous graph neural networks with tabular neural features through an interpretable Kolmogorov-Arnold gating mechanism. Our approach effectively combines structural, sequence, and tabular modalities while providing insights into their relative contributions to predictions.
![TabKAN-DTA Model Architecture](https://raw.githubusercontent.com/trybestxk/TabKAN-DTA/main/tabKANmodel.png)


## Key Features

- **Multimodal Integration**: Combines protein-ligand interaction graphs, sequence embeddings, and tabular features
- **Interpretable Gating**: Uses Kolmogorov-Arnold Networks to dynamically weight different modalities
- **State-of-the-Art Performance**: Achieves superior results on multiple benchmarks (CASF-2016, CASF-2013, Bind2020+)
- **Modality Contribution Analysis**: Provides insights into which modalities contribute most to predictions

## Installation

```
# Create conda environment from file

conda env create -f environment.yml

# Activate the environment

conda activate TabKAN_DTA
```

or

```
# Create a new conda environment
conda create -n TabKAN_DTA python=3.10
conda activate TabKAN_DTA

# Install CUDA and DGL
conda install -c nvidia/label/cuda-11.8.0 cuda-nvcc=11.8.89
conda install -c dglteam dgl=1.1.2.cu118

# Install PyTorch
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118

# Install other dependencies
pip install rdkit==2025.3.2 scikit-learn==1.6.1 pandas==2.2.3 
pip install matplotlib==3.10.3 seaborn==0.13.2 shap==0.48.0
pip install tabpfn==2.0.9 deep-kan==0.0.4 mamba-ssm==2.2.4
```

## Data Preparation

Datasets and Pre-trained Weights

**DataSet03**

- Processed protein-ligand complex graphs in binary format
- Includes train, validation, and multiple test sets (CASF-2013, CASF-2016, Bind2020+, NSCLC)
- https://huggingface.co/Trybestxk/TabKAN-DTA/blob/main/DataSet03.zip

**HGCN_PTH.zip**

- Pre-trained heterogeneous graph convolutional network weights
- Base model for capturing protein-ligand interaction patterns
- https://huggingface.co/Trybestxk/TabKAN-DTA/blob/main/HGCN_PTH.zip

**avg_embeddings.zip**

- Pre-computed protein and SMILES embeddings for all datasets
- Generated from ESM-2 and molecular language models
- https://huggingface.co/Trybestxk/TabKAN-DTA/blob/main/avg_embeddings.zip

**best_result.7z**

- Best performing TabKAN-DTA model weights for each benchmark dataset
- Includes models for CASF-2016, CASF-2013, Bind2020+.
- https://huggingface.co/Trybestxk/TabKAN-DTA/blob/main/best_result.7z

**tabpfn_learned_features.zip**

- Tabular neural features extracted using TabPFN
- Provides complementary information to graph and sequence modalities
- https://huggingface.co/Trybestxk/TabKAN-DTA/blob/main/tabpfn_learned_features.zip

```
TabKAN-DTA/
├── DataSet03/
│   └── processed/
│       ├── train.bin
│       ├── val.bin
│       ├── test2013.bin
│       ├── test2016.bin
│       ├── bind2020.bin
│       └── csar.bin
├── HGCN_PTH/
│   └── best_model.pt
├── avg_embeddings/
│   ├── train_protein_avg_embeddings.pt
│   ├── train_smiles_avg_embeddings.pt
│   ├── val_protein_avg_embeddings.pt
│   ├── val_smiles_avg_embeddings.pt
│   ├── test2013_protein_avg_embeddings.pt
│   ├── test2013_smiles_avg_embeddings.pt
│   ├── test2016_protein_avg_embeddings.pt
│   ├── test2016_smiles_avg_embeddings.pt
│   ├── bind2020_protein_avg_embeddings.pt
│   ├── bind2020_smiles_avg_embeddings.pt
│   ├── csar_protein_avg_embeddings.pt
│   └── csar_smiles_avg_embeddings.pt
├── best_result/
│   ├── test2016.bin/
│   │   ├── best_model.pt
│   │   └── best_model_corr.pt
│   ├── test2013.bin/
│   │   ├── best_model.pt
│   │   └── best_model_corr.pt
│   ├── bind2020.bin/
│   │   ├── best_model.pt
│   │   └── best_model_corr.pt
│   └── csar.bin/
│       ├── best_model.pt
│       └── best_model_corr.pt
├── tabpfn_learned_features/
│   ├── train.npy
│   ├── val.npy
│   ├── test2013.npy
│   ├── test2016.npy
│   ├── bind2020.npy
│   └── csar.npy
├── FastKANlob.py
├── HGCN.py
├── Loder.py
├── Utils.py
├── environment.yml
├── model.py
└── test.py

```

## Evaluation

```
python test.py
```


