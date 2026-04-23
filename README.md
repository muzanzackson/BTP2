# CNN Embedding Analysis Suite

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/YOUR_USERNAME/Embedding-Analysis/actions/workflows/lint.yml/badge.svg)

> A multi-page Streamlit application for extracting, analyzing, and visualizing feature embeddings from CNN and transformer models. Uses hierarchical relationship analysis of model embeddings via phylogenetic tree distances and Robinson-Foulds scoring.

---

## Reference Papers

This software is based on the following research articles. If you use this tool in your research, please consider citing them:

1. Chatterjee, A., Mukherjee, J., & Das, P. P. (2025). **Analyzing Hierarchical Relationships and Quality of Embedding in Latent Space**. *IEEE Transactions on Artificial Intelligence*, 6(4), 843-858. [DOI: 10.1109/TAI.2024.3497921](https://doi.org/10.1109/TAI.2024.3497921)
2. Chatterjee, A., Mukherjee, J., & Das, P. P. (2024). **ImageNet Classification Using WordNet Hierarchy**. *IEEE Transactions on Artificial Intelligence*, 5(4), 1718-1727. [DOI: 10.1109/TAI.2023.3297086](https://doi.org/10.1109/TAI.2023.3297086)
3. Mahapatra, A., & Mukherjee, J. (2021). **Deformity Index: A Semi-Reference Clade-Based Quality Metric of Phylogenetic Trees**. *Journal of Molecular Evolution*.

---

## What This Project Does

When you train or use a deep learning model, each layer produces a numeric vector for every image it processes — these are called **embeddings** or **feature vectors**. Similar images should produce similar vectors. This suite lets you:

- **Extract** feature vectors from 70+ pretrained models (ResNet, VGG, EfficientNet, ViT, ConvNeXt, and more) or your own custom models.
- **Visualize** how classes cluster in high-dimensional space using t-SNE, with Deformity Index scoring.
- **Analyze** whether the learned embedding hierarchy matches real-world class taxonomy using Robinson–Foulds tree distance.
- **Explore** interactive 3D phylogenetic trees to examine structural relationships.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
   - [Option A: Conda (Recommended)](#option-a-conda-recommended)
   - [Option B: Python venv](#option-b-python-venv)
3. [Running the Application](#running-the-application)
4. [Page-by-Page Guide](#page-by-page-guide)
   - [Page 1: Extract Embeddings](#page-1-extract-embeddings)
   - [Page 2: Cluster & t-SNE](#page-2-cluster--t-sne)
   - [Page 3: Robinson–Foulds Analysis](#page-3-robinsonfoulds-analysis)
   - [Page 4: Tree Visualization](#page-4-tree-visualization)
5. [Typical Workflow](#typical-workflow)
6. [Important Notes and Common Issues](#important-notes-and-common-issues)
7. [Citation](#citation)

---

## Project Structure

```
Embedding-Analysis/
│
├── main.py                      # Entry point — run with: streamlit run main.py
├── feature_extraction.py        # Page 1: model loading and embedding extraction
├── cluster_analysis.py          # Page 2: t-SNE visualization and cluster quality
├── hierarchical_analysis.py     # Page 3: UPGMA tree building and RF distance
├── tree_visualization.py        # Page 4: interactive 3D tree viewer
├── tree_rendering.py            # Helper: CLI renderer for tree HTML
│
├── data/                        # Bundled taxonomy reference files
│   ├── CIFAR10_WordNet.nwk      # WordNet reference tree for CIFAR-10
│   ├── CIFAR10_classes.json     # Class name mapping for CIFAR-10
│   ├── CIFAR100_WordNet.nwk     # WordNet reference tree for CIFAR-100
│   └── CIFAR100_classes.json    # Class name mapping for CIFAR-100
│
├── lib/                         # Vendored third-party libraries
│   └── DefIn/                   # Deformity Index library
│
├── embeddings/                  # OUTPUT: extracted .npz files (gitignored)
│   └── .gitkeep
│
├── trees/                       # OUTPUT: generated .nwk tree files (gitignored)
│   └── .gitkeep
│
├── logs/                        # OUTPUT: runtime logs (gitignored)
│   └── .gitkeep
│
├── assets/                      # Screenshots for this README
│
├── requirements.txt             # CPU Python dependencies (pinned)
├── requirements_gpu.txt         # GPU variant (CUDA)
├── requirements-dev.txt         # Developer tools (ruff, black, flake8)
├── environment.yml              # Conda environment spec
│
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Environment Setup

You need **Python 3.10 or 3.11**. Python 3.12+ is not recommended because `ete3` and `dendropy` may lack compatible wheels.

### Option A: Conda (Recommended)

Conda handles complex binary dependencies (PyTorch, `ete3`) more reliably than pip alone.

**Step 1: Install Miniconda** (if not already installed)

- <https://docs.conda.io/en/latest/miniconda.html>

**Step 2: Create and activate the environment**

```bash
conda create -n embedding-analysis python=3.10 -y
conda activate embedding-analysis
```

**Step 3: Install PyTorch**

Choose the correct command for your hardware:

- **CPU only (most laptops):**

  ```bash
  pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
  ```

- **NVIDIA GPU (CUDA 11.8):**

  ```bash
  pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
  ```

- **NVIDIA GPU (CUDA 12.1):**

  ```bash
  pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
  ```

  If unsure which CUDA version you have, run `nvidia-smi`. If the command is not found, use the CPU option.

**Step 4: Install remaining dependencies**

```bash
cd /path/to/Embedding-Analysis
pip install -r requirements.txt
```

**Step 5 (optional): Install `ete3` via conda for stability**

```bash
conda install -c etetoolkit ete3 -y
```

---

### Option B: Python venv

**Step 1: Check your Python version**

```bash
python3 --version   # Must be 3.10 or 3.11
```

**Step 2: Create and activate a virtual environment**

```bash
cd /path/to/Embedding-Analysis
python3 -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows Command Prompt
```

**Step 3: Install PyTorch** (same commands as Conda Option Step 3 above)

**Step 4: Install remaining dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Application

Activate your environment and navigate to the project folder:

```bash
conda activate embedding-analysis   # or: source venv/bin/activate
cd /path/to/Embedding-Analysis
python main.py
```

The script will launch four independent Streamlit servers in the background. You can access the pages using the following local URLs:
- **Feature Extraction:** http://localhost:8501
- **Cluster Analysis:** http://localhost:8502
- **Hierarchical Analysis:** http://localhost:8503
- **Tree Visualization:** http://localhost:8504

To shut down all servers, press `Ctrl + C` in the terminal.

> **Important:** Always run the command from inside the `Embedding-Analysis/` directory, not from a parent folder — the app reads relative paths for `embeddings/`, `trees/`, and `data/`.

---

## Page-by-Page Guide

### Page 1: Extract Embeddings

**File:** `feature_extraction.py`

Loads a neural network model and runs your images through it to extract feature vectors from a layer you choose. Results are saved as a `.npz` file in the `embeddings/` folder.

**How to use:**

1. **Choose a model mode** — *Pretrained* (70+ torchvision models) or *Custom* (your `.py` + `.pth` files).
2. **Select a layer** from the architecture browser. Use the search box to filter by name (e.g., `avgpool`, `fc`).
3. **Configure extraction** — choose dataset source (Standard: CIFAR-10/100/MNIST/FashionMNIST/STL10, or Local: upload your own images) and transformation type.
4. **Click Extract Features.** A progress bar tracks extraction. The `.npz` file is saved to `embeddings/` and a download button appears.

---

### Page 2: Cluster & t-SNE

**File:** `cluster_analysis.py`

Takes a `.npz` embedding file and produces:

- **t-SNE scatter plot** (2D or 3D) showing how classes cluster visually.
- **Boundary circles/spheres** around each class cluster.
- **Inter-centroid distance heatmap** across all class pairs.
- **Deformity Index** per class (0 = perfect, 1 = very messy).
- **Clustering metrics:** NMI, Purity Index, and Adjusted Rand Index.

**How to use:**

1. **Load data** — from the `embeddings/` folder, upload a `.npz`/`.npy`, or upload a folder of `.npy` files.
2. **Configure** in the sidebar: Dimension (2D/3D), Reduction Method (PCA/TruncatedSVD), Distance Metric (euclidean/manhattan/canberra).
3. **Click Run Visualization.** Results appear in tabs below.

---

### Page 3: Robinson–Foulds Analysis

**File:** `hierarchical_analysis.py`

The research core of the application. Answers: **does the learned embedding structure of a model match the known real-world hierarchy of classes?**

**Process:**

1. Loads embeddings and computes class centroids using a boundary-detection algorithm.
2. Builds UPGMA phylogenetic trees from centroids under 3 distance metrics × 2 centroid types = **6 trees**.
3. Compares each tree against the WordNet reference taxonomy using **Robinson–Foulds (RF) distance**.
4. Lower RF distance = better alignment between model embeddings and real-world class hierarchy.

**Reference files** (bundled in `data/`):

- `data/CIFAR10_WordNet.nwk` + `data/CIFAR10_classes.json`
- `data/CIFAR100_WordNet.nwk` + `data/CIFAR100_classes.json`

**How to use:**

1. Upload a `.npz` file from `embeddings/`.
2. For custom datasets, enable the Custom Taxonomy toggle in the sidebar and upload your `.nwk` + `.json` files.
3. Click **Run Full Analysis**. Generated trees are saved to the `trees/` folder.

---

### Page 4: Tree Visualization

**File:** `tree_visualization.py`

Renders an interactive 3D phylogenetic tree in the browser. Supports rotate, zoom, pan, and hover to see node names, depth, and branch lengths.

**How to use:**

1. Select class configuration (Default CIFAR or custom JSON).
2. Select a tree — default WordNet reference, from `trees/` folder, or upload a `.nwk` file.
3. Click **Generate Tree Visualization**. The tree appears inline.

---

## Typical Workflow

```
1. Page 1 → Load ResNet50 → CIFAR-10 → Extract Features
2. Page 2 → Load the NPZ → Run Visualization → inspect t-SNE and Deformity Index
3. Page 3 → Upload the same NPZ → Run Full Analysis → read the RF distance table
4. Page 4 → Select a tree from trees/ → Generate → explore the 3D view
```

---

## Important Notes and Common Issues

**NumPy version is strictly pinned**

`requirements.txt` pins `numpy==1.26.4`. NumPy 2.0 introduced breaking changes that crash `ete3`. Do not upgrade.

**First model load downloads weights**

PyTorch downloads pretrained weights on first use (50–500 MB depending on model). Internet access required. Weights are cached locally afterward.

**`ete3` on Windows**

Install via conda if pip fails:

```bash
conda install -c etetoolkit ete3
```

Pages 1 and 2 still work without `ete3`. Only Pages 3 and 4 require it.

**t-SNE is capped at 1000 samples**

For CIFAR-100 (10,000 test images), 1000 samples are randomly drawn. This is expected behavior.

**GPU acceleration**

PyTorch uses your NVIDIA GPU automatically (Page 1). If `cupy` is installed, Page 2 also uses the GPU for distance computations.

**Run from the project root**

Always run `streamlit run main.py` from inside the `Embedding-Analysis/` directory. Paths to `embeddings/`, `trees/`, and `data/` are relative to the working directory.

**`runtime.txt` removed**

The old `runtime.txt` was a Heroku artifact and has been replaced by `environment.yml` for Conda users.

---

## Citation

If you use this tool in your research, please cite the research articles listed in the [Reference Papers](#reference-papers) section at the top of this document.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style guide, and how to submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
