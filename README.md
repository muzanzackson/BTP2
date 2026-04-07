# CNN Embedding Analysis Suite

A multi-page Streamlit web application for extracting, analyzing, and visualizing feature embeddings from CNN and transformer models. The tool is designed for researchers studying how neural networks represent image data and how those representations compare across architectures.

The suite is built around the methodology from **Chatterjee et al., IEEE TAI 2025**, which proposes hierarchical relationship analysis of model embeddings using phylogenetic tree distances.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [Environment Setup](#environment-setup)
   - [Option A: Conda (Recommended)](#option-a-conda-recommended)
   - [Option B: Python venv](#option-b-python-venv)
   - [Option C: pip global install](#option-c-pip-global-install)
4. [Running the Application](#running-the-application)
5. [Page-by-Page Guide](#page-by-page-guide)
   - [Page 1: Extract Embeddings](#page-1-extract-embeddings)
   - [Page 2: Analyze and Visualize](#page-2-analyze-and-visualize)
   - [Page 3: Robinson-Foulds Analysis](#page-3-robinson-foulds-analysis)
   - [Page 4: Tree Visualization](#page-4-tree-visualization)
6. [Typical Workflow](#typical-workflow)
7. [Important Notes and Common Issues](#important-notes-and-common-issues)

---

## What This Project Does

When you train or use a deep learning model, each layer of the network produces a numeric vector for every image it processes. These vectors are called **embeddings** or **feature vectors**. Two similar images should ideally produce similar vectors.

This suite lets you:

- Extract those vectors from any of 70+ pretrained models (ResNet, VGG, EfficientNet, ViT, ConvNeXt, and more) or your own custom model.
- Visualize how classes cluster together in high-dimensional space using t-SNE.
- Compare the hierarchical structure of those clusters against a known reference taxonomy (e.g., WordNet class hierarchy) using Robinson-Foulds tree distance.
- Render interactive 3D phylogenetic trees to explore the structural relationships.

---

## Project Structure

```
Embedding-Analysis/
|
|-- main.py                  # Entry point; defines the 4-page navigation
|-- feature_extraction.py    # Page 1: model loading and embedding extraction
|-- cluster_analysis.py      # Page 2: t-SNE visualization and cluster quality
|-- hierarchical_analysis.py # Page 3: UPGMA tree building and RF distance
|-- tree_visualization.py    # Page 4: interactive 3D tree viewer
|-- tree_rendering.py        # Helper: CLI tool that generates tree HTML files
|
|-- requirements.txt         # All Python dependencies with pinned versions
|
|-- CIFAR10_WordNet.nwk      # Reference tree for CIFAR-10 (WordNet hierarchy)
|-- CIFAR10_classes.json     # Class name mapping for CIFAR-10
|-- CIFAR100_WordNet.nwk     # Reference tree for CIFAR-100
|-- CIFAR100_classes.json    # Class name mapping for CIFAR-100
|
|-- embeddings/              # Folder where extracted .npz files are saved
|-- trees/                   # Folder where generated .nwk tree files are saved
|-- DefIn/                   # External Deformity Index library
```

---

## Environment Setup

You need Python 3.10 or 3.11. Python 3.12+ is not recommended because some bioinformatics packages (`ete3`, `dendropy`) may not have wheels for it yet.

### Option A: Conda (Recommended)

Conda handles complex binary dependencies (like PyTorch) more reliably than pip alone.

**Step 1: Install Miniconda**

If you do not already have Conda, download and install Miniconda:
- Mac/Linux: https://docs.conda.io/en/latest/miniconda.html
- Windows: https://docs.conda.io/en/latest/miniconda.html

**Step 2: Create a new environment**

```bash
conda create -n embedding-analysis python=3.10 -y
```

**Step 3: Activate the environment**

```bash
conda activate embedding-analysis
```

Your terminal prompt should now show `(embedding-analysis)` at the start.

**Step 4: Install PyTorch**

Install PyTorch separately first because the correct command depends on whether you have a GPU.

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

If you are unsure which CUDA version you have, run `nvidia-smi` in a terminal. If the command is not found, you have no NVIDIA GPU and should use the CPU option.

**Step 5: Install remaining dependencies**

Navigate to the project directory first, then run:

```bash
cd /path/to/Embedding-Analysis
pip install -r requirements.txt
```

> Note: `torch` and `torchvision` are already listed in `requirements.txt`. pip will skip them if they are already installed from Step 4.

**Step 6: Install ete3 via conda (optional but more stable)**

`ete3` sometimes has issues installing via pip on certain systems. If you run into errors, try:

```bash
conda install -c etetoolkit ete3 -y
```

---

### Option B: Python venv

Use this if you prefer not to install Conda. This uses Python's built-in virtual environment.

**Step 1: Check your Python version**

```bash
python3 --version
```

You need 3.10 or 3.11. If your version is different, install the correct version from https://www.python.org/downloads/

**Step 2: Create a virtual environment**

Navigate to the project directory first:

```bash
cd /path/to/Embedding-Analysis
python3 -m venv venv
```

**Step 3: Activate the environment**

- Mac / Linux:
  ```bash
  source venv/bin/activate
  ```

- Windows (Command Prompt):
  ```cmd
  venv\Scripts\activate
  ```

- Windows (PowerShell):
  ```powershell
  venv\Scripts\Activate.ps1
  ```

Your prompt should change to show `(venv)`.

**Step 4: Upgrade pip**

```bash
pip install --upgrade pip
```

**Step 5: Install PyTorch**

Same as Conda Option Step 4 above. Choose CPU or GPU depending on your hardware.

**Step 6: Install remaining dependencies**

```bash
pip install -r requirements.txt
```

---

### Option C: pip global install

Not recommended because it can conflict with other projects on your system, but it works for a quick test.

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Running the Application

Make sure your environment is activated (you should see `(embedding-analysis)` or `(venv)` in your terminal prompt).

Navigate to the project folder:

```bash
cd /path/to/Embedding-Analysis
```

Start the Streamlit app:

```bash
streamlit run main.py
```

Streamlit will print a local URL, usually:

```
Local URL: http://localhost:8501
```

Open that URL in your web browser. The app will load with a sidebar on the left for navigation between the four pages.

To stop the app, press `Ctrl + C` in the terminal.

---

## Page-by-Page Guide

### Page 1: Extract Embeddings

**File:** `feature_extraction.py`
**Navigation label:** "1. Extract Embeddings"

#### What it does

This is the starting point. It loads a neural network model and runs your images through it to extract the numeric feature vectors from a layer you choose. The vectors are saved as a `.npz` file in the `embeddings/` folder for use in later pages.

#### How to use it

**Step 1: Choose a model mode**

You have two options:

- **Pretrained**: Choose from over 70 models built into PyTorch (ResNet18, VGG16, EfficientNet, ViT, ConvNeXt, etc.). These come with weights already trained on ImageNet. Select a model from the dropdown and click **Load Model**. The model downloads automatically on first use.

- **Custom (.py + .pth)**: If you have trained your own model, upload your model definition file (`.py`) and weight file (`.pth`). After uploading the `.py` file, click **Load Architecture**, then pick the class from the dropdown and click **Confirm Class Selection**. Upload the `.pth` file and click **Load Weights**.

**Step 2: View the model architecture and pick a layer**

After loading, a list of all layers appears with their output shapes. Use the search box to filter by name (e.g., type `fc` to find fully connected layers, or `avgpool` to find pooling layers). Select the layer you want to extract features from.

For pretrained models, the last pooling layer (e.g., `avgpool`) or the penultimate fully connected layer is a common choice for meaningful embeddings.

**Step 3: Configure feature extraction**

- **Dataset Source**: Choose **Standard** to use a built-in dataset (CIFAR-10, CIFAR-100, MNIST, FashionMNIST, STL10), or **Local** to upload your own images.

- **Transformation**: Choose how images are resized before being fed to the model.
  - **Resize**: Stretches the image to the target size.
  - **Crop**: Scales the image so the short side matches the target, then center-crops.
  - **None**: No spatial transform (only normalization is applied).

**Step 4: Extract**

Click **Extract Features**. A progress bar shows the extraction progress. When done, the `.npz` file is saved to the `embeddings/` folder and a download button appears so you can save a copy locally.

---

### Page 2: Analyze and Visualize

**File:** `cluster_analysis.py`
**Navigation label:** "2. Analyze & Visualize"

#### What it does

This page takes a `.npz` embedding file and produces:

- A **t-SNE scatter plot** (2D or 3D) that projects high-dimensional embeddings down to 2 or 3 dimensions so you can see how classes cluster visually.
- Boundary circles (2D) or spheres (3D) drawn around each class cluster to show how well-separated the classes are.
- An **inter-centroid distance heatmap** showing how far apart each pair of class centroids is.
- A **Deformity Index** for each class, which is a score from 0 to 1 measuring how clean and compact the cluster is (lower is better: 0 is perfect, 1 is very messy).
- Clustering metrics: **NMI** (Normalized Mutual Information), **Purity Index**, and **Rand Index**, which quantify how well the embeddings align with the true class labels.

#### How to use it

**Step 1: Load data**

Choose a source from the three options at the top:

- **Pre-loaded Embeddings**: If you extracted embeddings on Page 1 (or previously), they appear in this dropdown. Select one and click Load.
- **Upload NPY/NPZ File**: Upload a `.npy` or `.npz` file from your computer.
- **Upload Folder of NPY Files**: Upload multiple `.npy` files at once. File names are used to infer class labels (e.g., a file named `cat_001.npy` will be labeled class `cat`).

**Step 2: Configure settings (sidebar)**

- **Dimension**: `2D` or `3D`. 3D is more visually interesting but slower to compute.
- **Reduction Method**: `PCA` or `TruncatedSVD`. Applied to reduce the dimensionality before t-SNE kicks in. PCA is the standard choice.
- **Distance Metric**: `euclidean`, `cosine`, `cityblock`, or `canberra`. Controls how distances are measured when drawing cluster boundaries and the heatmap.

**Step 3: Run**

Click the **Run Visualization** button. t-SNE can take 30 seconds to several minutes depending on the number of samples (it is capped at 1000 samples; larger datasets are automatically downsampled). Watch the log panel at the bottom for status messages.

**Step 4: Interpret results**

In the t-SNE plot, each color represents a different class. Tight, well-separated clusters indicate that the model has learned distinct representations for each class. Overlapping clusters indicate the model is confused between those classes.

The Deformity Index table lists a score per class. Lower scores are better. The overall deformity is a single summary number.

---

### Page 3: Robinson-Foulds Analysis

**File:** `hierarchical_analysis.py`
**Navigation label:** "3. Robinson Foulds Analysis"

#### What it does

This is the research core of the application. It answers the question: **does the learned embedding structure of a model match the known real-world hierarchy of classes?**

It does this by:

1. Loading your embeddings and computing class centroids using a boundary-detection algorithm (from Chatterjee et al., 2025).
2. Building phylogenetic trees (using UPGMA clustering) from those centroids under three distance metrics (Euclidean, Manhattan/Cityblock, Canberra) and two centroid types (G and MPP).
3. Comparing each generated tree against a reference taxonomy (e.g., the WordNet hierarchy of CIFAR class names) using the **Robinson-Foulds (RF) distance**, which measures how different two trees are topologically.
4. Lower RF distance means the model's internal representation more closely mirrors how humans categorize the world.

It also renders cladogram and dendrogram visualizations of all 6 generated trees alongside the reference tree.

#### What Robinson-Foulds distance means

Two trees with the same topology have RF = 0 (identical). The more they differ in how they branch, the higher the RF score. A model where animals cluster together, vehicles cluster together, etc. will have a low RF distance to the WordNet tree.

#### How to use it

**Step 1: Upload an embedding file**

Drag and drop a `.npz` file from your `embeddings/` folder into the upload zone. The file should have been produced by Page 1. The app detects whether it is CIFAR-10 or CIFAR-100 automatically based on the number of classes.

**Step 2: Taxonomy settings (sidebar)**

- For CIFAR-10 and CIFAR-100, the reference taxonomies are bundled with the app (`CIFAR10_WordNet.nwk` and `CIFAR100_WordNet.nwk`). These are loaded automatically.
- If you have a custom dataset, you can upload your own `.nwk` (Newick format) reference tree and a `.json` class name mapping. Toggle the "Custom Taxonomy" switch in the sidebar.

**Step 3: Run analysis**

Click **Run Full Analysis**. The tool computes boundary groups and centroids (which can take a few minutes for large datasets), builds all 6 trees, computes RF distances for all 9 metric combinations, and renders all visualizations.

**Step 4: Read the RF table**

The table shows RF distances for every combination of group distance metric, tree distance metric, and centroid type. Lower values mean better alignment with the reference taxonomy. Look for the combination that consistently gives low RF scores — that tells you which metric best captures the semantic structure of the dataset for that model.

The generated `.nwk` tree files are saved to the `trees/` folder and can be used in Page 4.

---

### Page 4: Tree Visualization

**File:** `tree_visualization.py`
**Navigation label:** "4. Tree Visualization"

#### What it does

This page renders an interactive 3D tree in the browser. You can rotate, zoom, and pan the tree and hover over nodes to see class names, depth, and branch lengths. It takes any Newick `.nwk` file and an optional JSON label file and produces a self-contained HTML page with the visualization embedded.

This is primarily used to explore the UPGMA trees generated by Page 3, or to view the reference WordNet trees.

#### How to use it

**Step 1: Select class configuration**

- **Default (CIFAR)**: Choose CIFAR-10 or CIFAR-100. The bundled `.nwk` and `.json` files will be used.
- **Upload Custom JSON**: If you have a custom label mapping for your own dataset, upload it here.

**Step 2: Select a tree**

- **Use Default Tree**: Uses the bundled WordNet reference tree for the selected dataset.
- **Select from trees/ folder**: Choose from any `.nwk` file in the `trees/` folder (these are generated by Page 3 after running analysis).
- **Upload New Tree**: Upload any `.nwk` file from your computer.

**Step 3: Generate**

Click **Generate Tree Visualization**. The app runs the `tree_rendering.py` script in the background, which produces `tree_visualization.html`. This may take 5-10 seconds.

**Step 4: View**

The interactive 3D tree appears directly in the page. You can:
- **Drag** to rotate the 3D view.
- **Scroll** to zoom in and out.
- **Shift + drag** to pan.
- **Hover** over any node to see its name, depth, and branch length.

Leaf nodes (classes) are colored based on their depth in the tree using a blue-to-coral gradient.

---

## Typical Workflow

For a new user running this for the first time, the recommended sequence is:

1. Open the app and go to **Page 1**.
2. Select a pretrained model (e.g., ResNet50 — a good starting point).
3. Use the Standard dataset source and pick CIFAR-10. Click Extract Features.
4. Go to **Page 2**, load the embedding that was just created, and click Run Visualization.
5. Look at the t-SNE plot. Do the classes cluster cleanly? Check the Deformity Index.
6. Go to **Page 3**, upload the same embedding file. Click Run Full Analysis.
7. Look at the RF table. Which metric combination gives the lowest RF distance to the WordNet reference?
8. Go to **Page 4**, select "Select from trees/ folder", pick one of the generated trees, and click Generate Tree Visualization.

---

## Important Notes and Common Issues

**Dependency versions are strictly pinned**

The `requirements.txt` pins numpy to version 1.26.4. This is intentional. NumPy 2.0 introduced breaking changes that crash `ete3` and other bioinformatics packages. Do not upgrade numpy unless you have verified compatibility.

**First model load downloads weights**

When you load a pretrained model for the first time, PyTorch downloads the weights from the internet. This can be 50-500 MB depending on the model. Make sure you have internet access. The weights are cached locally after the first download.

**ete3 installation on Windows**

`ete3` can be tricky to install on Windows. If `pip install` fails, try:
```bash
conda install -c etetoolkit ete3
```

If that also fails, you can still use Pages 1 and 2. Only Pages 3 and 4 (which use tree analysis) require `ete3`.

**t-SNE is slow for large datasets**

t-SNE does not scale well. The app caps samples at 1000. For CIFAR-100 (10000 test images), 1000 samples are randomly drawn. This is normal and expected.

**GPU acceleration**

If you have an NVIDIA GPU with CUDA, PyTorch will use it automatically for feature extraction (Page 1). Pages 2, 3, and 4 run on CPU. If you also install `cupy`, Page 2 can use the GPU for distance computations.

**The `streamlit run main.py` command must be run from the project folder**

The app reads files from relative paths (e.g., `embeddings/`, `CIFAR10_classes.json`). Always `cd` into the `Embedding-Analysis/` folder before running `streamlit run main.py`.

**Page file references**

The `main.py` file references pages as `app.py`, `app_backup.py`, `visual_app.py`, and `tree_viewer.py`. If you get a FileNotFoundError on startup, it means `main.py` needs to be updated to point to the actual file names. Update the `st.Page(...)` calls in `main.py` to match the names of the files present in the directory:

```python
extractor_page  = st.Page("feature_extraction.py",    title="1. Extract Embeddings",      default=True)
visualizer_page = st.Page("cluster_analysis.py",      title="2. Analyze & Visualize")
robinson_foulds = st.Page("hierarchical_analysis.py", title="3. Robinson Foulds Analysis")
tree_vis        = st.Page("tree_visualization.py",     title="4. Tree Visualization")
```
