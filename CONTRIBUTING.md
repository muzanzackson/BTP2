# Contributing to CNN Embedding Analysis Suite

Thank you for your interest in contributing! This guide will help you get started.

---

## Getting Started

**1. Fork and clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Embedding-Analysis.git
cd Embedding-Analysis
```

**2. Create a Conda environment**
```bash
conda env create -f environment.yml
conda activate embedding-analysis
```

**3. Install PyTorch (GPU or CPU)**
```bash
# CPU only:
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU (CUDA 12.1):
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

**4. Install dev tools**
```bash
pip install -r requirements-dev.txt
```

**5. Run the app to verify your setup**
```bash
streamlit run main.py
```

---

## Submitting Changes

1. **Open an issue first** before starting large changes — discuss the approach.
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Follow PEP 8 code style.** Run linting before committing:
   ```bash
   ruff check .
   ```
4. **Add a brief entry** to `CHANGELOG.md` under `## [Unreleased]`.
5. **Submit a pull request** with a clear title and description of your change.

---

## Reporting Bugs

Use the **Bug Report** issue template. Please include:
- OS, Python version, PyTorch version
- GPU (yes/no) and CUDA version if applicable
- Full traceback from the terminal or Streamlit error box

---

## Project Structure

```
Embedding-Analysis/
├── main.py                     # Entry point (streamlit run main.py)
├── feature_extraction.py       # Page 1 — model loading & embedding extraction
├── cluster_analysis.py         # Page 2 — t-SNE, Deformity Index, cluster metrics
├── hierarchical_analysis.py    # Page 3 — UPGMA trees & Robinson-Foulds analysis
├── tree_visualization.py       # Page 4 — interactive 3D tree viewer
├── tree_rendering.py           # Helper — CLI renderer (called by Page 4)
├── data/                       # Bundled taxonomy files (CIFAR-10/100 WordNet)
├── lib/DefIn/                  # DefIn Deformity Index library
├── embeddings/                 # OUTPUT: extracted .npz files (gitignored)
├── trees/                      # OUTPUT: generated .nwk files (gitignored)
├── logs/                       # OUTPUT: runtime logs (gitignored)
└── assets/                     # Screenshots for README
```

---

## Code Style

- **Python:** PEP 8, enforced by `ruff`
- **Max line length:** 120 characters
- **Docstrings:** Google-style for all public functions
- **Type hints:** Encouraged on new functions

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
