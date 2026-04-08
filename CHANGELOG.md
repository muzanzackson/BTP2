# Changelog

All notable changes to this project follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

## [1.0.0] – 2025-XX-XX

### Added
- Feature extraction from 70+ pretrained torchvision models (ResNet, VGG, EfficientNet, ViT, ConvNeXt, Swin, MaxViT, and more)
- Support for custom model architectures via `.py` + `.pth` file upload
- t-SNE cluster visualization (2D/3D) with per-class boundary detection and Deformity Index scoring
- Inter-centroid distance heatmap across all class pairs
- Clustering quality metrics: NMI, Purity Index, Adjusted Rand Index
- Robinson–Foulds distance analysis against WordNet phylogenetic hierarchy (CIFAR-10 & CIFAR-100)
- UPGMA tree construction under 3 distance metrics (Euclidean, Manhattan, Canberra) × 2 centroid types (G, MPP)
- Interactive 3D phylogenetic tree visualization (via Plotly / custom HTML)
- Support for custom datasets, taxonomies, and label mappings
- Bundled CIFAR-10 and CIFAR-100 WordNet reference trees (`data/`)
- DefIn (Deformity Index) library integration for tree-level shape quality scoring
- Multi-page Streamlit navigation (`main.py`)
- Conda environment spec (`environment.yml`)
- MIT License
- GitHub Actions lint CI

### Structure
- Taxonomy reference files moved to `data/`
- DefIn library moved to `lib/DefIn/`
- Output directories (`embeddings/`, `trees/`, `logs/`) properly gitignored with `.gitkeep` stubs
- `assets/` directory for README screenshots
