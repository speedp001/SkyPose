# Skyline-Based Camera Orientation Estimation

This project provides a full pipeline for extracting skylines from real-world images, matching them against DEM (Digital Elevation Model)-based 360° skylines, and estimating the optimal viewing direction (azimuth) using NCC-based matching.

---

## 📂 Modules Overview

### `DEMProcessor`
Handles preprocessing of DEM data including:
- DEM (.img) → point cloud (.bin)
- Cubic interpolation & upsampling
- Poisson mesh generation
- 3D visualization using Open3D

### `SkylineExtractor`
Extracts skylines from:
- **RGB images** using semantic segmentation (SegFormer)
- **DEM** using 360° ray sampling and elevation interpolation

Outputs:
- `skyline.txt`: normalized skyline from image
- `skyline_360.txt`: normalized 360° skyline from DEM
- Visualization images: `skyline.png`, `skyline_360_plot.png`

### `SkylineMatcher`
Estimates best matching viewing direction (azimuth) by:
- Converting skyline vectors to pixel/elevation angles
- Performing sliding-window **NCC (Normalized Cross-Correlation)**
- Matching `skyline.txt` to best segment of `skyline_360.txt`
- Visualizing the match

---

## 🛠️ Requirements

Install the required Python libraries:

```bash
pip install -r requirements.txt
