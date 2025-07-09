# Skyline-Based large-scale VPS

This project provides a full pipeline for extracting skylines from real-world images, matching them against DEM (Digital Elevation Model)-based 360° skylines, and estimating the optimal viewing direction (azimuth) using NCC-based matching.

---

## Modules Overview

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

## Requirements

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## File Outputs

| File                   | Description                                      |
|------------------------|--------------------------------------------------|
| `skyline.txt`          | Normalized skyline from image (1D CSV format)    |
| `skyline_360.txt`      | Normalized 360° skyline from DEM                 |
| `resampled_DEM.bin`    | Resampled DEM as point cloud (for matching)      |
| `skyline.png`          | Skyline overlay image extracted from RGB         |
| `skyline_360_plot.png` | 360° skyline elevation plot from DEM             |
| `*.ply`                | Optional Poisson mesh output (if mesh enabled)   |

All files are saved under the same directory as the input image.

---
