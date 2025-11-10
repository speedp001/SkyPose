### ![IWAIT2026 Logo](https://i.imgur.com/UInLqku.png)

#  IWAIT 2026 Conference Paper

> 본 프로젝트는 2026년 **The International Workshop on Advanced Image Technology (IWAIT 2026)** 컨퍼런스에 채택되어,  
> SPIE, the international society for optics and photonics에 정식 출간되었습니다.  
> 컨퍼런스는 2026년 1월 12–14일, TAIWAN, KAOHSIUNG에서 개최되었습니다.

- **IWAIT 2026 공식 웹사이트**: [https://2025.hci.international]([https://2025.hci.international](https://iwait.online/paper-submission-for-iwait-2026/)
- **논문 시리즈 (SPIE)**: [SPIE Proceedings](https://link.springer.com/series/16277)
- **논문 링크**: [https://link.springer.com/chapter/10.1007/978-3-031-93163-5_9](https://link.springer.com/chapter/10.1007/978-3-031-93163-5_9)

> 논문 제목: *SkyPose: Real-Time Camera Pose Estimation via Skyline Matching in Mountainous Terrain*

# Large Scale VPS

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
| `*.obj`                | Optional Poisson mesh output (if mesh enabled)   |

All files are saved under the same directory as the input image.

---
