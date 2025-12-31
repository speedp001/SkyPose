### ![IWAIT2026 Logo](https://i.imgur.com/UInLqku.png)

#  IWAIT 2026 Conference Paper

> 본 프로젝트는 2026년 **The International Workshop on Advanced Image Technology (IWAIT 2026)** 컨퍼런스에 채택되어,  
> SPIE, the international society for optics and photonics에 정식 출간되었습니다.  
> 컨퍼런스는 2026년 1월 12–14일, TAIWAN, KAOHSIUNG에서 개최되었습니다.

- **IWAIT 2026 공식 웹사이트**: https://iwait.online/paper-submission-for-iwait-2026/
- **논문 시리즈 (SPIE)**: 
- **논문 링크**: 

> 논문 제목: *SkyPose: Real-Time Camera Pose Estimation via Skyline Matching in Mountainous Terrain*

## Index

- [Project Introduction](#project-introduction)  
- [Project Structure](#project-structure)
- [Modules Overview](#modules-overview)
- [Experiments](#experiments)
- [Requirements](#requirements)  
- [Demo Video](#demo-video)
<br></br>

## Project Introduction

 Visual Positioning System (VPS) estimates a camera’s pose from visual information of the surrounding environment and is widely used in autonomous driving, robotic navigation, and augmented reality (AR). Outdoor VPS commonly uses Global Navigation Satellite Systems (GNSS) and Inertial Measurement Units (IMU) as auxiliary cues. However, IMU accuracy can degrade due to magnetic disturbances and sensor noise. This paper presents a new outdoor VPS that precisely refines the IMU measurements by matching skylines extracted from input images with those generated from a Digital Elevation Model (DEM). The proposed system operates in real time and delivers high-accuracy pose estimates even under unreliable GNSS coverage and noisy IMU readings. The framework is robust and well-suited to mountainous terrain, enabling effective deployment in military, exploration, and AR-based applications.

---

## Project Structure

<img width="5324" height="1884" alt="Fig 1" src="https://i.imgur.com/Xby9gek.jpeg" />
The framework consists of three sequential stages:
1. **Skyline extraction** from the input RGB image  
2. **360° skyline generation** from DEM at the observation location  
3. **Camera pose estimation** via skyline matching  
An initial orientation provided by GNSS/IMU sensors is used to constrain the search space, and the final output is a refined camera azimuth corrected by skyline alignment.

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
<img width="5324" height="1884" alt="Fig 1" src="[https://i.imgur.com/Xby9gek.jpeg"](https://i.imgur.com/H8NnK4h.png) />

### `Outputs`
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

## Requirements

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

## Demo Video
> Supplemental Video
> https://youtu.be/qXvuMw8qoJo
<br></br>
> Presentation Video
> 
