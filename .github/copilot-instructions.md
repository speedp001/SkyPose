# SkyPose AI Agent Instructions

## Project Overview
SkyPose is a Large Scale Visual Positioning System (VPS) that matches real-world skylines against DEM (Digital Elevation Model) data for GPS-independent localization. The system uses semantic segmentation, 360° skyline extraction, and NCC correlation matching to estimate optimal viewing azimuth.

## Core Architecture

### Three-Component Pipeline
1. **`DEMProcessor`** - Converts DEM data (.img) → point clouds (.bin), handles GPS↔UTM coordinate transforms using EPSG:5179 (Korean coordinate system)
2. **`SkylineExtractor`** - Extracts skylines from images (SegFormer) and DEM (360° ray sampling), outputs normalized elevation profiles
3. **`SkylineMatcher`** - Performs sliding-window NCC correlation between user skyline and DEM skyline to find best azimuth match

### Key Data Flow
```
Image + GPS → SkylineExtractor.user_skyline() → user_skyline array
DEM + GPS → SkylineExtractor.skyline_360_DEM() → dem_skyline array
Both arrays → SkylineMatcher.match_skyline() → best azimuth + correlation
```

## Critical Implementation Details

### Coordinate System Handling
- **GPS (EPSG:4326) ↔ UTM (EPSG:5179)** transforms using `pyproj.Transformer`
- **UTM bounds checking** in `DEMProcessor.gps_to_utm()` prevents out-of-range errors
- **Pixel↔World coordinate mapping** via `rasterio.transform` for DEM data

### FOV and Camera Parameters
```python
# Device-specific FOV constants (critical for accurate skyline extraction)
FOV_V = 73.7  # Galaxy S23 vertical FOV
FOV_H = 57.6  # Galaxy S23 horizontal FOV
# XReal AR: FOV_V = 22.0, FOV_H = 36.5
```

### Skyline Processing Convention
- **Normalization**: All skylines stored as 0-1 normalized elevation values
- **Sampling**: `SAMPLE_STEP=20` determines angular resolution for skyline extraction
- **Output format**: CSV format saved to `skyline.txt` and `skyline_360.txt`

## Development Workflows

### Running the Pipeline
```bash
# Demo mode (local processing)
python demo.py

# Server mode (FastAPI endpoint)
python server.py  # Runs on /VPS endpoint
```

### Key Configuration Points
- **DEM files**: `./skyline_db/korea_DEM.img` (raster) and `korea_DEM.bin` (processed point cloud)
- **Visualization flags**: Set `visualization=True` for Open3D/matplotlib debugging
- **Search radius**: `search_radius=30` in SkylineMatcher controls azimuth search range

### File Output Patterns
All intermediate files saved to `./client_data/{lat:.5f}_{lon:.5f}/`:
- `image.png` - original input image
- `skyline.txt` - normalized user skyline (CSV)
- `skyline_360.txt` - normalized 360° DEM skyline (CSV)
- `skyline.png` - segmentation overlay visualization

## AI Agent Guidance

### When Modifying Skyline Extraction
- **SegFormer model**: `nvidia/segformer-b0-finetuned-ade-512-512` handles semantic segmentation
- **Sky detection**: Look for class indices representing sky in segmentation output
- **Elevation calculation**: Uses `angle_to_pixel()` conversion with camera pitch/roll compensation

### When Working with DEM Data
- **Interpolation**: Always use `method='cubic'` in `scipy.interpolate.griddata`
- **Radius filtering**: `make_bin(radius=50000)` crops DEM to 50km around GPS point
- **Upsampling**: `upsample` parameter controls DEM resolution (1=original, 2=2x, etc.)

### NCC Matching Logic
- **ZNCC implementation**: Zero-mean normalization handles different value ranges
- **Circular matching**: Handle wraparound at 360° boundary in `SkylineMatcher.ZNCC()`
- **Best correlation**: Higher values (closer to 1.0) indicate better matches

### FastAPI Server Integration
- **JSON payload**: `Client` model expects base64 image + sensor data (lat, lon, pitch, yaw, roll)
- **Response format**: Returns corrected yaw angle + altitude + segmentation overlay
- **CORS enabled**: Supports cross-origin requests from web clients

### Common Debugging Patterns
1. **Enable visualization**: `visualization=True` shows Open3D point clouds and matplotlib plots
2. **Check coordinate bounds**: GPS must be within DEM coverage area (Korean peninsula)
3. **Verify skyline arrays**: Both user and DEM skylines should be 0-1 normalized
4. **Correlation threshold**: NCC values below 0.3 typically indicate poor matches

Always test with known GPS coordinates within South Korea when developing new features.