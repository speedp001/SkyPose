import os
import rasterio
import numpy as np
import open3d as o3d

from pyproj import Transformer
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from open3d.visualization import VisualizerWithKeyCallback





##### DEMProcessor 클래스 정의
class DEMProcessor:

    def __init__(self, lat, lon, dem_path, bin_path, mesh_path, upsample):
        self.lat = lat
        self.lon = lon
        self.dem_path = dem_path
        self.bin_path = bin_path
        self.mesh_path = mesh_path
        self.upsample = upsample
        
    ##### GPS → UTM 좌표 변환 및 범위 검사
    def gps_to_utm(self, lat, lon):
        """
        lat: GPS 위도
        lon: GPS 경도
        """
        
        # GPS → UTM 좌표 변환기
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
        
        # GPS 좌표 → UTM 좌표
        x_utm, y_utm = transformer.transform(lon, lat)

        # DEM 파일에서 bounds 계산
        with rasterio.open(self.dem_path) as dataset:
            bounds = dataset.bounds
            x_min, x_max = bounds.left, bounds.right
            y_min, y_max = bounds.bottom, bounds.top

        # DEM 범위 확인
        if not (x_min <= x_utm <= x_max and y_min <= y_utm <= y_max):
            raise ValueError("GPS is out of the specified DEM bounds.")
        # print(f"GPS -> UTM: x={x_utm}, y={y_utm}")

        return x_utm, y_utm
    
    def get_altitude(self):
        """
        DEM에서 주어진 GPS 좌표의 고도 값을 반환
        """

        x_utm, y_utm = self.gps_to_utm(self.lat, self.lon)

        # DEM 데이터 읽기
        # DEM 파일은 고도(m)값이 2차원 배열로 저장되어 있음
        # UTM 좌표계가 따로 메타데이터에 저장되어 있어 변환 후에 이용
        with rasterio.open(self.dem_path) as dataset:
            dem = dataset.read(1)
            transform = dataset.transform

            # (x_utm, y_utm) → (col,row)
            # UTM -> Pixel 역행렬(~transform)을 이용하여 좌표 변환
            col_f, row_f = (~transform) * (x_utm, y_utm)
            h, w = dem.shape
            
            # 주변 4개 점 인덱스
            c0, r0 = int(np.floor(col_f)), int(np.floor(row_f))
            c1, r1 = c0 + 1, r0 + 1

            # 보간 가중치
            dc = col_f - c0
            dr = row_f - r0

            z00 = dem[r0, c0]
            z10 = dem[r0, c1]
            z01 = dem[r1, c0]
            z11 = dem[r1, c1]

            z0 = z00 * (1 - dc) + z10 * dc
            z1 = z01 * (1 - dc) + z11 * dc
            z_utm = z0 * (1 - dr) + z1 * dr
            print(f"Altitude at ({self.lat}, {self.lon}): {z_utm:.2f}m")

            return float(z_utm)

    ##### DEM 데이터를 읽고 BIN 파일로 변환
    def make_bin(self, radius, visualization):
        """
        DEM 데이터를 읽고 BIN 파일로 변환
        GPS 기준 반경 radius(m) DEM만 BIN 파일로 변환
        radius: DEM을 읽을 범위 (m)
        """
        
        # GPS 좌표 -> UTM 좌표
        x_utm, y_utm = self.gps_to_utm(self.lat, self.lon)

        # DEM 데이터 읽기
        with rasterio.open(self.dem_path) as dataset:
            dem = dataset.read(1)
            transform = dataset.transform
            height, width = dem.shape
            x = np.arange(width)
            y = np.arange(height)

            # Pixel 좌표 -> UTM 좌표
            x_grid, y_grid = np.meshgrid(x, y)
            x_utm_grid = transform.c + x_grid * transform.a
            y_utm_grid = transform.f + y_grid * transform.e

        # radius 범위 내의 DEM 데이터 필터링
        # 중심 UTM 좌표에서 전체 좌표의 거리 계산
        dist = np.sqrt((x_utm - x_utm_grid)**2 + (y_utm - y_utm_grid)**2)
        mask = dist <= radius
        x_utm_crop = x_utm_grid[mask]
        y_utm_crop = y_utm_grid[mask]
        z_utm_crop = dem[mask]

        # 업샘플링 그리드 생성
        crop_width = len(np.unique(x_utm_crop))
        crop_height = len(np.unique(y_utm_crop))
        x_coords_interp = np.linspace(x_utm_crop.min(), x_utm_crop.max(), crop_width * self.upsample)
        y_coords_interp = np.linspace(y_utm_crop.min(), y_utm_crop.max(), crop_height * self.upsample)
        x_utm_interp, y_utm_interp = np.meshgrid(x_coords_interp, y_coords_interp)

        # DEM 보간
        z_utm_interp = griddata(
            (x_utm_crop, y_utm_crop),
            z_utm_crop,
            (x_utm_interp, y_utm_interp),
            method='cubic'
        )
        
        # 최종 3D 포인트 병합 (utm x, utm y, dem z)
        pts = np.stack((x_utm_interp, y_utm_interp, z_utm_interp), axis=2)
        pts = pts.reshape(-1, 3).astype(np.float32)
        
        # NaN 값 제거
        # 거리 기준으로 뽑으면 원형 형태로 나오므로, grid 크기로 업샘플링하면 사격형 형태이므로 NaN 값이 생길 수 있음
        pts = pts[~np.isnan(pts[:, 2])]
        
        # BIN 저장
        pts.tofile(self.bin_path)
        print(f"Saved {self.bin_path}")

        # DEM 시각화
        if (visualization==True):
            # 포인트 클라우드 생성
            tree = KDTree(pts[:, :2])
            query_pt = self.gps_to_utm(self.lat, self.lon)
            _, idx = tree.query(query_pt)
            center_pt = [pts[idx, 0], pts[idx, 1], pts[idx, 2]]

            # 시각화 point 생성
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            vis = VisualizerWithKeyCallback()
            vis.create_window(window_name="DEM Visualization")
            vis.add_geometry(pcd)

            # 카메라 방향 설정
            ctr = vis.get_view_control()
            ctr.set_lookat(center_pt)     # 카메라 시선 목표점 (x,y,z)
            ctr.set_front([0, 0, 1])      # 카메라 앞 방향 벡터
            ctr.set_up([0, 1, 0])         # 카메라 위 방향 벡터
            
            vis.run()
            vis.destroy_window()