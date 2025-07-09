import os
import cv2
import torch
import rasterio
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from PIL import Image
from pyproj import Transformer
from scipy.spatial import KDTree
from open3d.visualization import VisualizerWithKeyCallback
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation





##### SkylineExtractor 클래스 정의
class SkylineExtractor:

    def __init__(self, dem_path, bin_path, image_path, fov_h):
        """
        image_path: 원본 이미지 경로
        fov_h: 시야각
        """

        # GPS → UTM 좌표 변환기
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
        
        # DEM 경로
        self.dem_path = dem_path
        
        # BIN 경로
        self.bin_path = bin_path
        
        # 원본 이미지 경로
        self.image_path = image_path
        
        # 원본 이미지 디렉토리 경로
        self.image_dir = os.path.dirname(image_path)
        
        # 시야각
        self.fov_h = fov_h

    ##### segmentation: SegFormer
    def SegFormer(self):
        # segmentation
        seg_output_path = os.path.join(self.image_dir, "segmentation.png")

        # 원본 이미지 읽기
        img = Image.open(self.image_path).convert("RGB")
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        # SegFormer 모델 준비
        processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", use_fast=False)
        model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        # 입력 텐서 변환 후 추론
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(pixel_values=inputs["pixel_values"])

        # 클래스 인덱스 맵 생성
        seg = outputs.logits.argmax(dim=1)[0].cpu().numpy()
        num_classes = model.config.num_labels

        # 랜덤 색상 할당
        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        """
        colors =
        [[124  43 200]   ← Class 0
        [ 67 159 231]   ← Class 1
        ...
        [ 23  87 192]]  ← Class 149
        """
        seg_rgb = colors[seg]
        """
        seg[100, 200] = 12 → colors[12] = [10, 240, 90]
        """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

        # 원본 크기로 리사이즈
        seg_rgb_resized = cv2.resize(seg_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

        # # 결과 저장
        # seg_output_path = f"{self.image_dir}/segmentation.png"
        # Image.fromarray(seg_rgb_resized).save(seg_output_path)
        # print(f"saved {seg_output_path}")
        
        return seg_rgb_resized

    ##### user skyline 추출
    def user_skyline(self, sample_step):
        """
        사용자 이미지에서 스카이라인 추출
        sample_step: 픽셀 단위 샘플링
        """
        
        img = np.array(Image.open(self.image_path).convert("RGB"))
        h, w = img.shape[:2]
        
        # 세그멘테이션 이미지 읽기
        seg_rgb = self.SegFormer()
        
        # 가장 많이 등장하는 최상단 색 → 하늘로 간주
        top_row = seg_rgb[0:10]
        sky_colors, counts = np.unique(top_row.reshape(-1, 3), axis=0, return_counts=True)
        sky_colors = sky_colors[np.argmax(counts)]
        # print("Detected sky color:", sky_colors)

        # 하늘 지정 색상 픽셀 찾기
        sky_mask = np.all(seg_rgb == sky_colors, axis=2)

        # 하늘 지정 색상 흰색으로 변경
        output_img = seg_rgb.copy()
        output_img[sky_mask] = [255, 255, 255]

        # 고립 영역 제거를 위한 마스크 생성
        # 결과적으로 테두리와 이어진 물체만 남게 되며, 고립된 작은 물체 등은 제거
        # 흰색이 아닌(하늘이 아닌) 픽셀은 1, 흰색(하늘)은 0인 마스크
        not_white_mask = np.any(output_img != [255, 255, 255], axis=2).astype(np.uint8)
        
        # 1로 연결된 영역을 각각 다른 숫자로 라벨링
        # num_labels는 라벨의 개수, labels는 라벨(각 연결된 영역에 부여되는 고유한 번호)
        num_labels, labels = cv2.connectedComponents(not_white_mask)
        """
        connectedComponents 함수는 연결된 픽셀 영역을 찾아서 각 영역에 고유한 라벨을 부여
        [[0, 0, 1, 1],       [[0 0 1 1]
        [0, 0, 1, 1],   ->   [0 0 1 1]
        [1, 0, 0, 1]]        [2 0 0 1]]
        """

        preserve_mask = np.zeros_like(not_white_mask, dtype=bool)
        h, w = not_white_mask.shape
        
        # 라벨 영역 순회하면서 테두리와 접촉하는 영역을 보존
        # 주위에 아무 테두리와 접촉하지 않는 영역은 제거
        for label in range(1, num_labels):
        # label 0 = background
            region = (labels == label)
            touches_border = (
                # 최상단
                # np.any(region[0, :]) or
                # 최하단
                # np.any(region[-1, :]) or
                # 좌측
                np.any(region[:, 0]) or
                # 우측
                np.any(region[:, -1])
            )
            if touches_border:
                # 원소별 OR 연산자
                preserve_mask |= region
                
        # preserve_mask가 False인 곳(고립된 영역)은 흰색으로 변경
        output_img[~preserve_mask] = [255, 255, 255]

        # 스카이라인 추출
        # 흰색이 아닌 픽셀 중 최상단 y값을 스카이라인으로 간주
        # output_img로 최종 not_white_mask 할당
        not_white_mask = np.any(output_img != [255, 255, 255], axis=2)
        
        # 일정 픽셀 간격으로 샘플링
        sample_indices = np.arange(0, w, sample_step)
        
        # 마지막 픽셀 포함
        if sample_indices[-1] != w - 1:
            sample_indices = np.append(sample_indices, w - 1)
            
        skyline = np.full(len(sample_indices), np.nan, dtype=np.float32)
        for i, x in enumerate(sample_indices):
            
            # 하늘이 아닌 최상단 y좌표 (첫 번째로 등장하는 하얀색이 아닌 픽셀의 y좌표)
            y = np.where(not_white_mask[:, x])[0]
            if len(y) > 0:
                skyline[i] = y[0]
        
        # skyline y좌표 계산
        # skyline y좌표 저장할 1D 배열 생성
        skyline = np.full(w, np.nan, dtype=np.float32)
        for x in range(w):
            y = np.where(not_white_mask[:, x])[0]
            if len(y) > 0:
                skyline[x] = y[0]

        # 결측값(NaN) 보간
        valid_mask = ~np.isnan(skyline)
        if valid_mask.sum() > 0:
            skyline[~valid_mask] = np.interp(
                np.flatnonzero(~valid_mask),
                np.flatnonzero(valid_mask),
                skyline[valid_mask]
            )

        # 스무딩 (이동평균, window_size=15)
        window_size = 15
        kernel = np.ones(window_size) / window_size
        padded = np.pad(skyline, (window_size // 2, window_size // 2), mode='edge')
        skyline = np.convolve(padded, kernel, mode='valid')

        # 픽셀 -> 각도 매핑
        # angle_x = -FOV/2 + (FOV * x / (w-1))
        angle_indices = -self.fov_h / 2 + (self.fov_h * sample_indices / (w - 1))

        # 정규화
        skyline_norm = skyline / h
        
        # skyline.txt 저장
        skyline_path = f"{self.image_dir}/skyline.txt"
        sampled_skyline = ",".join([f"{val:.5f}" for val in skyline_norm[sample_indices]])

        with open(skyline_path, "w") as f:
            f.write(sampled_skyline)
        print(f"Saved {skyline_path})")
        
        # 원본 이미지 크기 투명 배경 생성
        # (h, w, 4) RGBA, 모두 투명
        skyline_img = np.zeros((h, w, 4), dtype=np.uint8)
                
        # 스카이라인 곡선 좌표
        thickness = 15
        for x in range(w):
            y = int(round(skyline[x]))
            if 0 <= y < h:
                for dy in range(-thickness // 2, thickness // 2 + 1):
                    yy = y + dy
                    if 0 <= yy < h:
                        skyline_img[yy, x] = [255, 0, 0, 255]

        # skyline.png 저장
        skyline_png_path = os.path.join(self.image_dir, "skyline.png")
        Image.fromarray(skyline_img).save(skyline_png_path)
        print(f"Saved {skyline_png_path}")

        # 시각화
        # 원본 이미지와 크기 동일시
        plt.figure(figsize=(w / 100, h / 100))
        plt.imshow(img)
        plt.plot(range(w), skyline, color='red', linewidth=1)
        plt.title("User Skyline")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        return angle_indices, sample_indices

    ##### 360도 DEM 스카이라인 추출
    def skyline_360_DEM(self, lat, lon, fov_v, angles):
        """
        관측 지점에서 360도 방향으로 DEM을 리샘플링하여 360도 스카이라인 추출
        lat: 관측 지점 위도
        lon: 관측 지점 경도
        fov_v: 수직 시야각
        angles: 각도 배열(angle_step에 따라 생성)
        """

        # 각도 간격 계산
        if len(angles) > 1:
            pixel_angle_step = abs(angles[1] - angles[0])
            print(f"Pixel angle step: {pixel_angle_step:.5f} degrees")
        else:
            raise ValueError("Not enough angles provided")

        # 각도 간격 배열 (0~360)
        n_rays = int(360 / pixel_angle_step)
        angles_360 = np.linspace(0, 360, n_rays, endpoint=False)

        # 관측지점 UTM 좌표
        observer_x, observer_y = self.transformer.transform(lon, lat)

        # 거리별 샘플링 간격 정의
        d1, step1 = 5000, 5
        d2, step2 = 10000, 50
        d3, step3 = 100000, 500
        dists1 = np.arange(step1, d1 + step1, step1)
        dists2 = np.arange(d1 + step2, d2 + step2, step2)
        dists3 = np.arange(d2 + step3, d3 + step3, step3)
        distances = np.concatenate([dists1, dists2, dists3])

        sampled_points = []

        # Ray에 따른 DEM 리샘플링
        with rasterio.open(self.dem_path) as ds:
            dem = ds.read(1)
            transform = ds.transform
            h, w = dem.shape

            # 방위각 별 DEM 리샘플링
            for az in angles_360:
                
                # 방위각 -> 라디안 변환
                az_rad = np.deg2rad(az)
                cos_a, sin_a = np.cos(az_rad), np.sin(az_rad)
                
                # Ray 방향으로 x, y 좌표 이동(거리에 따라 샘플링)
                for dist in distances:
                    x = observer_x + dist * cos_a
                    y = observer_y + dist * sin_a
                    
                    # 역변환으로 다시 픽셀 좌표계로 변환
                    col_f, row_f = ~transform * (x, y)
                    r0, c0 = int(np.floor(row_f)), int(np.floor(col_f))
                    r1, c1 = r0 + 1, c0 + 1

                    if 0 <= r0 < h-1 and 0 <= c0 < w-1:
                        # z00 좌상단, z10 좌하단, z01 우상단, z11 우하단
                        z00 = dem[r0, c0]
                        z10 = dem[r1, c0]
                        z01 = dem[r0, c1]
                        z11 = dem[r1, c1]
                        
                        # 내부 상대 위치
                        dr = row_f - r0
                        dc = col_f - c0
                        
                        # 좌우 비, 상하 비율
                        z0 = z00 * (1-dc) + z01 * dc
                        z1 = z10 * (1-dc) + z11 * dc
                        # 최종 고도 보간
                        z = z0 * (1-dr) + z1 * dr
                    else:
                        z = np.nan
                    
                    # 유효한 z값 저장
                    sampled_points.append([x, y, z])

        pts = np.array(sampled_points, dtype=np.float32)

        # resampled DEM 저장
        resampled_bin = os.path.join(self.image_dir, "resampled_DEM.bin")
        pts.tofile(resampled_bin)
        print(f"Saved {resampled_bin}")

        # 360도 스카이라인 계산
        points = np.fromfile(resampled_bin, dtype=np.float32).reshape(-1, 3)
        kdtree = KDTree(points[:, :2])
        _, idx = kdtree.query([observer_x, observer_y])
        observer_z = points[idx, 2]
        observer = np.array([observer_x, observer_y, observer_z+2])

        # 각 점에 대한 방향 계산
        vecs = points - observer
        dists_xy = np.linalg.norm(vecs[:, :2], axis=1)
        azimuths = (np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0])) + 360 - 90) % 360
        elevations = np.degrees(np.arctan2(vecs[:, 2], dists_xy))

        # fov_v 범위 안 최대 고도 계산
        skyline = np.full_like(angles_360, np.nan, dtype=np.float32)
        for i, az in enumerate(angles_360):
            mask_az = np.abs((azimuths - az + 180) % 360 - 180) < (pixel_angle_step / 2)
            elevs = elevations[mask_az]
            elevs_in_fov = elevs[(elevs >= -fov_v/2) & (elevs <= fov_v/2)]
            if elevs_in_fov.size > 0:
                skyline[i] = np.max(elevs_in_fov)
            elif elevs.size > 0:
                skyline[i] = np.max(elevs)

        valid = ~np.isnan(skyline)
        if valid.sum() > 1:
            skyline[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), skyline[valid])
            
        # resampled BIN 시각화
        tree = KDTree(pts[:, :2])
        _, idx = tree.query([observer_x, observer_y])
        center_pt = [observer_x, observer_y, float(pts[idx, 2])]

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        vis = VisualizerWithKeyCallback()
        vis.create_window(window_name="Resampled DEM Visualization")
        vis.add_geometry(pcd)
        vis.register_key_callback(ord('q'), lambda vis: vis.destroy_window())

        ctr = vis.get_view_control()
        ctr.set_lookat(center_pt)
        ctr.set_front([0, 0, 1])
        ctr.set_up([0, 1, 0])

        vis.run()
        vis.destroy_window()

        # 스무딩
        window_size = 15
        kernel = np.ones(window_size) / window_size
        padded = np.pad(skyline, (window_size//2, window_size//2), mode='edge')
        skyline = np.convolve(padded, kernel, mode='valid')
        
        # 정규화
        skyline_norm = (skyline - (-fov_v/2)) / (fov_v + 1e-8)

        # skyline_360.txt 저장
        skyline_path = os.path.join(self.image_dir, "skyline_360.txt")
        with open(skyline_path, "w") as f:
            f.write(",".join([f"{val:.5f}" for val in skyline_norm]))
        print(f"Saved {skyline_path}")

        # 360 skyline 시각화
        plt.figure(figsize=(12, 4))
        plt.plot(angles_360, skyline, label="DEM Skyline")
        plt.xlabel("Azimuth (°)")
        plt.ylabel("Elevation (°)")
        plt.title("360° Skyline")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(self.image_dir, "skyline_360_plot.png")
        plt.savefig(plot_path)
        print(f"Saved skyline plot: {plot_path}")
        plt.show()