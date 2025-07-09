import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image





##### SkylineMatcher 클래스 정의
class SkylineMatcher:
    """
    NCC 매칭을 통해 DEM 스카이라인과 이미지 스카이라인을 비교하는 클래스
    """
    
    def __init__(self, image_path, fov_v, fov_h, center_angle, sample_step, search_radius=20):
        """
        image_path: 이미지 파일 경로
        fov_v: 수직 화각
        fov_h: 수평 화각
        center_angle: 중심 방위각
        sample_step: 이미지에서 샘플링할 픽셀 간격
        search_radius: NCC 매칭 시 검색 반경
        """
        self.image_path       = image_path
        self.image_dir        = os.path.dirname(image_path)
        self.skyline_path     = os.path.join(self.image_dir, "skyline.txt")
        self.skyline_360_path = os.path.join(self.image_dir, "skyline_360.txt")
        self.fov_v            = fov_v
        self.fov_h            = fov_h
        self.center_angle     = center_angle % 360
        self.search_radius    = search_radius
        self.sample_step      = sample_step

    def load_skyline_txt(self, path):
        with open(path) as f:
            return np.array([float(v) for v in f.read().split(",")])

    def to_pixel_heights(self, norm_vals, h, norm_type):
        # user skyline은 이미지 높이로 복원
        if norm_type == "user":
            return norm_vals * h
        # 360도 skyline은 화각에 따라 픽셀 높이로 복원
        elif norm_type == "360":
            elev_deg = norm_vals * self.fov_v - (self.fov_v / 2)
            return ((self.fov_v / 2 - elev_deg) / self.fov_v) * h

    def match_ncc(self, ref, target, allowed_starts):
        L = len(ref)
        best_corr, best_s = -1, allowed_starts[0]
        ref_m = ref - ref.mean()
        norm_ref = np.linalg.norm(ref_m)

        for s in allowed_starts:
            if s + L <= len(target):
                seg = target[s:s+L]
            else:
                seg = np.concatenate([target[s:], target[:(s+L)%len(target)]])
            seg_m = seg - seg.mean()
            norm_seg = np.linalg.norm(seg_m)
            if norm_ref * norm_seg == 0:
                continue
            corr = np.dot(ref_m, seg_m) / (norm_ref * norm_seg)
            if corr > best_corr:
                best_corr, best_s = corr, s
        return best_s, best_corr

    def match_skyline_NCC_angle(self):
        # 입력 스카이라인 및 이미지 로드
        sky_photo_n = self.load_skyline_txt(self.skyline_path)      # 이미지에서 추출된 스카이라인 (정규화)
        sky360_n    = self.load_skyline_txt(self.skyline_360_path)  # DEM에서 추출된 360도 스카이라인
        img = np.array(Image.open(self.image_path).convert("RGB"))
        h, w = img.shape[:2]

        # 이미지 픽셀 샘플 인덱스 계산
        x_idxs = np.arange(0, w, self.sample_step)
        if x_idxs[-1] != w-1:
            x_idxs = np.append(x_idxs, w-1)

        # DEM 스카이라인 길이 및 각도 스텝 계산
        n = len(sky360_n)
        angle_step = 360.0 / n
        L = len(sky_photo_n)  # 이미지 기반 스카이라인 길이

        # center_angle 기준 ± search_radius 범위 내에서 비교할 시작 인덱스들 계산
        center_idx = int(self.center_angle / angle_step)
        radius_idx = int(self.search_radius / angle_step)
        allowed_starts = [(center_idx + i) % n for i in range(-radius_idx, radius_idx + 1)]

        # 이미지 스카이라인 pixel height로 변환
        y_photo = self.to_pixel_heights(sky_photo_n, h, norm_type="user")

        # DEM 스카이라인도 정규화값을 픽셀 높이로 변환 (일단 전체를 변환)
        y_360 = self.to_pixel_heights(sky360_n, h, norm_type="360")

        # NCC 매칭 수행 (최적 시작 인덱스 및 상관계수 계산)
        best_start, best_corr = self.match_ncc(y_photo, y_360, allowed_starts)
        best_angle = (best_start * angle_step) % 360

        # 최적 세그먼트 추출 (롤링)
        if best_start + L <= n:
            seg = y_360[best_start:best_start+L]
        else:
            seg = np.concatenate([y_360[best_start:], y_360[:(best_start+L)%n]])

        # 시각화
        plt.figure(figsize=(w/100, h/100))
        plt.imshow(img, origin='upper')
        plt.plot(x_idxs, y_photo, color='red', linewidth=2, label="Photo Skyline")
        plt.plot(x_idxs, seg,     color='blue', linewidth=2, label="DEM Skyline (Best Match)")
        plt.title(
            f"Range NCC\nSearch: {self.center_angle:.1f}° ± {self.search_radius}°\n"
            f"Best: {best_angle:.2f}°, Corr: {best_corr:.4f}"
        )
        plt.axis('off')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

        print("Range NCC Matching 결과")
        print(f"Center Angle: {self.center_angle}° ± {self.search_radius}°")
        print(f"Best Match Angle: {best_angle:.2f}°, Corr: {best_corr:.4f}")
        return best_angle, best_corr