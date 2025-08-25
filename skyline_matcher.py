import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

class SkylineMatcher:
    """
    NCC 매칭을 통해 DEM 스카이라인과 이미지 스카이라인을 비교하는 클래스
    """
    
    def __init__(self, image_path, fov_v, fov_h, yaw, sample_step, visualization, search_radius=30):
        self.image_path       = image_path
        # # 저장된 txt 파일 사용하는 경우
        # self.image_dir        = os.path.dirname(image_path)
        # self.skyline_path     = os.path.join(self.image_dir, "skyline.txt")
        # self.skyline_360_path = os.path.join(self.image_dir, "skyline_360.txt")
        
        # 수직 화각
        self.fov_v            = fov_v
        
        # 수평 화각
        self.fov_h            = fov_h
        
        # yaw 각도
        self.yaw              = yaw
        
        # 탐색 허용 각도
        self.search_radius    = search_radius
        
        # 샘플링 간격
        self.sample_step      = sample_step
        
        # 시각화 여부
        self.visualization = visualization

    def load_skyline_txt(self, path):
        """
        skyline.txt 파일을 읽어 numpy 배열로 변환
        """
        
        # 구분자 ','로 저장된 텍스트 파일 읽기
        with open(path) as f:
            return np.array([float(v) for v in f.read().split(",")])

    def angle_to_pixel(self, elev_deg, h):
        """
        world elevation(°) -> 카메라 로컬 elevation(°) -> 픽셀 높이로 변환
        이미지 좌표계는 좌상단이 (0,0)이므로 위쪽이 0, 아래쪽이 h
        """
        
        # 카메라 로컬 좌표계에서의 높이 계산
        pixel_height = ((self.fov_v / 2.0 - elev_deg) / self.fov_v) * h
        
        return pixel_height

    def ZNCC(self, ref, target, starts):
        """
        주어진 ref와 target 사이의 NCC 매칭 수행
        두 값 범위가 달라도 상관없도록 평균 제거 후 정규화해서 비교
        """
        
        ref_m = ref - ref.mean()
        norm_ref = np.linalg.norm(ref_m)
        best_corr, best_s = -1, starts[0]
        
        for s in starts:
            if s + len(ref) <= len(target):
                tar = target[s:s+len(ref)]
            else:
                tar = np.concatenate([target[s:], target[:(s+len(ref))%len(target)]])
            tar_m = tar - tar.mean()
            norm_tar = np.linalg.norm(tar_m)

            if norm_ref * norm_tar == 0:
                continue
            corr = np.dot(ref_m, tar_m) / (norm_ref * norm_tar)
            if corr > best_corr:
                best_corr, best_s = corr, s
                
        return best_s, best_corr

    def match_skyline(self, user_skyline, dem_skyline, pitch):
        """
        이미지 스카이라인과 DEM 스카이라인을 NCC 매칭
        """

        # # 저장된 txt 파일 사용하는 경우
        # # txt 파일 로드
        # user_skyline = self.load_skyline_txt(self.skyline_path)
        # user_skyline = user_skyline[:-1]
        # dem_skyline = self.load_skyline_txt(self.skyline_360_path)

        # user_skyline 마지막 값 제거
        user_skyline = user_skyline[:-1]

        # 이미지 크기
        img = np.array(Image.open(self.image_path).convert("RGB"))
        h, w = img.shape[:2]

        # 탐색 구간 구성
        angle_step = 360.0 / len(dem_skyline)
        center_idx = int(round(self.yaw / angle_step))

        search_idx = []
        for off in range(-self.search_radius, self.search_radius + 1):
            idx = (center_idx + off) % len(dem_skyline)
            search_idx.append(idx)

        # ZNCC 매칭
        best_start, best_corr = self.ZNCC(user_skyline, dem_skyline, search_idx)
        best_angle = (best_start * angle_step) % 360

        # DEM에서 뽑은 매칭된 스카이라인
        # 매칭된 영역이 배열 끝을 넘어가지 않는 경우
        if best_start + len(user_skyline) <= len(dem_skyline):
            dem_seg_world = dem_skyline[best_start : best_start + len(user_skyline)]
        # 매칭된 영역이 배열 끝을 넘어가는 경우
        else:
            wrap = (best_start + len(user_skyline)) % len(dem_skyline)
            dem_seg_world = np.concatenate([dem_skyline[best_start:], dem_skyline[:wrap]])

        # Pitch 보정 후 픽셀로 변환
        y_user = self.angle_to_pixel(user_skyline - pitch, h)
        y_dem  = self.angle_to_pixel(dem_seg_world - pitch, h)

        # 시각화 x축 구성
        x_idxs = np.arange(0, w, self.sample_step)
        x_idxs = x_idxs[:len(user_skyline)]

        # 매칭 비교 시각화
        if (self.visualization==True):
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
            axes[0].imshow(img)
            axes[0].plot(x_idxs, y_user, color='red', linewidth=2, label="Photo Skyline")
            axes[0].set_xlim([0, w]); axes[0].set_ylim([h, 0]); axes[0].set_aspect('equal')
            axes[0].set_title("Original Image with Extracted Skyline"); axes[0].axis('off'); axes[0].legend(loc='lower right')

            axes[1].plot(x_idxs, y_user, color='red',  linewidth=2, label="Photo Skyline")
            axes[1].plot(x_idxs, y_dem,  color='blue', linewidth=2, label="DEM Skyline (Matched)")
            axes[1].set_xlim([0, w]); axes[1].set_ylim([h, 0]); axes[1].set_aspect('equal')
            axes[1].set_title(f"Matched DEM Segment (±{self.search_radius}° search)\nBest: {best_angle:.2f}° | Corr: {best_corr:.4f}")
            axes[1].set_xlabel("Pixel X (px)"); axes[1].set_ylabel("Pixel Height"); axes[1].legend(); axes[1].grid(True)
            
            # 'q' 키를 누르면 창 닫기
            def on_key(event):
                if event.key in ('q', 'ㅂ'):
                    plt.close(event.canvas.figure)
            plt.gcf().canvas.mpl_connect('key_press_event', on_key)
            
            plt.show()

        print(f"Search radius ±{self.search_radius}° — NCC result")
        print(f"Best azimuth: {best_angle:.2f}° | Corr: {best_corr:.4f}")

        return best_angle, best_corr