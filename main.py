from DEM_processor import DEMProcessor
from skyline_extractor import SkylineExtractor
from skyline_matcher import SkylineMatcher

"""
고쳐야 할 것들:
1. AR Glass의 대각 화각(52도 -> AR 카메라 화각 재확인 필요)를 고려하여 데이터를 취득
AR 글래스의 카메라를 사용하는 시나리오로 설계를 하여야 한다.
2. 사용자 이미지에서(화각을 잘 적용) 산 능선을 추출하여 DEM에서 해당하는 방위각을 구한다.
여기서 방위각에 해당하는 DEM 능 선 뷰를 넘겨줄 때, AR 글래스의 수평 수직 화각에 해당하는 부분을 크롭하여 비율에 맞게 넘겨준다.
3. 카메라 이미지는 얻을 수 있지만, Nreal light에서 GPS정보와 방향 정보를 취득할 수 없기에 안드로이드 디바이스를 연결하여 사용해야한다.
4. AR 글래스를 활용한 전방 방향 이미지의 캡쳐
5. 나침반 초기값이 주어질 경우, 해당 방향에 오차 범위 내 조회만 하여 계산(연산 시간 감축)
6. AR 글래스의 캡쳐 이미지로 적용 / 해당 카메라의 화각을 적용
7. 하나의 파이프라인으로 일체형으로 구현
8. DEM데이터의 Accuracy의 한계점 및 분포를 설명
9. DEM 기준으로 각도를 구할 때, 내 주변 특정 임계 거리 기반으로 세밀하게 한다.
"""

FOV_V = 50.0
FOV_H = 36.5
COMPASS = 68
LAT = 37.54536
LON = 126.98720
SAMPLE_STEP = 10

DEM_PATH  = "./data/namsan_DEM.img"
BIN_PATH = "./data/namsan_DEM.bin"
MESH_PATH = "./data/namsan_DEM.obj"
IMAGE_PATH = "./data/68_37.54536_126.98720_10/image.png"

##### DEMProcessor
dem = DEMProcessor(DEM_PATH, BIN_PATH, MESH_PATH, upsample=1)
# BIN 파일 생성 및 시각화
dem.make_bin()
# dem.visualize_bin()

# Mesh 파일 생성 및 시각화
# dem.make_mesh()
# dem.visualize_mesh()

# GPS -> UTM 변환 및 고도 계산
dem.gps_to_utm(LAT, LON)

##### SkylineExtractor
extractor = SkylineExtractor(DEM_PATH, BIN_PATH, IMAGE_PATH, FOV_H)
# 이미지 skyline 추출
pixel_angles, pixel_samples = extractor.user_skyline(SAMPLE_STEP)
# 360도 skyline 추출
extractor.skyline_360_DEM(LAT, LON, FOV_V, pixel_angles)

##### SkylineMatcher
matcher = SkylineMatcher(IMAGE_PATH, FOV_V, FOV_H, COMPASS, SAMPLE_STEP)
# skyline 비교 및 best match 계산
matcher.match_skyline_NCC_angle()

