from DEM_processor import DEMProcessor
from skyline_extractor import SkylineExtractor
from skyline_matcher import SkylineMatcher




# XReal FOV
FOV_V = 22.0
FOV_H = 36.5

# Galaxy S23 FOV
# FOV_V = 73.7
# FOV_H = 57.6

YAW = 180
LAT = 37.33435
LON = 128.00328
SAMPLE_STEP = 10

DEM_PATH  = "./data/korea_DEM.img"
BIN_PATH = "./data/korea_DEM.bin"
MESH_PATH = "./data/korea_DEM.obj"
IMAGE_PATH = "./data/180_37.33435_128.00328_5/image.png"





##### 카메라 기울기 및 고도 각도 -> IMU센서 이용해서 추정 예정
PITCH = 5
ROLL = 0





##### DEMProcessor
dem = DEMProcessor(LAT, LON, DEM_PATH, BIN_PATH, MESH_PATH, upsample=1)

# BIN 파일 생성 및 시각화
# dem.make_bin(radius=50000, visualization=True)

# Altitude 조회
altitude = dem.get_altitude()





##### SkylineExtractor
extractor = SkylineExtractor(DEM_PATH, BIN_PATH, IMAGE_PATH, FOV_V, FOV_H, visualization=False)

# 이미지 skyline 추출
pixel_angles, pixel_samples, user_skyline = extractor.user_skyline(SAMPLE_STEP, PITCH, ROLL)

# 360도 skyline 추출
dem_skyline = extractor.skyline_360_DEM(LAT, LON, pixel_angles)





##### SkylineMatcher
matcher = SkylineMatcher(IMAGE_PATH, FOV_V, FOV_H, YAW, SAMPLE_STEP, search_radius=30, visualization=False)

# skyline 비교 및 best match 계산
matcher.match_skyline(user_skyline, dem_skyline, PITCH)