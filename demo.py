from DEM_processor import DEMProcessor
from skyline_extractor import SkylineExtractor
from skyline_matcher import SkylineMatcher





# XReal FOV
# FOV_V = 22.0
# FOV_H = 36.5

# Galaxy S23 FOV
FOV_V = 48
FOV_H = 74

LAT = 37.55453
LON = 127.04611
SAMPLE_STEP = 10

DEM_PATH  = "./skyline_db/korea_DEM.img"
BIN_PATH = "./skyline_db/korea_DEM.bin"
MESH_PATH = "./skyline_db/korea_DEM.obj"
IMAGE_PATH = "./client_data/37.55453_127.04611/image.png"





##### 카메라 기울기 및 고도 각도 -> IMU센서 이용해서 추정 예정
YAW = 0
PITCH = 0
ROLL = 0





##### DEMProcessor
dem = DEMProcessor(LAT, LON, DEM_PATH, BIN_PATH, MESH_PATH, upsample=1, visualization=True)

# BIN 파일 생성 및 시각화
# dem.make_bin(radius=50000)

# Altitude 조회
# altitude = dem.get_altitude()





##### SkylineExtractor
extractor = SkylineExtractor(DEM_PATH, BIN_PATH, IMAGE_PATH, FOV_V, FOV_H, visualization=False, save=False)

# 이미지 skyline 추출
pixel_angles, user_skyline, seg_base64 = extractor.user_skyline(SAMPLE_STEP, PITCH, ROLL)

# DB 생성
# extractor.db_maker(pixel_angles)

# 360도 skyline 추출
dem_skyline = extractor.skyline_360_DEM(LAT, LON, pixel_angles)

# Real-time 360도 skyline 추출
dem_skyline = extractor.real_time_skyline_360_DEM(LAT, LON)

##### SkylineMatcher
matcher = SkylineMatcher(IMAGE_PATH, FOV_V, FOV_H, YAW, SAMPLE_STEP, search_radius=30, visualization=True)

# skyline 비교 및 best match 계산
matcher.match_skyline(user_skyline, dem_skyline, PITCH)