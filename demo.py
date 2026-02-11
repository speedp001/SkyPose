import time

from DEM_processor import DEMProcessor
from skyline_extractor import SkylineExtractor
from skyline_matcher import SkylineMatcher





# XReal FOV
FOV_V = 22.0
FOV_H = 36.5

# Galaxy S23 FOV
# FOV_V = 48
# FOV_H = 74

LAT = 37.33435
LON = 128.00328
# LAT = 37.54536
# LON = 126.98720
# LAT = 37.33435
# LON = 128.00328

SAMPLE_STEP = 10

DEM_PATH  = "./skyline_db/korea_DEM.img"
BIN_PATH = "./skyline_db/korea_DEM.bin"
MESH_PATH = "./skyline_db/korea_DEM.obj"
IMAGE_PATH = "./client_data/180_37.33435_128.00328_5/image.png"
# IMAGE_PATH = "./client_data/104_37.54536_126.98720_10/image.png"
# IMAGE_PATH = "./client_data/66_37.33435_128.00328_15/image.png"





##### Camera Orientation
YAW = 180
# YAW = 104
# YAW = 65
PITCH = 5
ROLL = 0



# 시간 기록
start_time = time.time()

##### DEMProcessor
# dem = DEMProcessor(LAT, LON, DEM_PATH, BIN_PATH, MESH_PATH, upsample=1, visualization=True)

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
# dem_skyline = extractor.skyline_360_DEM(LAT, LON, pixel_angles)

# 실시간 360도 skyline 추출
dem_skyline = extractor.real_time_skyline_360_DEM(LAT, LON)

##### SkylineMatcher
matcher = SkylineMatcher(IMAGE_PATH, FOV_V, FOV_H, YAW, SAMPLE_STEP, search_radius=30, visualization=False)

# skyline 비교 및 best match 계산
matcher.match_skyline(user_skyline, dem_skyline, PITCH)

# 기록 종료
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")