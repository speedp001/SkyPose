import os
import base64

from PIL import Image
from fastapi import Body
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from DEM_processor import DEMProcessor
from skyline_extractor import SkylineExtractor
from skyline_matcher import SkylineMatcher

##### 파일 경로 설정
DEM_PATH  = "./skyline_db/korea_DEM.img"
BIN_PATH  = "./skyline_db/korea_DEM.bin"
MESH_PATH = "./skyline_db/korea_DEM.obj"

##### FOV
FOV_V = 73.7
FOV_H = 57.6
SAMPLE_STEP = 10

##### Client JSON 파싱
class Client(BaseModel):
    client_latitude: float
    client_longitude: float
    client_pitch: float
    client_yaw: float
    client_roll: float
    client_image: str  # base64 인코딩된 이미지

##### FastAPI
app = FastAPI(title="Large Scale VPS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.post("/VPS")
async def VPS(payload: Client = Body(...)):
    """
    클라이언트에서 JSON으로 보내온 이미지(base64)와 센서값을 받아
    DEM/스카이라인 매칭 결과를 반환하는 엔드포인트.
    """

    # Body(...): 요청(JSON)을 함수 인자로 파싱해주는 FastAPI 기능
    # 저장 디렉토리 (파일명: lat_lon)
    save_dir = os.path.join("client_data", f"{payload.client_latitude:.5f}_{payload.client_longitude:.5f}")
    os.makedirs(save_dir, exist_ok=True)

    # base64 -> PNG 이미지 저장
    image_base64 = payload.client_image
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64)
    img_path = os.path.join(save_dir, "image.png")
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    ##### DEMProcessor
    dem = DEMProcessor(payload.client_latitude, payload.client_longitude, DEM_PATH, BIN_PATH, MESH_PATH, upsample=1)

    # BIN 파일 생성 및 시각화
    # dem.make_bin(radius=50000, visualization=True)

    # Altitude 조회
    altitude = dem.get_altitude()

    ##### SkylineExtractor
    extractor = SkylineExtractor(DEM_PATH, BIN_PATH, img_path, FOV_V, FOV_H, visualization=False, save=False)

    # 이미지 skyline 추출
    pixel_angles, user_skyline, seg_base64 = extractor.user_skyline(SAMPLE_STEP, payload.client_pitch, payload.client_roll)

    # 360도 skyline 추출
    # dem_skyline = extractor.skyline_360_DEM(payload.client_latitude, payload.client_longitude, pixel_angles)

    # 실시간 360도 skyline 추출
    dem_skyline = extractor.real_time_skyline_360_DEM(payload.client_latitude, payload.client_longitude, pixel_angles)

    ##### SkylineMatcher
    matcher = SkylineMatcher(img_path, FOV_V, FOV_H, payload.client_yaw, SAMPLE_STEP, search_radius=30, visualization=False)

    # skyline 비교 및 best match 계산
    best_angle, best_corr = matcher.match_skyline(user_skyline, dem_skyline, payload.client_pitch)

    # Client 센서값 출력
    print(f"[Client] pitch: {payload.client_pitch}, yaw: {payload.client_yaw}, roll: {payload.client_roll}")

    # Server 보정값 출력
    print(f"[Server] pitch: {payload.client_pitch}, yaw(best): {best_angle}, roll: {payload.client_roll}")

    ##### skyline.png 저장 시 사용
    # # skyline.png -> base64
    # seg = os.path.join(save_dir, "skyline.png")
    # if os.path.exists(seg):
    #     with open(seg, "rb") as f:
    #         seg_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Server 결과 반환
    return {
        "server_altitude": float(altitude),
        "server_pitch": float(payload.client_pitch),
        "server_yaw": float(best_angle),
        "server_roll": float(payload.client_roll),
        "server_image": seg_base64,
    }