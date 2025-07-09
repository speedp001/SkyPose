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

    def __init__(self, dem_path, bin_path, mesh_path, upsample=4):
        self.dem_path = dem_path
        self.bin_path = bin_path
        self.mesh_path = mesh_path
        self.upsample = upsample
        
        # GPS → UTM 좌표 변환기
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
        
        # # 시각화 배율 조정
        # self.scale_x = None
        # self.scale_y = None
        # self.scale_z = None

        # DEM 경계 및 중앙 UTM 좌표 계산
        with rasterio.open(self.dem_path) as ds:
            b = ds.bounds
            self.x_min, self.x_max = b.left,  b.right
            self.y_min, self.y_max = b.bottom, b.top
            self.dem_bounds = [self.x_min, self.x_max, self.y_min, self.y_max]
        
        # center UTM 좌표
        self.center_x = (self.x_min + self.x_max) / 2.0
        self.center_y = (self.y_min + self.y_max) / 2.0
        self.points = None
        
    ##### BIN 파일 로드
    def load_points(self):
        if os.path.exists(self.bin_path):
            self.points = np.fromfile(self.bin_path, dtype=np.float32).reshape(-1, 3)
            # print(f"DEM 포인트 수: {len(self.points):,}")
            # print(f"x 범위 (m): {self.points[:, 0].min()} ~ {self.points[:, 0].max()}")
            # print(f"y 범위 (m): {self.points[:, 1].min()} ~ {self.points[:, 1].max()}")
        else:
            raise FileNotFoundError(f"The BIN file does not exist {self.bin_path}.")
        
    ##### GPS → UTM 좌표 변환 및 범위 검사
    def gps_to_utm(self, lon, lat):
        """
        lon: GPS 경도
        lat: GPS 위도
        """
        
        # x_min, x_max, y_min, y_max = self.dem_bounds
        # x_utm, y_utm = self.transformer.transform(lat, lon)

        # # DEM 범위 확인
        # if not (x_min <= x_utm <= x_max and y_min <= y_utm <= y_max):
        #     raise ValueError("GPS is out of the specified DEM bounds.")
        # print(f"GPS UTM 좌표: x={x_utm}, y={y_utm} (within DEM bounds)")
    
        # UTM 변환
        x_utm, y_utm = self.transformer.transform(lat, lon)
        x_min, x_max, y_min, y_max = self.dem_bounds

        # 범위 검사
        if not (x_min <= x_utm <= x_max and y_min <= y_utm <= y_max):
            raise ValueError("GPS is out of the specified DEM bounds.")

        # BIN 파일 없으면 생성
        if not os.path.exists(self.bin_path):
            print("The BIN file does not exist, so it will be created.")
            self.make_bin()

        # BIN 파일 로드
        self.load_points()

        # 최근접 지점 탐색
        tree = KDTree(self.points[:, :2])
        dist, idx = tree.query([x_utm, y_utm])
        z_utm = self.points[idx, 2]

        # UTM 좌표 및 고도 출력
        print(f"GPS -> UTM: x={int(x_utm)}, y={int(y_utm)}, z={int(z_utm)} (m)")
        # print(f"Closest point distance {dist}m")
        
    ##### DEM 데이터를 읽고 BIN 파일로 변환
    def make_bin(self):
        """
        input_path: DEM 파일 경로 (예: namsan_DEM.img)
        output_path: 출력할 BIN 파일 경로 (예: namsan_DEM.bin)
        각 포인트는 (x(정규화 utm 좌표계), y(정규화 utm 좌표계), z(고도))
        """
        
        # DEM 읽기
        with rasterio.open(self.dem_path) as dataset:
            dem       = dataset.read(1)
            transform = dataset.transform
            height, width = dem.shape
            x = np.arange(width)
            y = np.arange(height)

            # 좌표계 생성
            x_coords, y_coords = np.meshgrid(x, y)
            x_utm = transform.c + x_coords * transform.a
            y_utm = transform.f + y_coords * transform.e

        # 업샘플링 그리드 생성
        x_coords_interp = np.linspace(x_utm.min(), x_utm.max(), width  * self.upsample)
        y_coords_interp = np.linspace(y_utm.min(), y_utm.max(), height * self.upsample)
        x_utm_interp, y_utm_interp = np.meshgrid(x_coords_interp, y_coords_interp)

        # DEM 보간
        dem_interp = griddata(
            (x_utm.flatten(), y_utm.flatten()),
            dem.flatten(),
            (x_utm_interp, y_utm_interp),
            method='cubic'
        )

        """
        ##### 정규화하지 않는 것으로 결정
        # 정규화 utm 좌표계 범위 계산 → 시각화 scale
        self.scale_x = x_utm_interp.max() - x_utm_interp.min()
        self.scale_y = y_utm_interp.max() - y_utm_interp.min()
        self.scale_z = dem.max() - dem.min()

        # 정규화
        x_utm_norm = (x_utm_interp - x_utm_interp.min()) / self.scale_x
        y_utm_norm = (y_utm_interp - y_utm_interp.min()) / self.scale_y
        dem_norm   = (dem_interp   - dem.min()) / self.scale_z

        # 3D 포인트 최종 병합 (정규화 utm x, 정규화 utm y, dem z)
        # points = np.stack((x_utm_norm, y_utm_norm, dem_norm), axis=2)
        """
        
        # 최종 3D 포인트 병합 (utm x, utm y, dem z)
        pts = np.stack((x_utm_interp, y_utm_interp, dem_interp), axis=2)
        pts = pts.reshape(-1, 3).astype(np.float32)

        # BIN 저장
        pts.tofile(self.bin_path)
        print(f"Saved {self.bin_path}")

    ##### BIN 파일 시각화
    def visualize_bin(self):
        """
        ##### 정규화 하지 않는 것으로 결정
        # 자동 스케일 설정
        if scale_x is None:
            scale_x = self.scale_x
        if scale_y is None:
            scale_y = self.scale_y
        if scale_z is None:
            scale_z = self.scale_z
        """
        
        # BIN 파일 로드
        self.load_points()
        pts = self.points

        # # 정규화 좌표 시각화 스케일 적용
        # pts[:, 0] *= scale_x
        # pts[:, 1] *= scale_y
        # pts[:, 2] *= scale_z

        # KDTree 생성 후 입력 좌표의 z값 추출
        tree = KDTree(pts[:, :2])
        # query_pt = [x_input * scale_x, y_input * scale_y]
        query_pt = [self.center_x, self.center_y]
        _, idx = tree.query(query_pt)
        center_pt = [query_pt[0], query_pt[1], pts[idx, 2]]

        # 시각화 point 생성
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        vis = VisualizerWithKeyCallback()
        vis.create_window(window_name="DEM Visualization")
        vis.add_geometry(pcd)

        # 'q' 누르면 창 종료
        vis.register_key_callback(ord("q"), lambda vis: vis.destroy_window())

        # 카메라 방향 설정
        ctr = vis.get_view_control()
        ctr.set_lookat(center_pt)     # 카메라 시선 목표점 (x,y,z)
        ctr.set_front([0, 0, 1])      # 카메라 앞 방향 벡터
        ctr.set_up([0, 1, 0])         # 카메라 위 방향 벡터

        # 중앙 좌표를 GPS/UTM으로 표기
        transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
        lon_center, lat_center = transformer.transform(self.center_x, self.center_y)
        print(f"Centered UTM: (x={self.center_x}, y={self.center_y}) → z={center_pt[2]}")
        print(f"Centered GPS: (lat={lat_center:.5f}, lon={lon_center:.5f}) → z={center_pt[2]}")
        
        vis.run()
        vis.destroy_window()
        
    ##### poisson 매쉬 생성
    def make_mesh(self):
        # BIN 로드
        self.load_points()
        pts = self.points

        # 중심점이 (0,0)에 오도록 이동
        pts[:, 0] -= self.center_x
        pts[:, 1] -= self.center_y

        # pointcloud 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # 노멀 추정
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_consistent_tangent_plane(10)

        # poisson 매쉬 생성
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

        # 저장
        o3d.io.write_triangle_mesh(self.mesh_path, mesh)
        print(f"Saved {self.mesh_path}")
        
    ##### Poisson 매쉬 시각화
    def visualize_mesh(self):
        """
        Poisson mesh를 Open3D 창에서 시각화
        """
        
        # 메쉬 파일 로드
        if not os.path.exists(self.mesh_path):
            raise FileNotFoundError(f"The Mesh file does not exist {self.mesh_path}")
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        mesh.compute_vertex_normals()

        # 높이 기반 컬러맵
        verts = np.asarray(mesh.vertices)
        z = verts[:, 2]
        zn = (z - z.min()) / (z.max() - z.min() + 1e-8)
        colors = np.zeros_like(verts)
        colors[:, 0] = zn
        colors[:, 1] = 1.0 - np.abs(zn - 0.5) * 2
        colors[:, 2] = 1.0 - zn
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        # 시각화
        vis = VisualizerWithKeyCallback()
        vis.create_window(window_name="Mesh Visualization")
        vis.add_geometry(mesh)
        
        # 'q' 누르면 창 종료
        vis.register_key_callback(ord("q"), lambda vis: vis.destroy_window())

        # 초기 카메라 뷰 설정
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, 1])      # 카메라 앞 방향 벡터
        ctr.set_up([0, -1, 0])         # 카메라 위 방향 벡터

        vis.run()
        vis.destroy_window()