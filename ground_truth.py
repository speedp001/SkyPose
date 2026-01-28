import math

# 측정 GPS
observer_lat = 37.5463424
observer_lon = 127.1182115

# 목표 GPS
target_lat = 37.5551549
target_lon = 127.1106859

# 도 → 라디안
lat1 = math.radians(observer_lat)
lon1 = math.radians(observer_lon)
lat2 = math.radians(target_lat)
lon2 = math.radians(target_lon)

dlon = lon2 - lon1

x = math.sin(dlon) * math.cos(lat2)
y = (math.cos(lat1) * math.sin(lat2)
     - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

theta = math.atan2(x, y)  # rad
azimuth = math.degrees(theta)
azimuth = (azimuth + 360.0) % 360.0  # 0~360 정규화

print("측정 GPS(관측자):", observer_lat, observer_lon)
print("목표 GPS(목표):", target_lat, target_lon)
print("방위각(도):",  azimuth)  # 도 단위