import json

azimuths = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]
elevations = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]
distances = [1.0, 1.5, 2.0]

configs = {}
config_id = 0
for az in azimuths:
    for el in elevations:
        for dist in distances:
            # 원본과 완전히 똑같은 경우의 수(방위각 0, 고도 0, 거리 1.0)는 제외
            if az == 0 and el == 0 and dist == 1.0:
                continue
                
            configs[str(config_id)] = {
                "azimuth_delta": az,
                "elevation_delta": el,
                "distance_scale": dist
            }
            config_id += 1

with open("camera_configs.json", "w") as f:
    json.dump(configs, f, indent=2)

print(f"Generated {config_id} camera configurations in camera_configs.json")
