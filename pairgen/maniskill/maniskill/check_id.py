import json
with open("camera_configs.json") as f:
    data = json.load(f)

for req_az, req_el, req_dist in [(0, 0, 1.5), (0, 0, 2.0), (15, 0, 1.0), (60, 0, 1.0), (75, 0, 1.0)]:
    exact = None
    for k, v in data.items():
        if v["azimuth_delta"] == req_az and v["elevation_delta"] == req_el and v["distance_scale"] == req_dist:
            exact = k
            break
            
    if exact:
        print(f"Azimuth {req_az:>3}, Elevation {req_el:>3}, Distance {req_dist:>3} -> Exact ID: {exact}")
    else:
        print(f"Azimuth {req_az:>3}, Elevation {req_el:>3}, Distance {req_dist:>3} -> No exact match!")

