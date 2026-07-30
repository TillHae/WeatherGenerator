import numpy as np
import os

def create_regular_grid(res_deg, filename):
    # Lats: 90 to -90 inclusive
    num_lats = int(180 / res_deg) + 1
    lats = np.linspace(90, -90, num_lats)
    
    # Lons: 0 to 360 exclusive (so 0 to 360-res)
    num_lons = int(360 / res_deg)
    lons = np.linspace(0, 360 - res_deg, num_lons)
    
    lon2d, lat2d = np.meshgrid(lons, lats)
    
    latitudes = lat2d.flatten()
    longitudes = lon2d.flatten()
    
    np.savez(filename, latitudes=latitudes, longitudes=longitudes)
    print(f"Created {filename} with {len(latitudes)} points (res: {res_deg} deg)")

if __name__ == "__main__":
    os.makedirs("datasets/grids", exist_ok=True)
    create_regular_grid(8.0, "datasets/grids/grid-8deg.npz")
    create_regular_grid(4.0, "datasets/grids/grid-4deg.npz")
    create_regular_grid(2.0, "datasets/grids/grid-2deg.npz")
