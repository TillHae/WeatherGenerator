import zarr
import numpy as np
import os
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse
import pandas as pd

def downsample(in_zarr_path, out_zarr_path, res_deg, freq_hours=6):
    print(f"Opening {in_zarr_path}...")
    in_group = zarr.open(in_zarr_path, mode='r')
    
    # 1. Handle dates and frequency
    print("Extracting dates...")
    old_dates = in_group['dates'][:]
    
    # Assuming dates are stored as ISO strings or similar, convert to pandas datetime
    if isinstance(old_dates[0], (str, bytes)):
        dates_pd = pd.to_datetime(old_dates)
    else:
        dates_pd = pd.to_datetime(old_dates) # works for np.datetime64 too
        
    # Find indices where hour is a multiple of freq_hours
    valid_indices = np.where(dates_pd.hour % freq_hours == 0)[0]
    new_dates = old_dates[valid_indices]
    num_dates = len(new_dates)
    print(f"Filtered dates from {len(old_dates)} (1h) to {num_dates} ({freq_hours}h).")
    
    # 2. Create new grid
    num_lats = int(180 / res_deg) + 1
    lats = np.linspace(90, -90, num_lats)
    num_lons = int(360 / res_deg)
    lons = np.linspace(0, 360 - res_deg, num_lons)
    lon2d, lat2d = np.meshgrid(lons, lats)
    new_lats = lat2d.flatten()
    new_lons = lon2d.flatten()
    new_nodes = len(new_lats)
    print(f"New grid resolution: {res_deg} deg, {new_nodes} nodes.")
    
    # 3. Build KDTree for nearest neighbor
    print("Building KDTree for old grid...")
    old_lats = in_group['latitudes'][:]
    old_lons = in_group['longitudes'][:]
    
    def to_cartesian(lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.column_stack((x, y, z))
    
    old_coords = to_cartesian(old_lats, old_lons)
    tree = cKDTree(old_coords)
    
    new_coords = to_cartesian(new_lats, new_lons)
    print("Querying nearest neighbors...")
    _, indices = tree.query(new_coords)
    
    # 4. Create new Zarr store
    print(f"Creating new Zarr store at {out_zarr_path}...")
    out_group = zarr.open_group(out_zarr_path, mode='w')
    
    # Copy attributes
    out_group.attrs.update(in_group.attrs)
    if 'resolution' in out_group.attrs:
        out_group.attrs['resolution'] = f"{res_deg}deg"
    
    # Copy 1D arrays except specific ones
    for key in in_group.array_keys():
        if key in ['data', 'latitudes', 'longitudes', 'dates']:
            continue
        print(f"Copying array {key}...")
        arr = in_group[key]
        out_group.create_dataset(key, data=arr[:], chunks=arr.chunks, dtype=arr.dtype, compressor=arr.compressor)
        out_group[key].attrs.update(arr.attrs)
        
    # Write new grid and dates
    print("Writing new latitudes, longitudes, and dates...")
    out_group.create_dataset('latitudes', data=new_lats, chunks=(new_nodes,), dtype=np.float32)
    out_group.create_dataset('longitudes', data=new_lons, chunks=(new_nodes,), dtype=np.float32)
    out_group.create_dataset('dates', data=new_dates, chunks=(num_dates,), dtype=in_group['dates'].dtype)
    out_group['dates'].attrs.update(in_group['dates'].attrs)
    
    # 5. Copy and downsample data
    old_data = in_group['data']
    shape = list(old_data.shape)
    shape[0] = num_dates
    shape[-1] = new_nodes
    
    chunks = list(old_data.chunks)
    chunks[-1] = new_nodes
    if chunks[0] > num_dates:
        chunks[0] = num_dates
    
    print(f"Creating new data array with shape {shape} and chunks {chunks}...")
    new_data = out_group.create_dataset(
        'data', shape=shape, chunks=tuple(chunks), dtype=old_data.dtype, 
        compressor=old_data.compressor
    )
    new_data.attrs.update(old_data.attrs)
    
    # Process in chunks to avoid memory issues
    # valid_indices tells us which indices in old_data to read.
    # We will read date by date, or chunk of dates by chunk of dates.
    
    batch_size = chunks[0]
    print(f"Processing data in batches of {batch_size} dates...")
    
    for start_idx in tqdm(range(0, num_dates, batch_size)):
        end_idx = min(start_idx + batch_size, num_dates)
        
        # Get the actual indices in the original dataset for this batch
        batch_old_indices = valid_indices[start_idx:end_idx]
        
        # Read each date sequentially to avoid giant fancy indexing on Zarr
        batch_mapped = np.zeros((end_idx - start_idx, shape[1], shape[2], shape[3]), dtype=old_data.dtype)
        for i, old_idx in enumerate(batch_old_indices):
            # old_data[old_idx] is shape (ensemble, variables, old_nodes)
            date_data = old_data[old_idx, ...] 
            # Nearest neighbor mapping
            batch_mapped[i, ...] = date_data[..., indices]
            
        new_data[start_idx:end_idx, ...] = batch_mapped

    print(f"Successfully created {out_zarr_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=float, required=True, help="Resolution in degrees (e.g. 8.0)")
    parser.add_argument("--out", type=str, required=True, help="Output zarr path")
    args = parser.parse_args()
    
    in_path = '/e/data1/slmet/ml_training/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr'
    downsample(in_path, args.out, args.res)
