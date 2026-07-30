import zarr
import xarray as xr
import numpy as np

def test_earthkit_regrid():
    try:
        from earthkit.regrid import interpolate
    except ImportError:
        print("earthkit-regrid not found!")
        return

    # Load the O96 dataset
    z_path = '/e/data1/slmet/ml_training/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr'
    print(f"Opening {z_path}...")
    ds = zarr.open(z_path, mode='r')
    
    # Check data shape
    data = ds['data'][:1, :1, :]
    print(f"Data shape: {data.shape}")
    
    # Try regridding
    print("Testing earthkit regrid from O96 to [8, 8]...")
    try:
        out = interpolate(data, in_grid="O96", out_grid=[8, 8], method="linear")
        print(f"Success! Output shape: {out.shape}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_earthkit_regrid()
