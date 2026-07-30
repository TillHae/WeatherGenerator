import zarr
import xarray as xr
import numpy as np
import traceback

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
    
    attempts = [
        ({"grid": "O96"}, {"grid": [8, 8]}),
        ("O96", {"grid": [8, 8]}),
        ({"grid": "O96"}, [8, 8]),
        ("O96", [8, 8])
    ]
    
    for in_g, out_g in attempts:
        print(f"\n--- Testing in_grid={in_g}, out_grid={out_g} ---")
        try:
            out = interpolate(data, in_grid=in_g, out_grid=out_g, method="linear")
            print(f"Success! Output shape: {out.shape}")
            break
        except Exception as e:
            print(f"Failed: {e}")
            # traceback.print_exc()

if __name__ == "__main__":
    test_earthkit_regrid()
