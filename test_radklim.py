import sys
import time
import logging
from pathlib import Path
import numpy as np

# Set up logging so we can see the _logger outputs
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Add the src directory to the Python path
sys.path.insert(0, str(Path("src").resolve()))

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.data_reader_radklim import DataReaderRadklim

def test_radklim():
    # Setup dummy time window for testing spanning 5 years
    t_start = np.datetime64("2001-01-01T00:00:00")
    t_end = np.datetime64("2021-12-31T23:00:00")
    t_window_len = np.timedelta64(24, "h")
    t_window_step = np.timedelta64(24, "h")

    tw_handler = TimeWindowHandler(t_start, t_end, t_window_len, t_window_step)

    # Mock stream_info based on the yaml config
    stream_info = {
        "name": "RADKLIM_TEST",
        "source": [],
        "target": ["RR"],
        "spatial_stride": 3,  # Subsample to keep things fast during testing
        "calc_stats": False,  # Keep False so we only measure index/fetch performance
    }

    dataset_path = Path("/p/data1/slmet/met_data/dwd/radklim-rw/netcdf/orig_grid")

    print(f"=== Initializing Radklim Reader ===")
    print(f"Path: {dataset_path}")
    print(f"Time Range: {t_start} to {t_end}")
    
    # Measure Initialization Time (This tests the new multiprocessing index builder)
    start_init = time.time()
    reader = DataReaderRadklim(
        tw_handler=tw_handler,
        filename=dataset_path,
        stream_info=stream_info
    )
    end_init = time.time()

    print(f"\n=== Reader Initialization Successful ===")
    print(f"Initialization Time: {end_init - start_init:.2f} seconds")
    print(f"Total time windows available: {reader.length()}")
    print(f"Grid shape (after stride): {reader.grid_shape}")
    
    if reader.length() > 0:
        print("\n=== Measuring Data Fetch Performance ===")
        num_tests = 5
        total_fetch_time = 0
        
        # Pick 5 indices evenly spread across the 5 years to test disk seeking
        test_indices = np.linspace(0, reader.length() - 1, num_tests, dtype=int)
        
        for i, idx in enumerate(test_indices):
            start_get = time.time()
            # Target channels index is [0] for RR
            reader_data = reader._get(idx=idx, channels_idx=[0])
            fetch_time = time.time() - start_get
            
            total_fetch_time += fetch_time
            print(f"  Fetched window {idx} in {fetch_time:.3f} seconds. (Data shape: {reader_data.data.shape})")
            
        avg_fetch = total_fetch_time / num_tests
        print(f"\nAverage fetch time per 24h window: {avg_fetch:.3f} seconds")
        
    else:
        print("\nReader is empty for the given time window! Try changing t_start/t_end.")

if __name__ == "__main__":
    test_radklim()