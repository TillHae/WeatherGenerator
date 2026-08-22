import logging

import numpy as np
import pandas as pd
import xarray as xr
from joblib import delayed
from numpy.typing import NDArray

from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel
from weathergen.evaluate.io.io_reader import ReaderOutput
from weathergen.evaluate.scores.score import VerifiedData, get_score
from weathergen.evaluate.scores.score_orchestration import get_next_fstep_data
from weathergen.evaluate.utils.regions import RegionBoundingBox

_logger = logging.getLogger(__name__)


def group_by_init_hour(
    output_data: ReaderOutput,
) -> dict[int, NDArray]:
    """Group sample indices by the hour of day of the initialisation time.

    The initialisation time is taken from the ``init_times`` coordinate
    on the sample dimension (which stores the end of the conditioning window, i.e.
    the forecast reference time).

    Returns only sample index arrays (no data copies) to avoid doubling memory.
    Use ``da.sel(sample=indices)`` on the original data when processing each group.

    Parameters
    ----------
    output_data : ReaderOutput
        Pre-loaded data with ``target`` and ``prediction`` dicts keyed by fstep.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from hour (0–23) to sample indices belonging to that hour.
    """
    first_tar = next(iter(output_data.target.values()))
    if "init_times" not in first_tar.coords:
        _logger.warning("Cannot group by init hour: 'init_times' coordinate not found.")
        return {}

    init_times = pd.DatetimeIndex(first_tar.init_times.values)
    hours = init_times.hour
    samples = first_tar.sample.values

    grouped: dict[int, NDArray] = {int(hour): samples[hours == hour] for hour in sorted(set(hours))}

    _logger.info(
        f"Grouped {len(samples)} samples into {len(grouped)} init-hour bins: "
        f"{sorted(grouped.keys())}."
    )
    return grouped


def _compute_scores_for_fstep(
    region: str,
    fstep: int,
    metric_names: list[str],
    metric_params: list,
    score_data: VerifiedData,
    preds_r: xr.DataArray,
) -> tuple[str, int, list, xr.DataArray, list[str]]:
    """Compute scores for a single (region, fstep) pair (parallelisable worker).

    Returns ``(region, fstep, score_results, preds_r, metric_names)``.
    """
    score_results: list[xr.DataArray | None] = [
        get_score(score_data, m, agg_dims="sample", parameters=p)
        for m, p in zip(metric_names, metric_params, strict=False)
    ]
    return region, fstep, score_results, preds_r, metric_names


def _compute_scores(
    regions: list[str],
    metrics_dict: dict,
    fsteps: list,
    da_preds: dict,
    da_tars: dict,
    aligned_clim_data: dict | None,
    n_workers: int | None = None,
) -> tuple[dict, dict]:
    """Compute scores for all (region, fstep) pairs. Score computation is parallelised across
      (region, fstep) pairs.

    Returns
    -------
    computed : dict[tuple, tuple]
        ``{(region, fstep): (score_results, preds_r, metric_names)}``
    raw_results : list[tuple]
        List of raw results from parallel score computation, each item is a tuple of
        ``(region, fstep, score_results, preds_r, metric_names)``.
    """
    # Build one task per (region, fstep) with pre-applied region masking.
    tasks = []
    for region in regions:
        bbox = RegionBoundingBox.from_region_name(region)
        metrics = metrics_dict[region]
        metric_names = list(metrics.keys())
        metric_params = list(metrics.values())
        for fstep in fsteps:
            tars_fs = da_tars[fstep]
            preds_fs = da_preds[fstep]
            preds_next, tars_next = get_next_fstep_data(fstep, da_preds, da_tars, fsteps)
            climatology = aligned_clim_data[fstep] if aligned_clim_data else None
            tars_r, preds_r, tars_next_r, preds_next_r = [
                bbox.apply_mask(x) if x is not None else None
                for x in (tars_fs, preds_fs, tars_next, preds_next)
            ]
            tasks.append(
                dict(
                    region=region,
                    fstep=fstep,
                    metric_names=metric_names,
                    metric_params=metric_params,
                    score_data=VerifiedData(
                        preds_r, tars_r, preds_next_r, tars_next_r, climatology
                    ),
                    preds_r=preds_r,
                )
            )

    # Compute scores in parallel across (region, fstep) pairs.
    calls = [delayed(_compute_scores_for_fstep)(**t) for t in tasks]
    raw_results = dispatch_parallel(
        calls, n_workers=n_workers, backend="loky", desc="Score computation"
    )

    # Accumulate per-channel colour ranges from the completed results.
    computed: dict[tuple, tuple] = {}
    for region, fstep, score_results, preds_r, metric_names in raw_results:
        computed[(region, fstep)] = (score_results, preds_r, metric_names)

    return computed, raw_results


def _compute_ranges(
    raw_results: list[tuple[str, int, list, xr.DataArray, list[str]]],
    global_plotting_options: dict | None = None,
) -> dict:
    """Compute colour ranges for each metric/region/channel from the raw score results.

    Returns
    -------
    score_ranges_dict : dict
        ``{metric: {region: {channel: {'vmin': float, 'vmax': float}}}}``
    """
    if global_plotting_options is None:
        global_plotting_options = {}
    
    score_maps_config = global_plotting_options.get("score_maps_config", {})
    
    # Standard fixed map maximums to ensure consistent scales across runs
    std_vmax_rmse = {
        "2t": 4.0, "10u": 5.0, "10v": 5.0,
        "msl": 500.0, "z": 500.0, "t": 4.0,
        "u": 5.0, "v": 5.0, "q": 0.005, "tp": 10.0
    }
    std_vmax_bias = {
        "2t": 2.0, "10u": 2.5, "10v": 2.5,
        "msl": 250.0, "z": 250.0, "t": 2.0,
        "u": 2.5, "v": 2.5, "q": 0.002, "tp": 5.0
    }
    std_vmax_mae = {
        "2t": 3.0, "10u": 4.0, "10v": 4.0,
        "msl": 400.0, "z": 400.0, "t": 3.0,
        "u": 4.0, "v": 4.0, "q": 0.004, "tp": 8.0
    }

    score_ranges_dict: dict = {}
    for region, fstep, score_results, _, metric_names in raw_results:
        for metric, result in zip(metric_names, score_results, strict=False):
            if result is None or "channel" not in result.coords:
                continue
            score_ranges_dict.setdefault(metric, {}).setdefault(region, {})
            for ch in result.coords["channel"].values:
                vals = result.sel(channel=ch).values.flatten()
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                ch_key = str(ch)
                score_ranges_dict[metric][region].setdefault(ch_key, {})
                
                vmin, vmax = float(vals.min()), float(vals.max())
                
                # Check for config overrides
                metric_cfg = score_maps_config.get(metric, {})
                ch_cfg = metric_cfg.get(ch_key, {})
                
                if "vmin" in ch_cfg and "vmax" in ch_cfg:
                    vmin = float(ch_cfg["vmin"])
                    vmax = float(ch_cfg["vmax"])
                else:
                    # Apply fixed scales for known variables, scaled dynamically by fstep!
                    # This ensures fstep 2 has a tighter colorbar than fstep 8, preventing "all blue" maps.
                    # We assume fstep 8 (48h) uses the full standard max scale.
                    fstep_scale = max(0.1, min(1.0, float(fstep) / 8.0))
                    
                    if metric == "rmse":
                        vmin = 0.0
                        if ch_key in std_vmax_rmse:
                            vmax = float(std_vmax_rmse[ch_key]) * fstep_scale
                    elif metric == "mae":
                        vmin = 0.0
                        if ch_key in std_vmax_mae:
                            vmax = float(std_vmax_mae[ch_key]) * fstep_scale
                    elif metric == "bias":
                        if ch_key in std_vmax_bias:
                            vmax = float(std_vmax_bias[ch_key]) * fstep_scale
                            vmin = -vmax
                        else:
                            # symmetric dynamic bounds for bias
                            abs_max = max(abs(vmin), abs(vmax))
                            vmin, vmax = -abs_max, abs_max
                    elif metric == "acc":
                        vmin, vmax = 0.0, 1.0
                    elif metric == "ssim":
                        vmin, vmax = 0.0, 1.0
                    
                    # Apply partial overrides from config
                    if "vmin" in ch_cfg:
                        vmin = float(ch_cfg["vmin"])
                    if "vmax" in ch_cfg:
                        vmax = float(ch_cfg["vmax"])

                # Save directly per fstep
                score_ranges_dict[metric][region][ch_key][fstep] = {
                    "vmin": vmin,
                    "vmax": vmax,
                }

    return score_ranges_dict
