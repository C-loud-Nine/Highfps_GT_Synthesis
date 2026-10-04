# Per-clip timing tables

The timing tables for all 51 iPhone-HighFPS clips are released together with the
clips at the camera-ready version of the paper. They can be regenerated from any
clip with the screening tool, which reads them directly from the container:

```bash
python ../screening_tool/check_timing.py /path/to/clips --window 7 --csv per_clip.csv
```

`per_clip.csv` has one row per clip with these columns:

| Column | Meaning |
|---|---|
| `clip` | file name |
| `frames` | number of samples in the video track |
| `timescale` | ticks per second of the track's media timescale |
| `median_interval_ticks` | median sample duration, in ticks |
| `residual_ticks` | maximum absolute residual of cumulative presentation times from a linear fit, in ticks |
| `timing` | `uniform` (residual at most 0.75 ticks), `non-uniform` (above 1.5) or `indeterminate` |
| `held_pct` | percentage of intervals longer than 1.5 times the median |
| `nominal_fps` | timescale divided by the median interval |
| `effective_fps` | frames per second of recording |
| `window`, `parity`, `parity_cost` | with `--window N`: whether N is odd, and the even-window offset 1/(2(N-1)) |
| `clean_window_pct` | with `--window N`: percentage of N-frame windows containing no held interval (blank if the clip is shorter than the window) |

Presentation times are computed with signed composition offsets (`ctts`), as the
paper describes; reading them as unsigned, as the ISO rule for a version-0 `ctts`
implies, inflates every timestamp by about 2^32 ticks.
