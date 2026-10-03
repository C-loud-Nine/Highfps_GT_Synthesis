Per-clip timing tables.

`per_clip.csv` is one row per clip: population, frames, duration, effective rate,
held fraction, linear-fit residual, onset and timescale.

`samples/<clip>.csv` is every sample of that clip in decode order: duration,
decode time, signed composition offset and presentation time, in timescale ticks.
Every timing measurement in the paper can be rebuilt from these files.
