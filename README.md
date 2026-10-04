# highfps-gt-synthesis

**Check whether ground truth synthesised from high-speed video is temporally valid.**

Screening tool, controlled experiment and timing tables for the iPhone-HighFPS corpus.

Code accompanying the paper *Revisiting Ground-Truth Synthesis from High-Speed
Video: Exact Validity Conditions and an Audited Consumer Capture Corpus*
(under review; author identity withheld).

Frame-averaged blur synthesis labels a blurred image with the central frame of the
averaged window. That label is guaranteed to be unbiased only when the window is odd
**and** the sample durations inside it are equal. Interpolation benchmarks label a
middle frame as t = 0.5, which is correct only when the durations are equal. Both
conditions are readable from a video container without decoding, which is what the
tool here checks.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.8 or later. The screening tool needs only numpy.

## Contents

| Path | What it is |
|---|---|
| `screening_tool/check_timing.py` | Standalone screening tool. Reads the sample-duration table of a QuickTime or MP4 file and reports whether the durations are uniform, what fraction of intervals are held, the effective capture rate, and the parity cost of an intended window size. No decoding; milliseconds per clip. |
| `controlled_test/` | Reproduces the controlled test of Section 3.1 and Table 1 (3 x 81 linear filter, five seeds): `python run.py SEED` for seeds 0 to 4, then `python summarise.py`. `hold_sequences.json` holds the measured hold patterns transplanted onto the synthetic scenes. |
| `timing/` | Format of the per-clip timing tables, which are released with the clips and can be regenerated from any clip with the screening tool (`timing/README.md`). |

## Screening tool

```bash
python screening_tool/check_timing.py CLIP.MOV --window 7   # one clip
python screening_tool/check_timing.py /path/to/clips --window 8 --csv results.csv
```

Reported per clip:

- `median_interval_ticks`, `residual_ticks`: a clip passes the duration test when the maximum absolute deviation of cumulative timestamps from a linear fit is at most 0.75 ticks, and fails above 1.5. Pure muxer rounding stays near half a tick; genuine variation accumulates.
- `held_pct`: share of intervals longer than 1.5 times the clip's median.
- `effective_fps`: frames per second of recording, which is what the sensor delivered, as against the nominal rate in the header.
- `parity`, `parity_cost`: for an even window the label is misaligned by 1/(2(N-1)) of the blur length however perfect the capture.
- `clean_window_pct`: share of averaging windows containing no held interval, which is what survives filtering.

Requires Python 3.8+ and numpy.

## Controlled test

```bash
cd controlled_test
for s in 0 1 2 3 4; do python run.py $s; done
python summarise.py
```

Requires numpy, scipy and scikit-image. About 45 seconds per seed on one CPU core; no GPU needed.
Reports, for each violation and blur length, the PSNR cost of the conventional
label against a correctly centred one, measured on identical inputs.

Expected output (mean ± standard deviation over the five seeds; paper Section 3.1,
Table 1):

| Condition | Blur length | Cost (dB) | Learned shift |
|---|---|---|---|
| Odd window, equal durations | any | 0.000 | none |
| Even window, equal durations | 16-32 px | 0.42 ± 0.05 | none |
| Even window, equal durations | 32-48 px | 0.73 ± 0.07 | none |
| Even window, direction fixed | 16-32 px | 0.85 ± 0.03 | 1.08 of 1.72 px |
| Odd window, real hold patterns | 8-48 px | 0.04 to 0.05 | none |

## Data

The 51 iPhone-HighFPS clips and their per-clip timing tables are released with the
camera-ready version of the paper. Faces and vehicle plates are blurred, the files
carry the video track only, and each file's sample-duration table and frame count
are preserved from the camera original, so every timing result can be reproduced
from the released files with the screening tool.

## Citation

A BibTeX entry will be added on acceptance.

## Licence

Code: MIT (see `LICENSE`). Data: released under a permissive research licence
with the corpus.
