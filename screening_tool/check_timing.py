"""
check_timing.py -- screening tool for frame-averaged blur synthesis.

Checks the two conditions a blur-synthesis window must satisfy for its sharp
label to be unbiased:

  1. equal sample durations within the averaging window;
  2. an odd window size N.

Both are read from the container's sample-duration table. No decoding, so a
clip is screened in milliseconds.

Usage:
    python check_timing.py CLIP.MOV [CLIP2.MOV ...] --window 7
    python check_timing.py /path/to/folder --window 7 --csv out.csv
"""
import argparse, csv, os, struct, sys
import numpy as np

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}

def _boxes(f, end):
    while f.tell() < end - 8:
        start = f.tell(); head = f.read(8)
        if len(head) < 8: return
        size, btype = struct.unpack(">I4s", head)
        if size == 1:                       # 64-bit extended size
            ext = f.read(8)
            if len(ext) < 8: return
            size = struct.unpack(">Q", ext)[0]; payload = start + 16
        elif size == 0:                     # box runs to end of file
            size = end - start; payload = start + 8
        else:
            payload = start + 8
        if size < 8: return
        box_end = min(start + size, end)
        yield btype, payload, box_end
        f.seek(box_end)

def _walk(f, start, end, wanted, ctx=None, out=None, depth=0):
    if out is None: out = []
    if ctx is None: ctx = {"trak": -1}
    if depth > 8: return out
    f.seek(start)
    for btype, payload, box_end in _boxes(f, end):
        if btype == b"trak": ctx["trak"] += 1
        if btype in wanted: out.append((btype, payload, box_end, ctx["trak"]))
        if btype in CONTAINERS:
            here = f.tell(); _walk(f, payload, box_end, wanted, ctx, out, depth+1)
            f.seek(here)
    return out

def signed32(v):
    """Apple writes negative composition offsets in a version-0 ctts even though
    the specification declares the field unsigned. Anything at or above 2**31 is
    the corresponding negative value. Getting this wrong inflates every
    presentation timestamp by roughly 2**32 ticks."""
    return v - 2**32 if v >= 2**31 else v

def read_track(path):
    """Return the video track's sample durations, signed composition offsets,
    media timescale and edit-list entry count."""
    tracks = {}
    wanted = {b"mdhd", b"hdlr", b"stts", b"ctts", b"elst"}
    with open(path, "rb") as f:
        for btype, payload, box_end, tk in _walk(f, 0, os.path.getsize(path), wanted):
            tracks.setdefault(tk, {}); f.seek(payload)
            if btype == b"mdhd":
                raw = f.read(min(32, box_end - payload))
                if len(raw) >= 32 and raw[0] == 1:
                    tracks[tk]["timescale"] = struct.unpack(">IQ", raw[20:32])[0]
                elif len(raw) >= 20:
                    tracks[tk]["timescale"] = struct.unpack(">II", raw[12:20])[0]
            elif btype == b"hdlr":
                raw = f.read(min(24, box_end - payload))
                # each track has TWO hdlr atoms: a media handler in mdia and a
                # data handler in minf. Only the first identifies the track type.
                if len(raw) >= 12 and raw[8:12] == b"vide" and "kind" not in tracks[tk]:
                    tracks[tk]["kind"] = "vide"
            elif btype == b"stts":
                raw = f.read(8)
                if len(raw) < 8: continue
                n = struct.unpack(">I", raw[4:8])[0]
                pay = f.read(min(n*8, max(0, box_end - f.tell())))
                tracks[tk]["stts"] = [struct.unpack(">II", pay[i*8:i*8+8])
                                      for i in range(min(n, len(pay)//8))]
            elif btype == b"ctts":
                raw = f.read(8)
                if len(raw) < 8: continue
                n = struct.unpack(">I", raw[4:8])[0]
                pay = f.read(min(n*8, max(0, box_end - f.tell())))
                tracks[tk]["ctts"] = [
                    (struct.unpack(">I", pay[i*8:i*8+4])[0],
                     signed32(struct.unpack(">I", pay[i*8+4:i*8+8])[0]))
                    for i in range(min(n, len(pay)//8))]
            elif btype == b"elst":
                raw = f.read(min(8, box_end - payload))
                if len(raw) >= 8: tracks[tk]["elst_n"] = struct.unpack(">I", raw[4:8])[0]
    vid = next((t for t in tracks.values()
                if t.get("kind") == "vide" and "stts" in t), None)
    if vid is None:                         # fall back to the longest track
        cands = [t for t in tracks.values() if "stts" in t and "timescale" in t]
        vid = max(cands, key=lambda t: sum(a for a, _ in t["stts"]), default=None)
    return vid

expand = lambda entries: (None if not entries else
    np.concatenate([np.full(c, v, dtype=np.int64) for c, v in entries]))



def analyse(path, window=None, resid_pass=0.75, resid_fail=1.5, held_factor=1.5):
    """Screen one file. Returns a dict of findings, or None if unreadable."""
    try:
        trk = read_track(path)
    except Exception as exc:
        print(f"  {os.path.basename(path)}: unreadable ({exc})", file=sys.stderr)
        return None
    if not trk or "stts" not in trk:
        return None
    dur = expand(trk["stts"])
    if dur is None or len(dur) < 2:
        return None
    timescale = trk.get("timescale", 0) or 0
    ctts = expand(trk.get("ctts")) if trk.get("ctts") else None
    dts = np.concatenate([[0], np.cumsum(dur)[:-1]])
    pts = dts + (ctts[:len(dts)] if ctts is not None and len(ctts) >= len(dts)
                 else np.zeros(len(dts), dtype=np.int64))
    order = np.argsort(pts, kind="stable")
    t = pts[order].astype(float)
    gaps = np.diff(t)

    # condition 1: are the durations equal, beyond muxer rounding?
    idx = np.arange(len(t))
    fit = np.polyfit(idx, t, 1)
    resid = float(np.max(np.abs(t - np.polyval(fit, idx))))
    uniform = resid <= resid_pass
    verdict = "uniform" if uniform else ("non-uniform" if resid > resid_fail else "indeterminate")

    med = float(np.median(gaps))
    held = gaps > held_factor * med
    nominal = timescale / med if med else float("nan")
    effective = (len(t) - 1) * timescale / (t[-1] - t[0]) if t[-1] > t[0] else float("nan")

    out = dict(clip=os.path.basename(path), frames=len(t), timescale=timescale,
               median_interval_ticks=round(med, 1),
               residual_ticks=round(resid, 3), timing=verdict,
               held_pct=round(100 * float(held.mean()), 3),
               nominal_fps=round(nominal, 2), effective_fps=round(effective, 2))
    if window:
        out["window"] = window
        out["parity"] = "odd (unbiased)" if window % 2 else "even (biased)"
        out["parity_cost"] = 0.0 if window % 2 else round(1 / (2 * (window - 1)), 4)
        # share of windows that contain no held interval
        if len(held) >= window - 1:
            conv = np.convolve(held.astype(int), np.ones(window - 1, int), mode="valid")
            out["clean_window_pct"] = round(100 * float((conv == 0).mean()), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="clip files, or folders of clips")
    ap.add_argument("--window", type=int, default=None,
                    help="intended averaging window N, to check parity")
    ap.add_argument("--csv", default=None, help="write results to this CSV")
    a = ap.parse_args()

    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p))
                      if f.lower().endswith((".mov", ".mp4", ".m4v"))]
        else:
            files.append(p)

    rows = [r for r in (analyse(f, a.window) for f in files) if r]
    if not rows:
        print("no readable clips found"); return

    cols = []                      # union of fields, first-seen order: rows can differ
    for r in rows:
        cols += [c for c in r if c not in cols]
    w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))

    bad = [r for r in rows if r["timing"] != "uniform"]
    print(f"\n{len(rows)} clips: {len(rows) - len(bad)} uniform, {len(bad)} non-uniform")
    if a.window and a.window % 2 == 0:
        print(f"window N={a.window} is even: every pair carries a misalignment of "
              f"{1 / (2 * (a.window - 1)):.4f} of its blur length, even on uniform clips")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, restval="")
            wr.writeheader(); wr.writerows(rows)
        print(f"written: {a.csv}")


if __name__ == "__main__":
    main()
