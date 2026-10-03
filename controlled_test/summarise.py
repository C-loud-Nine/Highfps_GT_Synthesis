import json, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
R=[json.load(open(os.path.join(HERE,f"res_{s}.json"))) for s in range(5)]
for key in R[0]:
    d=np.array([r[key]["oracle"]["psnr"]-r[key]["conventional"]["psnr"] for r in R])
    w=np.concatenate([np.array(r[key]["oracle"]["psnr_per"])-np.array(r[key]["conventional"]["psnr_per"]) for r in R])
    sh=np.mean([r[key]["conventional"]["shift"] for r in R]); off=np.mean([r[key]["label_offset_px"] for r in R])
    print(f"{key:38s} offset {off:5.2f} px  cost {d.mean():+.3f} +/- {d.std(ddof=1):.3f} dB  correct better {100*np.mean(w>1e-9):5.1f}%  shift {sh:+.2f} px")
