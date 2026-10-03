"""Controlled test of P1/P4: does a fitted L2 estimator inherit the label misalignment?"""
import numpy as np, json, sys, zlib
import skimage.data as D, skimage.color as C
from scipy.ndimage import correlate
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 0
rng=np.random.default_rng(SEED)
def g(n):
    im=getattr(D,n)(); return C.rgb2gray(im) if im.ndim==3 else im/255.0
IMGS=[g(n) for n in ['astronaut','coffee','chelsea','rocket','camera','brick','grass','gravel']]
# real hold patterns: the four affected rows of Figure 1(c), 320 intervals each
import os
HERE=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(os.path.join(HERE,'hold_sequences.json')))['sequences']
HOLDS=[np.isin(np.arange(320),r).astype(int) for r in R if r]
H,W,PAD,KY,KX=48,160,96,1,40           # crop, padding, filter half-sizes (3 x 81: support exceeds the longest blur)

def crop(split):
    im=IMGS[rng.integers(len(IMGS))]; h,w=im.shape
    y0,y1=(0,h//2-H) if split=='train' else (h//2,h-H)
    y=rng.integers(y0,y1); x=rng.integers(0,w-W-2*PAD)
    return im[y:y+H, x:x+W+2*PAD]
def shifted(F,s,n):                     # sub-pixel horizontal shift via Fourier
    k=np.fft.fftfreq(n)
    return np.real(np.fft.ifft(F*np.exp(-2j*np.pi*k*s)[None,:],axis=1))[:,PAD:PAD+W]
def window(kind,split,fixed_sign=False,Lr=(16,32)):
    im=crop(split); F=np.fft.fft(im,axis=1); n=im.shape[1]
    if kind=='odd':   N=7; b=np.zeros(N-1,int)
    elif kind=='even':N=8; b=np.zeros(N-1,int)
    else:             N=7; h=HOLDS[rng.integers(len(HOLDS))]; i=rng.integers(0,len(h)-(N-1)); b=h[i:i+N-1]
    T=np.concatenate([[0.0],np.cumsum(1+b)]); m=N//2
    L=rng.uniform(*Lr); v=L/(N-1)                 # blur length under uniform timing
    sgn=1.0 if fixed_sign else rng.choice([-1.0,1.0])
    pos=lambda t: sgn*v*(t-T.mean())                # centred on the true centroid
    blur=np.mean([shifted(F,pos(t),n) for t in T],axis=0)
    label=shifted(F,pos(T[m]),n); target=shifted(F,pos(T.mean()),n)
    dtil=(T[m]-T.mean())/(T[-1]-T[0])
    return blur,label,target,dtil,sgn*v*(T[m]-T.mean())

def feats(blur,ys,xs):
    return np.stack([blur[ys+dy, xs+dx] for dy in range(-KY,KY+1) for dx in range(-KX,KX+1)],1)
def fit_pair(kind,fixed_sign,Lr=(16,32),n=500,per=300,lam=1e-3):
    """One set of training windows, two targets: the conventional label and the
    frame at the true centroid. Only the label differs between the two fits."""
    X=[];Yc=[];Yo=[]
    for _ in range(n):
        b,l,t,_,_=window(kind,'train',fixed_sign,Lr)
        ys=rng.integers(KY,H-KY,per); xs=rng.integers(KX,W-KX,per)
        X.append(feats(b,ys,xs)); Yc.append(l[ys,xs]); Yo.append(t[ys,xs])
    X=np.concatenate(X); X=np.c_[X,np.ones(len(X))]; G=X.T@X+lam*np.eye(X.shape[1])
    out=[]
    for Y in (np.concatenate(Yc),np.concatenate(Yo)):
        w=np.linalg.solve(G,X.T@Y); out.append((w[:-1].reshape(2*KY+1,2*KX+1),w[-1]))
    return out
def psnr(a,b): return 10*np.log10(1/max(float(np.mean((a-b)**2)), 1e-12))   # identical images give a finite value
def gradE(a): return np.mean(np.diff(a,axis=1)**2)
def xshift(a,b):                        # sub-pixel horizontal offset of a relative to b
    A=np.fft.fft(a-a.mean(),axis=1); B=np.fft.fft(b-b.mean(),axis=1)
    c=np.real(np.fft.ifft((A*np.conj(B)).sum(0))); k=int(np.argmax(c)); n=len(c)
    y0,y1,y2=c[(k-1)%n],c[k],c[(k+1)%n]; d=0.5*(y0-y2)/(y0-2*y1+y2+1e-12)
    s=k+d; return s-n if s>n/2 else s
def evaluate(kind,fixed_sign,k,b0,Lr=(16,32),n=200,seed=0):
    global rng; rng=np.random.default_rng(10_000+seed)   # identical test windows for paired models
    r=dict(model=[],label=[],sharp=[],shift=[],offset=[],dtil=[])
    sl=np.s_[KY+2:H-KY-2, KX+2:W-KX-2]
    for _ in range(n):
        b,l,t,d,off=window(kind,'test',fixed_sign,Lr)
        p=correlate(b,k,mode='nearest')+b0
        r['model'].append(psnr(p[sl],t[sl])); r['label'].append(psnr(l[sl],t[sl]))
        r['sharp'].append(gradE(p[sl])/gradE(t[sl])); r['shift'].append(xshift(p[sl],t[sl]))
        r['offset'].append(off); r['dtil'].append(d)
    return {k2:np.array(v) for k2,v in r.items()}
out={}
CONDS=[('A odd, equal durations','odd',False),('B even, equal durations','even',False),
       ('C odd, real holds','holds',False),('B+ even, direction fixed','even',True)]
for LR in [(8,16),(16,32),(32,48)]:
    for name,kind,fs in CONDS:
        if name.startswith('B+') and LR!=(16,32): continue
        rng=np.random.default_rng(SEED*1000+zlib.crc32((name+str(LR)).encode())%997)
        (kc,bc),(ko,bo)=fit_pair(kind,fs,LR)
        res={}
        for tag,(k,b0) in (('conventional',(kc,bc)),('oracle',(ko,bo))):
            e=evaluate(kind,fs,k,b0,LR,seed=SEED)
            res[tag]=dict(psnr=float(e['model'].mean()),shift=float(np.mean(e['shift']*np.sign(e['offset']+1e-9))),psnr_per=e['model'].tolist())
        res['label_offset_px']=float(np.mean(np.abs(e['offset']))); res['sigma_dtil']=float(e['dtil'].std())
        out[f"{name} | L {LR[0]}-{LR[1]}"]=res
json.dump(out,open(os.path.join(HERE,f'res_{SEED}.json'),'w'))
print(f"seed {SEED} done")
