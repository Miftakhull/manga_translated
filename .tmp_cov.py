import sys, os, pickle, numpy as np, cv2
ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
import textmask as tm
from config import Region
page = cv2.cvtColor(cv2.imread(os.path.join(ROOT,'jepang_002.webp')), cv2.COLOR_BGR2RGB)
st = pickle.load(open(os.path.join(ROOT,'.pagediag.pkl'),'rb'))
regs=[]
for d in st['regs']:
    r=Region(idx=d['idx'],bbox=d['bbox'],det_class=d['det_class'],det_conf=d['det_conf'])
    r.bubble_bbox,r.bubble_mask=d['bubble_bbox'],d['bubble_mask']
    r.ink_mask=d['ink_mask'].copy(); r.est_font_size=d['est_font_size']
    r.fill_bbox,r.fill_mask=d['fill_bbox'],d['fill_mask']; regs.append(r)
tm.protect_bubble_outline(page,regs)
print(" r  ink_px  di_itr  frac  ada_bubble")
for r in regs:
    if r.ink_mask is None: continue
    x1,y1=r.bbox[0],r.bbox[1]
    itr=np.zeros(page.shape[:2],np.uint8)
    if r.bubble_mask is not None and r.bubble_bbox is not None:
        bx1,by1=r.bubble_bbox[0],r.bubble_bbox[1]
        bh,bw=r.bubble_mask.shape[:2]
        yy,xx=min(by1+bh,page.shape[0]),min(bx1+bw,page.shape[1])
        itr[by1:yy,bx1:xx]=r.bubble_mask[:yy-by1,:xx-bx1]
    ink=np.zeros(page.shape[:2],np.uint8)
    mh,mw=r.ink_mask.shape[:2]
    yy,xx=min(y1+mh,page.shape[0]),min(x1+mw,page.shape[1])
    ink[y1:yy,x1:xx]=r.ink_mask[:yy-y1,:xx-x1]
    a=int((ink>0).sum()); b=int(((ink>0)&(itr>0)).sum())
    print(f" r{r.idx:<2d} {a:6d} {b:7d} {b/max(a,1):5.3f}  {itr.any()}")
