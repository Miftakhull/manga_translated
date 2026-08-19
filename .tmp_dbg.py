import sys, os, pickle, numpy as np, cv2
ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
import erase, verify, textmask as tm
from config import SETTINGS, Region
page = cv2.cvtColor(cv2.imread(os.path.join(ROOT,'jepang_002.webp')), cv2.COLOR_BGR2RGB)
st = pickle.load(open(os.path.join(ROOT,'.pagediag.pkl'),'rb'))
regs=[]
for d in st['regs']:
    r=Region(idx=d['idx'],bbox=d['bbox'],det_class=d['det_class'],det_conf=d['det_conf'])
    r.bubble_bbox,r.bubble_mask=d['bubble_bbox'],d['bubble_mask']
    r.ink_mask=d['ink_mask'].copy(); r.est_font_size=d['est_font_size']
    r.fill_bbox,r.fill_mask=d['fill_bbox'],d['fill_mask']; regs.append(r)
tm.protect_bubble_outline(page,regs)
clean = erase.erase_page(page.copy(), regs, device="cpu")
def big(mask,bbox,shape):
    out=np.zeros(shape,np.uint8)
    x1,y1=bbox[0],bbox[1]; h,w=mask.shape[:2]
    yy,xx=min(y1+h,shape[0]),min(x1+w,shape[1])
    out[y1:yy,x1:xx]=mask[:yy-y1,:xx-x1]; return out
r=[x for x in regs if x.idx==6][0]
itr=big(r.bubble_mask,r.bubble_bbox,page.shape[:2])
ink=big(r.ink_mask,r.bbox,page.shape[:2])
er=cv2.erode(itr,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
tepi=(itr>0)&(er==0)&(ink>0)
print("r6 bbox",r.bbox,"bubble_bbox",r.bubble_bbox,"tepi",int(tepi.sum()))
ys,xs=np.where(tepi)
print("tepi y",ys.min(),ys.max(),"x",xs.min(),xs.max())
print("page gray di tepi", cv2.cvtColor(page,cv2.COLOR_RGB2GRAY)[tepi][:12])
print("clean gray di tepi", cv2.cvtColor(clean,cv2.COLOR_RGB2GRAY)[tepi][:12])
img=clean.copy(); img[tepi]=page[tepi]
dev,scope=verify._residue_scope(img,r)
x1,y1=r.bbox[0],r.bbox[1]; h,w=scope.shape
print("scope shape",scope.shape,"bbox h,w",r.bbox[3]-y1,r.bbox[2]-x1)
sb=np.zeros(page.shape[:2],bool); sb[y1:y1+h,x1:x1+w]=scope
db=np.zeros(page.shape[:2],bool); db[y1:y1+h,x1:x1+w]=dev
print("tepi dalam scope:",int((tepi&sb).sum()),"tepi dev:",int((tepi&db).sum()))
print("tepi di crop range:",int((tepi[y1:y1+h,x1:x1+w]).sum()))
