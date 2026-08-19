import sys, os, glob, numpy as np, cv2
ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
os.makedirs('.stage', exist_ok=True)
for p in glob.glob('_nbsrc/*.py'):
    src = open(p, encoding='utf-8').read().split('\n')
    if src and src[0].startswith('%%writefile'):
        src = src[1:]
    open(os.path.join('.stage', os.path.basename(p)), 'w', encoding='utf-8').write('\n'.join(src))
sys.path.insert(0, '.stage')
import config, detect, textmask as tm, erase
img = cv2.cvtColor(cv2.imread('jepang_002.webp'), cv2.COLOR_BGR2RGB)
print('page', img.shape, flush=True)
regs, bubs = detect.detect(img)
soft = tm.ctd_soft_mask(img)
for r in regs:
    tm.build_region_mask(img, r, soft)
tm.partition_shared_interiors(img, regs)
tm.disjoin_overlapping_interiors(img, regs)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(f'regions={len(regs)} bubbles={len(bubs)}', flush=True)
for r in regs:
    fm = erase._fill_on_page(r, img.shape[:2])
    if fm is None:
        print(f'  r{r.idx} bbox={r.bbox} bub={r.bubble_bbox} fill=None -> jalur ink_mask', flush=True)
        continue
    sel = fm > 0
    bx1, by1, bx2, by2 = r.fill_bbox
    box = (bx2-bx1)*(by2-by1)
    boxfrac = sel.sum()/max(box,1)
    ink = np.zeros(img.shape[:2], np.uint8)
    x1,y1,x2,y2 = r.bbox
    mh,mw = r.ink_mask.shape[:2]
    yy2,xx2 = min(y2,y1+mh), min(x2,x1+mw)
    ink[y1:yy2, x1:xx2] = r.ink_mask[:yy2-y1,:xx2-x1]
    inkin = float(((ink>0)&sel).sum())/max(float((ink>0).sum()),1)
    darkin = float((gray[sel] < 110).mean())
    print(f'  r{r.idx} bbox={r.bbox} fillbox={r.fill_bbox} px={int(sel.sum())} boxfrac={boxfrac:.3f} inkin={inkin:.3f} darkfrac={darkin:.4f}', flush=True)
print('--- bubbles tanpa region ---', flush=True)
for b in bubs:
    owned = any(r.bubble_bbox == b or (r.shared_bubble_bbox == b) for r in regs)
    if owned: continue
    bx1,by1,bx2,by2 = b
    sub = gray[by1:by2, bx1:bx2]
    print(f'  ORPHAN {b} {bx2-bx1}x{by2-by1} dark={float((sub<110).mean()):.4f}', flush=True)
