import os, glob, io
src = "_nbsrc"; dst = ".stage"
os.makedirs(dst, exist_ok=True)
n = 0
for f in sorted(glob.glob(os.path.join(src, "*.py"))):
    txt = io.open(f, encoding="utf-8").read().split("\n")
    if txt and txt[0].startswith("%%writefile"):
        txt = txt[1:]
    io.open(os.path.join(dst, os.path.basename(f)), "w", encoding="utf-8",
            newline="\n").write("\n".join(txt))
    n += 1
print("staged", n)
