import pandas as pd, glob, os

node="between_deep_clear"
a="left_clean"
b="overlap_boundary_only"

for fp in sorted(glob.glob("phase29b_rollouts_wN=*.csv")):
    df=pd.read_csv(fp)
    ca=cb=ct=0
    for p in df["path"].dropna().astype(str):
        nodes=[x.strip() for x in p.split("->") if x.strip()]
        for i in range(len(nodes)-1):
            if nodes[i]==node:
                ct+=1
                ca += (nodes[i+1]==a)
                cb += (nodes[i+1]==b)
    print(os.path.basename(fp), "steps@",node,"=",ct," A=",ca," B=",cb)
