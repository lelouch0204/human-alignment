"""Compare local shape substitutions averaged across the seven shared outlines."""
from itertools import combinations
import csv
import numpy as np
from scipy.stats import rankdata, spearmanr
from probe_navon_shapes import root, run, labels

data = np.load(root / "data/human.npz")
a,b = labels[data["pairs"]].transpose(1,0,2)
dist = [data["human_distance"]]
for variant in ("unalignet", "alignet"):
    dist.append(np.load(run / f"siglip_{variant}_pre_logits-distances.npz")["frozen"])
rows = []
for x,y in combinations(range(1,8),2):
    sel = (a[:,0]==b[:,0]) & (np.minimum(a[:,1],b[:,1])==x) & (np.maximum(a[:,1],b[:,1])==y)
    assert sel.sum()==7
    rows.append([x,y,*(d[sel].mean() for d in dist)])
ranks = rankdata(np.array(rows)[:,2:],axis=0)
gain = abs(ranks[:,0]-ranks[:,1])-abs(ranks[:,0]-ranks[:,2])
with (run/"siglip-local-substitutions.csv").open("w") as out:
    w=csv.writer(out)
    w.writerow(["local_a","local_b","human","unalignet","alignet","human_rank","unalignet_rank","alignet_rank","rank_error_reduction"])
    for i in np.argsort(-gain):
        w.writerow([*rows[i],*ranks[i],gain[i]])
print(spearmanr(ranks).statistic[0],[(rows[i][:2],ranks[i].tolist()) for i in np.argsort(-gain)[:3]])
