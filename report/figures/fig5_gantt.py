# Builds fig5_gantt.png. Run: python3 fig5_gantt.py (writes into the current directory).
# Data: report Table 3 (actual cells) / Table 2; 25-27 Nov = remaining (planned) part of M9.
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
rows = [  # (label, [(start_day, end_day_inclusive, kind)])
 ("Phases 1-5: topic to roadmap", [(12,15,"a")]),
 ("M0 Setup", [(16,16,"a")]),
 ("M1-M3 Data, features, training", [(17,18,"a")]),
 ("M4 Grid code and run", [(17,18,"a"),(19,20,"run"),(21,21,"a")]),
 ("M5 Serving and Docker", [(21,22,"a")]),
 ("M6 Monitoring and policy", [(22,22,"a")]),
 ("M7 Replay (E3)", [(22,22,"a")]),
 ("M8 Freeze and E4", [(22,22,"a"),(23,24,"a")]),
 ("M9 Report", [(12,15,"d"),(23,24,"a"),(25,27,"p")]),
]
style = {"a":dict(color="#4a6fa5"), "run":dict(color="#9db4d6"),
         "d":dict(color="#c9d6e8"), "p":dict(facecolor="white", edgecolor="#4a6fa5", hatch="///")}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
fig, ax = plt.subplots(figsize=(7.5,3.9), dpi=200)
for i,(lab,segs) in enumerate(rows):
    y = len(rows)-1-i
    for s,e,k in segs:
        ax.barh(y, e-s+1, left=s, height=0.6, **style[k], linewidth=0.8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xticks(range(12,29)); ax.set_xticklabels([str(d) if d<28 else "" for d in range(12,29)])
ax.set_xlim(12,28); ax.set_xlabel("Day of November 2026")
ax.grid(axis="x", color="#dddddd", linewidth=0.6); ax.set_axisbelow(True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.legend(handles=[Patch(**style["a"],label="Work (actual)"),Patch(**style["run"],label="Overnight grid run"),
    Patch(**style["d"],label="Early draft (1.1-1.2)"),Patch(**style["p"],label="Remaining (planned)")],
    loc="lower center", bbox_to_anchor=(0.4,1.0), fontsize=8, frameon=False, ncol=4)
fig.tight_layout(); fig.savefig("fig5_gantt.png")
