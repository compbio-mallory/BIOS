"""
plot_loss.py — total loss vs epoch from training logs.

Usage:
  python plot_loss.py --out results/loss_UCEC_matched_600.png \
      --title "UCEC — linear, 4 GSVA anchors, bio_dim 4, K=4, lambda_bio 0.1, 600 epochs" \
      logs/ucec_matched_12152922.out
"""
import re, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--title", required=True)
ap.add_argument("--marks", default="150,200,300",
                help="comma-separated epochs to mark with vertical lines")
ap.add_argument("logs", nargs="+")
a = ap.parse_args()

fig, ax = plt.subplots(figsize=(10, 6))
for f in a.logs:
    ep, loss = [], []
    for line in open(f):
        m = re.search(r"Epoch \[(\d+)/\d+\] Total Loss: ([\d.]+)", line)
        if m:
            ep.append(int(m.group(1))); loss.append(float(m.group(2)))
    if not ep:
        print(f"{f}: no loss lines found"); continue
    label = f.split('/')[-1].replace('.out', '')
    ax.plot(ep, loss, lw=1.2, label=f"{label} (n={len(ep)})")
    for cand in (100, 150, 200, 300, 400, 600, 1000):
        if len(loss) > cand:
            print(f"{label}: @{cand} loss={loss[cand]:.3f}  "
                  f"delta(prev 50)={loss[cand]-loss[cand-50]:+.3f}")
    print(f"{label}: final@{ep[-1]} = {loss[-1]:.3f}\n")

for c in [int(x) for x in a.marks.split(",") if x]:
    ax.axvline(c, ls="--", c="gray", lw=0.9)
ax.set_xlabel("Epoch"); ax.set_ylabel("Total loss")
ax.set_title(a.title, fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(a.out, dpi=200)
print(f"saved {a.out}")