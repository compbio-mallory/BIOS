"""List every save dir with its stored config, from the last checkpoint."""
import glob, os, re, torch

rows = []
for d in sorted(glob.glob('save/*/')):
    cks = sorted(glob.glob(f'{d}/checkpoint_*.tar'),
                 key=lambda f: int(re.search(r'_(\d+)\.tar', f).group(1)))
    if not cks:
        rows.append((os.path.basename(d.rstrip('/')), 0, '', '', '', '', ''))
        continue
    last_ep = int(re.search(r'_(\d+)\.tar', cks[-1]).group(1))
    try:
        c = torch.load(cks[-1], map_location='cpu', weights_only=False)
        rows.append((os.path.basename(d.rstrip('/')), len(cks), last_ep,
                     c.get('head_type'), c.get('bio_dim'),
                     c.get('n_anchors'), c.get('n_clusters')))
    except Exception as e:
        rows.append((os.path.basename(d.rstrip('/')), len(cks), last_ep,
                     f'ERR {e}', '', '', ''))

print(f"{'dir':52s} {'#ck':>4s} {'last':>5s} {'head':22s} {'bd':>4s} {'anch':>5s} {'k':>3s}")
print('-'*105)
for r in rows:
    print(f"{r[0]:52s} {r[1]:4d} {str(r[2]):>5s} {str(r[3]):22s} "
          f"{str(r[4]):>4s} {str(r[5]):>5s} {str(r[6]):>3s}")
