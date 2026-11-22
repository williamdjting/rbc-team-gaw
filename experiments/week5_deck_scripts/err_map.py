import json, sys
import numpy as np, h3
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
d = json.load(open(sys.argv[1]))['mapday']
poly = lambda cells: [[(lon, lat) for lat, lon in h3.cell_to_boundary(c)] for c in cells]
fig, ax = plt.subplots(figsize=(9, 8.2))
ax.set_aspect(1 / np.cos(np.deg2rad(51))); ax.axis('off')
ax.add_collection(PolyCollection(poly(d['all']), facecolor='#f1eeea', edgecolor='#e2ddd7', linewidth=0.1))
for k, c in [('fp', '#f2a33a'), ('fn', '#d23b3b'), ('tp', '#2a78d6')]:
    ax.add_collection(PolyCollection(poly(d[k]), facecolor=c, edgecolor=c, linewidth=0.4))
ax.autoscale_view()
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110, facecolor='white')
print('ok', d['date'])
