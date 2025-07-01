import numpy as np
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

genre_map = pd.read_csv("/u/abhutani/story-surprisal/results/gutenberg-LOC-genres.csv")
colors = 41*list(mcolors.XKCD_COLORS.values())

color_map = dict(zip(genre_map.genre.unique(),colors))
genre_map = dict(zip(genre_map.gutenberg_id, genre_map.genre))


rng = np.random.RandomState(42)
genres = []
X = []
for fp in glob.glob("/u/abhutani/story-surprisal/results/**-**/embedding-files/**.csv")[:100]:
    g_id = int(os.path.splitext(os.path.basename(fp))[0])
    genres.append(genre_map[g_id])
    df = pd.read_csv(fp)
    X.append(df['0'].values)

resampler = TimeSeriesResampler(sz=1000)
X = resampler.fit_transform(X)
scaler = TimeSeriesScalerMeanVariance()
X_scaled = scaler.fit_transform(X)  #  shape = (n_series, series_length)

n_clusters = 10
km_dtw = TimeSeriesKMeans(n_clusters=n_clusters,
                          metric="dtw",
                          max_iter=10,
                          random_state=0,
                          verbose=True)
labels = km_dtw.fit_predict(X_scaled)


D = cdist_dtw(X_scaled)

sil_score = silhouette_score(D, labels, metric="precomputed")
print(f"Silhouette Score (DTW): {sil_score:.3f}")
print(f"n_clusters {n_clusters}")

fig, axes = plt.subplots(
    n_clusters, 1,
    figsize=(6, 2 * n_clusters),
    sharex=True,
    sharey=True
)

all_handles, all_labels = [], []

for cluster_idx, ax in enumerate(axes):
    for ts, lbl, genre in zip(X_scaled, labels, genres):
        if lbl == cluster_idx:
            handle, = ax.plot(ts.ravel(), alpha=0.4, color=color_map[genre], 
                              label=genre if genre not in all_labels else "")
            if genre not in all_labels:
                all_handles.append(handle)
                all_labels.append(genre)
    ax.set_title(f"Cluster {cluster_idx} — {np.sum(labels == cluster_idx)} series")

fig.legend(
    all_handles, all_labels,
    loc="lower center",
    ncol=5,               
    frameon=False,
    bbox_to_anchor=(0.5, -0.02)
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.05)

plt.savefig("embedding_clusters.png", dpi=300, bbox_inches="tight")
plt.show()