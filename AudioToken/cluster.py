import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
import umap.umap_ as umap
import os
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity


def compute_cluster_metrics_to_file(all_tokens, save_dir):

    if len(all_tokens) == 0:
        raise ValueError("all_tokens is empty")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cluster_metrics.txt")

    labels = []
    emb_list = []

    for lab, tok in all_tokens:
        labels.append(str(lab))

        try:
            arr = tok.detach().cpu().numpy()
        except Exception:
            arr = np.array(tok)

        emb_list.append(arr.reshape(-1))

    X = np.stack(emb_list)
    labels = np.array(labels)

    # Normalize embeddings (recommended for cosine metrics)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels])

    # Global metrics
    global_sil = float(silhouette_score(X, y))
    db_index = float(davies_bouldin_score(X, y))

    sil_samples = silhouette_samples(X, y)

    with open(save_path, "w") as f:

        f.write("===== GLOBAL CLUSTER METRICS =====\n")
        f.write(f"Global Silhouette Score: {global_sil:.6f}\n")
        f.write(f"Davies-Bouldin Index: {db_index:.6f}\n\n")

        f.write("===== PER-LABEL METRICS =====\n\n")

        for lab in unique_labels:

            mask = labels == lab
            X_lab = X[mask]
            sil_lab = sil_samples[mask]

            n = X_lab.shape[0]

            centroid = X_lab.mean(axis=0)

            # Within cluster variance
            wcv = np.mean(np.sum((X_lab - centroid) ** 2, axis=1))

            # Intra-cluster cosine similarity
            if n > 1:
                cos_sim = cosine_similarity(X_lab)
                cos_sim = cos_sim[~np.eye(n, dtype=bool)]
                mean_cos = float(np.mean(cos_sim))
            else:
                mean_cos = 0.0

            mean_sil = float(np.mean(sil_lab))

            f.write(f"Label: {lab}\n")
            f.write(f"  Samples: {n}\n")
            f.write(f"  Within Cluster Variance: {wcv:.6f}\n")
            f.write(f"  Mean Intra Cosine Similarity: {mean_cos:.6f}\n")
            f.write(f"  Mean Silhouette Score: {mean_sil:.6f}\n")
            f.write("\n")

    print(f"Cluster metrics saved to: {save_path}")



def visualize_tokens(
    all_tokens,
    reduce_algo='pca',
    n_components=2,
    highlight_each_label=True,
    save_dir=None,
    save_path=None,
    show=False,
    random_state=42,
    highlight_marker_size=120,
    other_marker_size=30,
    highlight_alpha=0.95,
    other_alpha=0.12
):
    if len(all_tokens) == 0:
        raise ValueError("all_tokens is empty")

    # extract labels and embeddings
    labels = []
    emb_list = []
    for lab, tok in all_tokens:
        labels.append(str(lab))
        try:
            arr = tok.detach().cpu().numpy()
        except Exception:
            arr = np.array(tok)
        arr = arr.reshape(-1)
        emb_list.append(arr.astype(np.float32))

    X = np.stack(emb_list, axis=0)

    # standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # reduce
    D = X_scaled.shape[1]
    n_comp = min(n_components, D) if reduce_algo.lower() == 'pca' else n_components

    if reduce_algo.lower() == 'pca':
        pca = PCA(n_components=n_comp, random_state=random_state)
        X_reduced = pca.fit_transform(X_scaled)
    elif reduce_algo.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_comp, random_state=random_state)
        X_reduced = reducer.fit_transform(X_scaled)
    else:
        raise ValueError("reduce_algo must be 'pca' or 'umap'")

    plot_2d = (X_reduced.shape[1] >= 2)
    if not plot_2d:
        raise ValueError("n_components must be >= 2 for plotting")

    unique_labels = sorted(list(set(labels)))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    label_idx = np.array([label_to_idx[lab] for lab in labels])

    # color map for up to 50+ labels
    n_colors = len(unique_labels)
    if n_colors <= 20:
        base_cmap = cm.get_cmap('tab20', n_colors)
    else:
        base_cmap = cm.get_cmap('hsv', n_colors)
    label_colors = [base_cmap(i % base_cmap.N) for i in range(n_colors)]

    # combined plot (all labels colored)
    if not highlight_each_label:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        colors = np.array([label_colors[idx] for idx in label_idx])
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colors, s=45, alpha=0.95, edgecolor='k', linewidth=0.18)
        ax.set_xlabel(f"{reduce_algo.upper()} 1")
        ax.set_ylabel(f"{reduce_algo.upper()} 2")
        ax.set_title(f"{reduce_algo.upper()} (n_comp={X_reduced.shape[1]}) — colored by label")

        # legend (limit to 50)
        from matplotlib.lines import Line2D
        legend_elems = []
        max_legend = 50
        for lab, idx in list(label_to_idx.items())[:max_legend]:
            legend_elems.append(Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=label_colors[idx % len(label_colors)], markersize=8,
                                       markeredgecolor='k', markeredgewidth=0.3, label=str(lab)))
        if legend_elems:
            ax.legend(handles=legend_elems, bbox_to_anchor=(1.05, 1), loc='upper left', title='Labels')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.savefig('visualization_all_labels.png', dpi=200, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()
        return

    # per-label highlight plots
        # per-label highlight plots
    if highlight_each_label:
        if save_dir is None:
            save_dir = 'visualizations'
        os.makedirs(save_dir, exist_ok=True)

        for main_lab in unique_labels:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()

            # Plot ALL points as grey background
            ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                c='grey',                 # darker than lightgrey
                s=other_marker_size,
                alpha=0.7,               # increase visibility
                edgecolor='white',        # gives contrast
                linewidth=0.2
            )
            # Highlight ONLY current label
            main_idx = label_to_idx[main_lab]
            this_mask = (label_idx == main_idx)

            if this_mask.any():
                ax.scatter(
                    X_reduced[this_mask, 0],
                    X_reduced[this_mask, 1],
                    c=[label_colors[main_idx % len(label_colors)]],
                    s=highlight_marker_size,
                    alpha=highlight_alpha,
                    edgecolor='k',
                    linewidth=0.3
                )

            ax.set_xlabel(f"{reduce_algo.upper()} 1")
            ax.set_ylabel(f"{reduce_algo.upper()} 2")
            ax.set_title(f"Highlighting label: {main_lab} — {int(this_mask.sum())} points")

            plt.tight_layout()

            fname = os.path.join(save_dir, f'highlight_{main_lab}.png')
            plt.savefig(fname, dpi=200, bbox_inches='tight')

            if show:
                plt.show()

            plt.close()
        
    compute_cluster_metrics_to_file(all_tokens, save_dir)    
    return 0
    