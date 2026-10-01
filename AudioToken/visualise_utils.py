# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.decomposition import PCA
# # from sklearn.preprocessing import StandardScaler
# # from matplotlib import cm
# # from matplotlib.lines import Line2D
# # import umap.umap_ as umap
# # from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
# # from sklearn.metrics.pairwise import cosine_similarity
# # import torch
# # import clip

# # # ── Class names (fixed vocabulary) ─────────────────────────────────────────
# # _CLASS_NAMES = [
# #     "airplane flyby", "ambulance siren", "baby crying", "baby laughter", "car engine idling",
# #     "cat purring", "chainsawing trees", "church bell ringing", "train whistling", "train horning",
# #     "cow lowing", "dog barking", "elk bugling", "fire truck siren", "gibbon howling",
# #     "hail", "lawn mowing", "volcano explosion", "stream burbling", "waterfall burbling",
# #     "orchestra", "owl hooting", "people cheering", "people crowd", "people marching",
# #     "playing bassoon", "playing drum kit", "playing harp", "playing harpsichord", "playing snare drum",
# #     "playing timbales", "printer printing", "scuba diving", "sea waves", "sheep bleating",
# #     "singing choir", "skiing", "slot machine", "snake hissing", "tractor digging",
# #     "underwater bubbling", "railroad car, train wagon", "train wheels squealing", "baltimore oriole calling",
# #     "car engine knocking", "driving snowmobile", "wood thrush calling", "hedge trimmer running",
# #     "ice cream truck, ice cream van", "woodpecker pecking tree"
# # ]

# # # ── Lazy CLIP singleton ─────────────────────────────────────────────────────
# # _CLIP_MODEL  = None
# # _CLIP_DEVICE = None

# # def _get_clip_text_embeddings(device_str: str = None) -> dict:
# #     """
# #     Load ViT-L/14 once (cached in module globals), encode all _CLASS_NAMES
# #     with template 'a photo of a {}', return dict {label: np.ndarray (D,)}.
# #     """
# #     global _CLIP_MODEL, _CLIP_DEVICE

# #     if device_str is None:
# #         device_str = "cuda" if torch.cuda.is_available() else "cpu"

# #     if _CLIP_MODEL is None:
# #         print("[CLIP] Loading ViT-L/14 …")
# #         _CLIP_MODEL, _ = clip.load("ViT-L/14", device=device_str)
# #         _CLIP_DEVICE   = device_str
# #         print("[CLIP] Model loaded.")

# #     prompts = [f"a photo of a {c}" for c in _CLASS_NAMES]
# #     tokens  = clip.tokenize(prompts).to(_CLIP_DEVICE)

# #     with torch.no_grad():
# #         text_emb = _CLIP_MODEL.encode_text(tokens)          # (N, D)
# #         text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# #     return {
# #         name: text_emb[i].cpu().numpy()
# #         for i, name in enumerate(_CLASS_NAMES)
# #     }


# # def compute_cluster_metrics_to_file(all_tokens, save_dir, text_embeddings=None):

# #     if len(all_tokens) == 0:
# #         raise ValueError("all_tokens is empty")

# #     os.makedirs(save_dir, exist_ok=True)
# #     save_path = os.path.join(save_dir, "cluster_metrics.txt")

# #     labels = []
# #     emb_list = []

# #     for lab, tok in all_tokens:
# #         labels.append(str(lab))

# #         try:
# #             arr = tok.detach().cpu().numpy()
# #         except Exception:
# #             arr = np.array(tok)

# #         # ── BEATs: mean-pool [B, T, D] → [B, D] before flattening ──────────
# #         if arr.ndim == 3:
# #             arr = arr.mean(axis=1)          # (B, T, D) → (B, D)
# #         # ────────────────────────────────────────────────────────────────────

# #         emb_list.append(arr.reshape(-1))

# #     X_raw = np.stack(emb_list)              # (N, D) — raw, used for centroids
# #     labels = np.array(labels)

# #     # Normalize per-sample for silhouette / DB / cosine-sim metrics
# #     X = X_raw / np.linalg.norm(X_raw, axis=1, keepdims=True)

# #     unique_labels = sorted(set(labels))
# #     label_to_idx = {l: i for i, l in enumerate(unique_labels)}
# #     y = np.array([label_to_idx[l] for l in labels])

# #     # Global metrics
# #     global_sil = float(silhouette_score(X, y))
# #     db_index = float(davies_bouldin_score(X, y))
# #     sil_samples = silhouette_samples(X, y)

# #     def _centroid_distances(centroid_raw, t_arr):
# #         """
# #         Compute L2 and cosine distance between a raw (un-normalised) centroid
# #         and a text embedding (already unit-norm from CLIP encode_text).

# #         The centroid is the mean of raw embeddings — it is NOT unit-norm.
# #         We normalise it here before cosine distance so the comparison is valid.
# #         L2 is computed on the raw centroid vs unit-norm text emb, which is the
# #         geometrically correct distance in the embedding space.
# #         """
# #         l2  = float(np.linalg.norm(centroid_raw - t_arr))
# #         c_n = centroid_raw / (np.linalg.norm(centroid_raw) + 1e-8)   # normalise centroid
# #         t_n = t_arr        / (np.linalg.norm(t_arr)        + 1e-8)   # already unit, safety only
# #         cos = float(1.0 - np.dot(c_n, t_n))
# #         return l2, cos

# #     with open(save_path, "w") as f:

# #         f.write("===== GLOBAL CLUSTER METRICS =====\n")
# #         f.write(f"Global Silhouette Score: {global_sil:.6f}\n")
# #         f.write(f"Davies-Bouldin Index: {db_index:.6f}\n\n")

# #         f.write("===== PER-LABEL METRICS =====\n\n")

# #         for lab in unique_labels:
# #             mask  = labels == lab
# #             X_lab = X[mask]                         # normalised, for cosine metrics
# #             X_lab_raw = X_raw[mask]                 # raw, for centroid
# #             sil_lab = sil_samples[mask]
# #             n = X_lab.shape[0]

# #             # Centroid computed on RAW embeddings
# #             centroid_raw = X_lab_raw.mean(axis=0)

# #             # Within-cluster variance on normalised embeddings (consistent with cosine space)
# #             centroid_norm = centroid_raw / (np.linalg.norm(centroid_raw) + 1e-8)
# #             wcv = np.mean(np.sum((X_lab - centroid_norm) ** 2, axis=1))

# #             if n > 1:
# #                 cos_sim = cosine_similarity(X_lab)
# #                 cos_sim = cos_sim[~np.eye(n, dtype=bool)]
# #                 mean_cos = float(np.mean(cos_sim))
# #             else:
# #                 mean_cos = 0.0

# #             mean_sil = float(np.mean(sil_lab))

# #             # ── Centroid ↔ text embedding distances ────────────────────────
# #             centroid_to_text_l2     = None
# #             centroid_to_text_cosine = None

# #             if text_embeddings is not None:
# #                 t_emb = text_embeddings.get(lab)
# #                 if t_emb is not None:
# #                     t_arr = np.array(t_emb, dtype=np.float32).reshape(-1)
# #                     if t_arr.shape == centroid_raw.shape:
# #                         centroid_to_text_l2, centroid_to_text_cosine = \
# #                             _centroid_distances(centroid_raw, t_arr)

# #             f.write(f"Label: {lab}\n")
# #             f.write(f"  Samples: {n}\n")
# #             f.write(f"  Within Cluster Variance: {wcv:.6f}\n")
# #             f.write(f"  Mean Intra Cosine Similarity: {mean_cos:.6f}\n")
# #             f.write(f"  Mean Silhouette Score: {mean_sil:.6f}\n")
# #             if centroid_to_text_l2 is not None:
# #                 f.write(f"  Centroid → Text Emb L2 Distance:     {centroid_to_text_l2:.6f}\n")
# #                 f.write(f"  Centroid → Text Emb Cosine Distance: {centroid_to_text_cosine:.6f}\n")
# #             f.write("\n")

# #         # ── Aggregate centroid ↔ text distance stats across all classes ────
# #         if text_embeddings is not None:
# #             per_class_l2     = []
# #             per_class_cosine = []

# #             for lab in unique_labels:
# #                 mask        = labels == lab
# #                 centroid_raw = X_raw[mask].mean(axis=0)
# #                 t_emb       = text_embeddings.get(lab)

# #                 if t_emb is not None:
# #                     t_arr = np.array(t_emb, dtype=np.float32).reshape(-1)
# #                     if t_arr.shape == centroid_raw.shape:
# #                         l2, cos = _centroid_distances(centroid_raw, t_arr)
# #                         per_class_l2.append(l2)
# #                         per_class_cosine.append(cos)

# #             if per_class_l2:
# #                 f.write("===== CENTROID ↔ TEXT EMBEDDING DISTANCE (AGGREGATE) =====\n")
# #                 f.write(f"  Classes with valid distances: {len(per_class_l2)}\n\n")
# #                 f.write(f"  L2 Distance — Mean: {np.mean(per_class_l2):.6f}  "
# #                         f"Var: {np.var(per_class_l2):.6f}  "
# #                         f"Min: {np.min(per_class_l2):.6f}  "
# #                         f"Max: {np.max(per_class_l2):.6f}\n")
# #                 f.write(f"  Cosine Distance — Mean: {np.mean(per_class_cosine):.6f}  "
# #                         f"Var: {np.var(per_class_cosine):.6f}  "
# #                         f"Min: {np.min(per_class_cosine):.6f}  "
# #                         f"Max: {np.max(per_class_cosine):.6f}\n")

# #     print(f"Cluster metrics saved to: {save_path}")


# # def _pool_and_flatten(tok):
# #     """
# #     Convert a raw token tensor / array to a 1-D numpy float32 vector.

# #     Shapes handled
# #     --------------
# #     [T, D]      – single sample with sequence dim  → mean over T
# #     [B, T, D]   – batched (e.g. BEATs [8, 992, 2304]) → mean over T per sample,
# #                   then the B rows are treated as B individual embeddings by the
# #                   caller; here we just return shape (B, D) so the caller can
# #                   decide.  For the common single-sample case B=1, caller squeezes.
# #     [D]         – already flat, nothing to do
# #     """
# #     try:
# #         arr = tok.detach().cpu().numpy()
# #     except Exception:
# #         arr = np.array(tok)

# #     arr = arr.astype(np.float32)

# #     if arr.ndim == 3:
# #         # BEATs style [B, T, D] → mean pool over time → [B, D]
# #         arr = arr.mean(axis=1)
# #     elif arr.ndim == 2:
# #         # [T, D] single sample → mean pool → [D]
# #         arr = arr.mean(axis=0)
# #     # ndim == 1 → already fine

# #     return arr


# # def visualize_tokens(
# #     all_tokens,
# #     tokenizer=None,          # Optional CLIP tokenizer for text embedding extraction (if plot_labels=True)
# #     reduce_algo='pca',
# #     n_components=2,
# #     highlight_each_label=True,
# #     save_dir=None,
# #     save_path=None,
# #     show=False,
# #     random_state=42,
# #     highlight_marker_size=120,
# #     other_marker_size=30,
# #     highlight_alpha=0.95,
# #     other_alpha=0.12,
# #     plot_labels=False,      # If True: mark cluster centroid (star) + CLIP text emb (diamond) per plot
# #     clip_device=None,       # Device for CLIP inference; defaults to cuda if available
# # ):
# #     """
# #     Parameters
# #     ----------
# #     plot_labels : bool
# #         If True, for each per-label highlight plot also marks:
# #           * The cluster centroid       -- gold star
# #           * The CLIP text embedding    -- red diamond
# #           * A dashed line connecting both
# #         CLIP (ViT-L/14) is loaded automatically on first call and cached.
# #         Labels are matched against _CLASS_NAMES; unrecognised labels are skipped.

# #     clip_device : str or None
# #         e.g. 'cuda', 'cpu'. Defaults to cuda if available.
# #     """
# #     if len(all_tokens) == 0:
# #         raise ValueError("all_tokens is empty")

# #     # ── 1. Extract labels + embeddings ─────────────────────────────────────
# #     labels = []
# #     emb_list = []

# #     for lab, tok in all_tokens:
# #         arr = _pool_and_flatten(tok)

# #         if arr.ndim == 2:
# #             # BEATs batched [B, D] – add each row as its own sample
# #             for row in arr:
# #                 labels.append(str(lab))
# #                 emb_list.append(row)
# #         else:
# #             labels.append(str(lab))
# #             emb_list.append(arr)

# #     X = np.stack(emb_list, axis=0)          # (N, D)
# #     labels = np.array(labels)

# #     # ── 2. Standardise ─────────────────────────────────────────────────────
# #     scaler = StandardScaler()
# #     X_scaled = scaler.fit_transform(X)

# #     # ── 3. Fit dimensionality reducer on audio embeddings ONLY ─────────────
# #     D = X_scaled.shape[1]
# #     n_comp = min(n_components, D) if reduce_algo.lower() == 'pca' else n_components

# #     if reduce_algo.lower() == 'pca':
# #         reducer = PCA(n_components=n_comp, random_state=random_state)
# #     elif reduce_algo.lower() == 'umap':
# #         reducer = umap.UMAP(n_components=n_comp, random_state=random_state)
# #     else:
# #         raise ValueError("reduce_algo must be 'pca' or 'umap'")

# #     X_reduced = reducer.fit_transform(X_scaled)     # (N, n_comp)

# #     if X_reduced.shape[1] < 2:
# #         raise ValueError("n_components must be >= 2 for plotting")

# #     # ── 4. Compute + project CLIP text embeddings (only when plot_labels=True) ─
# #     #      CLIP is loaded lazily and cached. Projection uses the SAME fitted
# #     #      scaler + reducer as the audio embeddings so spaces are consistent.
# #     text_reduced = {}   # label → (n_comp,) in reduced space
# #     if plot_labels:
# #         # text_embeddings = _get_clip_text_embeddings(clip_device)   # {label: (D,)}
# #         text_embeddings = _get_clip_text_embeddings(tokenizer)   # {label: (D,)}

# #         for lab, t_emb in text_embeddings.items():
# #             t_arr = np.array(t_emb, dtype=np.float32).reshape(1, -1)

# #             if t_arr.shape[1] != X.shape[1]:
# #                 # CLIP dim != audio dim — can't project into the same space
# #                 print(f"[WARN] CLIP text emb dim {t_arr.shape[1]} != audio dim "
# #                       f"{X.shape[1]} for label '{lab}'; skipping text marker.")
# #                 continue

# #             t_scaled = scaler.transform(t_arr)

# #             if reduce_algo.lower() == 'pca':
# #                 t_proj = reducer.transform(t_scaled)[0]
# #             else:
# #                 # UMAP .transform() requires the reducer was fit normally
# #                 # (works out-of-the-box with standard umap-learn)
# #                 try:
# #                     t_proj = reducer.transform(t_scaled)[0]
# #                 except Exception:
# #                     t_proj = None
# #                     print(f"[WARN] UMAP transform failed for label '{lab}'; "
# #                           "text marker will be skipped.")

# #             text_reduced[str(lab)] = t_proj

# #     # ── 5. Label bookkeeping + colours ─────────────────────────────────────
# #     unique_labels = sorted(list(set(labels)))
# #     label_to_idx  = {lab: i for i, lab in enumerate(unique_labels)}
# #     label_idx     = np.array([label_to_idx[lab] for lab in labels])

# #     n_colors = len(unique_labels)
# #     base_cmap = cm.get_cmap('tab20', n_colors) if n_colors <= 20 \
# #         else cm.get_cmap('hsv', n_colors)
# #     label_colors = [base_cmap(i % base_cmap.N) for i in range(n_colors)]

# #     # ── 6a. Combined plot (no per-label highlighting) ──────────────────────
# #     if not highlight_each_label:
# #         plt.figure(figsize=(10, 8))
# #         ax = plt.gca()
# #         colors = np.array([label_colors[idx] for idx in label_idx])
# #         ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
# #                    c=colors, s=45, alpha=0.95, edgecolor='k', linewidth=0.18)
# #         ax.set_xlabel(f"{reduce_algo.upper()} 1")
# #         ax.set_ylabel(f"{reduce_algo.upper()} 2")
# #         ax.set_title(f"{reduce_algo.upper()} (n_comp={X_reduced.shape[1]}) — coloured by label")

# #         legend_elems = []
# #         for lab, idx in list(label_to_idx.items())[:50]:
# #             legend_elems.append(
# #                 Line2D([0], [0], marker='o', color='w',
# #                        markerfacecolor=label_colors[idx % len(label_colors)],
# #                        markersize=8, markeredgecolor='k', markeredgewidth=0.3,
# #                        label=str(lab))
# #             )
# #         if legend_elems:
# #             ax.legend(handles=legend_elems,
# #                       bbox_to_anchor=(1.05, 1), loc='upper left', title='Labels')

# #         plt.tight_layout()
# #         out = save_path or 'visualization_all_labels.png'
# #         plt.savefig(out, dpi=200, bbox_inches='tight')
# #         if show:
# #             plt.show()
# #         plt.close()
# #         return

# #     # ── 6b. Per-label highlight plots ──────────────────────────────────────
# #     if save_dir is None:
# #         save_dir = 'visualizations'
# #     os.makedirs(save_dir, exist_ok=True)

# #     for main_lab in unique_labels:
# #         fig, ax = plt.subplots(figsize=(10, 8))

# #         # Background – all points grey
# #         ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
# #                    c='grey', s=other_marker_size, alpha=0.7,
# #                    edgecolor='white', linewidth=0.2)

# #         # Highlighted cluster points
# #         main_idx  = label_to_idx[main_lab]
# #         this_mask = (label_idx == main_idx)
# #         cluster_color = label_colors[main_idx % len(label_colors)]

# #         if this_mask.any():
# #             ax.scatter(X_reduced[this_mask, 0], X_reduced[this_mask, 1],
# #                        c=[cluster_color],
# #                        s=highlight_marker_size, alpha=highlight_alpha,
# #                        edgecolor='k', linewidth=0.3)

# #         legend_elems = [
# #             Line2D([0], [0], marker='o', color='w',
# #                    markerfacecolor=cluster_color, markersize=9,
# #                    markeredgecolor='k', markeredgewidth=0.4,
# #                    label=f'{main_lab} samples ({int(this_mask.sum())})')
# #         ]

# #         # ── Extra markers when plot_labels=True ────────────────────────────
# #         if plot_labels and this_mask.any():

# #             # ① Cluster centroid  (gold star)
# #             centroid_2d = X_reduced[this_mask].mean(axis=0)
# #             ax.scatter(centroid_2d[0], centroid_2d[1],
# #                        marker='*', s=400, c='gold',
# #                        edgecolor='black', linewidth=0.8, zorder=5)
# #             legend_elems.append(
# #                 Line2D([0], [0], marker='*', color='w',
# #                        markerfacecolor='gold', markersize=14,
# #                        markeredgecolor='black', markeredgewidth=0.8,
# #                        label='Cluster centroid')
# #             )

# #             # ② CLIP text embedding  (red diamond)
# #             t_proj = text_reduced.get(str(main_lab))
# #             if t_proj is not None:
# #                 ax.scatter(t_proj[0], t_proj[1],
# #                            marker='D', s=220, c='red',
# #                            edgecolor='black', linewidth=0.8, zorder=6)
# #                 legend_elems.append(
# #                     Line2D([0], [0], marker='D', color='w',
# #                            markerfacecolor='red', markersize=10,
# #                            markeredgecolor='black', markeredgewidth=0.8,
# #                            label='CLIP text embedding')
# #                 )

# #                 # ③ Dashed line between centroid and text embedding
# #                 ax.plot([centroid_2d[0], t_proj[0]],
# #                         [centroid_2d[1], t_proj[1]],
# #                         linestyle='--', color='red', linewidth=0.9,
# #                         alpha=0.6, zorder=4)

# #         # ── Axes & legend ──────────────────────────────────────────────────
# #         ax.set_xlabel(f"{reduce_algo.upper()} 1")
# #         ax.set_ylabel(f"{reduce_algo.upper()} 2")
# #         ax.set_title(f"Highlighting: '{main_lab}'  —  {int(this_mask.sum())} points")
# #         ax.legend(handles=legend_elems, loc='upper right', framealpha=0.85)

# #         plt.tight_layout()
# #         fname = os.path.join(save_dir, f'highlight_{main_lab}.png')
# #         plt.savefig(fname, dpi=200, bbox_inches='tight')
# #         if show:
# #             plt.show()
# #         plt.close()

# #     # Pass text_embeddings so metrics file includes centroid↔text distances
# #     _text_embs_for_metrics = _get_clip_text_embeddings(clip_device) if plot_labels else None
# #     compute_cluster_metrics_to_file(all_tokens, save_dir, text_embeddings=_text_embs_for_metrics)
# #     return 0


# # # ── Example usage ───────────────────────────────────────────────────────────
# # # visualize_tokens(
# # #     all_tokens,              # list of (label_str, tensor)
# # #     tokenizer=None,
# # #     reduce_algo='umap',
# # #     highlight_each_label=True,
# # #     save_dir='viz_out',
# # #     plot_labels=True,        # CLIP loaded + cached automatically
# # # )


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from matplotlib import cm
# from matplotlib.lines import Line2D
# import umap.umap_ as umap
# from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
# from sklearn.metrics.pairwise import cosine_similarity
# import torch

# # ── Class names (fixed vocabulary) ─────────────────────────────────────────
# _CLASS_NAMES = [
#     "airplane flyby", "ambulance siren", "baby crying", "baby laughter", "car engine idling",
#     "cat purring", "chainsawing trees", "church bell ringing", "train whistling", "train horning",
#     "cow lowing", "dog barking", "elk bugling", "fire truck siren", "gibbon howling",
#     "hail", "lawn mowing", "volcano explosion", "stream burbling", "waterfall burbling",
#     "orchestra", "owl hooting", "people cheering", "people crowd", "people marching",
#     "playing bassoon", "playing drum kit", "playing harp", "playing harpsichord", "playing snare drum",
#     "playing timbales", "printer printing", "scuba diving", "sea waves", "sheep bleating",
#     "singing choir", "skiing", "slot machine", "snake hissing", "tractor digging",
#     "underwater bubbling", "railroad car, train wagon", "train wheels squealing", "baltimore oriole calling",
#     "car engine knocking", "driving snowmobile", "wood thrush calling", "hedge trimmer running",
#     "ice cream truck, ice cream van", "woodpecker pecking tree"
# ]


# def _get_lm_text_embeddings(model, tokenizer) -> dict:
#     """
#     For each class name, tokenize → look up in the model's own input embedding
#     table → mean pool over tokens → returns dict {label: np.ndarray (D,)}.

#     Produces vectors in the EXACT same space as the audio embedder output
#     (both target the LLM input embedding table), making L2 and cosine
#     distances geometrically meaningful.

#     No transformer forward pass — purely a table lookup + mean pool.
#     """
#     # Unwrap DDP / accelerate wrappers if present
#     try:
#         text_encoder = model.text_encoder
#     except AttributeError:
#         text_encoder = model

#     embed_table = text_encoder.get_input_embeddings().weight.data  # (vocab, D)

#     text_embs = {}
#     for name in _CLASS_NAMES:
#         enc      = tokenizer(name, return_tensors="pt", add_special_tokens=True)
#         ids      = enc["input_ids"].squeeze(0)           # (T,)
#         vecs     = embed_table[ids]                      # (T, D)
#         mean_vec = vecs.mean(dim=0).cpu().numpy()        # (D,)
#         text_embs[name] = mean_vec.astype(np.float32)

#     return text_embs


# def _centroid_distances(centroid_raw, t_arr):
#     """
#     L2 and cosine distance between a raw (un-normalised) centroid and a
#     text embedding vector. Centroid is normalised on the fly for cosine.
#     Both values are in [0, 2] for unit-norm inputs.
#     """
#     l2  = float(np.linalg.norm(centroid_raw - t_arr))
#     c_n = centroid_raw / (np.linalg.norm(centroid_raw) + 1e-8)
#     t_n = t_arr        / (np.linalg.norm(t_arr)        + 1e-8)
#     cos = float(1.0 - np.dot(c_n, t_n))
#     return l2, cos


# def _pool_and_flatten(tok):
#     """
#     Convert a raw token tensor/array to a numpy float32 array.

#     Shapes handled
#     --------------
#     [B, T, D]  BEATs style → mean pool over T → (B, D)  [caller unrolls rows]
#     [T, D]     single sample with sequence dim → mean pool → (D,)
#     [D]        already flat
#     """
#     try:
#         arr = tok.detach().cpu().numpy()
#     except Exception:
#         arr = np.array(tok)

#     arr = arr.astype(np.float32)

#     if arr.ndim == 3:
#         arr = arr.mean(axis=1)   # (B, T, D) → (B, D)
#     elif arr.ndim == 2:
#         arr = arr.mean(axis=0)   # (T, D)    → (D,)
#     # ndim == 1 → already fine

#     return arr


# def compute_cluster_metrics_to_file(all_tokens, save_dir,
#                                     text_embeddings=None):
#     """
#     Compute and write cluster quality metrics to <save_dir>/cluster_metrics.txt.

#     Parameters
#     ----------
#     all_tokens      : list of (label_str, tensor)
#     save_dir        : output directory
#     text_embeddings : dict {label_str: np.ndarray (D,)} — optional.
#                       When provided, also reports centroid ↔ text embedding
#                       L2 and cosine distances per class and in aggregate.
#                       Build with _get_lm_text_embeddings(model, tokenizer).
#     """
#     if len(all_tokens) == 0:
#         raise ValueError("all_tokens is empty")

#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "cluster_metrics.txt")

#     labels   = []
#     emb_list = []

#     for lab, tok in all_tokens:
#         labels.append(str(lab))

#         try:
#             arr = tok.detach().cpu().numpy()
#         except Exception:
#             arr = np.array(tok)

#         if arr.ndim == 3:
#             arr = arr.mean(axis=1)   # BEATs mean pool

#         emb_list.append(arr.reshape(-1).astype(np.float32))

#     X_raw  = np.stack(emb_list)                                       # (N, D) raw
#     labels = np.array(labels)

#     # Per-sample unit normalisation for silhouette / DB / intra cosine metrics
#     X = X_raw / np.linalg.norm(X_raw, axis=1, keepdims=True)         # (N, D) normed

#     unique_labels = sorted(set(labels))
#     label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
#     y             = np.array([label_to_idx[l] for l in labels])

#     global_sil  = float(silhouette_score(X, y))
#     db_index    = float(davies_bouldin_score(X, y))
#     sil_samples = silhouette_samples(X, y)

#     with open(save_path, "w") as f:

#         f.write("===== GLOBAL CLUSTER METRICS =====\n")
#         f.write(f"Global Silhouette Score: {global_sil:.6f}\n")
#         f.write(f"Davies-Bouldin Index:    {db_index:.6f}\n\n")

#         f.write("===== PER-LABEL METRICS =====\n\n")

#         for lab in unique_labels:
#             mask      = labels == lab
#             X_lab     = X[mask]          # normalised — for cosine metrics
#             X_lab_raw = X_raw[mask]      # raw — for centroid computation
#             sil_lab   = sil_samples[mask]
#             n         = X_lab.shape[0]

#             # Centroid on raw embeddings; normalise only for WCV computation
#             centroid_raw  = X_lab_raw.mean(axis=0)
#             centroid_norm = centroid_raw / (np.linalg.norm(centroid_raw) + 1e-8)
#             wcv           = np.mean(np.sum((X_lab - centroid_norm) ** 2, axis=1))

#             if n > 1:
#                 cos_sim  = cosine_similarity(X_lab)
#                 cos_sim  = cos_sim[~np.eye(n, dtype=bool)]
#                 mean_cos = float(np.mean(cos_sim))
#             else:
#                 mean_cos = 0.0

#             mean_sil = float(np.mean(sil_lab))

#             # Centroid ↔ text embedding distances (same embedding space)
#             c2t_l2, c2t_cos = None, None
#             if text_embeddings is not None:
#                 t_emb = text_embeddings.get(lab)
#                 if t_emb is not None:
#                     t_arr = np.array(t_emb, dtype=np.float32).reshape(-1)
#                     if t_arr.shape == centroid_raw.shape:
#                         c2t_l2, c2t_cos = _centroid_distances(centroid_raw, t_arr)
#                     else:
#                         print(f"[WARN] Dim mismatch for '{lab}': "
#                               f"centroid {centroid_raw.shape} vs text {t_arr.shape}")

#             f.write(f"Label: {lab}\n")
#             f.write(f"  Samples:                      {n}\n")
#             f.write(f"  Within Cluster Variance:      {wcv:.6f}\n")
#             f.write(f"  Mean Intra Cosine Similarity: {mean_cos:.6f}\n")
#             f.write(f"  Mean Silhouette Score:        {mean_sil:.6f}\n")
#             if c2t_l2 is not None:
#                 f.write(f"  Centroid→Text L2 Distance:    {c2t_l2:.6f}\n")
#                 f.write(f"  Centroid→Text Cosine Dist:    {c2t_cos:.6f}\n")
#             f.write("\n")

#         # ── Aggregate centroid ↔ text distance stats across all classes ────
#         if text_embeddings is not None:
#             per_l2, per_cos = [], []

#             for lab in unique_labels:
#                 centroid_raw = X_raw[labels == lab].mean(axis=0)
#                 t_emb = text_embeddings.get(lab)
#                 if t_emb is not None:
#                     t_arr = np.array(t_emb, dtype=np.float32).reshape(-1)
#                     if t_arr.shape == centroid_raw.shape:
#                         l2, cos = _centroid_distances(centroid_raw, t_arr)
#                         per_l2.append(l2)
#                         per_cos.append(cos)

#             if per_l2:
#                 f.write("===== CENTROID → TEXT EMBEDDING DISTANCE (AGGREGATE) =====\n")
#                 f.write(f"  Classes with valid distances: {len(per_l2)}\n\n")
#                 f.write(f"  L2 Distance     — "
#                         f"Mean: {np.mean(per_l2):.6f}  "
#                         f"Var: {np.var(per_l2):.6f}  "
#                         f"Min: {np.min(per_l2):.6f}  "
#                         f"Max: {np.max(per_l2):.6f}\n")
#                 f.write(f"  Cosine Distance — "
#                         f"Mean: {np.mean(per_cos):.6f}  "
#                         f"Var: {np.var(per_cos):.6f}  "
#                         f"Min: {np.min(per_cos):.6f}  "
#                         f"Max: {np.max(per_cos):.6f}\n")

#     print(f"Cluster metrics saved to: {save_path}")


# def visualize_tokens(
#     all_tokens,
#     reduce_algo='pca',
#     n_components=2,
#     highlight_each_label=True,
#     save_dir=None,
#     save_path=None,
#     show=False,
#     random_state=42,
#     highlight_marker_size=120,
#     other_marker_size=30,
#     highlight_alpha=0.95,
#     other_alpha=0.12,
#     # ── Label markers ──────────────────────────────────────────────────────
#     plot_labels=False,   # mark cluster centroid (★) + LM text embedding (◆)
#     model=None,          # required when plot_labels=True
#     tokenizer=None,      # required when plot_labels=True
# ):
#     """
#     Parameters
#     ----------
#     plot_labels : bool
#         If True, each per-label highlight plot also marks:
#           ★  gold star   — cluster centroid (mean of cluster in reduced space)
#           ◆  red diamond — LLM input-embedding table vector for the label text
#           -- dashed line connecting both
#         Requires `model` and `tokenizer`.

#     model : nn.Module
#         The full model. Must expose model.text_encoder.get_input_embeddings().
#         Accelerate / DDP wrappers are unwrapped automatically.

#     tokenizer : transformers tokenizer
#         Tokenizer matching the LLM text encoder.
#     """
#     if len(all_tokens) == 0:
#         raise ValueError("all_tokens is empty")

#     if plot_labels and (model is None or tokenizer is None):
#         raise ValueError("`model` and `tokenizer` must be provided when `plot_labels=True`")

#     # ── 1. Extract labels + embeddings ─────────────────────────────────────
#     labels   = []
#     emb_list = []

#     for lab, tok in all_tokens:
#         arr = _pool_and_flatten(tok)

#         if arr.ndim == 2:
#             # BEATs batched [B, D] — each row is its own sample
#             for row in arr:
#                 labels.append(str(lab))
#                 emb_list.append(row)
#         else:
#             labels.append(str(lab))
#             emb_list.append(arr)

#     X      = np.stack(emb_list, axis=0)   # (N, D)
#     labels = np.array(labels)

#     # ── 2. Standardise ─────────────────────────────────────────────────────
#     scaler   = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # ── 3. Fit reducer on audio embeddings ONLY ────────────────────────────
#     D      = X_scaled.shape[1]
#     n_comp = min(n_components, D) if reduce_algo.lower() == 'pca' else n_components

#     if reduce_algo.lower() == 'pca':
#         reducer = PCA(n_components=n_comp, random_state=random_state)
#     elif reduce_algo.lower() == 'umap':
#         reducer = umap.UMAP(n_components=n_comp, random_state=random_state)
#     else:
#         raise ValueError("reduce_algo must be 'pca' or 'umap'")

#     X_reduced = reducer.fit_transform(X_scaled)   # (N, n_comp)

#     if X_reduced.shape[1] < 2:
#         raise ValueError("n_components must be >= 2 for plotting")

#     # ── 4. Build + project LM text embeddings into the SAME fitted space ───
#     text_reduced   = {}    # label → (n_comp,) in reduced space
#     text_embs_dict = None  # kept for passing to metrics function

#     if plot_labels:
#         text_embs_dict = _get_lm_text_embeddings(model, tokenizer)

#         for lab, t_emb in text_embs_dict.items():
#             t_arr = np.array(t_emb, dtype=np.float32).reshape(1, -1)

#             if t_arr.shape[1] != X.shape[1]:
#                 print(f"[WARN] Text emb dim {t_arr.shape[1]} != audio dim "
#                       f"{X.shape[1]} for '{lab}'; skipping plot marker.")
#                 continue

#             t_scaled = scaler.transform(t_arr)

#             if reduce_algo.lower() == 'pca':
#                 t_proj = reducer.transform(t_scaled)[0]
#             else:
#                 try:
#                     t_proj = reducer.transform(t_scaled)[0]
#                 except Exception:
#                     t_proj = None
#                     print(f"[WARN] UMAP transform failed for '{lab}'; "
#                           "text marker skipped.")

#             if t_proj is not None:
#                 text_reduced[str(lab)] = t_proj

#     # ── 5. Label bookkeeping + colours ─────────────────────────────────────
#     unique_labels = sorted(list(set(labels)))
#     label_to_idx  = {lab: i for i, lab in enumerate(unique_labels)}
#     label_idx     = np.array([label_to_idx[lab] for lab in labels])

#     n_colors     = len(unique_labels)
#     base_cmap    = cm.get_cmap('tab20', n_colors) if n_colors <= 20 \
#         else cm.get_cmap('hsv', n_colors)
#     label_colors = [base_cmap(i % base_cmap.N) for i in range(n_colors)]

#     # ── 6a. Combined plot (no per-label highlighting) ──────────────────────
#     if not highlight_each_label:
#         plt.figure(figsize=(10, 8))
#         ax     = plt.gca()
#         colors = np.array([label_colors[idx] for idx in label_idx])
#         ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
#                    c=colors, s=45, alpha=0.95, edgecolor='k', linewidth=0.18)
#         ax.set_xlabel(f"{reduce_algo.upper()} 1")
#         ax.set_ylabel(f"{reduce_algo.upper()} 2")
#         ax.set_title(f"{reduce_algo.upper()} (n_comp={X_reduced.shape[1]}) — coloured by label")

#         legend_elems = []
#         for lab, idx in list(label_to_idx.items())[:50]:
#             legend_elems.append(
#                 Line2D([0], [0], marker='o', color='w',
#                        markerfacecolor=label_colors[idx % len(label_colors)],
#                        markersize=8, markeredgecolor='k', markeredgewidth=0.3,
#                        label=str(lab))
#             )
#         if legend_elems:
#             ax.legend(handles=legend_elems,
#                       bbox_to_anchor=(1.05, 1), loc='upper left', title='Labels')

#         plt.tight_layout()
#         out = save_path or 'visualization_all_labels.png'
#         plt.savefig(out, dpi=200, bbox_inches='tight')
#         if show:
#             plt.show()
#         plt.close()
#         return

#     # ── 6b. Per-label highlight plots ──────────────────────────────────────
#     if save_dir is None:
#         save_dir = 'visualizations'
#     os.makedirs(save_dir, exist_ok=True)

#     for main_lab in unique_labels:
#         fig, ax = plt.subplots(figsize=(10, 8))

#         # Background — all points grey
#         ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
#                    c='grey', s=other_marker_size, alpha=0.7,
#                    edgecolor='white', linewidth=0.2)

#         # Highlighted cluster points
#         main_idx      = label_to_idx[main_lab]
#         this_mask     = (label_idx == main_idx)
#         cluster_color = label_colors[main_idx % len(label_colors)]

#         if this_mask.any():
#             ax.scatter(X_reduced[this_mask, 0], X_reduced[this_mask, 1],
#                        c=[cluster_color],
#                        s=highlight_marker_size, alpha=highlight_alpha,
#                        edgecolor='k', linewidth=0.3)

#         legend_elems = [
#             Line2D([0], [0], marker='o', color='w',
#                    markerfacecolor=cluster_color, markersize=9,
#                    markeredgecolor='k', markeredgewidth=0.4,
#                    label=f'{main_lab} samples ({int(this_mask.sum())})')
#         ]

#         # ── Extra markers when plot_labels=True ────────────────────────────
#         if plot_labels and this_mask.any():

#             # ① Cluster centroid — gold star
#             centroid_2d = X_reduced[this_mask].mean(axis=0)
#             ax.scatter(centroid_2d[0], centroid_2d[1],
#                        marker='*', s=400, c='gold',
#                        edgecolor='black', linewidth=0.8, zorder=5)
#             legend_elems.append(
#                 Line2D([0], [0], marker='*', color='w',
#                        markerfacecolor='gold', markersize=14,
#                        markeredgecolor='black', markeredgewidth=0.8,
#                        label='Cluster centroid')
#             )

#             # ② LM text embedding — red diamond
#             t_proj = text_reduced.get(str(main_lab))
#             if t_proj is not None:
#                 ax.scatter(t_proj[0], t_proj[1],
#                            marker='D', s=220, c='red',
#                            edgecolor='black', linewidth=0.8, zorder=6)
#                 legend_elems.append(
#                     Line2D([0], [0], marker='D', color='w',
#                            markerfacecolor='red', markersize=10,
#                            markeredgecolor='black', markeredgewidth=0.8,
#                            label='LM text embedding')
#                 )

#                 # ③ Dashed line connecting centroid → text embedding
#                 ax.plot([centroid_2d[0], t_proj[0]],
#                         [centroid_2d[1], t_proj[1]],
#                         linestyle='--', color='red', linewidth=0.9,
#                         alpha=0.6, zorder=4)

#         # ── Axes & legend ──────────────────────────────────────────────────
#         ax.set_xlabel(f"{reduce_algo.upper()} 1")
#         ax.set_ylabel(f"{reduce_algo.upper()} 2")
#         ax.set_title(f"Highlighting: '{main_lab}'  —  {int(this_mask.sum())} points")
#         ax.legend(handles=legend_elems, loc='upper right', framealpha=0.85)

#         plt.tight_layout()
#         fname = os.path.join(save_dir, f'highlight_{main_lab}.png')
#         plt.savefig(fname, dpi=200, bbox_inches='tight')
#         if show:
#             plt.show()
#         plt.close()

#     # Reuse already-computed text_embs_dict — no second lookup
#     compute_cluster_metrics_to_file(
#         all_tokens, save_dir,
#         text_embeddings=text_embs_dict   # None when plot_labels=False
#     )
#     return 0


# # ── Example usage ───────────────────────────────────────────────────────────
# # visualize_tokens(
# #     all_tokens,
# #     reduce_algo='umap',
# #     highlight_each_label=True,
# #     save_dir='viz_out',
# #     plot_labels=True,
# #     model=model,           # full model with .text_encoder
# #     tokenizer=tokenizer,   # matching tokenizer
# # )





import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
from matplotlib.lines import Line2D
import umap.umap_ as umap
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import torch

# ── Class names (fixed vocabulary) ─────────────────────────────────────────
_CLASS_NAMES = [
    "airplane flyby", "ambulance siren", "baby crying", "baby laughter", "car engine idling",
    "cat purring", "chainsawing trees", "church bell ringing", "train whistling", "train horning",
    "cow lowing", "dog barking", "elk bugling", "fire truck siren", "gibbon howling",
    "hail", "lawn mowing", "volcano explosion", "stream burbling", "waterfall burbling",
    "orchestra", "owl hooting", "people cheering", "people crowd", "people marching",
    "playing bassoon", "playing drum kit", "playing harp", "playing harpsichord", "playing snare drum",
    "playing timbales", "printer printing", "scuba diving", "sea waves", "sheep bleating",
    "singing choir", "skiing", "slot machine", "snake hissing", "tractor digging",
    "underwater bubbling", "railroad car, train wagon", "train wheels squealing", "baltimore oriole calling",
    "car engine knocking", "driving snowmobile", "wood thrush calling", "hedge trimmer running",
    "ice cream truck, ice cream van", "woodpecker pecking tree"
]


def _get_lm_text_embeddings(model, tokenizer) -> dict:
    """
    Tokenize each class name → look up in model's input embedding table
    → mean pool over tokens → dict {label: np.ndarray (D,)}.
    """
    try:
        text_encoder = model.text_encoder
    except AttributeError:
        text_encoder = model

    embed_table = text_encoder.get_input_embeddings().weight.data  # (vocab, D)

    text_embs = {}
    for name in _CLASS_NAMES:
        enc      = tokenizer(name, return_tensors="pt", add_special_tokens=True)
        ids      = enc["input_ids"].squeeze(0)
        vecs     = embed_table[ids]
        mean_vec = vecs.mean(dim=0).cpu().numpy()
        text_embs[name] = mean_vec.astype(np.float32)

    return text_embs


def _centroid_distances(centroid_raw, t_arr):
    l2  = float(np.linalg.norm(centroid_raw - t_arr))
    c_n = centroid_raw / (np.linalg.norm(centroid_raw) + 1e-8)
    t_n = t_arr        / (np.linalg.norm(t_arr)        + 1e-8)
    cos = float(1.0 - np.dot(c_n, t_n))
    return l2, cos


def _pool_and_flatten(tok):
    """
    [B, T, D] → mean over T → (B, D)   [BEATs batched]
    [T, D]    → mean over T → (D,)     [single sample]
    [D]       → unchanged
    """
    try:
        arr = tok.detach().cpu().numpy()
    except Exception:
        arr = np.array(tok)

    arr = arr.astype(np.float32)

    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    elif arr.ndim == 2:
        arr = arr.mean(axis=0)

    return arr


def _extract_labels_and_embeddings(all_tokens):
    """Unroll all_tokens into parallel lists of labels and flat embeddings."""
    labels   = []
    emb_list = []
    for lab, tok in all_tokens:
        arr = _pool_and_flatten(tok)
        if arr.ndim == 2:
            for row in arr:
                labels.append(str(lab))
                emb_list.append(row)
        else:
            labels.append(str(lab))
            emb_list.append(arr)
    return np.array(labels), np.stack(emb_list, axis=0)


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_cluster_metrics_to_file(all_tokens_beats, all_tokens_clap, save_dir,
                                    text_embeddings=None):
    """
    Compute and save cluster metrics for BEATs and CLAP tokens separately
    into <save_dir>/cluster_metrics_beats.txt and cluster_metrics_clap.txt.
    """
    os.makedirs(save_dir, exist_ok=True)

    for tag, all_tokens in [("beats", all_tokens_beats), ("clap", all_tokens_clap)]:
        if len(all_tokens) == 0:
            print(f"[WARN] all_tokens_{tag} is empty — skipping metrics.")
            continue

        save_path = os.path.join(save_dir, f"cluster_metrics_{tag}.txt")

        labels, emb_list_raw = [], []
        for lab, tok in all_tokens:
            labels.append(str(lab))
            try:
                arr = tok.detach().cpu().numpy()
            except Exception:
                arr = np.array(tok)
            if arr.ndim == 3:
                arr = arr.mean(axis=1)
            emb_list_raw.append(arr.reshape(-1).astype(np.float32))

        X_raw  = np.stack(emb_list_raw)
        labels = np.array(labels)
        X      = X_raw / np.linalg.norm(X_raw, axis=1, keepdims=True)

        unique_labels = sorted(set(labels))
        label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
        y             = np.array([label_to_idx[l] for l in labels])

        global_sil  = float(silhouette_score(X, y))
        db_index    = float(davies_bouldin_score(X, y))
        sil_samples = silhouette_samples(X, y)

        with open(save_path, "w") as f:
            f.write(f"===== GLOBAL CLUSTER METRICS ({tag.upper()}) =====\n")
            f.write(f"Global Silhouette Score: {global_sil:.6f}\n")
            f.write(f"Davies-Bouldin Index:    {db_index:.6f}\n\n")
            f.write("===== PER-LABEL METRICS =====\n\n")

            for lab in unique_labels:
                mask      = labels == lab
                X_lab     = X[mask]
                X_lab_raw = X_raw[mask]
                sil_lab   = sil_samples[mask]
                n         = X_lab.shape[0]

                centroid_raw  = X_lab_raw.mean(axis=0)
                centroid_norm = centroid_raw / (np.linalg.norm(centroid_raw) + 1e-8)
                wcv           = np.mean(np.sum((X_lab - centroid_norm) ** 2, axis=1))

                if n > 1:
                    cos_sim  = cosine_similarity(X_lab)
                    cos_sim  = cos_sim[~np.eye(n, dtype=bool)]
                    mean_cos = float(np.mean(cos_sim))
                else:
                    mean_cos = 0.0

                mean_sil = float(np.mean(sil_lab))

                c2t_l2, c2t_cos = None, None
                if text_embeddings is not None:
                    t_emb = text_embeddings.get(lab)
                    if t_emb is not None:
                        t_arr = np.array(t_emb, dtype=np.float32).reshape(-1)
                        if t_arr.shape == centroid_raw.shape:
                            c2t_l2, c2t_cos = _centroid_distances(centroid_raw, t_arr)
                        else:
                            print(f"[WARN] Dim mismatch for '{lab}' ({tag}): "
                                  f"centroid {centroid_raw.shape} vs text {t_arr.shape}")

                f.write(f"Label: {lab}\n")
                f.write(f"  Samples:                      {n}\n")
                f.write(f"  Within Cluster Variance:      {wcv:.6f}\n")
                f.write(f"  Mean Intra Cosine Similarity: {mean_cos:.6f}\n")
                f.write(f"  Mean Silhouette Score:        {mean_sil:.6f}\n")
                if c2t_l2 is not None:
                    f.write(f"  Centroid→Text L2 Distance:    {c2t_l2:.6f}\n")
                    f.write(f"  Centroid→Text Cosine Dist:    {c2t_cos:.6f}\n")
                f.write("\n")

            if text_embeddings is not None:
                per_l2, per_cos = [], []
                for lab in unique_labels:
                    centroid_raw = X_raw[labels == lab].mean(axis=0)
                    t_emb = text_embeddings.get(lab)
                    if t_emb is not None:
                        t_arr = np.array(t_emb, dtype=np.float32).reshape(-1)
                        if t_arr.shape == centroid_raw.shape:
                            l2, cos = _centroid_distances(centroid_raw, t_arr)
                            per_l2.append(l2)
                            per_cos.append(cos)

                if per_l2:
                    f.write("===== CENTROID → TEXT EMBEDDING DISTANCE (AGGREGATE) =====\n")
                    f.write(f"  Classes with valid distances: {len(per_l2)}\n\n")
                    f.write(f"  L2 Distance     — "
                            f"Mean: {np.mean(per_l2):.6f}  "
                            f"Var: {np.var(per_l2):.6f}  "
                            f"Min: {np.min(per_l2):.6f}  "
                            f"Max: {np.max(per_l2):.6f}\n")
                    f.write(f"  Cosine Distance — "
                            f"Mean: {np.mean(per_cos):.6f}  "
                            f"Var: {np.var(per_cos):.6f}  "
                            f"Min: {np.min(per_cos):.6f}  "
                            f"Max: {np.max(per_cos):.6f}\n")

        print(f"Cluster metrics saved to: {save_path}")


# ── Main visualisation ──────────────────────────────────────────────────────

def visualize_tokens(
    all_tokens_beats,
    all_tokens_clap,
    reduce_algo='pca',
    n_components=2,
    highlight_each_label=True,
    save_dir=None,
    save_path=None,
    show=False,
    random_state=42,
    highlight_marker_size=120,
    other_marker_size=20,
    highlight_alpha=0.90,
    other_alpha=0.10,
    # ── Label markers ──────────────────────────────────────────────────────
    plot_labels=False,   # mark cluster centroid (★) + LM text embedding (◆)
    model=None,          # required when plot_labels=True
    tokenizer=None,      # required when plot_labels=True
):
    """
    Produce a single scatter plot (per-label highlight or combined) with three
    kinds of points:

        ●  circle  — BEATs audio embeddings
        ▲  triangle — CLAP audio embeddings
        ◆  diamond (red) — LM text embedding for the label  [if plot_labels]

    Extra per-label markers when plot_labels=True:
        ★  gold star   — BEATs cluster centroid
        ☆  cyan star   — CLAP  cluster centroid
        --  dashed lines connecting each centroid to the text embedding

    Parameters
    ----------
    all_tokens_beats : list of (label_str, tensor)
    all_tokens_clap  : list of (label_str, tensor)
    reduce_algo      : 'pca' | 'umap'
    highlight_each_label : bool
        True  → one saved figure per label (highlighted cluster, grey rest)
        False → single combined figure coloured by label
    plot_labels      : bool  — add centroid + text embedding markers
    model            : nn.Module  — needed when plot_labels=True
    tokenizer        : transformers tokenizer — needed when plot_labels=True
    """
    if not all_tokens_beats and not all_tokens_clap:
        raise ValueError("Both all_tokens_beats and all_tokens_clap are empty")

    if plot_labels and (model is None or tokenizer is None):
        raise ValueError("`model` and `tokenizer` must be provided when `plot_labels=True`")

    # ── 1. Extract labels + embeddings for each modality ───────────────────
    labels_b, X_b = _extract_labels_and_embeddings(all_tokens_beats) \
        if all_tokens_beats else (np.array([]), np.empty((0,)))
    labels_c, X_c = _extract_labels_and_embeddings(all_tokens_clap) \
        if all_tokens_clap else (np.array([]), np.empty((0,)))

    # Sanity-check dimensionality matches across modalities
    if X_b.size and X_c.size and X_b.shape[1] != X_c.shape[1]:
        raise ValueError(
            f"BEATs embedding dim ({X_b.shape[1]}) ≠ CLAP embedding dim ({X_c.shape[1]}). "
            "Both must share the same LLM token dimension for joint projection."
        )

    # ── 2. Stack everything for a SINGLE shared reducer ────────────────────
    n_beats = X_b.shape[0] if X_b.size else 0
    n_clap  = X_c.shape[0] if X_c.size else 0

    X_all    = np.vstack([x for x in [X_b, X_c] if x.size])   # (N_b+N_c, D)
    labels_all = np.concatenate([l for l in [labels_b, labels_c] if l.size])

    # ── 3. Standardise on the full joint set ───────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # ── 4. Fit reducer ─────────────────────────────────────────────────────
    D      = X_scaled.shape[1]
    n_comp = min(n_components, D) if reduce_algo.lower() == 'pca' else n_components

    if reduce_algo.lower() == 'pca':
        reducer = PCA(n_components=n_comp, random_state=random_state)
    elif reduce_algo.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_comp, random_state=random_state)
    else:
        raise ValueError("reduce_algo must be 'pca' or 'umap'")

    X_reduced = reducer.fit_transform(X_scaled)   # (N_b+N_c, n_comp)

    if X_reduced.shape[1] < 2:
        raise ValueError("n_components must be >= 2 for plotting")

    # Split back into per-modality arrays
    X_b_red = X_reduced[:n_beats]                 # (N_b, 2)
    X_c_red = X_reduced[n_beats:]                 # (N_c, 2)

    # ── 5. Project LM text embeddings into the SAME fitted space ───────────
    text_reduced   = {}
    text_embs_dict = None

    if plot_labels:
        text_embs_dict = _get_lm_text_embeddings(model, tokenizer)

        for lab, t_emb in text_embs_dict.items():
            t_arr = np.array(t_emb, dtype=np.float32).reshape(1, -1)

            if t_arr.shape[1] != X_all.shape[1]:
                print(f"[WARN] Text emb dim {t_arr.shape[1]} ≠ audio dim "
                      f"{X_all.shape[1]} for '{lab}'; skipping text marker.")
                continue

            t_scaled = scaler.transform(t_arr)

            if reduce_algo.lower() == 'pca':
                t_proj = reducer.transform(t_scaled)[0]
            else:
                try:
                    t_proj = reducer.transform(t_scaled)[0]
                except Exception:
                    t_proj = None
                    print(f"[WARN] UMAP transform failed for '{lab}'; text marker skipped.")

            if t_proj is not None:
                text_reduced[str(lab)] = t_proj

    # ── 6. Label bookkeeping + colours ─────────────────────────────────────
    unique_labels = sorted(set(labels_all.tolist()))
    label_to_idx  = {lab: i for i, lab in enumerate(unique_labels)}

    n_colors     = len(unique_labels)
    base_cmap    = cm.get_cmap('tab20', n_colors) if n_colors <= 20 \
        else cm.get_cmap('hsv', n_colors)
    label_colors = [base_cmap(i % base_cmap.N) for i in range(n_colors)]

    label_idx_b = np.array([label_to_idx[l] for l in labels_b]) if n_beats else np.array([])
    label_idx_c = np.array([label_to_idx[l] for l in labels_c]) if n_clap  else np.array([])

    # Shared marker / style constants
    MARKER_BEATS = 'o'    # circle
    MARKER_CLAP  = '^'    # triangle-up
    MARKER_TEXT  = 'D'    # diamond

    # ── 7a. Combined plot (all labels, one figure) ─────────────────────────
    if not highlight_each_label:
        fig, ax = plt.subplots(figsize=(12, 9))

        if n_beats:
            colors_b = [label_colors[i] for i in label_idx_b]
            ax.scatter(X_b_red[:, 0], X_b_red[:, 1],
                       c=colors_b, marker=MARKER_BEATS,
                       s=45, alpha=0.85, edgecolor='k', linewidth=0.18,
                       label='_nolegend_')

        if n_clap:
            colors_c = [label_colors[i] for i in label_idx_c]
            ax.scatter(X_c_red[:, 0], X_c_red[:, 1],
                       c=colors_c, marker=MARKER_CLAP,
                       s=45, alpha=0.85, edgecolor='k', linewidth=0.18,
                       label='_nolegend_')

        ax.set_xlabel(f"{reduce_algo.upper()} 1")
        ax.set_ylabel(f"{reduce_algo.upper()} 2")
        ax.set_title(
            f"{reduce_algo.upper()} — BEATs (●) + CLAP (▲) + Text (◆), coloured by label"
        )

        # Legend: one entry per label (colour), plus modality shape entries
        legend_elems = []
        for lab, idx in list(label_to_idx.items())[:50]:
            legend_elems.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=label_colors[idx % len(label_colors)],
                       markersize=8, markeredgecolor='k', markeredgewidth=0.3,
                       label=str(lab))
            )

        # Modality shape legend
        modality_elems = [
            Line2D([0], [0], marker=MARKER_BEATS, color='grey', linestyle='None',
                   markersize=9, markeredgecolor='k', markeredgewidth=0.5,
                   label='BEATs tokens'),
            Line2D([0], [0], marker=MARKER_CLAP, color='grey', linestyle='None',
                   markersize=9, markeredgecolor='k', markeredgewidth=0.5,
                   label='CLAP tokens'),
        ]

        # Second legend for modality shapes (top-right), label legend in separate box
        leg1 = ax.legend(handles=modality_elems, loc='upper right',
                         framealpha=0.85, title='Modality')
        ax.add_artist(leg1)
        if legend_elems:
            ax.legend(handles=legend_elems,
                      bbox_to_anchor=(1.05, 1), loc='upper left', title='Labels')

        plt.tight_layout()
        out = save_path or 'visualization_combined.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        compute_cluster_metrics_to_file(
            all_tokens_beats, all_tokens_clap, save_dir or os.path.dirname(out) or '.',
            text_embeddings=text_embs_dict
        )
        return

    # ── 7b. Per-label highlight plots ──────────────────────────────────────
    if save_dir is None:
        save_dir = 'visualizations'
    os.makedirs(save_dir, exist_ok=True)

    for main_lab in unique_labels:
        fig, ax = plt.subplots(figsize=(11, 8))

        main_idx = label_to_idx[main_lab]

        # ── Background: all BEATs (grey circles) ──────────────────────────
        if n_beats:
            ax.scatter(X_b_red[:, 0], X_b_red[:, 1],
                       c='grey', marker=MARKER_BEATS,
                       s=other_marker_size, alpha=other_alpha,
                       edgecolor='white', linewidth=0.15)

        # ── Background: all CLAP (grey triangles) ─────────────────────────
        if n_clap:
            ax.scatter(X_c_red[:, 0], X_c_red[:, 1],
                       c='grey', marker=MARKER_CLAP,
                       s=other_marker_size, alpha=other_alpha,
                       edgecolor='white', linewidth=0.15)

        cluster_color = label_colors[main_idx % len(label_colors)]
        legend_elems  = []

        # ── Highlighted BEATs points ───────────────────────────────────────
        mask_b = (label_idx_b == main_idx) if n_beats else np.array([], dtype=bool)
        if mask_b.any():
            ax.scatter(X_b_red[mask_b, 0], X_b_red[mask_b, 1],
                       c=[cluster_color], marker=MARKER_BEATS,
                       s=highlight_marker_size, alpha=highlight_alpha,
                       edgecolor='k', linewidth=0.4)
            legend_elems.append(
                Line2D([0], [0], marker=MARKER_BEATS, color='w',
                       markerfacecolor=cluster_color, markersize=10,
                       markeredgecolor='k', markeredgewidth=0.5,
                       label=f'BEATs — {main_lab} ({int(mask_b.sum())})')
            )

        # ── Highlighted CLAP points ────────────────────────────────────────
        mask_c = (label_idx_c == main_idx) if n_clap else np.array([], dtype=bool)
        if mask_c.any():
            ax.scatter(X_c_red[mask_c, 0], X_c_red[mask_c, 1],
                       c=[cluster_color], marker=MARKER_CLAP,
                       s=highlight_marker_size, alpha=highlight_alpha,
                       edgecolor='k', linewidth=0.4)
            legend_elems.append(
                Line2D([0], [0], marker=MARKER_CLAP, color='w',
                       markerfacecolor=cluster_color, markersize=10,
                       markeredgecolor='k', markeredgewidth=0.5,
                       label=f'CLAP — {main_lab} ({int(mask_c.sum())})')
            )

        # ── Extra markers: centroids + text embedding ──────────────────────
        if plot_labels:

            # ① BEATs centroid — gold star
            if mask_b.any():
                c_b = X_b_red[mask_b].mean(axis=0)
                ax.scatter(c_b[0], c_b[1],
                           marker='*', s=450, c='gold',
                           edgecolor='black', linewidth=0.8, zorder=6)
                legend_elems.append(
                    Line2D([0], [0], marker='*', color='w',
                           markerfacecolor='gold', markersize=14,
                           markeredgecolor='black', markeredgewidth=0.8,
                           label='BEATs centroid')
                )

            # ② CLAP centroid — cyan star
            if mask_c.any():
                c_c = X_c_red[mask_c].mean(axis=0)
                ax.scatter(c_c[0], c_c[1],
                           marker='*', s=450, c='cyan',
                           edgecolor='black', linewidth=0.8, zorder=6)
                legend_elems.append(
                    Line2D([0], [0], marker='*', color='w',
                           markerfacecolor='cyan', markersize=14,
                           markeredgecolor='black', markeredgewidth=0.8,
                           label='CLAP centroid')
                )

            # ③ LM text embedding — red diamond
            t_proj = text_reduced.get(str(main_lab))
            if t_proj is not None:
                ax.scatter(t_proj[0], t_proj[1],
                           marker=MARKER_TEXT, s=250, c='red',
                           edgecolor='black', linewidth=0.8, zorder=7)
                legend_elems.append(
                    Line2D([0], [0], marker=MARKER_TEXT, color='w',
                           markerfacecolor='red', markersize=10,
                           markeredgecolor='black', markeredgewidth=0.8,
                           label='LM text embedding')
                )

                # Dashed lines from each centroid to text embedding
                if mask_b.any():
                    ax.plot([c_b[0], t_proj[0]], [c_b[1], t_proj[1]],
                            linestyle='--', color='goldenrod', linewidth=0.9,
                            alpha=0.65, zorder=5)
                if mask_c.any():
                    ax.plot([c_c[0], t_proj[0]], [c_c[1], t_proj[1]],
                            linestyle='--', color='teal', linewidth=0.9,
                            alpha=0.65, zorder=5)

        # ── Axes & legend ──────────────────────────────────────────────────
        ax.set_xlabel(f"{reduce_algo.upper()} 1")
        ax.set_ylabel(f"{reduce_algo.upper()} 2")
        n_b_str = int(mask_b.sum()) if mask_b.size else 0
        n_c_str = int(mask_c.sum()) if mask_c.size else 0
        ax.set_title(
            f"Highlighting: '{main_lab}'  —  "
            f"BEATs: {n_b_str}  CLAP: {n_c_str}"
        )
        ax.legend(handles=legend_elems, loc='upper right', framealpha=0.88)

        plt.tight_layout()
        safe_name = main_lab.replace(' ', '_').replace(',', '')
        fname = os.path.join(save_dir, f'highlight_{safe_name}.png')
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    compute_cluster_metrics_to_file(
        all_tokens_beats, all_tokens_clap, save_dir,
        text_embeddings=text_embs_dict
    )
    return 0


# ── Example usage ───────────────────────────────────────────────────────────
# visualize_tokens(
#     all_tokens_beats,
#     all_tokens_clap,
#     reduce_algo='umap',
#     highlight_each_label=True,
#     save_dir='viz_out',
#     plot_labels=True,
#     model=model,
#     tokenizer=tokenizer,
# )