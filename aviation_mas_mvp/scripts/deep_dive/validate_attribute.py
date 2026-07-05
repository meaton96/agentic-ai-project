# """Controlled test: build a world where the maintenance label IS the cyl-3 fault,
# train the pipeline's model on it, and confirm occlusion attribution fingers the
# E1 EGT3 / E1 CHT3 channels on a held-out flagged flight."""
# import sys; sys.path.insert(0, "scripts")
# import numpy as np, pandas as pd
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.metrics import roc_auc_score
# from make_synth_flight import make_flight, SENSORS
# from attribute import attribute_prediction, compute_background

# def channel_features(v):
#     x = np.asarray(v, float); x = x[np.isfinite(x)]
#     if x.size == 0: return {k: 0.0 for k in
#         ["mean","std","min","max","range","p10","p50","p90","slope","mean_abs_diff","max_abs_diff","last"]}
#     d = np.diff(x) if x.size > 1 else np.array([0.0])
#     sl = float(np.polyfit(np.arange(x.size), x, 1)[0]) if x.size > 1 else 0.0
#     return {"mean":x.mean(),"std":x.std(),"min":x.min(),"max":x.max(),"range":np.ptp(x),
#             "p10":np.percentile(x,10),"p50":np.percentile(x,50),"p90":np.percentile(x,90),
#             "slope":sl,"mean_abs_diff":np.abs(d).mean(),"max_abs_diff":np.abs(d).max(),"last":x[-1]}

# def featurize(df):
#     row = {}
#     for c in SENSORS:
#         for k, val in channel_features(df[c].values).items():
#             row[f"{c}__{k}"] = val
#     row["__n_steps"] = len(df)
#     return row

# # --- build a labelled world: label ~ cyl-3 fault, with noise + variable severity ---
# rng = np.random.default_rng(0)
# rows, meta = [], []
# fid = 0
# for plane in range(14):
#     for _ in range(9):
#         has_fault = fid % 2                    # balanced
#         label = has_fault if rng.random() > 0.10 else 1 - has_fault   # 15% label noise
#         anom = None
#         if has_fault:
#             anom = {"cyl": 3, "egt_delta": float(rng.normal(80, 15)),
#                     "cht_delta": float(rng.normal(18, 4)), "phase": "climb"}
#         df = make_flight(seed=1000+fid, label=label, plane_id=plane, flight_id=fid, anomaly=anom)
#         r = featurize(df); r["label"]=label; r["fault"]=has_fault; r["plane"]=plane; r["fid"]=fid
#         rows.append(r); fid += 1
# tbl = pd.DataFrame(rows).fillna(0.0)
# feat_cols = [c for c in tbl.columns if c not in ("label","fault","plane","fid")]

# # hold out planes 12,13 (tail-disjoint), train on the rest
# te = tbl["plane"].isin([12,13]); tr = ~te
# clf = HistGradientBoostingClassifier(random_state=0).fit(tbl[tr][feat_cols].values, tbl[tr]["label"].values)
# bundle = {"model": clf, "feature_columns": feat_cols}
# auc = roc_auc_score(tbl[te]["label"].values, clf.predict_proba(tbl[te][feat_cols].values)[:,1])
# print(f"model held-out AUC = {auc:.3f}  (should be high; cyl-3 fault IS the label)")

# # attribute a held-out flight that is genuinely faulty AND flagged high
# background = compute_background(tbl[tr], feat_cols, normal_mask=(tbl[tr]["fault"]==0))
# cand = tbl[te & (tbl["fault"]==1)]
# probs = clf.predict_proba(cand[feat_cols].values)[:,1]
# flagged = cand.iloc[int(np.argmax(probs))]
# res = attribute_prediction(flagged[feat_cols], bundle, background, top_k=6)
# print(f"\nflagged faulty flight p_maintenance = {res['p_maintenance']}")
# print("top channels driving the flag:")
# for a in res["channel_attribution"][:6]:
#     print(f"  {a['channel']:12s} prob_drop={a['prob_drop']:+.3f}")
# ranks = [a["channel"] for a in res["channel_attribution"]]
# print(f"\ncyl-3 channels: E1 EGT3 rank #{ranks.index('E1 EGT3')+1}, "
#       f"E1 CHT3 rank #{ranks.index('E1 CHT3')+1}  (of {len(ranks)}; ground truth = cyl 3)")
