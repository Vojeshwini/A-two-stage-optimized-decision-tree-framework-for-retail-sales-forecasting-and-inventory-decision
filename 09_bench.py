"""Base-paper benchmarks (Naive, sNaive, MA, ES, CRO, ADIDA) on the same 9,119 series and test days,
one-step-ahead (next-day) rolling forecasts, scored with the base paper's RMSSE / SME, plus the inventory table."""
import numpy as np, pandas as pd
from scipy.stats import norm
S = "."   # project folder containing m5full/, figs/ and res/
cal = pd.read_csv(f"{S}/m5full/calendar.csv", parse_dates=["date"])
sales = pd.read_csv(f"{S}/m5full/sales_train_evaluation.csv")
pp = pd.read_csv(f"{S}/figs/figdata/per_product.csv")
sales = pp[["store_id", "item_id"]].merge(sales, on=["store_id", "item_id"], how="left")      # same order as pp
dcols = [c for c in sales.columns if c.startswith("d_")]
Y = sales[dcols].values.astype(np.float64)                     # N x 1941
dates = cal.date.iloc[:len(dcols)].values
T0 = int(np.where(dates == np.datetime64("2016-03-22"))[0][0]); T1 = T0 + 62
N, L = Y.shape; A = Y[:, T0:T1]
print("series", N, "test", T0, T1, L)
# ---- one-step-ahead forecasts F[:, t] for t in test (use data up to t-1) ----
def naive(): return Y[:, T0 - 1:T1 - 1]
def snaive(): return Y[:, T0 - 7:T1 - 7]
def ma():
    cs = np.concatenate([np.zeros((N, 1)), np.cumsum(Y, axis=1)], axis=1)
    tr0 = T0 - 365; best_k = np.full(N, 7); best = np.full(N, np.inf)
    for k in range(2, 15):                                     # k chosen per series by in-sample one-step MSE (M5 benchmark code)
        t = np.arange(tr0, T0); f = (cs[:, t] - cs[:, t - k]) / k
        mse = np.mean((Y[:, t] - f) ** 2, axis=1); m = mse < best; best[m] = mse[m]; best_k[m] = k
    t = np.arange(T0, T1); F = np.zeros((N, 62))
    for k in range(2, 15):
        m = best_k == k; F[m] = ((cs[:, t] - cs[:, t - k]) / k)[m]
    return F, best_k
def ses(alpha_grid=np.round(np.arange(0.05, 1.0, 0.05), 2)):
    tr0 = T0 - 365; best = np.full(N, np.inf); Fbest = np.zeros((N, 62))
    for a in alpha_grid:                                       # alpha chosen per series by in-sample one-step MSE
        lvl = Y[:, tr0 - 28:tr0].mean(axis=1); sse = np.zeros(N); F = np.zeros((N, 62))
        for t in range(tr0, T1):
            if t >= T0: F[:, t - T0] = lvl
            else: sse += (Y[:, t] - lvl) ** 2
            lvl = lvl + a * (Y[:, t] - lvl)
        m = sse < best; best[m] = sse[m]; Fbest[m] = F[m]
    return Fbest
def croston(a=0.1):
    z = np.full(N, np.nan); p = np.full(N, np.nan); q = np.ones(N); F = np.zeros((N, 62))
    for t in range(L if False else T1):
        if t >= T0: F[:, t - T0] = np.where(np.isnan(z), 0, z / p)
        if t < T1:
            y = Y[:, t]; pos = y > 0; first = pos & np.isnan(z)
            z[first] = y[first]; p[first] = q[first]
            upd = pos & ~first
            z[upd] = z[upd] + a * (y[upd] - z[upd]); p[upd] = p[upd] + a * (q[upd] - p[upd])
            q = np.where(pos, 1, q + 1)
    return F
def adida(a=0.1, b=7):
    # weekly (7-day) temporal aggregation, SES on the weekly totals, disaggregated evenly; one SES state per weekly phase
    cs = np.concatenate([np.zeros((N, 1)), np.cumsum(Y, axis=1)], axis=1)
    st = np.full((b, N), np.nan); F = np.zeros((N, 62))
    for t in range(b, T1):
        if t >= T0: F[:, t - T0] = np.nan_to_num(st[(t - 1) % b]) / b
        s = cs[:, t + 1] - cs[:, t + 1 - b]                     # sum of days t-b+1 .. t
        ph = t % b; st[ph] = np.where(np.isnan(st[ph]), s, st[ph] + a * (s - st[ph]))
    return F
FC = {"Naive": naive(), "sNaive": snaive()}
FC["MA"], ks = ma(); print("MA k distribution", np.bincount(ks)[2:])
FC["ES"] = ses(); FC["CRO"] = croston(); FC["ADIDA"] = adida()
# ---- model forecasts (same products, same days) ----
p3 = pd.read_csv(f"{S}/res/predictions_opt3_opt4_ens.csv.gz", parse_dates=["date"])
p5 = pd.read_csv(f"{S}/res/opt5/optimization5_predictions.csv.gz", parse_dates=["date"])
P = p3.merge(p5[["store_id", "item_id", "date", "pred"]], on=["store_id", "item_id", "date"])
P = pp[["store_id", "item_id"]].reset_index().merge(P, on=["store_id", "item_id"]).sort_values(["index", "date"])
assert len(P) == N * 62
assert np.allclose(P.sales.values.reshape(N, 62), A), "actuals mismatch"
FC["Opt 3"] = P.pred_3.values.reshape(N, 62); FC["Opt 4"] = P.pred_4.values.reshape(N, 62); FC["Opt 5"] = P.pred.values.reshape(N, 62)
FC["TSO-DTF (final)"] = (FC["Opt 4"] + FC["Opt 5"]) / 2
# ---- metrics: base-paper RMSSE (full-history scale) and SME (365-day MAD, as in the notebooks) ----
scale = pp.scale_full.values
mad = np.mean(np.abs(np.diff(Y[:, T0 - 365:T0], axis=1)), axis=1)
w = pp.weight.values
def score(F):
    r = np.sqrt(np.mean((A - F) ** 2, axis=1) / scale); s = np.mean(A - F, axis=1) / mad
    return r, s
chk, _ = score(FC["TSO-DTF (final)"]); print("check TSO mean RMSSE", chk.mean(), "stored", pp["rmsse|Ens 4+5"].mean())
_, s5 = score(FC["Opt 5"]); print("check SME opt5", np.nanmean(s5), "stored", pp["sme|Opt 5"].mean())
# ---- inventory: daily review, next-morning delivery, lost sales, 95% target, safety stock from past errors only ----
WARM, Z = 14, norm.ppf(0.95)
price = pd.read_csv(f"{S}/biz/biz_data.json") if False else None
def inventory(F):
    E = A - F; inv = np.zeros(N); O, I = [], []; so = np.zeros(N)
    for t in range(WARM, 62):
        sig = np.sqrt(np.mean(E[:, :t] ** 2, axis=1)); S_ = np.maximum(F[:, t], 0) + Z * sig
        o = np.maximum(0, np.ceil(S_ - inv)); av = inv + o; d = A[:, t]; so += d > av
        inv = av - np.minimum(av, d); O.append(o); I.append(inv.copy())
    O, I = np.array(O).T, np.array(I).T; D = A[:, WARM:]
    vd = D.var(axis=1); ok = vd > 0
    sl = 1 - so / (62 - WARM)
    return dict(inv=I.mean(axis=1).mean(), sl=sl.mean(), below=int((sl < 0.95).sum()),
                bwo=np.mean(O.var(axis=1)[ok] / vd[ok]), bwi=np.mean(I.var(axis=1)[ok] / vd[ok]))
rows = []
base_r = pp["rmsse|Baseline DTF"].values; base_s = pp["sme|Baseline DTF"].values
rows.append(dict(method="Baseline DTF (base paper)", RMSSE=base_r.mean(), WRMSSE=(w * base_r).sum() / w.sum(), SME=np.nanmean(base_s)))
order = ["TSO-DTF (final)", "Opt 5", "Opt 4", "Opt 3", "Naive", "sNaive", "MA", "ES", "CRO", "ADIDA"]
for m in order:
    r, s = score(FC[m]); inv = inventory(FC[m])
    rows.append(dict(method=m, RMSSE=r.mean(), WRMSSE=(w * r).sum() / w.sum(), SME=np.nanmean(s),
                     better_than_baseline=100 * np.mean(r < base_r), **inv))
R = pd.DataFrame(rows); b = R.loc[0, "RMSSE"]
R["impr_vs_baseline_%"] = 100 * (b - R.RMSSE) / b
R["impr_vs_ES_%"] = 100 * (R.loc[R.method == "ES", "RMSSE"].values[0] - R.RMSSE) / R.loc[R.method == "ES", "RMSSE"].values[0]
pd.set_option("display.width", 250); print(R.round(4).to_string())
R.to_csv(f"{S}/bench/benchmark_results.csv", index=False)
