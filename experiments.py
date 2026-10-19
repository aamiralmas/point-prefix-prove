"""CORA experiment driver. Usage:  python experiments.py [e1 e2 ...]
Writes JSON results to results/. All randomness from integer seeds."""
import json, os, sys, time
import numpy as np
from cora_sim import *

SEED = 20261017
N_SEEDS = 10
N_CAL, N_TEST = 500, 800
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def save(name, obj):
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)


def gen(cfg, rng, n):
    return [gen_failing_trace(cfg, rng) for _ in range(n)]


N_FIT = 300


def setup(cfg, seed, n_cal=N_CAL, n_test=N_TEST):
    """Three disjoint sets: N_FIT traces fit the likelihood tables and judge
    temperature; n_cal traces calibrate the conformal threshold; n_test
    traces are evaluated. Fitting and calibrating on the same traces would
    break exchangeability between calibration and test scores."""
    rng = np.random.default_rng(seed)
    fit = gen(cfg, rng, N_FIT)
    cal, test = gen(cfg, rng, n_cal), gen(cfg, rng, n_test)
    model = fit_cora(cfg, rng, fit)
    model.qhat, _ = calibrate_conformal(cfg, rng, cal, model, cfg.alpha)
    return rng, cal, test, model


def ms(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0]


# ------------------------------------------------------------------ E1
def e1():
    cfg = Config()
    names = ["AllAtOnce", "StepByStep", "BinarySearch", "Ensemble", "LOO-CF",
             "CORA"]
    acc = {n: {"agent": [], "step": [], "cost": [], "wall": []} for n in names}
    cset = {"cov": [], "acov": [], "size": [], "asize": [], "nrep": []}
    conf = {"CORA": np.zeros((N_ROLES, N_ROLES)),
            "Ensemble": np.zeros((N_ROLES, N_ROLES))}
    for k in range(N_SEEDS):
        rng, cal, test, model = setup(cfg, SEED + k)
        # tune step-by-step threshold on calibration
        best, tau = -1, 1.0
        for tt in np.linspace(0.4, 1.6, 13):
            a = np.mean([m_step_by_step(cfg, rng, tr, tt)[0] == tr.root
                         for tr in cal[:200]])
            if a > best:
                best, tau = a, tt
        meth = {"AllAtOnce": m_all_at_once,
                "StepByStep": lambda c, r, t: m_step_by_step(c, r, t, tau),
                "BinarySearch": m_binary, "Ensemble": m_ensemble,
                "LOO-CF": lambda c, r, t: m_loo(c, r, t, model)}
        for n, f in meth.items():
            out = [f(cfg, rng, tr) for tr in test]
            acc[n]["step"].append(np.mean([o[0] == tr.root for o, tr in zip(out, test)]))
            acc[n]["agent"].append(np.mean([tr.agent[o[0]] == tr.agent[tr.root]
                                            for o, tr in zip(out, test)]))
            acc[n]["cost"].append(np.mean([o[2] for o in out]))
            acc[n]["wall"].append(np.mean([o[1] for o in out]))
            if n == "Ensemble":
                for o, tr in zip(out, test):
                    conf[n][tr.agent[tr.root], tr.agent[o[0]]] += 1
        st, ag, co, wa, cv, acv, sz, asz, nr = ([] for _ in range(9))
        for tr in test:
            post, w, c, n_rep = cora_posterior(cfg, rng, tr, model)
            t = int(np.argmax(post))
            S = conformal_set(post, model.qhat)
            st.append(t == tr.root); ag.append(tr.agent[t] == tr.agent[tr.root])
            co.append(c); wa.append(w); cv.append(tr.root in S)
            A = set(int(a) for a in tr.agent[S])
            acv.append(int(tr.agent[tr.root]) in A); sz.append(len(S))
            asz.append(len(A)); nr.append(n_rep)
            conf["CORA"][tr.agent[tr.root], tr.agent[t]] += 1
        for key, v in zip(["step", "agent", "cost", "wall"], [st, ag, co, wa]):
            acc["CORA"][key].append(np.mean(v))
        for key, v in zip(["cov", "acov", "size", "asize", "nrep"],
                          [cv, acv, sz, asz, nr]):
            cset[key].append(np.mean(v))
        print("e1 seed", k, {n: round(acc[n]["step"][-1], 3) for n in names},
              "cov", round(cset["cov"][-1], 3), flush=True)
    res = {n: {k: ms(v) for k, v in d.items()} for n, d in acc.items()}
    res["CORA_set"] = {k: ms(v) for k, v in cset.items()}
    res["confusion"] = {k: (v / v.sum(1, keepdims=True).clip(1)).tolist()
                        for k, v in conf.items()}
    res["confusion_counts"] = {k: v.tolist() for k, v in conf.items()}
    save("e1", res)


# ------------------------------------------------------------------ E2
def e2():
    cfg = Config()
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    out = {m: {"cov": [], "size": []} for m in ["CORA", "JudgeConformal",
                                                 "Uncalibrated"]}
    for k in range(5):
        rng = np.random.default_rng(SEED + 100 + k)
        fit = gen(cfg, rng, N_FIT)
        cal, test = gen(cfg, rng, N_CAL), gen(cfg, rng, N_TEST)
        model = fit_cora(cfg, rng, fit)
        posts = {}
        for name, kw in [("CORA", {}), ("JudgeConformal", {"use_replay": False})]:
            pc = [cora_posterior(cfg, rng, tr, model, **kw)[0] for tr in cal]
            pt = [cora_posterior(cfg, rng, tr, model, **kw)[0] for tr in test]
            posts[name] = (pc, pt)
        rc, rt = np.array([t.root for t in cal]), np.array([t.root for t in test])
        rowc = {m: [] for m in out}
        rows = {m: [] for m in out}
        for a in alphas:
            for name in ["CORA", "JudgeConformal"]:
                pc, pt = posts[name]
                q = conformal_quantile(np.array([1 - p[r] for p, r in zip(pc, rc)]), a)
                S = [conformal_set(p, q) for p in pt]
                rowc[name].append(np.mean([r in s for r, s in zip(rt, S)]))
                rows[name].append(np.mean([len(s) for s in S]))
            # uncalibrated: smallest top-p set of judge posterior with mass 1-a
            S = []
            for p in posts["JudgeConformal"][1]:
                o = np.argsort(-p); c = np.cumsum(p[o])
                S.append(o[:int(np.searchsorted(c, 1 - a) + 1)])
            rowc["Uncalibrated"].append(np.mean([r in s for r, s in zip(rt, S)]))
            rows["Uncalibrated"].append(np.mean([len(s) for s in S]))
        for m in out:
            out[m]["cov"].append(rowc[m]); out[m]["size"].append(rows[m])
        print("e2 seed", k, flush=True)
    res = {"alphas": alphas}
    for m in out:
        c, s = np.array(out[m]["cov"]), np.array(out[m]["size"])
        res[m] = {"cov": c.mean(0).tolist(), "cov_sd": c.std(0, ddof=1).tolist(),
                  "size": s.mean(0).tolist(), "size_sd": s.std(0, ddof=1).tolist()}
    save("e2", res)


# ------------------------------------------------------------------ E3
def e3():
    cfg = Config()
    Bs = [0, 2, 4, 6, 8, 10, 12, 16]
    variants = {"CORA": {}, "PointOnly": {"probes": ("point",)},
                "PrefixOnly": {"probes": ("prefix",)},
                "RandomPolicy": {"policy": "random"}}
    acc = {v: np.zeros((3, len(Bs))) for v in variants}
    cost = {v: np.zeros((3, len(Bs))) for v in variants}
    loo = []
    for k in range(3):
        rng = np.random.default_rng(SEED + 200 + k)
        cal, test = gen(cfg, rng, 300), gen(cfg, rng, 500)
        model = fit_cora(cfg, rng, cal)
        loo.append(np.mean([m_loo(cfg, rng, tr, model)[0] == tr.root for tr in test]))
        for v, kw in variants.items():
            for j, B in enumerate(Bs):
                c2 = cfg.with_(stop_post=1.01)  # fixed budget for the sweep
                o = [cora_posterior(c2, rng, tr, model, B=B, **kw) for tr in test]
                acc[v][k, j] = np.mean([np.argmax(p[0]) == tr.root for p, tr in zip(o, test)])
                cost[v][k, j] = np.mean([p[2] for p in o])
        print("e3 seed", k, flush=True)
    res = {"B": Bs, "LOO": ms(loo)}
    for v in variants:
        res[v] = {"acc": acc[v].mean(0).tolist(), "acc_sd": acc[v].std(0, ddof=1).tolist(),
                  "cost": cost[v].mean(0).tolist()}
    save("e3", res)


# ------------------------------------------------------------------ E4
def mttr_samples(cfg, rng, test, model):
    strat = {k: {"mttr": [], "ok": [], "coll": []} for k in
             ["FullRestart", "RetryCommit", "JudgeTargeted", "LOOTargeted",
              "CORATop1", "CORASet"]}
    for tr in test:
        def rep_time(c):
            return step_times(cfg, rng, c).sum() + c * cfg.twin_step

        def fin(name, attr_wall, ok, c, coll):
            t = cfg.t_detect + attr_wall + rep_time(c) + (0 if ok else cfg.t_human)
            strat[name]["mttr"].append(t); strat[name]["ok"].append(ok)
            strat[name]["coll"].append(coll)
        ok, c, _ = repair(cfg, rng, tr, [0], quarantine=False)
        fin("FullRestart", 0.0, ok, c, 0)
        lc = int(tr.failed_commits.max())
        ok, c, _ = repair(cfg, rng, tr, [lc], quarantine=False)
        fin("RetryCommit", 0.0, ok, c, 0)
        t, w, _ = m_ensemble(cfg, rng, tr)
        ok, c, coll = repair(cfg, rng, tr, [t])
        fin("JudgeTargeted", w, ok, c, coll)
        t, w, _ = m_loo(cfg, rng, tr, model)
        ok, c, coll = repair(cfg, rng, tr, [t])
        fin("LOOTargeted", w, ok, c, coll)
        post, w, _, _ = cora_posterior(cfg, rng, tr, model)
        ok, c, coll = repair(cfg, rng, tr, [int(np.argmax(post))])
        fin("CORATop1", w, ok, c, coll)
        S = conformal_set(post, model.qhat)
        ok, c, coll = repair(cfg, rng, tr, S)
        fin("CORASet", w, ok, c, coll)
    return strat


def e4():
    cfg = Config()
    agg = None
    for k in range(5):
        rng, cal, test, model = setup(cfg, SEED + 300 + k)
        s = mttr_samples(cfg, rng, test, model)
        if agg is None:
            agg = {n: {kk: [] for kk in v} for n, v in s.items()}
        for n in s:
            for kk in s[n]:
                agg[n][kk] += [float(x) for x in s[n][kk]]
        print("e4 seed", k, flush=True)
    res = {}
    for n, d in agg.items():
        m = np.array(d["mttr"])
        res[n] = {"success": float(np.mean(d["ok"])), "mean": float(m.mean()),
                  "p50": float(np.percentile(m, 50)), "p95": float(np.percentile(m, 95)),
                  "collateral": float(np.mean(d["coll"])),
                  "cdf_x": np.percentile(m, np.arange(0, 101, 2)).tolist()}
    save("e4", res)
    save("e4_samples", agg)


# ------------------------------------------------------------------ E5
def e5():
    Ts = [12, 24, 48, 96, 192]
    res = {"T": Ts}
    for name in ["LOO-CF", "CORA", "CORA-logB", "Ensemble"]:
        res[name] = {"acc": [], "cost": [], "wall": [], "cpu_ms": []}
    res["B_log"] = [int(np.ceil(1.75 * np.log2(T))) for T in Ts]
    for T in Ts:
        cfg = Config(T=T)
        rng = np.random.default_rng(SEED + 400 + T)
        ncal, ntest = (300, 300) if T <= 48 else (200, 150)
        cal, test = gen(cfg, rng, ncal), gen(cfg, rng, ntest)
        model = fit_cora(cfg, rng, cal)
        for name in res:
            if name in ("T", "B_log"):
                continue
            a, c, w, cpu = [], [], [], []
            for tr in test:
                t0 = time.perf_counter()
                if name == "LOO-CF":
                    t, ww, cc = m_loo(cfg, rng, tr, model)
                elif name == "CORA":
                    post, ww, cc, _ = cora_posterior(cfg, rng, tr, model)
                    t = int(np.argmax(post))
                elif name == "CORA-logB":
                    post, ww, cc, _ = cora_posterior(
                        cfg, rng, tr, model, B=int(np.ceil(1.75 * np.log2(T))))
                    t = int(np.argmax(post))
                else:
                    t, ww, cc = m_ensemble(cfg, rng, tr)
                cpu.append(1e3 * (time.perf_counter() - t0))
                a.append(t == tr.root); c.append(cc); w.append(ww)
            for kk, v in zip(["acc", "cost", "wall", "cpu_ms"], [a, c, w, cpu]):
                res[name][kk].append(float(np.mean(v)))
        print("e5 T", T, {n: round(res[n]["acc"][-1], 3) for n in ["LOO-CF", "CORA", "CORA-logB", "Ensemble"]},
              {n: round(res[n]["cost"][-1], 1) for n in ["LOO-CF", "CORA", "CORA-logB"]}, flush=True)
    save("e5", res)


# ------------------------------------------------------------------ E6
def e6():
    rhos = [0.0, 0.25, 0.5, 0.75, 1.0]
    res = {"rho": rhos, "CORA": [], "CORA_cov": [], "LOO-CF": [], "Ensemble": [],
           "CORA_size": []}
    for rho in rhos:
        cfg = Config(rho=rho)
        a = {k: [] for k in ["CORA", "CORA_cov", "LOO-CF", "Ensemble", "CORA_size"]}
        for k in range(3):
            rng, cal, test, model = setup(cfg, SEED + 500 + k, 300, 500)
            o = [cora_posterior(cfg, rng, tr, model)[0] for tr in test]
            a["CORA"].append(np.mean([np.argmax(p) == tr.root for p, tr in zip(o, test)]))
            S = [conformal_set(p, model.qhat) for p in o]
            a["CORA_cov"].append(np.mean([tr.root in s for s, tr in zip(S, test)]))
            a["CORA_size"].append(np.mean([len(s) for s in S]))
            a["LOO-CF"].append(np.mean([m_loo(cfg, rng, tr, model)[0] == tr.root for tr in test]))
            a["Ensemble"].append(np.mean([m_ensemble(cfg, rng, tr)[0] == tr.root for tr in test]))
        for kk in a:
            res[kk].append(ms(a[kk]))
        print("e6 rho", rho, {kk: round(res[kk][-1][0], 3) for kk in a}, flush=True)
    save("e6", res)


# ------------------------------------------------------------------ E7
def e7():
    base = Config()
    shifts = {"none": base,
              "subtle_mix": base.with_(fault_mix=(0.15, 0.20, 0.50, 0.15)),
              "evasive_10": base.with_(evasive_frac=0.10),
              "evasive_20": base.with_(evasive_frac=0.20),
              "evasive_40": base.with_(evasive_frac=0.40),
              "judge_drift": base.with_(sig_ind=0.9, sig_shared=0.6),
              "sticky_agents": base.with_(p_adopt=(0.9,) * 6 + (0.6, 0.95)),
              "longer_T36": base.with_(T=36)}
    res = {}
    for name, sc in shifts.items():
        r = {"stale_cov": [], "stale_size": [], "recal_cov": [], "recal_size": [],
             "acc": []}
        for k in range(3):
            rng, cal, _, model = setup(base, SEED + 600 + k, 400, 10)
            win, test = gen(sc, rng, 300), gen(sc, rng, 500)
            # stale: model and threshold from the unshifted calibration
            o = [cora_posterior(sc, rng, tr, model)[0] for tr in test]
            S = [conformal_set(p, model.qhat) for p in o]
            r["stale_cov"].append(np.mean([tr.root in s for s, tr in zip(S, test)]))
            r["stale_size"].append(np.mean([len(s) for s in S]))
            r["acc"].append(np.mean([np.argmax(p) == tr.root for p, tr in zip(o, test)]))
            # recalibrated: refresh only the conformal threshold on a labelled
            # window of 300 recent shifted incidents (model tables unchanged)
            q, _ = calibrate_conformal(sc, rng, win, model, base.alpha)
            S = [conformal_set(p, q) for p in o]
            r["recal_cov"].append(np.mean([tr.root in s for s, tr in zip(S, test)]))
            r["recal_size"].append(np.mean([len(s) for s in S]))
        res[name] = {kk: ms(v) for kk, v in r.items()}
        print("e7", name, {kk: round(res[name][kk][0], 3) for kk in r}, flush=True)
    save("e7", res)


# ------------------------------------------------------------------ E8
def e8():
    """Slice SLA impact: incidents arrive Poisson(lam)/h on one slice cluster,
    served FIFO by a single attribution+repair pipeline. Downtime of an
    incident = waiting + MTTR. MTTR samples come from E4."""
    s = json.load(open(os.path.join(OUT, "e4_samples.json")))
    lams = [1, 2, 4, 6, 8, 10, 12, 14]
    rng = np.random.default_rng(SEED + 700)
    res = {"lam": lams}
    for name in ["FullRestart", "JudgeTargeted", "LOOTargeted", "CORASet"]:
        m = np.array(s[name]["mttr"])
        out = []
        for lam in lams:
            H = 400.0 * 3600
            arr = np.cumsum(rng.exponential(3600.0 / lam, size=int(lam * 420)))
            arr = arr[arr < H]
            free, down = 0.0, 0.0
            for a in arr:
                st = max(a, free)
                svc = m[rng.integers(len(m))]
                free = st + svc
                down += free - a
            out.append(down / 60.0 / (H / 3600.0))   # SLA-violation min per hour
        res[name] = out
    save("e8", res)


# ------------------------------------------------------------------ E9
def e9():
    cfg = Config()
    variants = {"CORA (full)": {},
                "w/o DAG pruning": {"use_dag": False},
                "w/o judge prior": {"use_prior": False},
                "w/o prefix probes": {"probes": ("point",)},
                "w/o point probes": {"probes": ("prefix",)},
                "w/o cost-aware selection": {"cost_aware": False},
                "random probe order": {"policy": "random"},
                "w/o replay (judge only)": {"use_replay": False}}
    r = {v: {"step": [], "agent": [], "cov": [], "size": [], "cost": []} for v in variants}
    for k in range(5):
        rng = np.random.default_rng(SEED + 800 + k)
        fit = gen(cfg, rng, N_FIT)
        cal, test = gen(cfg, rng, 400), gen(cfg, rng, 600)
        model = fit_cora(cfg, rng, fit)
        for v, kw in variants.items():
            q, _ = calibrate_conformal(cfg, rng, cal, model, cfg.alpha, **kw)
            o = [cora_posterior(cfg, rng, tr, model, **kw) for tr in test]
            S = [conformal_set(p[0], q) for p in o]
            r[v]["step"].append(np.mean([np.argmax(p[0]) == tr.root for p, tr in zip(o, test)]))
            r[v]["agent"].append(np.mean([tr.agent[np.argmax(p[0])] == tr.agent[tr.root]
                                          for p, tr in zip(o, test)]))
            r[v]["cov"].append(np.mean([tr.root in s for s, tr in zip(S, test)]))
            r[v]["size"].append(np.mean([len(s) for s in S]))
            r[v]["cost"].append(np.mean([p[2] for p in o]))
        print("e9 seed", k, flush=True)
    save("e9", {v: {kk: ms(x) for kk, x in d.items()} for v, d in r.items()})


ALL = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6,
       "e7": e7, "e8": e8, "e9": e9}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(ALL)
    for e in todo:
        t0 = time.time()
        ALL[e]()
        print(e, "done in %.0fs" % (time.time() - t0), flush=True)
