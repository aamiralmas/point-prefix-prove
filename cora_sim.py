"""CORA simulator: agentic LLM workflows for 5G/6G slice operations,
fault injection, failure attribution baselines, conformal counterfactual
attribution (CORA), and targeted repair.

Everything is seeded with integer seeds through numpy's default_rng.
No Python hash() is used in any seed path.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, replace

ROLES = ["Intent", "Planner", "RAN", "Transport", "Core",
         "Security", "Verifier", "Executor"]
N_ROLES = len(ROLES)
VERIFIER, EXECUTOR = 6, 7
FAULT_TYPES = ["hallucination", "wrong_argument", "ignored_input",
               "premature_stop"]


@dataclass
class Config:
    T: int = 24                  # steps per workflow trace
    rho: float = 0.5             # P(fault is persistent: recurs on re-execution)
    eps: float = 0.003           # spontaneous fault rate per step
    eps_re: float = 0.10         # transient fault recurrence on re-execution
    p_adopt: tuple = (0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.35, 0.85)
    p_adopt_fb: float = 0.30     # fallback (quarantine replacement) adoption
    q_ref: float = 0.99          # P(reference substitution of one step is correct)
    fault_mix: tuple = (0.35, 0.30, 0.20, 0.15)
    mu_type: tuple = (1.00, 0.80, 0.45, 1.20)   # judge salience per type
    evasive_frac: float = 0.0    # twin-aware faults (correct under replay)
    sig_shared: float = 0.35     # judge noise shared across judge samples
    sig_ind: float = 0.45        # judge noise per sample
    K_judge: int = 5             # ensemble size
    B_max: int = 8               # CORA replay budget
    stop_post: float = 0.95      # CORA early stop
    alpha: float = 0.10
    # timing model (seconds)
    step_med: float = 1.2
    step_sig: float = 0.5
    twin_step: float = 0.25
    judge_base: float = 0.8
    judge_per_step: float = 0.04
    t_detect: float = 5.0
    t_human: float = 900.0
    loo_workers: int = 8

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass
class Trace:
    T: int
    agent: np.ndarray
    parents: list
    commits: np.ndarray
    origins: list            # list of (step, persistent, evasive, ftype)
    faulty: np.ndarray
    origin_mask: np.ndarray
    depth: np.ndarray
    failed_commits: np.ndarray
    root: int
    anc: np.ndarray          # anc[i, j] True iff i is ancestor of j (i<j)
    judge_mu: np.ndarray = field(default=None)


# ----------------------------------------------------------------- structure
def gen_structure(rng, T):
    agent = np.empty(T, dtype=int)
    agent[0], agent[1] = 0, 1
    for t in range(2, T):
        agent[t] = rng.choice([2, 3, 4, 5, 6], p=[0.24, 0.2, 0.2, 0.14, 0.22])
    commit_pos = [t for t in range(5, T, 6)] + [T - 1]
    for t in commit_pos:
        agent[t] = EXECUTOR
    parents = [[]]
    for t in range(1, T):
        p = {t - 1}
        if t >= 3 and rng.random() < 0.5:
            p.add(int(rng.integers(0, t - 1)))
        if t >= 3 and rng.random() < 0.3:
            p.add(1)
        if agent[t] == EXECUTOR:
            p.update(range(max(0, t - 3), t))
        parents.append(sorted(p))
    anc = np.zeros((T, T), dtype=bool)
    for t in range(T):
        for p in parents[t]:
            anc[p, t] = True
            anc[:, t] |= anc[:, p]
    commits = np.array(sorted(set(commit_pos)))
    return agent, parents, commits, anc


def _forward(cfg, rng, agent, parents, commits, origins_spec, start=0,
             frozen=None, override=frozenset(), replaced=frozenset(),
             replay=False):
    """Execute steps [start, T). Steps < start copied from frozen state.
    origins_spec: dict step -> (persistent, evasive, ftype) of original
    intrinsic faults. Returns faulty, origin_mask, depth, origin list."""
    T = len(agent)
    faulty = np.zeros(T, bool)
    om = np.zeros(T, bool)
    depth = np.zeros(T, int)
    new_origins = []
    if frozen is not None and start > 0:
        faulty[:start], om[:start], depth[:start] = (
            frozen[0][:start], frozen[1][:start], frozen[2][:start])
    for t in range(start, T):
        par = parents[t]
        pf = [p for p in par if faulty[p]]
        fb = agent[t] in replaced
        if t in override:
            if rng.random() > cfg.q_ref:
                faulty[t], om[t] = True, True
            continue
        intrinsic = False
        if (not fb) and t in origins_spec:
            pers, eva, _ = origins_spec[t]
            if not replay:
                intrinsic = True
            elif eva:
                intrinsic = False
            elif pers:
                intrinsic = True
            else:
                intrinsic = rng.random() < cfg.eps_re
        if not intrinsic and not fb and rng.random() < cfg.eps:
            intrinsic = True
            if not replay:
                new_origins.append(t)
        if intrinsic:
            faulty[t], om[t] = True, True
            continue
        if pf:
            pa = cfg.p_adopt_fb if fb else cfg.p_adopt[agent[t]]
            if rng.random() < pa:
                faulty[t] = True
                depth[t] = 1 + max(depth[p] for p in pf)
    return faulty, om, depth, new_origins


def gen_failing_trace(cfg, rng, max_tries=200):
    T = cfg.T
    for _ in range(max_tries):
        agent, parents, commits, anc = gen_structure(rng, T)
        r = int(rng.integers(0, T - 1))
        ftype = int(rng.choice(4, p=cfg.fault_mix))
        pers = rng.random() < cfg.rho
        eva = rng.random() < cfg.evasive_frac
        spec = {r: (pers, eva, ftype)}
        faulty, om, depth, spont = _forward(cfg, rng, agent, parents,
                                            commits, spec)
        for s in spont:
            spec[s] = (rng.random() < cfg.rho, False,
                       int(rng.choice(4, p=cfg.fault_mix)))
        fc = commits[faulty[commits]]
        if len(fc) == 0:
            continue
        # decisive step: earliest origin with a faulty path to a failed commit
        cand = []
        for o in np.flatnonzero(om):
            reach = np.zeros(T, bool)
            reach[o] = True
            for t in range(o + 1, T):
                if faulty[t] and not om[t] and any(reach[p] for p in parents[t]):
                    reach[t] = True
            if reach[fc].any():
                cand.append(o)
        if not cand:
            continue
        root = int(min(cand))
        origins = [(s, v[0], v[1], v[2]) for s, v in spec.items()]
        tr = Trace(T, agent, parents, commits, origins, faulty, om, depth,
                   fc, root, anc)
        tr.spec = spec
        return tr
    raise RuntimeError("could not generate failing trace")


# ----------------------------------------------------------------- judges
def judge_means(cfg, tr):
    mu = np.zeros(tr.T)
    for t in range(tr.T):
        if tr.origin_mask[t]:
            ft = tr.spec[t][2] if t in tr.spec else 0
            mu[t] = cfg.mu_type[ft]
        elif tr.faulty[t]:
            mu[t] = min(0.6 + 0.12 * tr.depth[t], 1.4)
        elif any(tr.faulty[p] for p in tr.parents[t]):
            mu[t] = 0.2
    return mu


def judge_scores(cfg, rng, tr, K=1):
    mu = judge_means(cfg, tr)
    shared = cfg.sig_shared * rng.standard_normal(tr.T)
    s = np.stack([mu + shared + cfg.sig_ind * rng.standard_normal(tr.T)
                  for _ in range(K)])
    return s.mean(0)


def candidates(tr):
    """Steps that can be the decisive step given observables: failed commits
    and their DAG ancestors."""
    m = np.zeros(tr.T, bool)
    for c in tr.failed_commits:
        m[c] = True
        m |= tr.anc[:, c]
    return np.flatnonzero(m)


# ----------------------------------------------------------------- replay
def replay(cfg, rng, tr, t, probe="point", replaced=frozenset()):
    """Counterfactual replay on the twin.
    probe='point'  : substitute step t by the reference output, freeze steps
                     < t, re-execute steps > t.          cost T - t
    probe='prefix' : substitute steps 0..t by reference outputs, re-execute
                     steps > t.                           cost T
    probe='rerun'  : re-execute steps >= t with agents in `replaced` swapped
                     for fallbacks (used for repair).    cost T - t
    Returns (success, step executions)."""
    if probe == "prefix":
        ov, start, cost = frozenset(range(0, t + 1)), 0, tr.T
    elif probe == "point":
        ov, start, cost = frozenset([t]), t, tr.T - t
    else:
        ov, start, cost = frozenset(), t, tr.T - t
    frozen = (tr.faulty, tr.origin_mask, tr.depth)
    f, _, _, _ = _forward(cfg, rng, tr.agent, tr.parents, tr.commits,
                          tr.spec, start=start, frozen=frozen, override=ov,
                          replaced=replaced, replay=True)
    return not f[tr.commits].any(), cost


def rel(tr, t, r):
    if t < r:
        return 0
    if t == r:
        return 1
    return 2 if tr.anc[r, t] else 3


# ----------------------------------------------------------------- timing
def step_times(cfg, rng, n):
    return rng.lognormal(np.log(cfg.step_med), cfg.step_sig, size=n)


def judge_time(cfg, T, K=1):
    return K * (cfg.judge_base + cfg.judge_per_step * T)


# ----------------------------------------------------------------- baselines
def m_all_at_once(cfg, rng, tr):
    s = judge_scores(cfg, rng, tr)
    return int(np.argmax(s)), judge_time(cfg, tr.T), 0


def m_ensemble(cfg, rng, tr):
    s = judge_scores(cfg, rng, tr, K=cfg.K_judge)
    return int(np.argmax(s)), judge_time(cfg, tr.T, cfg.K_judge), 0


def m_step_by_step(cfg, rng, tr, tau=1.0):
    s = judge_scores(cfg, rng, tr)
    hits = np.flatnonzero(s > tau)
    t = int(hits[0]) if len(hits) else int(np.argmax(s))
    n_calls = (t + 1)
    return t, n_calls * (cfg.judge_base + cfg.judge_per_step * (t + 1)) * 0.5, 0


def m_binary(cfg, rng, tr):
    s_base = judge_scores(cfg, rng, tr)
    lo, hi, calls = 0, tr.T, 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        a = s_base[lo:mid].max() + cfg.sig_ind * rng.standard_normal()
        b = s_base[mid:hi].max() + cfg.sig_ind * rng.standard_normal()
        calls += 1
        if a >= b:
            hi = mid
        else:
            lo = mid
    return lo, calls * judge_time(cfg, tr.T), 0


# ----------------------------------------------------------------- CORA
PROBES = ("point", "prefix")


@dataclass
class CoraModel:
    L: np.ndarray            # L[probe, persistent, rel] = P(success)
    rho_hat: float           # prior P(persistent)
    beta: float              # judge-prior temperature
    qhat: float = 1.0        # conformal threshold on score 1-posterior


def _rel_matrix(tr, cand):
    return np.array([[rel(tr, int(t), int(r)) for r in cand] for t in cand])


def fit_cora(cfg, rng, cal_traces, n_probe=6):
    counts = np.ones((2, 2, 4, 2))          # Laplace smoothing
    pers = []
    for tr in cal_traces:
        p = int(tr.spec[tr.root][0])
        pers.append(p)
        cand = candidates(tr)
        for t in rng.choice(cand, size=min(n_probe, len(cand)), replace=False):
            for k, pr in enumerate(PROBES):
                ok, _ = replay(cfg, rng, tr, int(t), probe=pr)
                counts[k, p, rel(tr, int(t), tr.root), int(ok)] += 1
    L = counts[..., 1] / counts.sum(-1)
    scores = [(judge_scores(cfg, rng, tr), candidates(tr), tr.root)
              for tr in cal_traces]
    best, bb = -1e18, 1.0
    for b in np.linspace(0.0, 6.0, 31):
        ll = 0.0
        for s, c, r in scores:
            z = b * s[c]
            z = z - z.max()
            ll += z[np.searchsorted(c, r)] - np.log(np.exp(z).sum())
        if ll > best:
            best, bb = ll, b
    return CoraModel(L=L, rho_hat=float(np.mean(pers)), beta=float(bb))


def _entropy(p):
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())


def cora_posterior(cfg, rng, tr, model, use_prior=True, use_replay=True,
                   use_dag=True, probes=PROBES, B=None, cost_aware=True,
                   policy="ig"):
    """Joint posterior over hypotheses h = (root r, persistence p).
    Returns (posterior over steps, wall time, step executions, #replays)."""
    B = cfg.B_max if B is None else B
    cand = candidates(tr) if use_dag else np.arange(tr.T)
    n = len(cand)
    s = judge_scores(cfg, rng, tr)
    if use_prior:
        z = model.beta * s[cand]
        pr = np.exp(z - z.max())
    else:
        pr = np.ones(n)
    pr /= pr.sum()
    post = np.concatenate([pr * (1 - model.rho_hat), pr * model.rho_hat])
    R = _rel_matrix(tr, cand)
    # Lk[probe][i, h]: P(success | probe at cand i, hypothesis h)
    Lk = {}
    for k, name in enumerate(PROBES):
        Lk[name] = np.concatenate([model.L[k, 0][R], model.L[k, 1][R]], 1)
    actions = [(name, i) for name in probes for i in range(n)]
    tried = set()
    cost, wall, n_rep = 0, judge_time(cfg, tr.T), 0
    if use_replay:
        for _ in range(B):
            marg = post[:n] + post[n:]
            if marg.max() >= cfg.stop_post:
                break
            H0 = _entropy(post)
            best, best_g = None, -1e9
            for a in actions:
                if a in tried:
                    continue
                name, i = a
                if policy == "random":
                    g = rng.random()
                else:
                    Li = Lk[name][i]
                    ps = float(Li @ post)
                    p1, p0 = post * Li, post * (1 - Li)
                    g = H0
                    if p1.sum() > 0:
                        g -= ps * _entropy(p1 / p1.sum())
                    if p0.sum() > 0:
                        g -= (1 - ps) * _entropy(p0 / p0.sum())
                    if cost_aware:
                        c = tr.T if name == "prefix" else tr.T - cand[i]
                        g /= np.sqrt(c)
                if g > best_g:
                    best_g, best = g, a
            if best is None:
                break
            tried.add(best)
            name, i = best
            ok, c = replay(cfg, rng, tr, int(cand[i]), probe=name)
            cost += c
            n_rep += 1
            wall += step_times(cfg, rng, c).sum() + c * cfg.twin_step
            Li = Lk[name][i]
            post = post * (Li if ok else (1 - Li))
            post /= post.sum()
    full = np.zeros(tr.T)
    full[cand] = post[:n] + post[n:]
    return full, wall, cost, n_rep


def m_loo(cfg, rng, tr, model):
    """Exhaustive leave-one-out counterfactual: point-probe EVERY candidate,
    decide by maximum likelihood with the same calibrated tables as CORA."""
    cand = candidates(tr)
    n = len(cand)
    R = _rel_matrix(tr, cand)
    L = np.concatenate([model.L[0, 0][R], model.L[0, 1][R]], 1)
    ll = np.zeros(2 * n)
    times, cost = [], 0
    for i, t in enumerate(cand):
        ok, c = replay(cfg, rng, tr, int(t), probe="point")
        cost += c
        times.append(step_times(cfg, rng, c).sum() + c * cfg.twin_step)
        ll += np.log(L[i] if ok else 1 - L[i])
    ll[:n] += np.log(1 - model.rho_hat)
    ll[n:] += np.log(model.rho_hat)
    h = int(np.argmax(ll))
    wall = max(sum(times) / cfg.loo_workers, max(times))
    return int(cand[h % n]), wall, cost


def calibrate_conformal(cfg, rng, cal_traces, model, alpha, **kw):
    sc = []
    for tr in cal_traces:
        post, *_ = cora_posterior(cfg, rng, tr, model, **kw)
        sc.append(1.0 - post[tr.root])
    return conformal_quantile(np.array(sc), alpha), np.array(sc)


def conformal_quantile(scores, alpha):
    sc = np.sort(scores)
    n = len(sc)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return 1.0 if k > n else float(sc[k - 1])


def conformal_set(post, qhat):
    return np.flatnonzero(1.0 - post <= qhat + 1e-12)


# ----------------------------------------------------------------- repair
def repair(cfg, rng, tr, steps, quarantine=True):
    """Quarantine the agents of `steps` (swap in fallbacks) and re-execute
    from the earliest attributed step. Returns (success, step executions,
    healthy agents quarantined)."""
    steps = list(steps) if len(steps) else [tr.T - 1]
    agents = (frozenset(int(tr.agent[t]) for t in steps)
              if quarantine else frozenset())
    t0 = int(min(steps))
    ok, c = replay(cfg, rng, tr, t0, probe="rerun", replaced=agents)
    collateral = len(agents - {int(tr.agent[tr.root])})
    return ok, c, collateral
