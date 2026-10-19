"""Turn results/*.json into LaTeX: macros.tex (numbers quoted in the text),
fig_*.tex (pgfplots figures) and tab_*.tex (tables). Never hand-edit the
outputs; re-run this script after any experiment."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "paper_assets")
os.makedirs(OUT, exist_ok=True)


def J(n):
    return json.load(open(os.path.join(R, n + ".json")))


def w(name, s):
    open(os.path.join(OUT, name), "w").write(s)


macros = []


def M(name, val, fmt="{:.1f}"):
    assert name.isalpha(), name
    macros.append("\\newcommand{\\%s}{%s}" % (name, fmt.format(val)))


def pct(x):
    return 100.0 * x


def coords(xs, ys):
    return " ".join("(%g,%.4g)" % (x, y) for x, y in zip(xs, ys))


AX = r"""grid=major, grid style={dotted,gray!40},
    legend style={font=\scriptsize, at={(0.5,-0.36)}, anchor=north,
        legend columns=2, draw=gray!50},
    tick label style={font=\footnotesize}, label style={font=\small}"""
S = {"CORA": "color=blue, mark=o, thick",
     "LOO": "color=red, mark=square, thick",
     "ENS": "color=green!60!black, mark=triangle, thick",
     "B3": "color=orange, mark=diamond, thick",
     "B4": "color=violet, mark=x, thick",
     "REF": "color=gray, thin, dashed, mark=none"}

# ------------------------------------------------------------------ E1
e1 = J("e1")
names = ["AllAtOnce", "StepByStep", "BinarySearch", "Ensemble", "LOO-CF", "CORA"]
for n, key in [("AllAtOnce", "AAO"), ("StepByStep", "SBS"), ("BinarySearch", "BIN"),
               ("Ensemble", "ENS"), ("LOO-CF", "LOO"), ("CORA", "CORA")]:
    k = key.replace("-", "")
    M("E" + "one" + k.lower().capitalize() + "Step", pct(e1[n]["step"][0]))
    M("E" + "one" + k.lower().capitalize() + "Agent", pct(e1[n]["agent"][0]))
    M("E" + "one" + k.lower().capitalize() + "Cost", e1[n]["cost"][0], "{:.0f}")
    M("E" + "one" + k.lower().capitalize() + "Wall", e1[n]["wall"][0], "{:.1f}")
cs = e1["CORA_set"]
M("EoneCov", pct(cs["cov"][0])); M("EoneAcov", pct(cs["acov"][0]))
M("EoneSize", cs["size"][0], "{:.2f}"); M("EoneAsize", cs["asize"][0], "{:.2f}")
M("EoneNrep", cs["nrep"][0], "{:.2f}")
M("EoneCostRatio", e1["LOO-CF"]["cost"][0] / e1["CORA"]["cost"][0], "{:.2f}")
M("EoneStepGainLoo", pct(e1["CORA"]["step"][0] - e1["LOO-CF"]["step"][0]))
M("EoneStepGainSbs", pct(e1["CORA"]["step"][0] - e1["StepByStep"]["step"][0]))
M("EoneStepRatioEns", e1["CORA"]["step"][0] / e1["Ensemble"]["step"][0], "{:.2f}")

disp = {"AllAtOnce": "All-at-once judge", "StepByStep": "Step-by-step judge",
        "BinarySearch": "Binary-search judge", "Ensemble": "Ensemble judge ($K{=}5$)",
        "LOO-CF": "Exhaustive counterfactual (LOO-CF)", "CORA": r"\SYS{} (top-1)"}
rows = []
for n in names:
    d = e1[n]
    rows.append("%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %s & %.1f \\\\" % (
        disp[n], pct(d["agent"][0]), pct(d["agent"][1]), pct(d["step"][0]),
        pct(d["step"][1]), ("%.0f" % d["cost"][0]) if d["cost"][0] > 0 else "0",
        d["wall"][0]))
rows.append(r"\midrule")
rows.append(r"\SYS{} set ($\alpha{=}0.1$) & %.1f$^{\dagger}$ & %.1f$^{\dagger}$ & %.0f & %.1f \\" % (
    pct(cs["acov"][0]), pct(cs["cov"][0]), e1["CORA"]["cost"][0], e1["CORA"]["wall"][0]))
tab = r"""\begin{table*}[!t]
\centering
\begin{threeparttable}
\caption{Failure Attribution on Failing Slice-Reconfiguration Workflows
($T{=}24$, eight agent roles; mean $\pm$ std.\ over ten seeds,
800 test incidents per seed)}
\label{tab:main}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.4}
\begin{tabularx}{\textwidth}{@{}lCCCC@{}}
\toprule
\textbf{Method} & \textbf{Agent acc.\ (\%%)} & \textbf{Step acc.\ (\%%)}
 & \textbf{Twin step executions} & \textbf{Attribution time (s)} \\
\midrule
%s
\midrule
\multicolumn{5}{@{}l}{\footnotesize $^{\dagger}$ = coverage of the true
agent / decisive step by the conformal set;\quad $\pm$ = one standard
deviation across seeds}\\
\bottomrule
\end{tabularx}
\begin{tablenotes}[flushleft]\scriptsize
\item Twin step executions: agent steps re-executed on the digital twin
during attribution. Attribution time uses the timing model of
Table~\ref{tab:params}; LOO-CF runs its replays on %d parallel twin workers,
\SYS{} runs its adaptive replays sequentially. The \SYS{} set row reports
coverage rather than top-1 accuracy; its mean size is %.2f steps
(%.2f distinct agents).
\end{tablenotes}
\end{threeparttable}
\end{table*}
""" % ("\n".join(rows), 8, cs["size"][0], cs["asize"][0])
w("tab_main.tex", tab)

# confusion matrices
conf = e1["confusion"]
roles = ["Int", "Pln", "RAN", "Trn", "Cor", "Sec", "Ver", "Exe"]


def heat(mat, title, ylab):
    pts = []
    for i in range(8):
        for j in range(8):
            pts.append("(%d,%d) [%.3f]" % (j, i, mat[i][j]))
    return r"""\begin{axis}[width=0.36\textwidth, height=0.30\textwidth,
    title={%s}, title style={font=\small},
    enlargelimits=false, axis on top,
    xtick={0,...,7}, ytick={0,...,7}, y dir=reverse,
    xticklabels={%s}, yticklabels={%s},
    xlabel={Attributed role}, %s
    tick label style={font=\scriptsize}, label style={font=\small},
    colormap={bw}{rgb255=(255,255,255) rgb255=(30,80,170)},
    point meta min=0, point meta max=1,
    colorbar, colorbar style={width=2.5mm, ytick={0,0.5,1},
        tick label style={font=\scriptsize}}]
\addplot[matrix plot*, mesh/cols=8, point meta=explicit] coordinates {%s};
\end{axis}""" % (title, ",".join(roles), ",".join(roles),
                 ("ylabel={True role}," if ylab else ""), " ".join(pts))


fig = r"""\begin{figure*}[!t]
\centering
\begin{tikzpicture}
%s
\end{tikzpicture}\hfill
\begin{tikzpicture}
%s
\end{tikzpicture}
\caption{Row-normalised role-level confusion matrices over all ten seeds
(8{,}000 incidents). Left: ensemble LLM judge. Right: \SYS{} top-1.
Diagonal entries are per-role agent accuracies. The judge rarely blames
the upstream Intent (Int) and Planner (Pln) roles, whose symptoms surface
downstream; the residual error of \SYS{} concentrates in the Executor
(Exe) column, i.e.\ on commit steps adjacent to the true root.}
\label{fig:confusion}
\end{figure*}
""" % (heat(conf["Ensemble"], "Ensemble judge", True), heat(conf["CORA"], r"\SYS{}", False))
w("fig_confusion.tex", fig)
ce, cc = np.array(e1["confusion_counts"]["Ensemble"]), np.array(e1["confusion_counts"]["CORA"])
M("ConfEnsExe", pct(ce[:, 7].sum() / ce.sum()))
M("ConfCoraExe", pct(cc[:, 7].sum() / cc.sum()))
M("ConfTrueExe", pct(cc[7, :].sum() / cc.sum()))
M("ConfCoraVer", pct(cc[6, 6] / max(cc[6].sum(), 1)))
M("ConfCoraUp", pct((cc[0, 0] + cc[1, 1]) / cc[:2].sum()))
M("ConfEnsUp", pct((ce[0, 0] + ce[1, 1]) / ce[:2].sum()))
M("ConfCoraDomExe", pct(cc[2:7, 7].sum() / cc[2:7].sum()))
M("ConfEnsDomExe", pct(ce[2:7, 7].sum() / ce[2:7].sum()))
M("ConfCoraExeExe", pct(cc[7, 7] / cc[7].sum()))
M("ConfEnsVer", pct(ce[6, 6] / max(ce[6].sum(), 1)))

# ------------------------------------------------------------------ E2
e2 = J("e2")
al = e2["alphas"]
nom = [1 - a for a in al]
fig = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth, height=4.3cm,
    xlabel={Nominal coverage $1-\alpha$}, ylabel={Empirical coverage},
    xmin=0.68, xmax=0.97, ymin=0.66, ymax=1.0, %s]
\addplot[%s] coordinates {%s};
\addlegendentry{\SYS{}}
\addplot[%s] coordinates {%s};
\addlegendentry{Judge-only conformal}
\addplot[%s] coordinates {%s};
\addlegendentry{Uncalibrated top-$p$}
\addplot[%s] coordinates {(0.68,0.68)(0.97,0.97)};
\addlegendentry{$y=x$}
\end{axis}
\end{tikzpicture}
\caption{Empirical coverage of the decisive step versus the nominal level
(five seeds, 800 test incidents each). Both conformal sets track the
diagonal; the uncalibrated top-$p$ set of the judge posterior is
conservative.}
\label{fig:coverage}
\end{figure}
""" % (AX, S["CORA"], coords(nom, e2["CORA"]["cov"]), S["LOO"],
       coords(nom, e2["JudgeConformal"]["cov"]), S["ENS"],
       coords(nom, e2["Uncalibrated"]["cov"]), S["REF"])
w("fig_coverage.tex", fig)
fig = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth, height=4.3cm,
    xlabel={Miscoverage level $\alpha$}, ylabel={Mean set size (steps)},
    xmin=0.04, xmax=0.31, ymin=0, ymax=18,
    xtick={0.05,0.10,0.15,0.20,0.25,0.30},
    xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2}, %s]
\addplot[%s, error bars/.cd, y dir=both, y explicit] coordinates {%s};
\addlegendentry{\SYS{}}
\addplot[%s] coordinates {%s};
\addlegendentry{Judge-only conformal}
\addplot[%s] coordinates {%s};
\addlegendentry{Uncalibrated top-$p$}
\end{axis}
\end{tikzpicture}
\caption{Mean attribution-set size versus $\alpha$. Error bars: one
standard deviation across seeds. Counterfactual replay shrinks the set by
a factor of %.1f at $\alpha=0.1$ relative to judge-only conformal sets at
equal coverage.}
\label{fig:setsize}
\end{figure}
""" % (AX, S["CORA"], " ".join("(%g,%.3f) +- (0,%.3f)" % (a, s, d) for a, s, d in
                               zip(al, e2["CORA"]["size"], e2["CORA"]["size_sd"])),
       S["LOO"], coords(al, e2["JudgeConformal"]["size"]), S["ENS"],
       coords(al, e2["Uncalibrated"]["size"]),
       e2["JudgeConformal"]["size"][1] / e2["CORA"]["size"][1])
w("fig_setsize.tex", fig)
i = al.index(0.10)
M("EtwoCovTen", pct(e2["CORA"]["cov"][i])); M("EtwoSizeTen", e2["CORA"]["size"][i], "{:.2f}")
M("EtwoJudgeCovTen", pct(e2["JudgeConformal"]["cov"][i]))
M("EtwoJudgeSizeTen", e2["JudgeConformal"]["size"][i], "{:.2f}")
M("EtwoUncCovTen", pct(e2["Uncalibrated"]["cov"][i]))
M("EtwoUncSizeTen", e2["Uncalibrated"]["size"][i], "{:.2f}")
M("EtwoShrink", e2["JudgeConformal"]["size"][i] / e2["CORA"]["size"][i])
j = al.index(0.05)
M("EtwoCovFive", pct(e2["CORA"]["cov"][j])); M("EtwoSizeFive", e2["CORA"]["size"][j], "{:.2f}")
k = al.index(0.30)
M("EtwoCovThirty", pct(e2["CORA"]["cov"][k])); M("EtwoSizeThirty", e2["CORA"]["size"][k], "{:.2f}")
M("EtwoMaxDev", pct(max(abs(c - n) for c, n in zip(e2["CORA"]["cov"], nom))))

# ------------------------------------------------------------------ E3
e3 = J("e3")
B = e3["B"]
fig = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth, height=4.3cm,
    xlabel={Replay budget $B$ (probes per incident)},
    ylabel={Step accuracy}, xmin=0, xmax=16, ymin=0.15, ymax=0.72,
    xtick={0,2,4,6,8,10,12,16}, %s]
\addplot[%s] coordinates {%s};
\addlegendentry{\SYS{} (point + prefix, IG)}
\addplot[%s] coordinates {%s};
\addlegendentry{Point probes only}
\addplot[%s] coordinates {%s};
\addlegendentry{Prefix probes only}
\addplot[%s] coordinates {%s};
\addlegendentry{Random probe order}
\addplot[%s] coordinates {(0,%.3f)(16,%.3f)};
\addlegendentry{LOO-CF (all candidates)}
\end{axis}
\end{tikzpicture}
\caption{Top-1 step accuracy versus replay budget (three seeds, early
stopping disabled). LOO-CF point-probes every candidate (mean %.0f twin
step executions per incident); \SYS{} passes it at $B=%d$.}
\label{fig:budget}
\end{figure}
""" % (AX, S["CORA"], coords(B, e3["CORA"]["acc"]), S["LOO"], coords(B, e3["PointOnly"]["acc"]),
       S["ENS"], coords(B, e3["PrefixOnly"]["acc"]), S["B3"], coords(B, e3["RandomPolicy"]["acc"]),
       S["REF"], e3["LOO"][0], e3["LOO"][0], e1["LOO-CF"]["cost"][0],
       next(b for b, a in zip(B, e3["CORA"]["acc"]) if a > e3["LOO"][0]))
w("fig_budget.tex", fig)
bi = B.index(16); b8 = B.index(8); b4 = B.index(4)
M("EthreeCoraSixteen", pct(e3["CORA"]["acc"][bi])); M("EthreeCostSixteen", e3["CORA"]["cost"][bi], "{:.0f}")
M("EthreePointSixteen", pct(e3["PointOnly"]["acc"][bi])); M("EthreePrefixSixteen", pct(e3["PrefixOnly"]["acc"][bi]))
M("EthreeRandSixteen", pct(e3["RandomPolicy"]["acc"][bi]))
M("EthreeCoraEight", pct(e3["CORA"]["acc"][b8])); M("EthreeCoraFour", pct(e3["CORA"]["acc"][b4]))
M("EthreeCoraZero", pct(e3["CORA"]["acc"][0])); M("EthreeLoo", pct(e3["LOO"][0]))
M("EthreeCross", next(b for b, a in zip(B, e3["CORA"]["acc"]) if a > e3["LOO"][0]), "{:d}")
M("EthreeCostCross", e3["CORA"]["cost"][B.index(next(b for b, a in zip(B, e3["CORA"]["acc"]) if a > e3["LOO"][0]))], "{:.0f}")

# ------------------------------------------------------------------ E4
e4 = J("e4")
order = [("CORASet", r"\SYS{} set repair", "CORA"), ("CORATop1", r"\SYS{} top-1 repair", "B4"),
         ("LOOTargeted", "LOO-CF targeted", "LOO"), ("JudgeTargeted", "Judge targeted", "ENS"),
         ("FullRestart", "Full restart", "B3")]
ps = list(range(0, 101, 2))
plots = []
for key, lab, st in order:
    xs = e4[key]["cdf_x"]
    plots.append(r"\addplot[%s, mark repeat=5] coordinates {%s};" % (S[st], coords(xs, [p / 100 for p in ps])) +
                 "\n\\addlegendentry{%s}" % lab)
fig = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth, height=4.5cm, xmode=log,
    xlabel={Mean time to repair, MTTR (s)}, ylabel={Empirical CDF},
    xmin=20, xmax=2500, ymin=0, ymax=1.02, %s]
%s
\end{axis}
\end{tikzpicture}
\caption{CDF of slice MTTR per incident (five seeds, 4{,}000 incidents).
The step near 900\,s is the human on-call escalation that follows a
failed automated repair; its height is the repair failure rate.}
\label{fig:mttr}
\end{figure}
""" % (AX, "\n".join(plots))
w("fig_mttr.tex", fig)
lab4 = {"FullRestart": "Full restart (same agents)", "RetryCommit": "Retry failed commit",
        "JudgeTargeted": "Judge top-1 quarantine", "LOOTargeted": "LOO-CF top-1 quarantine",
        "CORATop1": r"\SYS{} top-1 quarantine", "CORASet": r"\SYS{} set quarantine"}
rows = []
for k in ["FullRestart", "RetryCommit", "JudgeTargeted", "LOOTargeted", "CORATop1", "CORASet"]:
    d = e4[k]
    rows.append("%s & %.1f & %.0f & %.0f & %.0f & %.2f \\\\" % (
        lab4[k], pct(d["success"]), d["mean"], d["p50"], d["p95"], d["collateral"]))
tab = r"""\begin{table*}[!t]
\centering
\begin{threeparttable}
\caption{Automated Repair Outcome and Slice MTTR (five seeds, 4{,}000 incidents)}
\label{tab:repair}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.4}
\begin{tabularx}{\textwidth}{@{}lCCCCC@{}}
\toprule
\textbf{Repair strategy} & \textbf{Repair success (\%%)} & \textbf{Mean MTTR (s)}
 & \textbf{Median MTTR (s)} & \textbf{p95 MTTR (s)} & \textbf{Healthy agents quarantined} \\
\midrule
%s
\midrule
\multicolumn{6}{@{}l}{\footnotesize MTTR = detection + attribution +
repair replay + escalation (on failure);\quad p95 = 95th percentile}\\
\bottomrule
\end{tabularx}
\begin{tablenotes}[flushleft]\scriptsize
\item Quarantine swaps the attributed agents for their fallback
implementations and re-executes the workflow from the earliest attributed
step. A failed repair escalates to the human on-call engineer
($\tau_{\mathrm{h}}=900$\,s). ``Healthy agents quarantined'' counts
attributed agents other than the true faulty role.
\end{tablenotes}
\end{threeparttable}
\end{table*}
""" % "\n".join(rows)
w("tab_repair.tex", tab)
for k, nm in [("CORASet", "Cset"), ("CORATop1", "Ctop"), ("LOOTargeted", "Loo"),
              ("JudgeTargeted", "Judge"), ("FullRestart", "Full"), ("RetryCommit", "Retry")]:
    d = e4[k]
    M("Efour" + nm + "Succ", pct(d["success"])); M("Efour" + nm + "Mean", d["mean"], "{:.0f}")
    M("Efour" + nm + "Med", d["p50"], "{:.0f}"); M("Efour" + nm + "Pnf", d["p95"], "{:.0f}")
    M("Efour" + nm + "Coll", d["collateral"], "{:.2f}")
M("EfourMeanRedLoo", pct(1 - e4["CORASet"]["mean"] / e4["LOOTargeted"]["mean"]))
M("EfourMeanRedFull", pct(1 - e4["CORASet"]["mean"] / e4["FullRestart"]["mean"]))
M("EfourPnfRedLoo", pct(1 - e4["CORASet"]["p95"] / e4["LOOTargeted"]["p95"]))

# ------------------------------------------------------------------ E5
e5 = J("e5")
T = e5["T"]
TlogT = [t * np.log2(t) * 1.0 for t in T]
fig = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth, height=4.3cm, xmode=log, ymode=log,
    log basis x=2, xlabel={Workflow length $T$ (steps)},
    ylabel={Twin step executions}, xmin=10, xmax=230, ymin=10, ymax=40000,
    xtick={12,24,48,96,192}, xticklabels={12,24,48,96,192}, %s]
\addplot[%s] coordinates {%s};
\addlegendentry{LOO-CF}
\addplot[%s] coordinates {%s};
\addlegendentry{\SYS{}, $B=8$}
\addplot[%s] coordinates {%s};
\addlegendentry{\SYS{}, $B=\lceil 1.75\log_2 T\rceil$}
\addplot[%s] coordinates {%s};
\addlegendentry{$\propto T^2/2$}
\end{axis}
\end{tikzpicture}

\vspace{1mm}
\begin{tikzpicture}
\begin{axis}[width=\columnwidth, height=4.0cm, xmode=log, log basis x=2,
    xlabel={Workflow length $T$ (steps)}, ylabel={Step accuracy},
    xmin=10, xmax=230, ymin=0, ymax=0.75,
    xtick={12,24,48,96,192}, xticklabels={12,24,48,96,192}, %s]
\addplot[%s] coordinates {%s};
\addlegendentry{LOO-CF}
\addplot[%s] coordinates {%s};
\addlegendentry{\SYS{}, $B=8$}
\addplot[%s] coordinates {%s};
\addlegendentry{\SYS{}, $B=\lceil 1.75\log_2 T\rceil$}
\addplot[%s] coordinates {%s};
\addlegendentry{Ensemble judge}
\end{axis}
\end{tikzpicture}
\caption{Scalability in workflow length. Top: attribution cost (log--log).
Bottom: top-1 step accuracy. A fixed budget keeps \SYS{} linear in $T$ but
loses accuracy as the candidate set grows; a budget logarithmic in $T$
restores it at $O(T\log T)$ cost.}
\label{fig:scale}
\end{figure}
""" % (AX, S["LOO"], coords(T, e5["LOO-CF"]["cost"]), S["CORA"], coords(T, e5["CORA"]["cost"]),
       S["B4"], coords(T, e5["CORA-logB"]["cost"]), S["REF"], coords(T, [t * t / 2.0 for t in T]),
       AX, S["LOO"], coords(T, e5["LOO-CF"]["acc"]), S["CORA"], coords(T, e5["CORA"]["acc"]),
       S["B4"], coords(T, e5["CORA-logB"]["acc"]), S["ENS"], coords(T, e5["Ensemble"]["acc"]))
w("fig_scale.tex", fig)
t192 = T.index(192)
M("EfiveLooCostMax", e5["LOO-CF"]["cost"][t192], "{:.0f}")
M("EfiveCoraCostMax", e5["CORA"]["cost"][t192], "{:.0f}")
M("EfiveLogCostMax", e5["CORA-logB"]["cost"][t192], "{:.0f}")
M("EfiveLooAccMax", pct(e5["LOO-CF"]["acc"][t192])); M("EfiveCoraAccMax", pct(e5["CORA"]["acc"][t192]))
M("EfiveLogAccMax", pct(e5["CORA-logB"]["acc"][t192])); M("EfiveEnsAccMax", pct(e5["Ensemble"]["acc"][t192]))
M("EfiveRatioMax", e5["LOO-CF"]["cost"][t192] / e5["CORA-logB"]["cost"][t192])
M("EfiveBlogMax", e5["B_log"][t192], "{:d}")
M("EfiveCpuMax", e5["CORA-logB"]["cpu_ms"][t192], "{:.0f}")
M("EfiveLooCpuMax", e5["LOO-CF"]["cpu_ms"][t192], "{:.0f}")
sl = np.polyfit(np.log(T), np.log(e5["LOO-CF"]["cost"]), 1)[0]
sc = np.polyfit(np.log(T), np.log(e5["CORA"]["cost"]), 1)[0]
M("EfiveSlopeLoo", sl, "{:.2f}"); M("EfiveSlopeCora", sc, "{:.2f}")
t48 = T.index(48)
M("EfiveLooAccFE", pct(e5["LOO-CF"]["acc"][t48])); M("EfiveLogAccFE", pct(e5["CORA-logB"]["acc"][t48]))

# ------------------------------------------------------------------ E6
e6 = J("e6")
rho = e6["rho"]
fig = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth, height=4.3cm,
    xlabel={Fraction of persistent faults $\rho$}, ylabel={Step accuracy},
    xmin=-0.03, xmax=1.03, ymin=0.1, ymax=0.8, %s]
\addplot[%s, error bars/.cd, y dir=both, y explicit] coordinates {%s};
\addlegendentry{\SYS{}}
\addplot[%s, error bars/.cd, y dir=both, y explicit] coordinates {%s};
\addlegendentry{LOO-CF}
\addplot[%s] coordinates {%s};
\addlegendentry{Ensemble judge}
\end{axis}
\end{tikzpicture}
\caption{Sensitivity to the persistent-fault fraction (three seeds). Point
probes lose identifiability as faults become persistent, while the
prefix probe gains it; \SYS{} uses both. Conformal coverage stays within
$[%.1f, %.1f]$\%% over the sweep.}
\label{fig:rho}
\end{figure}
""" % (AX, S["CORA"], " ".join("(%g,%.3f) +- (0,%.3f)" % (r, a[0], a[1]) for r, a in zip(rho, e6["CORA"])),
       S["LOO"], " ".join("(%g,%.3f) +- (0,%.3f)" % (r, a[0], a[1]) for r, a in zip(rho, e6["LOO-CF"])),
       S["ENS"], coords(rho, [a[0] for a in e6["Ensemble"]]),
       pct(min(c[0] for c in e6["CORA_cov"])), pct(max(c[0] for c in e6["CORA_cov"])))
w("fig_rho.tex", fig)
M("EsixCoraZero", pct(e6["CORA"][0][0])); M("EsixCoraOne", pct(e6["CORA"][-1][0]))
M("EsixLooZero", pct(e6["LOO-CF"][0][0])); M("EsixLooOne", pct(e6["LOO-CF"][-1][0]))
M("EsixCovMin", pct(min(c[0] for c in e6["CORA_cov"]))); M("EsixCovMax", pct(max(c[0] for c in e6["CORA_cov"])))
M("EsixSizeZero", e6["CORA_size"][0][0], "{:.2f}"); M("EsixSizeOne", e6["CORA_size"][-1][0], "{:.2f}")

# ------------------------------------------------------------------ E7
e7 = J("e7")
lab7 = [("none", "No shift (control)"), ("subtle_mix", "Fault mix: 50\\% ignored-input"),
        ("evasive_10", "Twin-aware evasion, 10\\%"), ("evasive_20", "Twin-aware evasion, 20\\%"),
        ("evasive_40", "Twin-aware evasion, 40\\%"),
        ("judge_drift", "Judge model drift ($2\\times$ noise)"),
        ("sticky_agents", "Higher error adoption"), ("longer_T36", "Longer workflows ($T{=}36$)")]
rows = []
for k, lab in lab7:
    if k not in e7:
        continue
    d = e7[k]
    rows.append("%s & %.1f & %.1f $\\pm$ %.1f & %.2f & %.1f $\\pm$ %.1f & %.2f \\\\" % (
        lab, pct(d["acc"][0]), pct(d["stale_cov"][0]), pct(d["stale_cov"][1]), d["stale_size"][0],
        pct(d["recal_cov"][0]), pct(d["recal_cov"][1]), d["recal_size"][0]))
tab = r"""\begin{table*}[!t]
\centering
\begin{threeparttable}
\caption{Robustness of the Coverage Guarantee to Distribution Shift and
Twin-Aware Adversaries ($\alpha=0.1$, three seeds, 500 shifted test
incidents per seed)}
\label{tab:shift}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.4}
\begin{tabularx}{\textwidth}{@{}lCCCCC@{}}
\toprule
\textbf{Test-time condition} & \textbf{Step acc.\ (\%%)}
 & \textbf{Coverage, stale $\hat q$ (\%%)} & \textbf{Set size, stale}
 & \textbf{Coverage, recalibrated (\%%)} & \textbf{Set size, recal.} \\
\midrule
%s
\midrule
\multicolumn{6}{@{}l}{\footnotesize stale = threshold from unshifted
calibration;\quad recalibrated = threshold refreshed on 300 shifted incidents}\\
\bottomrule
\end{tabularx}
\begin{tablenotes}[flushleft]\scriptsize
\item Likelihood tables and judge temperature are always those fitted on
the unshifted calibration set; recalibration refreshes only the conformal
threshold $\hat q$. Twin-aware evasion: the faulty agent behaves correctly
whenever it detects that it is executing inside the twin.
\end{tablenotes}
\end{threeparttable}
\end{table*}
""" % "\n".join(rows)
w("tab_shift.tex", tab)
for k, nm in [("none", "None"), ("judge_drift", "Drift"), ("evasive_40", "Evafo"),
              ("sticky_agents", "Sticky"), ("longer_T36", "Long"), ("subtle_mix", "Mix")]:
    if k in e7:
        d = e7[k]
        M("Eseven" + nm + "Stale", pct(d["stale_cov"][0])); M("Eseven" + nm + "Recal", pct(d["recal_cov"][0]))
        M("Eseven" + nm + "StaleSize", d["stale_size"][0], "{:.2f}")
        M("Eseven" + nm + "RecalSize", d["recal_size"][0], "{:.2f}")
        M("Eseven" + nm + "Acc", pct(d["acc"][0]))

# ------------------------------------------------------------------ E8 + Pollaczek-Khinchine
e8 = J("e8")
smp = J("e4_samples")
lam = e8["lam"]


def pk(m, l):
    """M/G/1 mean sojourn time (s) for arrival rate l per hour."""
    lam_s = l / 3600.0
    ES, ES2 = m.mean(), (m ** 2).mean()
    r = lam_s * ES
    if r >= 1:
        return np.inf
    return ES + lam_s * ES2 / (2 * (1 - r))


plots = []
for key, lab, st in [("CORASet", r"\SYS{} set", "CORA"), ("LOOTargeted", "LOO-CF targeted", "LOO"),
                     ("JudgeTargeted", "Judge targeted", "ENS"), ("FullRestart", "Full restart", "B3")]:
    plots.append(r"\addplot[%s, only marks] coordinates {%s};" % (S[st], coords(lam, e8[key])) +
                 "\n\\addlegendentry{%s (sim.)}" % lab)
    m = np.array(smp[key]["mttr"])
    xs = np.linspace(1, 14, 53)
    ys = [l * pk(m, l) / 60.0 for l in xs]
    pts = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y) and y < 3000]
    st2 = S[st].replace("thick", "thin").split(", mark")[0] + ", mark=none, forget plot"
    plots.append(r"\addplot[%s] coordinates {%s};" % (st2, " ".join("(%.3f,%.4g)" % p for p in pts)))
fig = r"""\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth, height=4.5cm, ymode=log,
    xlabel={Incident arrival rate $\lambda$ (per hour)},
    ylabel={SLA-violation slice-min per hour},
    xmin=0.5, xmax=14.5, ymin=1, ymax=3000, %s]
%s
\end{axis}
\end{tikzpicture}
\caption{Aggregate slice SLA-violation time under Poisson incident load
served by one attribution-and-repair pipeline. Markers: discrete-event
simulation (400 simulated hours per point). Lines: the M/G/1
Pollaczek--Khinchine prediction of Proposition~\ref{prop:queue} from the
measured MTTR moments; each line ends where the pipeline saturates.}
\label{fig:sla}
\end{figure}
""" % (AX, "\n".join(plots))
w("fig_sla.tex", fig)
for key, nm in [("CORASet", "Cset"), ("LOOTargeted", "Loo"), ("JudgeTargeted", "Judge"), ("FullRestart", "Full")]:
    m = np.array(smp[key]["mttr"])
    M("Eeight" + nm + "Cap", 3600.0 / m.mean(), "{:.1f}")
    M("Eeight" + nm + "Eight", e8[key][lam.index(8)], "{:.0f}")
    M("Eeight" + nm + "Two", e8[key][lam.index(2)], "{:.1f}")
    M("Eeight" + nm + "PkEight", 8 * pk(m, 8) / 60.0 if np.isfinite(pk(m, 8)) else 0, "{:.0f}")
M("EeightRatioEight", e8["LOOTargeted"][lam.index(8)] / e8["CORASet"][lam.index(8)])
M("EeightRatioTwo", e8["FullRestart"][lam.index(2)] / e8["CORASet"][lam.index(2)])
errs = []
for key in ["CORASet", "LOOTargeted", "JudgeTargeted", "FullRestart"]:
    m = np.array(smp[key]["mttr"])
    for l, v in zip(lam, e8[key]):
        if l * m.mean() / 3600.0 < 0.8:
            errs.append(abs(l * pk(m, l) / 60.0 - v) / v)
M("EeightPkErr", pct(np.median(errs)))
M("EeightPkErrMax", pct(np.max(errs)))

# ------------------------------------------------------------------ E9
e9 = J("e9")
rows = []
for k, d in e9.items():
    rows.append("%s & %.1f $\\pm$ %.1f & %.1f & %.1f & %.2f & %.0f \\\\" % (
        k.replace("CORA", r"\SYS{}").replace("w/o", "w/o"), pct(d["step"][0]), pct(d["step"][1]),
        pct(d["agent"][0]), pct(d["cov"][0]), d["size"][0], d["cost"][0]))
tab = r"""\begin{table*}[!t]
\centering
\begin{threeparttable}
\caption{Ablation of \SYS{} Components ($\alpha=0.1$, $B_{\max}=8$, five seeds,
600 test incidents per seed)}
\label{tab:ablation}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.4}
\begin{tabularx}{\textwidth}{@{}lCCCCC@{}}
\toprule
\textbf{Variant} & \textbf{Step acc.\ (\%%)} & \textbf{Agent acc.\ (\%%)}
 & \textbf{Coverage (\%%)} & \textbf{Set size (steps)} & \textbf{Twin step executions} \\
\midrule
%s
\midrule
\multicolumn{6}{@{}l}{\footnotesize w/o = component removed; each variant
is recalibrated, so every row carries its own valid coverage guarantee}\\
\bottomrule
\end{tabularx}
\begin{tablenotes}[flushleft]\scriptsize
\item IG = expected-information-gain probe selection; cost-aware selection
divides the gain by the square root of the probe's twin cost. DAG pruning
restricts hypotheses to ancestors of the failed commits.
\end{tablenotes}
\end{threeparttable}
\end{table*}
""" % "\n".join(rows)
w("tab_ablation.tex", tab)
nm9 = {"CORA (full)": "Full", "w/o DAG pruning": "Dag", "w/o judge prior": "Prior",
       "w/o prefix probes": "Prefix", "w/o point probes": "Point",
       "w/o cost-aware selection": "Cost", "random probe order": "Rand",
       "w/o replay (judge only)": "Noreplay"}
for k, nm in nm9.items():
    d = e9[k]
    M("Enine" + nm + "Step", pct(d["step"][0])); M("Enine" + nm + "Size", d["size"][0], "{:.2f}")
    M("Enine" + nm + "Cost", d["cost"][0], "{:.0f}"); M("Enine" + nm + "Cov", pct(d["cov"][0]))

w("macros.tex", "%% auto-generated by sim/make_figures.py -- do not edit\n" + "\n".join(macros) + "\n")
print("wrote", len(macros), "macros and figures/tables to", OUT)

# ------------------------------------------------------------------ extras
M("EninePrefixDrop", pct(e9["CORA (full)"]["step"][0] - e9["w/o prefix probes"]["step"][0]))
M("EninePointDrop", pct(e9["CORA (full)"]["step"][0] - e9["w/o point probes"]["step"][0]))
M("EnineRandDrop", pct(e9["CORA (full)"]["step"][0] - e9["random probe order"]["step"][0]))
M("EnineCostGain", pct(e9["w/o cost-aware selection"]["step"][0] - e9["CORA (full)"]["step"][0]))
M("EnineCostExtra", pct(e9["w/o cost-aware selection"]["cost"][0] / e9["CORA (full)"]["cost"][0] - 1))
M("EfiveCpuTwentyFour", e5["CORA"]["cpu_ms"][T.index(24)], "{:.0f}")
import sys
sys.path.insert(0, HERE)
from cora_sim import Config, gen_failing_trace, candidates
_r = np.random.default_rng(7)
_tr = [gen_failing_trace(Config(), _r) for _ in range(2000)]
M("CandFrac", pct(np.mean([len(candidates(t)) / t.T for t in _tr])))
M("CandMean", np.mean([len(candidates(t)) for t in _tr]))
M("RootPersist", pct(np.mean([t.spec[t.root][0] for t in _tr])))
M("FailCommits", np.mean([len(t.failed_commits) for t in _tr]), "{:.2f}")
w("macros.tex", "%% auto-generated by sim/make_figures.py -- do not edit\n" + "\n".join(macros) + "\n")
print("extras done", len(macros))
