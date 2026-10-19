# Point, Prefix, Prove

Simulation code for the paper **"Point, Prefix, Prove: Logarithmic-Replay
Failure Attribution with Coverage Guarantees for Agentic Network
Operations"** (manuscript under review).

When a team of LLM agents pushes a configuration that breaks a 5G/6G network
slice, **CORA** works out which agent caused the failure and at which step.
It does this by replaying suspect steps on a network digital twin with two
kinds of counterfactual probe:

- **point probe:** replace one step's output with a correct reference
  output and re-run everything after it;
- **prefix probe:** replace every step up to a point and re-run the rest.

A Bayesian posterior picks the most informative probe next. Split conformal
calibration turns the result into a set of suspect steps that contains the
true cause with probability at least 1 − α. That set then drives targeted
quarantine and re-execution of the workflow.

> **Scope.** This is a seeded stochastic simulator. It does **not** run real
> LLMs. Agents, LLM judges and the digital twin are stochastic models whose
> parameters are in `cora_sim.Config`. The coverage guarantee is
> distribution-free; the absolute accuracies are specific to this model.

## Quick start

```bash
git clone https://github.com/aamiralmas/point-prefix-prove.git
cd point-prefix-prove
pip install -r requirements.txt
python make_figures.py          # rebuild the paper's figures/tables from results/
python experiments.py           # re-run all experiments (about 25 min on 2 cores)
python experiments.py e1 e4 e8  # or just some of them (e8 needs e4's output)
```

`make_figures.py` writes LaTeX (pgfplots figures, tables and the numeric
macros quoted in the text) to `paper_assets/`.

## Files

| File | Purpose |
|---|---|
| `cora_sim.py` | Workflow generator, fault model, judges, twin replays, CORA, baselines, repair |
| `experiments.py` | Experiments e1–e9; writes `results/*.json` |
| `make_figures.py` | `results/*.json` → `paper_assets/*.tex` |
| `results/` | Outputs of the runs reported in the paper |

## Experiments

| Exp | Content | In the paper |
|---|---|---|
| e1 | Attribution accuracy, cost, role confusion matrices | Table III, Fig. 6 |
| e2 | Coverage and set size versus α | Figs. 7–8 |
| e3 | Accuracy versus replay budget; probe-type variants | Fig. 9 |
| e4 | Repair success and time to repair (MTTR) | Table V, Fig. 10 |
| e5 | Scalability in workflow length | Fig. 12 |
| e6 | Sensitivity to the persistent-fault fraction | Fig. 13 |
| e7 | Distribution shift and twin-aware evasion | Table VI |
| e8 | SLA impact under Poisson load versus the M/G/1 prediction | Fig. 11 |
| e9 | Ablation | Table IV |

## Reproducibility

All randomness comes from `numpy.random.default_rng` with integer seeds
(base `SEED` in `experiments.py`). Each seed uses three disjoint sets of
incidents: 300 to fit the model, 500 to calibrate the conformal threshold,
and 800 to test. Results were produced with Python 3 and NumPy 2.x.

## Citation

A citation entry will be added once the paper is published.

## License

MIT, see `LICENSE`.
