# Exp10 final report



All reported Exp10 full trainings are newly executed: 120 search + 24 review + 40 formal = 184.

Search: 20261101, five epochs, exactly 30 configurations per task/method (24 initial + six internal-validation refinements).

Top-2 reviewed on 20261102/03/04. Winner selected by four-seed mean internal accuracy; loss and trial ID break ties.

Formal paired seeds: 20261111–20261120. CV T=250, NLP T=310; search CV T=225.

Physical GPUs 1/2/3 only. FIFO; one CV or two NLP workers per GPU; no task mixing on a GPU.

Per-run epsilon=8, delta=1e-5, no sampling amplification. No epsilon=8 claim for composition of model selection and repeated training.

Internal validation is assumed public/non-protected. Diagnostics use DP optimizer state and public matrix/Gaussian geometry; raw training losses and clipping statistics are not logged.

Exp9 results are historical references, not new Exp10 trainings. Historical CV test-based tuning and prior viewing of SST-2 validation preclude any new blind-test claim.

Uncertainty: sample standard deviation (ddof=1), two-sided Student t 95% confidence intervals over ten seeds. Paired differences use each common seed.



| Task | Method | Mean accuracy | SD | 95% CI | LR | C | Module |
|---|---|---:|---:|---|---:|---:|---:|
| cv | adam-aware-mf | 0.7696 | 0.0055 | [0.7657, 0.7736] | 0.003 | 24.0 | 0.1 |
| cv | blockadam-mf | 0.0214 | 0.0089 | [0.0150, 0.0278] | 0.001 | 8.0 | 2 |
| nlp | adam-aware-mf | 0.7829 | 0.0077 | [0.7774, 0.7884] | 0.005 | 0.8 | 0.3 |
| nlp | blockadam-mf | 0.6604 | 0.0423 | [0.6302, 0.6907] | 0.0008 | 10 | 2 |



Paired accuracy differences (A minus B):

- cv: mean=0.74823, SD=0.01282, 95% CI=[0.73906, 0.75740].

- nlp: mean=0.12248, SD=0.04547, 95% CI=[0.08995, 0.15500].



![Search trajectory](figures/search_trajectory.png)

![Mechanisms](figures/mechanisms.png)



[Module A](module_a_report.md) · [Module B](module_b_report.md) · [Freeze](frozen_manifest.json) · [Audit](audit_summary.json)

Audit: passed; protected historical/data/cache files unchanged: True.

Full numeric accuracy/loss/epsilon/runtime/memory statistics are in summary_statistics.csv. All candidate and five-epoch results are retained in search_results.json, recheck_results.json and final_results.json.
