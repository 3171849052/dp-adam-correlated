# adam-aware-mf



Standard PyTorch Adam is unchanged. A 4-band inverse MF minimizes normalized Exp9 full Momentum-Bias J1 plus lambda times normalized Gaussian J2. C_clip=1 during optimization; each training recalibrates true strategy sensitivity at its chosen C.



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
| cv | adam-aware-mf | 0.7100 | 0.0000 | [0.7100, 0.7100] | 0.0024 | 100 | 3.0 |
| cv | blockadam-mf | 0.7000 | 0.0000 | [0.7000, 0.7000] | 0.006 | 30 | 25 |
| nlp | adam-aware-mf | 0.7100 | 0.0000 | [0.7100, 0.7100] | 0.004 | 1 | 3.0 |
| nlp | blockadam-mf | 0.7000 | 0.0000 | [0.7000, 0.7000] | 0.004 | 20 | 25 |



Formal mechanism geometry:

- cv: d=[1.0, -0.21647249300436122, -0.25550449875710424, -0.31500965599665587], sensitivity=2.9509028, J1=8815.3164, J2(C=1)=7.0560498e-13, J2(actual C)=7.0560498e-05.

- nlp: d=[1.0, -0.21237595395333064, -0.26118206189336185, -0.33187039520858463], sensitivity=3.0289378, J1=9324.9111, J2(C=1)=6.7108799e-13, J2(actual C)=6.7108799e-13.



![Mechanisms](figures/mechanisms.png)



Preconditioner diagnostics use the first 64 flattened coordinates per parameter tensor. Denominator lag is the mean absolute relative difference between the frozen and live DP denominator.

Pure MF cumulative first-moment noise includes (1-beta1)=0.1; it excludes gradients, adaptive preconditioning, and parameter feedback. It is a mechanism diagnostic, not total optimization-error variance.
