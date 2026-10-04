# Frozen sequential search

Only tuning seed 20261001 was used. Final validation has not been run.

| Stage | Method | New trials | Top-1 (%) | Muon LR | Adam LR | C | lambda | kappa | rho |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | nonprivate_hybrid | 8 | 85.44 | 0.0003 | 0.001 | 100.0 | 1.0 | 4.0 | 0.1 |
| B | iid_dp_hybrid | 9 | 69.82 | 0.001 | 0.001 | 30.0 | 1.0 | 4.0 | 0.1 |
| C | mf_muon_standard | 12 | 79.05 | 0.006 | 0.012 | 30.0 | 1.0 | 4.0 | 0.1 |
| D | mf_muon_normscale | 6 | 79.05 | 0.006 | 0.012 | 30.0 | 1.0 | 1.0 | 0.1 |
| E | mf_muon_spectralscale | 5 | 79.05 | 0.006 | 0.012 | 30.0 | 1.0 | 1.0 | 0.1 |

## Stage A

Stable small-Muon basin (.0003-.001); increasing Muon to .003/.01/.03 hurts. Adam .001 improves learning speed versus .0002 and finishes highest; .002 loses utility. No search to resolve sub-0.3 pp ties.

Stopped: Eight-trial budget reached; Adam optimum bracketed and large-Muon overshoot established.

## Stage B

C10-30 plateau, larger C60/100 harms utility despite similar normalized update RMS; best C30 has less late clipping. Muon .001 modestly improves .0003, whereas .003 sharply degrades. Adam .0003 underfits and .003 has a declining later curve.

Stopped: Best LR region bracketed by inferior neighbors; clipping trend clear. Stop after nine trials rather than expend the remaining three.

## Stage C

MF requires higher LRs than the IID basin: .001/.001 reaches only42.99%, while .006/.012 reaches79.05%. Adam .024 overshoots (74.54%); .006/.009 are inferior Adam neighbors. C10 and C60 do not improve on C30 at matched LRs. MF frozen perturbation RMSE scales with Muon LR; its lower value at underfitting LRs does not predict better utility. Update-gain temporal CV is about .035, lower than IID but not a standalone utility score. Best final clipping .614 supports a reasonable regime. Muon .006 remains a boundary of the tested LR range.

Stopped: Reached12-trial budget with a clear Adam basin and clipping trend; freeze highest final_test_top1 rather than extend search.

## Stage D

Exact identity lambda1 remains best79.05%. Lambda .1/.01/.0001 at C30 gives76.61/75.80/71.69%; expansion4/100 collapses to10.51/9.33% with near-total clipping and regressing curves. Identity is bracketed by worse moderate neighbors .1 and4. Raising C from30 to60 at lambda .01 restores final clipping .6103 close to Standard .6141 but gives75.53%, so a clipping-operating-point change does not rescue geometry. Noise-weighted JVP and frozen final RMSE decrease while utility worsens; actual Muon update RMS stays near .000342. Temporal gain CV rises from identity .0351 to .1871/.5787/1.033 under stronger compression, consistent with disruption and utility loss. No credible nonidentity geometry basin warrants LR tuning.

Stopped: Two completed rounds without any improvement over identity, lower-performing neighbors bracket lambda1, and clipping compensation fails. Stop at6 new trials plus1 reused identity, without spending the16-trial limit.

## Stage E

Identity kappa1 remains highest79.05%. At rho.1, kappa1.1/1.25/2/4 gives77.64/76.58/74.56/73.38%, a clear monotone decline away from the allowed lower bound. Raising C30->60 at kappa1.25 lowers final clipping .6693->.5933 but gives76.28%, so clipping compensation does not rescue utility. Raw innovation sigma doubles, while noise-weighted JVP and frozen RMSE stay essentially unchanged at matched kappa/LRs; do not explain the small loss as a doubled measured update-noise amplitude. Temporal gain CV and frozen cumulative RMSE rise with kappa and utility deteriorates; within-probe CV and instantaneous gain fall without utility benefit. Best active Version B77.64% exceeds best active Version A76.61% by1.03pp, but neither beats Standard. The frozen choices for both versions are exact identity.

Stopped: Two rounds without improvement over the79.05% identity control, monotone performance loss at higher kappa, and failed clipping compensation. Stop at5 new trials plus1 reused identity; no nonidentity basin warrants LR tuning. Keep rho.1: neither utility nor temporal-cancellation diagnostics supports an extra rho search, and reducing strength toward identity has already been tested.

## Diagnostics

Frozen-state first-order estimates, not exact nonlinear training dynamics; different LRs scale cumulative RMSE, so these diagnostics do not rank utility.

Detailed diagnostics, epoch curves, and identity reuse references are in search_report.json and search_history.json.

## Final validation command

```bash
conda activate curve
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp3/tmp" python -m exp3.final_runner --frozen-config exp3/results/selected_configs.json --result-dir exp3/results/final
```
