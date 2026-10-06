# Stage 2 completed: tuning and freeze

Search seed: 20261001. Selection: maximum final_test_top1, then smaller C, then smaller lr. Scores below are tuning results, not final multi-seed results.

| Method | C | lr | geom_eps | Tuning final_test_top1 |
|---|---:|---:|---:|---:|
| dp-lora-iid | 1.5 | 0.002 | inactive | 76.76% |
| dp-lora-bandinvmf | 5.0 | 0.015 | inactive | 81.50% |
| dp-lora-iid-scale | 0.3 | 0.002 | 1.0 | 76.81% |
| dp-lora-bandinvmf-scale | 5.0 | 0.02 | 1.0 | 81.62% |

85 unique full tuning trials were trained once. Eight distinct completed configurations were reused across stages. A complete recovery replay reused all 85 trials, trained zero additional trials, and preserved selected_configs.json byte for byte.

Verification: 13 unit tests passed; selected trial privacy, geometry, paired initialization/data order/augmentation/Gaussian innovations and identical raw/scale BandInvMF matrices passed. All 6896 baseline files under existing experiment/data/cache roots have unchanged size and modification time.

Frozen configuration: search/selected_configs.json. Full provenance, stage summaries, completed trial index, stopping decisions and prepared 12-trial manifest are in search/. Detailed checks: stage2_verification.json and stage2_write_scope_verification.json.

Final seeds: 20261011, 20261012, 20261013. The 12 formal trials are prepared and have not been launched. From the repository root, run only after reviewing the frozen configuration:

```bash
source /home/longt29/miniconda3/etc/profile.d/conda.sh && conda activate curve && TMPDIR="$PWD/exp6/runtime/tmp" python -B -m exp6.final_runner --selected exp6/results/search/selected_configs.json --seeds 20261011 20261012 20261013 --gpus 1,2,3 > exp6/results/final_launcher.log 2>&1
```

The launcher uses only physical GPUs 1,2,3 with one training process per GPU, reads data/ and cache/ offline, and writes all logs/caches/results under exp6/. After all final trials complete, report.json/report.md include sample std (ddof=1), per-seed results, metadata/diagnostics, all four factorial effects and the paired interaction MF_gain_scale - MF_gain_raw.
