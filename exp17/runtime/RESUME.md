# GPU 2/3 resume operation

Launch from the repository root:

    /home/longt29/miniconda3/envs/curve/bin/python -B exp17/runtime/resume_gpus_2_3.py

Use this launcher for this experiment's continuation. The original launcher still selects GPUs 1/2/3 and lacks the metadata compatibility adapter.

The runtime worker adapter normalizes pretrained model metadata through JSON before the existing strict checkpoint comparison. This resolves tuple/list representation differences for mean/std without disabling any config, input, initialization, or checkpoint identity check. Training formulas and the accepted source files are unchanged. Runtime adapters are operational additions outside the frozen source manifest; their hashes and failed-attempt artifacts are recorded in results/resume_repair.json.
