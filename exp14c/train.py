from exp14c.reuse import load
load('train', globals(), (
    ("assert not cfg.scaled and cfg.seed == 20261111 and cfg.epsilon == 8", "assert not cfg.scaled and cfg.epsilon == 8"),
    ("file_hash(historical/'frozen_configs.csv')", "file_hash(RESULTS/'frozen_manifest.json')"),
    ("'mechanism_metrics.csv')}", "'mechanism_metrics.csv','optimizer_metrics.jsonl')}")
))
