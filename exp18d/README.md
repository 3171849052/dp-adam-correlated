# Exp18d

Offline SST-2 / BERT-Tiny experiment at epsilon=8, delta=1e-5. All new artifacts live here.

The exact Exp18 `Update` and `OnlineNoise` are imported. Exp16e clipping is copied unchanged; its model wrapper changes only the local MODEL_PATH import, avoiding Exp16e package initialization writes. `matrices.py` is the Exp18 solver with T=310, k=5, K=P=62 and epsilon=8; no Exp16 matrix is reused. Raw data is tokenized locally and compared tensor-for-tensor with Exp16e, with its split reproduced exactly. No downloads.

Parameters are frozen in `hyperparameters.json`; protocol in `protocol.json`. Adam/Momentum reuse Exp16e per-method NLP winners. STP-Ex2 transfers the root Exp18c frozen configuration, unchanged, to NLP. Historical results are reference tables under `history/` only.

Run the accepted full experiment with `bash exp18d/start_background.sh`. The launcher activates `curve`, checks the accepted seal, starts a pipeline in an independent session with all stdio redirected, and writes `pipeline.pid`. The scheduler has fixed slots [1,1,2,2,3,3], binds each child by GPU UUID through CUDA_VISIBLE_DEVICES, and records PIDs in `formal_status.json`. Each child writes to its own method/seed directory. Duplicate launch is rejected. No automatic retries; failures stop the stage, preserve outputs, and are never marked complete.

Logs: `logs/pipeline.log`, `logs/formal_*.log`; per-step records: `formal/*/steps.jsonl`. `summary.csv`, `per_seed.csv`, `integrity_audit.json`, and `final_report.md` are refreshed as results complete. They explicitly remain incomplete until all 60 verified formal results exist. Completed results are reused only after artifact hashes pass; incomplete directories require inspection.

One training is (8,1e-5)-DP under the stated GDP fixed-slot accounting without sampling amplification. The 60-run composition is not epsilon=8. Raw losses and clipping diagnostics are internal non-DP audit artifacts.
