# Exp14a startup

Tests: passed. Eight full-batch two-step GPU smokes: passed.

Launch: `conda run --no-capture-output -n curve python -B -m exp14a.start_background`

Scheduler PID: 481168; detached session ID: 481168; parent PID: 1.

- cv dp-adam-iid: running, GPU 0, PID 481169
- cv dp-adam-sgd-bandinvmf: running, GPU 1, PID 481170
- cv dp-adam-momentum-bandinvmf: running, GPU 2, PID 481171
- cv dp-adam-momentum-bias-bandinvmf: running, GPU 3, PID 481172
- nlp dp-adam-iid: queued, GPU pending, PID pending
- nlp dp-adam-sgd-bandinvmf: queued, GPU pending, PID pending
- nlp dp-adam-momentum-bandinvmf: queued, GPU pending, PID pending
- nlp dp-adam-momentum-bias-bandinvmf: queued, GPU pending, PID pending

Live status: scheduler_status.json; logs: scheduler.log and scheduler.jsonl.
