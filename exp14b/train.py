from exp14b.reuse import load
load('train', globals(), (
    ("        winner = frozen_row(cfg.task, cfg.method)\n        assert cfg.lr == float(winner['lr']) and cfg.C == float(winner['C'])", "        assert cfg.C == 30"),
    ('eps=1e-8,weight_decay', 'eps=cfg.eps,weight_decay'),
))
