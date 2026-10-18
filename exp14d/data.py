from exp14d.reuse import load
load('data', globals(), (
    ("from exp14d import ROOT", "from exp14d import ROOT, RESULTS as LOCAL_RESULTS"),
    ("RESULTS/'nlp_tokens_128.pt'", "LOCAL_RESULTS/'nlp_tokens_128.pt'"),
    ("RESULTS/'nlp_official_tokens_128.pt'", "LOCAL_RESULTS/'nlp_official_tokens_128.pt'"),
))
