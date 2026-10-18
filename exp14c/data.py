from exp14c.reuse import load
load('data', globals(), (
    ("from exp14c import ROOT", "from exp14c import ROOT, RESULTS as LOCAL_RESULTS"),
    ("RESULTS/'nlp_tokens_128.pt'", "LOCAL_RESULTS/'nlp_tokens_128.pt'"),
    ("RESULTS/'nlp_official_tokens_128.pt'", "LOCAL_RESULTS/'nlp_official_tokens_128.pt'"),
))
