"""Normalize model metadata to its JSON representation before resume validation.

No identity checks are skipped; numerical training code remains frozen.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import exp17.train as train

original_pretrained_vit = train.pretrained_vit

def pretrained_vit_json_metadata():
    model, metadata = original_pretrained_vit()
    return model, json.loads(json.dumps(metadata))

train.pretrained_vit = pretrained_vit_json_metadata
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    train.run(json.loads(parser.parse_args().config))
