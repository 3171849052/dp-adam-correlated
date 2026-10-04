"""Recompute the saved frozen-trajectory MF+Muon diagnostic offline."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from exp3.spec import EXP3
from exp3.diagnostics import frozen_trajectory_diagnostic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-dir', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu','cuda:0'), default='cpu')
    parser.add_argument('--probes', type=int, default=2)
    args=parser.parse_args()
    directory=args.trial_dir.resolve()
    assert directory.is_relative_to(EXP3) and args.probes>0
    if args.device=='cuda:0':
        assert os.environ['CUDA_VISIBLE_DEVICES'] in ('1','2','3')
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['method'].startswith('mf_')
    trajectory=torch.load(directory/'muon_trajectory.pt', map_location='cpu', weights_only=True)
    with np.load(directory/'matrices.npz') as matrices:
        report=frozen_trajectory_diagnostic(trajectory,matrices['D'],summary['innovation_std_sum'],1000,
                                            summary['muon_lr'],args.device,args.probes)
    target=directory/f'frozen_trajectory_muon_mf_offline_{args.probes}.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(str(target))


if __name__=='__main__':
    main()
