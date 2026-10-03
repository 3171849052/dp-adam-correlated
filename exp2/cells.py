"""The fixed 2 geometries × 3 noise mechanisms and experiment grids."""
CELLS = {
    'iid_standard': dict(geometry='standard', noise='iid'),
    'iid_scale': dict(geometry='scale', noise='iid'),
    'prefix_standard': dict(geometry='standard', noise='prefix_bandinvmf'),
    'prefix_scale': dict(geometry='scale', noise='prefix_bandinvmf'),
    'momentum_standard': dict(geometry='standard', noise='momentum_bandinvmf'),
    'momentum_scale': dict(geometry='scale', noise='momentum_bandinvmf'),
}
SEARCH_SEED = 20261001
FINAL_SEEDS = (20261011, 20261012, 20261013)
CLIPS = (50, 100, 200, 300, 500)
LR_GRIDS = {
    'iid_scale': (5e-4, 1e-3, 2e-3, 3e-3),
    'prefix_standard': (5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3),
    'momentum_scale': (1e-3, 2e-3, 3e-3, 5e-3, 7e-3),
}
ANCHORS = {
    'iid_standard': dict(lr=5e-4, max_grad_norm=1.),
    'prefix_scale': dict(lr=2e-3, max_grad_norm=200., eps_scale=.1, num_bands=4),
    'momentum_standard': dict(lr=5e-3, max_grad_norm=1., num_bands=4),
}


def job(method, lr, clip, seed=SEARCH_SEED):
    values = dict(method=method, lr=lr, max_grad_norm=float(clip), seed=seed)
    if CELLS[method]['geometry'] == 'scale':
        values['eps_scale'] = .1
    if CELLS[method]['noise'] != 'iid':
        values['num_bands'] = 4
    return values


def first_search_trials():
    clips = [dict(job(method, 2e-3, clip), name=f'{method}/clip_{clip}', stage='clip')
             for method in ('iid_scale', 'momentum_scale') for clip in CLIPS]
    prefix = [dict(job('prefix_standard', lr, 1),
                   name=f'prefix_standard/lr_{lr:g}', stage='lr')
              for lr in LR_GRIDS['prefix_standard']]
    return clips + prefix


def second_search_trials(best_clips):
    return [dict(job(method, lr, best_clips[method]),
                 name=f'{method}/lr_{lr:g}_clip_{best_clips[method]:g}', stage='lr')
            for method in ('iid_scale', 'momentum_scale') for lr in LR_GRIDS[method]
            if lr != 2e-3]


def final_trials(configs):
    assert set(configs) == set(CELLS)
    return [dict(configs[method], method=method, seed=seed,
                 name=f'{method}/seed_{seed}', stage='final')
            for seed in FINAL_SEEDS for method in CELLS]
