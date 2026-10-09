# Exp8b final report

70 full trials: 5 epochs, 310 logical steps, physical batch=1000, epsilon=8, delta=1e-5.
Accuracy is evaluated on official SST-2 validation (872 examples). Search used only the held-out 5,349 train examples.
Intervals use Student t(df=9), sample std(ddof=1); accuracy is a fraction. Wins exclude ties.
The final-best comparison is descriptive, chosen by final mean, without multiple-comparison correction; it never changes frozen hyperparameters.

| Method | Mean | Sample std | SE | 95% CI |
|---|---:|---:|---:|---|
| dp-adam-iid | 0.75894 | 0.00483 | 0.00153 | [0.7554903246225967, 0.7623995836342842] |
| dp-adam-sgd-bandinvmf | 0.78142 | 0.00578 | 0.00183 | [0.7772893259227901, 0.7855547107744577] |
| dp-adam-momentum-bandinvmf | 0.75837 | 0.00499 | 0.00158 | [0.7548051009980244, 0.7619380182680308] |
| dp-adam-momentum-bias-bandinvmf | 0.77947 | 0.00382 | 0.00121 | [0.7767365653858994, 0.782208388742541] |
| dp-adam-sgd-bandinvmf-scale | 0.78612 | 0.00811 | 0.00257 | [0.7803197860103676, 0.7919279204116506] |
| dp-adam-momentum-bandinvmf-scale | 0.76628 | 0.00635 | 0.00201 | [0.7617447299374509, 0.7708240774019985] |
| dp-adam-momentum-bias-bandinvmf-scale | 0.78739 | 0.00405 | 0.00128 | [0.7844861877625398, 0.7902844544392953] |

## Raw accuracy in seed order 20261011–20261020

- dp-adam-iid: [0.7603211009174312, 0.7603211009174312, 0.7568807339449541, 0.7557339449541285, 0.7511467889908257, 0.768348623853211, 0.7545871559633027, 0.7591743119266054, 0.7591743119266054, 0.7637614678899083]
- dp-adam-sgd-bandinvmf: [0.7763761467889908, 0.7786697247706422, 0.783256880733945, 0.7763761467889908, 0.7752293577981652, 0.7878440366972477, 0.7752293577981652, 0.7878440366972477, 0.783256880733945, 0.7901376146788991]
- dp-adam-momentum-bandinvmf: [0.7637614678899083, 0.7511467889908257, 0.7626146788990825, 0.7568807339449541, 0.7522935779816514, 0.7614678899082569, 0.7545871559633027, 0.7568807339449541, 0.7580275229357798, 0.7660550458715596]
- dp-adam-momentum-bias-bandinvmf: [0.7752293577981652, 0.7763761467889908, 0.783256880733945, 0.7821100917431193, 0.7740825688073395, 0.7752293577981652, 0.7821100917431193, 0.7809633027522935, 0.7809633027522935, 0.7844036697247706]
- dp-adam-sgd-bandinvmf-scale: [0.7912844036697247, 0.7844036697247706, 0.7878440366972477, 0.7729357798165137, 0.7717889908256881, 0.7878440366972477, 0.7855504587155964, 0.7947247706422018, 0.7889908256880734, 0.7958715596330275]
- dp-adam-momentum-bandinvmf-scale: [0.7740825688073395, 0.7580275229357798, 0.768348623853211, 0.7694954128440367, 0.7545871559633027, 0.7694954128440367, 0.7660550458715596, 0.768348623853211, 0.7614678899082569, 0.7729357798165137]
- dp-adam-momentum-bias-bandinvmf-scale: [0.786697247706422, 0.786697247706422, 0.7844036697247706, 0.7912844036697247, 0.7809633027522935, 0.783256880733945, 0.7912844036697247, 0.7935779816513762, 0.7901376146788991, 0.7855504587155964]

## Paired effects

| Comparison | Mean | Sample std | 95% CI | Wins / 10 |
|---|---:|---:|---|---:|
| dp-adam-sgd-bandinvmf - dp-adam-iid | 0.02248 | 0.00405 | [0.01958, 0.02538] | 10/10 |
| dp-adam-momentum-bandinvmf - dp-adam-sgd-bandinvmf | -0.02305 | 0.00508 | [-0.02669, -0.01941] | 0/10 |
| dp-adam-momentum-bias-bandinvmf - dp-adam-momentum-bandinvmf | 0.02110 | 0.00519 | [0.01739, 0.02481] | 10/10 |
| dp-adam-sgd-bandinvmf-scale - dp-adam-sgd-bandinvmf | 0.00470 | 0.00576 | [0.00058, 0.00882] | 7/10 |
| dp-adam-momentum-bandinvmf-scale - dp-adam-momentum-bandinvmf | 0.00791 | 0.00352 | [0.00539, 0.01043] | 10/10 |
| dp-adam-momentum-bias-bandinvmf-scale - dp-adam-momentum-bias-bandinvmf | 0.00791 | 0.00392 | [0.00511, 0.01071] | 10/10 |
| dp-adam-momentum-bias-bandinvmf-scale - dp-adam-momentum-bandinvmf-scale | 0.02110 | 0.00667 | [0.01633, 0.02587] | 10/10 |
| final best - IID | 0.02844 | 0.00666 | [0.02368, 0.03321] | 10/10 |

## Three workloads × two geometries

| Workload | Standard mean | Scale mean | Paired Scale − Standard |
|---|---:|---:|---:|
| SGD | 0.78142 | 0.78612 | 0.00470 |
| Momentum | 0.75837 | 0.76628 | 0.00791 |
| Momentum-Bias | 0.77947 | 0.78739 | 0.00791 |

## Frozen provenance

Config SHA256: `d5bc9a25308ac9b5d1f1a9187de9fb0fd41491dffef1f9c3d7bf0651277837cd`
Manifest SHA256: `9acc38450c199d6ba0c816f21cd690284ea51afcc2029cd9bd010f2cc156e459`
All trials passed matrix, privacy, training pairing, finite checkpoint/Adam, pretrained, tokenizer, split, and frozen-hash audits.

## Execution and failure history

GPU 0, 1, 2; at most two concurrent isolated trials per GPU.
41 unit tests passed; seven canonical GPU smokes and six additional simultaneous Scale smokes passed.
102 complete search trials: four Standard methods × 12 and three Scale methods × 18. No search OOM or non-finite trials.
Four initial final workers failed before training because concurrent downloads shared a temporary official-validation file. Logs and summaries are archived under `failures/validation_download_race/`.
The final runner now verifies frozen settings and prepares official validation once before launching parallel workers. Only final orchestration and its regression test changed; the original manifest is preserved and the amendment is recorded in the current manifest.
Frozen hyperparameter SHA256 remained unchanged. All four failed starts were rerun; the final CSV contains exactly 70 successful five-epoch trials.

## Compute diagnostics across ten final seeds

Logical step times include forward, both backwards, DP noise, Adam, finite-value checks and device synchronization. Throughput is training examples per second; evaluation time is excluded.

| Method | Mean seconds / logical step | Mean samples / s | Max allocated GiB | Max reserved GiB |
|---|---:|---:|---:|---:|
| dp-adam-iid | 0.2641 | 3804.6 | 3.393 | 3.676 |
| dp-adam-sgd-bandinvmf | 0.2688 | 3734.1 | 3.442 | 3.684 |
| dp-adam-momentum-bandinvmf | 0.2639 | 3797.9 | 3.442 | 3.684 |
| dp-adam-momentum-bias-bandinvmf | 0.2689 | 3734.1 | 3.442 | 3.684 |
| dp-adam-sgd-bandinvmf-scale | 0.2853 | 3512.9 | 3.783 | 3.926 |
| dp-adam-momentum-bandinvmf-scale | 0.2813 | 3563.8 | 3.783 | 3.926 |
| dp-adam-momentum-bias-bandinvmf-scale | 0.2763 | 3623.6 | 3.783 | 3.926 |

## Search and final figures

[Search curves (PNG)](search/search_curves.png) · [Search curves (PDF)](search/search_curves.pdf)
[Final summary (PNG)](final_summary.png) · [Final summary (PDF)](final_summary.pdf)

Actual execution command:

```bash
conda run --no-capture-output -n curve python -B -m exp8b.stage2 --gpus 0 1 2 --per-gpu 2
```

Single-GPU entry remains available:

```bash
conda run --no-capture-output -n curve python -B -m exp8b.stage2 --gpu 0
```
