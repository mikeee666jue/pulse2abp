# Result files

One `.json` per evaluation, written by `scripts/evaluate.py` unless noted. `python scripts/results_table.py`
turns them into the tables of the main README.

| File | What was evaluated |
|---|---|
| `b0.json` | reference forecasts without a network: climatology, training mean, persistence |
| `b1.json`, `m1.json`, `m3.json` | first-round checkpoints; M3 with a same-patient calibration (`--cal fixed`) |
| `m3_far.json`, `m3_other.json` | M3 with a stale calibration, and with another patient's calibration |
| `m3_noise.json` | M3 with white noise on the pulse (`--ppg noise:0.5`: noise SD 0.75 of the signal SD) |
| `m3_info.json` | `scripts/information.py`: what the pulse adds to M3, 4,000 segments |
| `reanalysis_map.json` | `scripts/reanalyze.py` on the first-round draws of B1, M1 and M3, for MAP |
| `n7200_k19_s50/*_s15k.json`, `*_s15k_seed1.json` | second-round checkpoints (15,000 training steps), seeds 0 and 1 |
| `n7200_k19_s50/m3_nocal.json`, `m3_w0.json`, `m3_w2.json` | first-round M3 without calibration, with the pulse switched off, with the pulse weight doubled |
| `n7200_k19_s50/b1_noise.json`, `m1_noise.json`, `m3_noise.json` | first-round checkpoints with the noisy pulse |
| `n7200_k19_s25/m3.json`, `n7200_k19_s100/m3.json` | first-round M3 sampled with 25 and with 100 steps |
| `comparisons/*.txt` | output of `scripts/steps_compare.py`, `scripts/reanalyze.py` and `scripts/gaps_table.py`: paired differences with bootstrap intervals |
| `general/<run>/abp_from_ppg.json` | general model, ABP from PPG, for `g1`, `g1_seed1`, `g1_abpw`, `g1_abpw_seed1`, `s_ppg`, `g1_ft_ppg`, `g1_twin` (written by `scripts/evaluate_general.py`, like everything under `general/`) |
| `general/g1/`, `general/g1_twin/`: `<target>_from_<visible>[_cal].json` | signal grid: one signal generated from the visible ones, `_cal` = with a calibration reading |
| `general/g1/`, `general/g1_twin/`: `<target>_gap<seconds>_<alone or others>.json` | gap filling: one span of the target hidden, with nothing else visible or with the two other signals visible |
| `general/g1_twin_last/` | the same gap cells and ABP from PPG for the twin's last checkpoint (step 60,000) instead of its kept one (step 6,000) |
| `general/base/<target>_gap<seconds>_<copy or linear>.json` | the same gaps filled without a model |

Settings: files directly in `results/` use 20,000 test segments and 39 draws per segment. The folders
`n7200_k19_s<steps>/` use 7,200 test segments, 19 draws and the stated number of sampling steps. The 7,200
segments are a subset of the 20,000, with the same calibration readings. Everything under `general/` uses
those same 7,200 segments with 19 draws and 50 sampling steps.

## Fields

    config        the command-line options of the evaluation
    train_args    the options the checkpoint was trained with
    n             number of test segments
    metrics       for sbp, dbp and map: mae, me (mean error), sd (of the error), crps, cov90, cov90_nominal,
                  width90; hypo (prevalence and AUROC of MAP below 65 mmHg); tracking_map (see the note below)
    wave_mae      mean absolute error between the true waveform and the mean of the drawn waveforms, in mmHg
    flat_fraction share of drawn waveforms in which the checker found no beat
    by_change     the MAP metrics split by how far the true MAP is from the calibration reading

Files under `general/` also have `target`, `visible`, `gap_sec` and `wave`: the pointwise `crps`, the
error of the mean of the draws (`mae_mean`, `rmse_mean`) on the hidden samples, in the model's units (for
ABP also in mmHg, with the suffix `_mmhg`; for PPG and ECG in units of each segment's standard deviation).
Only whole-signal ABP files carry `metrics`.

All pressures are in mmHg. `tracking_map` is kept for completeness but is not a valid measure (a constant
prediction scores 0.50); the main README uses pulse-following from `reanalysis_map.json` instead.

## Not in this repository

* The draws for every segment (`.npz`, 71 MB in total) and the file with the patient code of every test
  segment (`results/ids/test/meta.npz`). `scripts/reanalyze.py`, `scripts/steps_compare.py` and
  `scripts/pooled_noise.py` need them; `scripts/evaluate.py` writes the draws again, and `--data` can point
  at the prepared dataset instead of `results/ids`.
* Checkpoints: in [pulse2abp-checkpoints](https://github.com/mikeee666jue/pulse2abp-checkpoints), except the PaPaGei arms.

In `config` and `train_args` the cluster account name inside the data path has been replaced by `USER`.
Nothing else in these files was edited.
