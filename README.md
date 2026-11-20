# pulse2abp

Sample arterial blood pressure (ABP) waveforms from the pulse signal (PPG), optionally given one earlier
blood-pressure reading from the same patient (the "calibration"), and measure what the generator relies
on when the pulse and the calibration disagree.

A second stage extends the same network to three signals, ABP, PPG and ECG: one general model that
generates whatever is hidden, whether that is a whole signal or a stretch of time.

Course project for CMU 10-423/623 (Generative AI), fall 2026. **Work in progress:** the numbers below are
midway results on one dataset, from one or two training runs per model.

## Models

Every network maps a 10 s PPG segment (1,250 samples at 125 Hz) to a 10 s ABP waveform. Systolic, diastolic
and mean pressure (SBP, DBP, MAP) are then read from the waveform by one beat-detection routine,
`p2a/checker.py`, which scores the true and the generated waveform in the same way.

| Arm | Option of `scripts/train.py` | What it is |
|---|---|---|
| B0 | none (`evaluate.py --baselines-only`) | reference forecasts without a network: climatology (draws from the training distribution) and persistence (the calibration reading plus draws from the training distribution of within-patient changes) |
| B1 | `--mode reg` | 1D U-Net regression, PPG to ABP |
| M1 | `--mode gen` | the same U-Net as a conditional diffusion model |
| M1 flow matching | `--mode gen --path linear` | flow matching instead of diffusion, everything else identical |
| M2 | `--mode gen --encoder {frozen,lora,full,scratch}` | M1 plus features from PaPaGei, a pretrained PPG model; `scratch` is the same encoder started from random weights, the control for pretraining |
| M3 | `--mode gen --use-cal` | M1 plus the calibration reading; pulse and calibration are dropped independently during training, so each can be switched off or re-weighted when sampling |

### The general model

The second stage keeps the U-Net and the loss and extends them to three signals and to time. The network
is given ABP, PPG and ECG together with a mask that says which samples are visible, and it generates
everything that is hidden. Hiding a whole signal is translation (ABP from PPG, PPG from ABP, and so on);
hiding one stretch of a signal is fill-in. The first-stage task is then one mask among many: PPG visible,
ABP and ECG hidden. Every generator has a regression twin, the same network trained on the same masks
that returns one prediction instead of draws.

| Run | Option of `scripts/train_general.py` | What it is |
|---|---|---|
| `g1` | `--mode gen` | general generator, random masks |
| `g1_twin` | `--mode reg` | its regression twin, same masks |
| `g1_abpw` | `--mode gen --p-hide 0.6 0.25 0.4` | general generator with "ABP from PPG" trained more often |
| `s_ppg` | `--mode gen --fixed-visible ppg --steps 15000 --val-every 1000` | one mask only: PPG visible, ABP and ECG hidden |
| `g1_ft_ppg` | `--mode gen --init runs/g1/best.pt --fixed-visible ppg --steps 5000 --lr 5e-5 --warmup 200 --val-every 500` | `g1` trained further on that one mask |

Settings that define what these runs measure:

* **Training masks.** Each signal of each example is hidden whole with probability 0.4, gets one hidden
  span with probability 0.2 (length log-uniform between 0.25 and 8 s), and is visible otherwise. An example
  with nothing hidden gets one signal hidden. The mask "PPG visible, ABP and ECG hidden" is then 6% of the
  training examples.
* **Loss.** On hidden samples only. Each example counts once, as the mean over its hidden samples.
* **Calibration.** The reading is given in half of the training examples.
* **Checkpoint.** The one with the lowest validation loss over a fixed set of random masks; 60,000 training
  steps by default.
* **No guidance weights.** An input is switched off by hiding it in the mask.

## Results so far

Data: the VitalDB part of PulseDB. Training uses 442,080 segments from 1,228 patients and validation
23,400 segments from 65 further patients. The test set has 57,600 segments from 144 patients, none of
whom appear in training or validation. Each evaluation below scores a fixed random subset of the test
set. All errors are in mmHg. The section "How to read the numbers" explains the columns.

Every table is the output of `python scripts/results_table.py`, which reads the metric files in
`results/` and needs neither data nor a GPU.

**Table 1. First round: 20,000 test segments, 39 draws per segment, 50 sampling steps**

| Arm | Given | SBP MAE | DBP MAE | MAP MAE | MAP CRPS | MAP 90% coverage | Hypotension AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 climatology | nothing | 15.05 | 9.44 | 11.32 | 7.99 | 0.904 | 0.500 |
| B0 persistence | calibration | 15.77 | 8.30 | 11.32 | 8.26 | 0.895 | 0.757 |
| B1 regression | pulse | 12.75 | 7.91 | 9.04 | – | – | 0.826 |
| M1 diffusion | pulse | 12.75 | 8.00 | 9.19 | 6.45 | 0.854 | 0.812 |
| M3 diffusion | pulse + calibration | 11.38 | 6.30 | 7.77 | 5.48 | 0.855 | 0.879 |

**Table 2. The same 20,000 segments, MAP, read again from the saved draws (144 patients)**

| Arm | MAE | CRPS, point + fixed bar | CRPS, own draws | Pulse-following | Calibration-following | Between-patient R2 |
|---|---:|---:|---:|---:|---:|---:|
| B1 regression | 9.04 | 6.46 | – | 0.47 | 0.00 | 0.17 |
| M1 diffusion | 9.19 | 6.53 | 6.45 | 0.36 | 0.00 | 0.17 |
| M3 diffusion | 7.77 | 5.57 | 5.48 | 0.43 | 0.27 | 0.59 |

Differences with 95% intervals from a bootstrap over patients:

    MAE   M1 - B1                               +0.14  [-0.13, +0.42]
    MAE   M3 - B1                               -1.28  [-1.70, -0.84]
    MAE   M3 - (B1 + calibration, linear)       -0.58  [-0.80, -0.34]
    CRPS  M1 own draws - (B1 + fixed bar)       -0.02  [-0.21, +0.18]
    CRPS  M3 own draws - (B1 + fixed bar)       -0.98  [-1.27, -0.69]

**Table 3. Second round, 15,000 training steps: 7,200 test segments, 19 draws, 50 sampling steps**

| Arm | Seed | SBP MAE | DBP MAE | MAP MAE | MAP CRPS | MAP 90% coverage |
|---|---:|---:|---:|---:|---:|---:|
| B1 regression | 0 | 13.25 | 8.11 | 9.24 | – | – |
| M1 diffusion | 0 | 12.79 | 8.14 | 9.30 | 6.46 | 0.860 |
| M1 diffusion | 1 | 12.66 | 8.11 | 9.24 | 6.44 | 0.859 |
| M1 flow matching | 0 | 12.49 | 7.99 | 9.13 | 6.35 | 0.863 |
| M1 flow matching | 1 | 12.60 | 8.10 | 9.22 | 6.40 | 0.871 |
| M2 PaPaGei, frozen | 0 | 12.78 | 8.13 | 9.31 | 6.48 | 0.858 |
| M2 PaPaGei, LoRA | 0 | 12.50 | 7.97 | 9.11 | 6.35 | 0.857 |
| M2 PaPaGei, LoRA | 1 | 12.49 | 8.02 | 9.14 | 6.37 | 0.862 |
| M2 PaPaGei, fully fine-tuned | 0 | 12.73 | 8.14 | 9.30 | 6.47 | 0.861 |
| M2 same encoder, random start | 0 | 12.83 | 8.25 | 9.46 | 6.63 | 0.848 |
| M3 diffusion + calibration | 0 | 11.53 | 6.38 | 7.87 | 5.49 | 0.870 |

Each arm against M1 on the same segments, with intervals, is in `results/comparisons/`.

**Table 4. M3 (first-round checkpoint) when its inputs are removed, corrupted or wrong, MAP**

| Test | Segments | MAE | CRPS | 90% coverage | 90% interval width |
|---|---:|---:|---:|---:|---:|
| as trained | 20,000 | 7.77 | 5.48 | 0.855 | 29.2 |
| stale calibration (`--cal far`) | 20,000 | 12.55 | 9.16 | 0.682 | 32.7 |
| another patient's calibration (`--cal other`) | 20,000 | 9.94 | 7.21 | 0.753 | 29.4 |
| noisy pulse (`--ppg noise:0.5`) | 20,000 | 10.58 | 7.30 | 0.865 | 42.8 |
| no calibration (`--cal none`) | 7,200 | 9.08 | 6.35 | 0.849 | 34.7 |
| pulse switched off (`--w-ppg 0`) | 7,200 | 9.75 | 6.81 | 0.869 | 40.3 |
| pulse weight doubled (`--w-ppg 2`) | 7,200 | 9.57 | 7.10 | 0.696 | 25.0 |

**Table 5. M3 with 25 and 100 sampling steps instead of 50, MAP**

| Sampling steps | Segments | MAE | CRPS | 90% coverage | 90% interval width |
|---:|---:|---:|---:|---:|---:|
| 25 | 7,200 | 7.76 | 5.50 | 0.817 | 27.5 |
| 100 | 7,200 | 7.79 | 5.42 | 0.881 | 32.8 |

### What the numbers show

* **The pulse alone says little about a patient's pressure level.** B1 and M1 lower the MAP error from
  11.3 (climatology) to about 9.1, and they explain 17% of the differences between patients (Table 2).
* **One calibration reading helps more than any change to the generator.** M3 is 1.28 mmHg better than B1
  on MAP and explains 59% of the differences between patients. It is also 0.58 better than a linear
  combination of B1 and the calibration reading.
* **Without calibration, generation has not beaten regression.** M1 and B1 have the same point error
  within the interval, and M1's own draws score no better than B1 with one fixed error bar
  (CRPS difference -0.02, interval -0.21 to +0.18).
* **The intervals are too narrow.** A nominal 90% interval covers 85 to 87% of the true values. Part of
  this is the sampler: coverage is 0.817, 0.856 and 0.881 with 25, 50 and 100 sampling steps, while the
  error does not change (Table 5; the 50-step value on those segments is in
  `results/comparisons/sampling_steps.txt`).
* **PaPaGei helps only with LoRA, and only a little.** LoRA minus M1 on the MAP error is
  -0.194 [-0.314, -0.073] on seed 0 and -0.099 [-0.169, -0.015] on seed 1. Frozen and fully fine-tuned
  encoders do not differ from M1 (+0.011 and +0.003, both intervals include zero), and the same encoder
  started from random weights is worse, +0.156 [+0.005, +0.311].
* **Flow matching against diffusion is undecided.** Flow matching minus M1 on the MAP error is
  -0.174 [-0.321, -0.035] on seed 0 and -0.021 [-0.060, +0.021] on seed 1.
* **M3 does not signal a wrong calibration, but it does signal a noisy pulse** (Table 4). With a stale
  calibration the error rises to 12.55 and coverage falls to 0.68, while the interval is hardly wider
  (29.2 to 32.7). With noise on the pulse the interval widens from 29.2 to 42.8 and coverage holds.
* **Doubling the pulse weight is not a fix.** It narrows the interval to 25.0 and coverage falls to 0.70.
  Guidance weights other than 1 do not sample a posterior.

### Limits

* One dataset, one hospital's surgical patients, 10 s segments. The MIMIC-III part of PulseDB is not run.
* The calibration is the SBP and DBP of another segment of the same patient, taken from the arterial line.
  It stands in for an earlier cuff reading. The segments carry no timestamps, so "earlier" is not enforced.
* One training seed for most arms and two for three of them. Intervals reflect the choice of test
  patients, not run-to-run variation, and are not corrected for the number of comparisons.
* The fixed error bar and the linear reference in Table 2 are fitted on half of the test patients and
  scored on the other half.
* Training settings are first guesses and are not tuned. In the first round every arm reached its best
  validation score within 30,000 of 100,000 steps and got worse afterwards; the reported checkpoint is the
  one with the lowest validation MAP CRPS. The second round uses a shorter schedule, which also shortens
  the learning-rate decay, so compare arms within a round.

## Results so far: the general model

Data: the same VitalDB files converted again with the ECG. `scripts/compare_roots.py` confirmed that the
patients, the PPG and the ABP equal the first-stage data, and no ECG segment was flat or held missing
values. Every evaluation uses the 7,200 test segments of Table 3, checked to be the same segments, with
19 draws and 50 sampling steps. Tables 6 to 11 are also printed by `python scripts/results_table.py`.

**Table 6. General model, ABP from PPG: 7,200 test segments, 19 draws, 50 sampling steps**

| Model | Seed | SBP MAE | DBP MAE | MAP MAE | MAP CRPS | MAP 90% coverage |
|---|---:|---:|---:|---:|---:|---:|
| g1, general generator | 0 | 12.93 | 8.06 | 9.30 | 6.49 | 0.848 |
| g1, general generator | 1 | 12.92 | 8.11 | 9.36 | 6.52 | 0.848 |
| g1_abpw, ABP from PPG trained more often | 0 | 12.51 | 7.89 | 9.05 | 6.33 | 0.839 |
| g1_abpw, ABP from PPG trained more often | 1 | 12.82 | 8.08 | 9.32 | 6.53 | 0.833 |
| s_ppg, one mask only (PPG visible) | 0 | 12.76 | 8.11 | 9.31 | 6.48 | 0.849 |
| g1_ft_ppg, g1 fine-tuned on that mask | 0 | 12.79 | 8.07 | 9.28 | 6.46 | 0.850 |
| g1_twin, regression twin of g1 | 0 | 12.94 | 7.96 | 9.10 | – | – |
| M1, single-task diffusion (Table 3) | 0 | 12.79 | 8.14 | 9.30 | 6.46 | 0.860 |
| M1, single-task diffusion (Table 3) | 1 | 12.66 | 8.11 | 9.24 | 6.44 | 0.859 |

**Table 7. Signal grid, ABP generated from other signals: `g1` and its regression twin**

| Visible | Calibration | g1 SBP MAE | g1 DBP MAE | g1 MAP MAE | g1 MAP CRPS | g1 MAP 90% coverage | twin SBP MAE | twin DBP MAE | twin MAP MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPG | no | 12.93 | 8.06 | 9.30 | 6.49 | 0.848 | 12.94 | 7.96 | 9.10 |
| PPG | yes | 11.79 | 6.47 | 8.01 | 5.60 | 0.862 | 11.83 | 6.23 | 7.74 |
| ECG | no | 14.42 | 9.09 | 10.86 | 7.64 | 0.837 | 14.48 | 8.83 | 10.49 |
| ECG | yes | 13.13 | 7.26 | 9.39 | 6.59 | 0.849 | 13.25 | 6.99 | 9.06 |
| PPG + ECG | no | 12.15 | 7.99 | 9.07 | 6.41 | 0.820 | 12.47 | 7.85 | 8.98 |
| PPG + ECG | yes | 11.21 | 6.44 | 7.82 | 5.48 | 0.849 | 11.44 | 6.26 | 7.73 |

**Table 8. Signal grid, PPG and ECG generated from other signals (pointwise scores in units of each segment's SD)**

| Target | Visible | g1 CRPS | g1 error of mean | twin error |
|---|---|---:|---:|---:|
| PPG | ABP | 0.149 | 0.211 | 0.210 |
| PPG | ECG | 0.187 | 0.266 | 0.265 |
| PPG | ABP + ECG | 0.149 | 0.211 | 0.209 |
| ECG | PPG | 0.338 | 0.495 | 0.467 |
| ECG | ABP | 0.298 | 0.425 | 0.411 |
| ECG | PPG + ABP | 0.299 | 0.425 | 0.412 |

**Table 9. Filling a gap in the ABP: error on the hidden span (mmHg)**

| Gap | g1 alone, CRPS | g1 alone, error of mean | g1 others, CRPS | g1 others, error of mean | twin alone | twin others | twin at step 60,000, alone | twin at step 60,000, others | copy | line |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 s | 0.53 | 0.77 | 0.35 | 0.50 | 2.17 | 1.35 | 1.48 | 0.76 | 2.21 | 7.39 |
| 1 s | 0.94 | 1.39 | 0.49 | 0.70 | 2.90 | 1.51 | 2.05 | 0.98 | 2.47 | 17.08 |
| 2 s | 1.47 | 2.30 | 0.63 | 0.91 | 4.67 | 1.60 | 2.83 | 1.18 | 3.12 | 17.11 |
| 4 s | 3.73 | 6.60 | 0.71 | 1.03 | 9.07 | 1.64 | 6.80 | 1.27 | 3.88 | 17.12 |
| 8 s | 7.43 | 11.71 | 1.01 | 1.47 | 11.83 | 2.31 | 10.59 | 1.71 | 14.57 | 17.23 |

In Table 9, "alone" means nothing but the rest of the ABP is visible and "others" means the PPG and the
ECG are visible as well. "Copy" repeats the neighbouring beat and "line" draws a straight line across the
gap; neither uses a model. Tables 10 and 11, the same for gaps in the PPG and in the ECG, show the same
pattern and are printed by the script.

### What the numbers show

Differences are on the same test segments with 95% intervals from a bootstrap over patients; the full
output is in `results/comparisons/general_*.txt`.

* **One general model costs nothing measurable on MAP for ABP from PPG.** `g1` minus M1 on the MAP error
  is -0.004 [-0.164, +0.172] and +0.053 [-0.095, +0.208] against the two M1 seeds, and for the second seed
  of `g1` +0.054 [-0.121, +0.230] and +0.111 [-0.032, +0.260]. The single-mask model `s_ppg` minus `g1` is
  +0.017 [-0.107, +0.150]. Two smaller differences do repeat: the systolic error is higher than that of M1
  seed 1 (+0.269 [+0.075, +0.468] and +0.253 [+0.049, +0.451] for the two seeds of `g1`), and coverage is
  about 0.01 lower than M1's.
* **Training "ABP from PPG" more often did not reproducibly help.** `g1_abpw` minus `g1` is
  -0.249 [-0.416, -0.080] on seed 0 and -0.037 [-0.140, +0.067] on seed 1, and the second seed of `g1_abpw`
  is 0.271 [0.129, 0.423] higher than its first. The seed-0 run was the unusual one.
* **Training `g1` further on the single task changed nothing measurable:** -0.020 [-0.128, +0.086].
* **Generation against regression, as in the first stage.** The point error of `g1` is above its twin's,
  +0.20 [+0.06, +0.33] and +0.25 [+0.13, +0.38] for the two seeds, and its own draws score level with the
  twin plus one fixed error bar (CRPS +0.02 [-0.12, +0.17] and +0.06 [-0.08, +0.20]).
* **The ECG alone says less about blood pressure than the PPG alone** (MAP error +1.568 [+0.777, +2.359]),
  and adding it to the PPG is not established (-0.231 [-0.600, +0.154], with coverage 0.027 lower).
* **With a calibration reading** the general model from the PPG is slightly behind M3
  (+0.138 [+0.023, +0.256]) and level with it once the ECG is added (-0.054 [-0.326, +0.225]).
* **Gaps are where the other signals matter most.** With PPG and ECG visible, `g1` fills a gap in the ABP
  to 0.5 to 1.5 mmHg at every length. Alone, its draws score below the copy filler at 0.25, 1, 2 and 8 s
  (CRPS minus copy error -1.685 [-1.922, -1.518] at 0.25 s) and level with it at 4 s
  (-0.147 [-0.406, +0.067]). At 4 s the mean of the draws is behind copy by 2.72 [2.48, 2.98]: with a long
  gap and no other signal the beat positions are uncertain, and averaging the draws blurs them.
* **How the twin does on gaps depends on its checkpoint.** The checkpoint rule kept the twin at step 6,000.
  At step 60,000 it is better in every gap cell (Table 9) and not distinguishable on ABP from PPG
  (MAP error +0.169 [-0.115, +0.440]). Against the twin at step 60,000, the mean of `g1`'s draws is ahead
  by about 0.25 mmHg when the other signals are visible (-0.257 [-0.275, -0.242] at 0.25 s,
  -0.243 [-0.264, -0.222] at 8 s) and behind by 1.13 [1.07, 1.20] at 8 s alone.

### Limits

* Two training seeds for `g1` and `g1_abpw`, one for the other runs. The `g1_abpw` result shows that a
  difference of 0.25 mmHg between two runs can be run-to-run variation even when its interval excludes
  zero: the interval covers the choice of test patients, not the training seed.
* `g1`, `g1_twin` and `g1_abpw` trained for 60,000 steps, `s_ppg` and the M1 runs for 15,000.
* `s_ppg` is not M1 trained again: with only the PPG visible it also generates the ECG, which is half of its
  loss, and like every general-model run it is given the calibration reading in half of its training
  examples.
* The checkpoint rule looks at all masks together. For the twin it picked a checkpoint that is far from
  its best on gaps; no gap score is tracked during training.
* Scores for generated PPG and ECG are pointwise, so a beat drawn slightly early or late is penalised
  twice. There is no beat-timing score yet.
* The fixed error bar is fitted on half of the test patients and scored on the other half.

## Run it

Self-test without any data, about 10 minutes on a laptop:

    pip install -r requirements.txt
    python -m pytest tests/test_core.py -q
    python scripts/make_synthetic.py --root /tmp/p2a_synth
    python scripts/train.py --data /tmp/p2a_synth --mode gen --use-cal --base 16 --bs 32 --steps 300 \
        --val-every 300 --val-n 64 --val-k 9 --val-steps 10 --workers 0 --out /tmp/p2a_run

Print the tables above from the metric files:

    python scripts/results_table.py

### Data

The data are not in this repository. We used the VitalDB subset files of PulseDB, which are on Kaggle
(https://doi.org/10.34740/KAGGLE/DS/2447469): `VitalDB_Train_Subset.mat` for training and validation, and
`VitalDB_CalFree_Test_Subset.mat`, PulseDB's calibration-free test set, for testing.

    export DATA=/path/to/p2a_vital
    python scripts/prep_pulsedb.py --mat VitalDB_Train_Subset.mat        --root $DATA --split train --val-frac 0.05
    python scripts/prep_pulsedb.py --mat VitalDB_CalFree_Test_Subset.mat --root $DATA --split test

Read the sanity report it prints: the ABP percentiles must look like mmHg, and the checker must agree with
the dataset's own SBP and DBP labels to within a few mmHg. On our copy the 1st, 50th and 99th percentiles
of the training ABP were 43.4, 78.3 and 147.6 mmHg, and the mean absolute difference from the labels was
1.33 mmHg for SBP and 0.65 for DBP.

### Train

    python scripts/train.py --data $DATA --mode reg           --out runs/b1
    python scripts/train.py --data $DATA --mode gen           --out runs/m1
    python scripts/train.py --data $DATA --mode gen --use-cal --out runs/m3

    # PaPaGei arms: clone https://github.com/Nokia-Bell-Labs/papagei-foundation-model into $PAPAGEI
    # and download papagei_s.pt from https://zenodo.org/records/13983110
    python scripts/train.py --data $DATA --mode gen --encoder lora --papagei-repo $PAPAGEI \
        --papagei-weights $PAPAGEI/weights/papagei_s.pt --out runs/m2_lora

The first round used the defaults (100,000 steps, batch 256, base width 64, learning rate 2e-4). The second
round added `--steps 15000 --val-every 1000 --val-n 2048`. Each run writes `best.pt`, the checkpoint with
the lowest validation MAP CRPS, and `last.pt` for `--resume`. Evaluate `best.pt`. The exact settings and
the training and validation history of every run are in `runs/<name>/args.json` and `log.jsonl`.

### Evaluate

    python scripts/evaluate.py --data $DATA --baselines-only --out results/b0.json
    python scripts/evaluate.py --data $DATA --ckpt runs/b1/best.pt --out results/b1.json
    python scripts/evaluate.py --data $DATA --ckpt runs/m1/best.pt --out results/m1.json
    python scripts/evaluate.py --data $DATA --ckpt runs/m3/best.pt --cal fixed --out results/m3.json

    # M3 with wrong, missing or re-weighted inputs (Table 4)
    python scripts/evaluate.py --data $DATA --ckpt runs/m3/best.pt --cal far   --out results/m3_far.json
    python scripts/evaluate.py --data $DATA --ckpt runs/m3/best.pt --cal other --out results/m3_other.json
    python scripts/evaluate.py --data $DATA --ckpt runs/m3/best.pt --cal fixed --ppg noise:0.5 --out results/m3_noise.json
    python scripts/evaluate.py --data $DATA --ckpt runs/m3/best.pt --cal fixed --w-ppg 2.0 --n 7200 --k 19 \
        --out results/n7200_k19_s50/m3_w2.json

    # what the pulse adds, as information
    python scripts/information.py --data $DATA --ckpt runs/m3/best.pt --cal fixed --out results/m3_info.json

Each evaluation writes a `.json` with the metrics and a `.npz` with the draws for every segment. The
defaults are 20,000 segments and 39 draws; `--n 7200 --k 19` gives the smaller setting, which we used for
the second round and wherever a guidance weight other than 1 triples the cost. `scripts/reanalyze.py`,
`scripts/steps_compare.py` and `scripts/pooled_noise.py` read the `.npz` files and produce Table 2 and the
paired comparisons without a GPU.

### General model

    # data with the ECG, into a new folder; then confirm it is the first-stage data plus the ECG
    export DATA3=/path/to/p2a_vital3
    python scripts/prep_pulsedb.py --mat VitalDB_Train_Subset.mat        --root $DATA3 --split train --val-frac 0.05 --ecg
    python scripts/prep_pulsedb.py --mat VitalDB_CalFree_Test_Subset.mat --root $DATA3 --split test --ecg
    python scripts/compare_roots.py --data $DATA3 --other $DATA

    python scripts/train_general.py --data $DATA3 --mode gen --out runs/g1
    python scripts/train_general.py --data $DATA3 --mode reg --out runs/g1_twin

    # one cell of the signal grid, one gap, and a filler without a model
    python scripts/evaluate_general.py --data $DATA3 --ckpt runs/g1/best.pt --target abp --visible ppg \
        --out results/general/g1/abp_from_ppg.json
    python scripts/evaluate_general.py --data $DATA3 --ckpt runs/g1/best.pt --target abp --gap 2 --visible ppg,ecg \
        --out results/general/g1/abp_gap2_others.json
    python scripts/evaluate_general.py --data $DATA3 --baseline copy --target abp --gap 2 \
        --out results/general/base/abp_gap2_copy.json

`--target abp` without `--gap` writes the first-stage output format, so `scripts/steps_compare.py` and
`scripts/reanalyze.py` compare the general model with M1 and M3 on the same segments.
`scripts/gaps_table.py` does the same for the gap cells. The self-test is
`python -m pytest tests/test_general.py -q`; `P2A_SLOW=1` adds a four-minute end-to-end run.

`slurm/` holds the job files we used to run all of this on a Slurm cluster, for example
`sbatch slurm/train.sbatch --mode gen --out runs/m1`. Their partition, QoS, GPU and path lines are
specific to that cluster.

## Layout

    p2a/checker.py     SBP, DBP and MAP from a waveform; the only place blood-pressure numbers are computed
    p2a/data.py        dataset, calibration pairing (the tests of Table 4 are pairing modes), synthetic data
    p2a/unet1d.py      1D U-Net shared by regression and generation
    p2a/diffusion.py   noise paths, loss with condition dropout, two-weight guidance, sampler, information estimate
    p2a/encoders.py    PaPaGei wrapper: frozen, LoRA, full, scratch
    p2a/evalcore.py    model construction and prediction shared by training and evaluation
    p2a/metrics.py     MAE, CRPS, coverage, hypotension AUROC, reference forecasts
    p2a/corrupt.py     PPG corruptions of known strength
    p2a/masks.py       general model: which samples are visible (whole signals, gaps, random training masks)
    p2a/general.py     general model: network with three signal channels and a mask, loss on hidden samples,
                       sampler, pointwise scores
    p2a/tri.py         dataset with the ECG; synthetic data with an ECG
    p2a/gapfill.py     gap fillers without a model: copy the neighbouring beat, straight line
    scripts/           prep_pulsedb.py, train.py, evaluate.py, information.py, make_synthetic.py,
                       reanalyze.py, steps_compare.py, pooled_noise.py, results_table.py;
                       general model: train_general.py, evaluate_general.py, compare_roots.py, gaps_table.py
    slurm/             job files for a Slurm cluster
    tests/             test_core.py and test_general.py (seconds), test_information.py (minutes)
    results/           metric files of every evaluation; see results/README.md
    runs/              settings and training history of every run (no checkpoints)

## How to read the numbers

* **MAE** is the mean absolute error of the point prediction. For a generator the point prediction is the
  mean of its draws.
* **CRPS** (lower is better) scores a whole predictive distribution against the true value. For a single
  point prediction it equals the MAE, so it is left blank for B1 in Tables 1 and 3. "Point + fixed bar"
  in Table 2 puts one Gaussian error bar of fixed width around an arm's point prediction, which is the
  fair reference for a generator's own draws.
* **90% coverage** is how often the true value falls inside the central 90% interval of the draws, and
  **width** is how wide that interval is. With 19 or 39 draws a perfectly calibrated model scores exactly
  0.90. Both depend on the number of sampling steps (Table 5), so compare arms only at the same number.
* **Pulse-following and calibration-following** (Table 2) are computed within each patient, after removing
  the patient's mean from every variable: the prediction is regressed on the true pressure and on the
  calibration reading. A constant prediction scores (0, 0), repeating the calibration scores (0, 1), and
  a perfect predictor scores (1, 0).
* **Between-patient R2** is the share of the differences between patients' mean pressures that the
  predictions capture.
* **Hypotension AUROC** is how well a model ranks segments with a true MAP below 65 mmHg: a generator by
  the share of its draws below 65, a point prediction by its predicted MAP.
* **Error of mean** (general-model tables) is the absolute error of the mean of the draws at each hidden
  sample, averaged over the hidden samples. The **CRPS** in those tables is computed sample by sample in
  the same way. For the twin and the fillers, which return one prediction, the two are the same number.
  PPG and ECG have no physical unit, so their scores are in units of each segment's own standard deviation.
* The metric files also hold a `tracking_map` slope. We no longer use it: a constant prediction scores
  0.50 on it, so it cannot be read as "follows this share of a change".

## What has and has not been checked

Unit tests (`tests/`): the checker on waveforms with known SBP and DBP; CRPS against the Gaussian closed
form; coverage against its exact finite-sample target; AUROC and the tracking slope on constructed cases;
both noise paths invert exactly; guidance reduces to each of the four conditionals, and a dropped condition
cannot leak; LoRA starts as an exact no-op, its adapters receive gradient and the frozen base does not; the
pairing rules; the information estimate against a closed-form answer; the PulseDB reader on a file built to
mimic the MATLAB v7.3 layout.

Unit tests for the general model (`tests/test_general.py`): the training masks cover all eight
whole-signal patterns and a gap is a single span of the stated length; the loss has zero gradient at
visible samples; true values of hidden samples and noise at visible samples cannot reach the network; the
sampler returns visible samples unchanged; the pointwise CRPS equals the scalar one; the gap fillers on a
signal with a known period; `--ecg` writes the ECG and leaves PPG, ABP, labels and the patient split
identical; on synthetic data both scripts run, resume, and learn that the pulse helps and that a gap is
easier than a missing signal.

On real data: the reader's sanity report quoted above; the pretrained PaPaGei weights file, whose checksum
matches the Zenodo record and which loads into the wrapper without missing or unexpected weights; and for
the general model, that the data with the ECG equals the first-stage data plus the ECG and that every
evaluation used the same 7,200 test segments as Table 3.

Not done: MIMIC-III; tuning; more than two seeds; a calibration that respects time order; a beat-timing
score for generated PPG and ECG; a checkpoint rule that looks at gap filling.

## Sources

* PulseDB: W. Wang, P. Mohseni, K. L. Kilgore, L. Najafizadeh, "PulseDB: A large, cleaned dataset based on
  MIMIC-III and VitalDB for benchmarking cuff-less blood pressure estimation methods", Frontiers in
  Digital Health, 2023. https://github.com/pulselabteam/PulseDB
* VitalDB: H.-C. Lee et al., "VitalDB, a high-fidelity multi-parameter vital signs database in surgical
  patients", Scientific Data, 2022. https://vitaldb.net
* PaPaGei: A. Pillai, D. Spathis, F. Kawsar, M. Malekzadeh, "PaPaGei: Open Foundation Models for Optical
  Physiological Signals", ICLR 2025. https://github.com/Nokia-Bell-Labs/papagei-foundation-model

PaPaGei's code and weights and the PulseDB data are external and are not included here.
