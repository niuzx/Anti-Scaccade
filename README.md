# Anti-Scaccade

Core research code for evaluating anti-saccade eye movements and odor identification against Montreal Cognitive Assessment (MoCA) scores.

## Evidence boundary

The evaluated cohort had 157 participants. The multimodal network did **not** outperform training-fold mean/median baselines for recorded-total prediction. Four domain scores were missing across two participants, and nine complete records had inconsistent recorded totals versus domain sums. Labels remain unresolved; no automatic correction or imputation is performed. This is research software, not a validated diagnostic system.

The original balanced training policy is the primary retained analysis. The natural-frequency policy is a post-hoc sensitivity analysis that jointly removes replacement sampling and class-weighted cross-entropy. Neither policy is selected as a winner using outer-test results. Restricted evaluation on 155/146 records is not clean-subset retraining.

## Included

- Shared Transformer and wavelet models; eye-only, odor-only, fusion and no-wavelet variants.
- Subject-level five-fold evaluation with inner validation and outer-training refitting.
- Training-fold mean/median baselines, nested ridge controls, MAE/RMSE/R2 and correlations.
- Original and post-hoc training policies, with locked parameter templates.
- Private-data input validation and a CPU model smoke test.

`model_source/` contains the legacy architecture modules imported by the evaluation runner. Run the evaluation entry points below rather than those legacy scripts' standalone training routines.

## Installation

Use Python 3.11 and install `pip install -r requirements.txt`. GPU training requires a compatible NVIDIA driver and PyTorch CUDA build. The paper environment used PyTorch 2.10.0+cu128. Version pins describe that environment; CPU smoke tests do not reproduce training results.

## Private input contract

Raw data, participant identifiers, score sheets, participant-level predictions, trained checkpoints, and credentials are intentionally absent. Requesting data access and obtaining appropriate authorization are separate from accessing this repository.

Create a private JSON list of participant objects using the schema in `docs/data_contract.md`. Do not commit it. Then run:

```bash
python scripts/prepare_private_data.py --manifest /absolute/path/private_manifest.json --data-root /absolute/path/private_data --policy balanced
python experiments/balanced/run_revision.py ridge
python experiments/balanced/run_revision.py olfactory_only
python experiments/balanced/run_revision.py eye_only
python experiments/balanced/run_revision.py multimodal_no_wavelet
python experiments/balanced/run_revision.py multimodal
python experiments/balanced/run_revision.py summarize
```

For the post-hoc sensitivity analysis, initialize `--policy natural`, and run the same commands under `experiments/natural/`. Use identical participant order and seed across policies. Do not edit original experiment outputs or select seeds using held-out performance. The runner resumes completed checkpoint files only when its signature and fold checks match.

```bash
python tests/smoke_models.py
```

The CPU test uses generated random tensors only; these are not study data. It checks forward/backward execution, excluded-modality invariance, fold separation, and the exact-error metric identity.

## Reproducibility and limitations

CWT `fs=60` is an algorithm reference after normalized-time resampling, not a verified physiological sampling frequency. Eye recordings were processed into 40 paired arrays per participant. Domain labels are discrete and missing domain values must be JSON null. Recorded total and complete domain sum remain separate evaluation targets. Fixed-OOF bootstrap intervals do not include refitting uncertainty.

Published code does not imply public availability of the clinical dataset or a trained model. This initial source release has no granted reuse license yet; the author team should select an appropriate license before describing it as open-source. Preserve authors' rights and third-party dependency licenses.
