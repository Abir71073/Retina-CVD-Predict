# Retina — authenticated dual-stream microvascular review

This update keeps your approved specialist CNN / Grad-CAM / U-Net architecture
and all five original model files. It adds a working login, integrates U-Net into
every analysis, improves Grad-CAM, and ends with a findings-based clinical interpretation. The original fundus gatekeeper now screens every upload before both analysis streams.

## Start on Windows

1. Install **64-bit Python 3.12**.
2. Extract the ZIP completely.
3. Double-click `start_windows.bat` and wait for dependencies to install.
4. Open **http://127.0.0.1:5000**.
5. On the first visit, **create your own username and password** (12+ characters).
   There is no default account or password. Subsequent visits use the login page.
6. Select a retinal fundus image and choose **Run both streams**. The gatekeeper checks the image first; rejected uploads receive no disease analysis.
7. Review the classifier maps, U-Net outputs, and **Clinical interpretation** at the bottom.

The first analysis loads the CNN and U-Net weights and takes longer. Sample images
are in `test/`. Uploaded filenames are not used as ground-truth labels.

Manual Windows setup:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Runtime and tests were executed on Linux CPU/Python 3.12. Windows packages are
available for the pinned principal dependencies, but Windows execution was not tested.
No CUDA is required. The server binds to localhost; debug mode is disabled.

## Mandatory fundus image check

The gatekeeper runs before the CNN scores, Grad-CAM, U-Net and clinical interpretation.
Rejected images show an explanation and no disease scores. If the gatekeeper cannot
load or returns an invalid result, analysis stops; it never silently bypasses the gate.
Any temporary rejected upload is removed. Accepted results and printable reports
record that the image-type check passed. Passing does not establish clinical quality
or disease status.

## New Grad-CAM behaviour

- Uses EfficientNet's **`top_activation`** spatial feature tensor for all three
  real classifiers, including the nested hypertension model.
- Explains the positive logit for **Regions increasing the model score**, or
  the negative logit for **Regions decreasing the model score**.
- Changing the target changes the explanation only. It never changes model
  predictions, weights, sigmoid activation or disease thresholds.
- Grad-CAM uses inference mode and supports gradients even for frozen feature layers.
- Overlay opacity follows attribution intensity: zero-activation pixels retain
  the original image. Background attribution is not artificially hidden.
- Toggle **Activation grid** to inspect the native coarse heatmap without smoothing.
  Overlay uses a colour legend; the activation grid uses a grayscale legend.
- No activation for the selected target is reported explicitly. No fake heatmap
  is substituted. A missing map does not discard the numerical score.
- When most attribution lies outside the estimated retinal field, the final report
  identifies possible border/background sensitivity instead of treating it as anatomy.

These explanations remain coarse model attributions, not lesion annotations.

## U-Net is now part of the web pipeline

Every successful analysis calls the original `unet_vessel_segmentation.keras`.
It uses RGB/255 input, 256-pixel patches, 128-pixel stride and Gaussian overlap
blending, following the original training convention. Large images are reduced
proportionally to a maximum edge of 768 pixels to bound CPU work.

The dashboard shows:

- Vessel overlay on the image.
- Binary vessel mask using a **0.50 segmentation cutoff**.
- Grayscale vessel probability map.
- Vessel area as a percentage of the estimated image field.
- Skeleton pixel count and processing dimensions.

The field is estimated from brightness and is not a learned quality/fundus check.
Reported measurements are at the processing resolution, have no validated
normal/abnormal cutoff, and must not be compared across different acquisition
settings as clinical measurements. The original narrow/wide width-group method
is retained as an exploratory ratio in the report. It does not identify arteries
and veins and is not a clinical AVR.

U-Net failure is isolated: classifier outputs remain available, and the final
interpretation explicitly marks the vessel analysis as missing.

## Final clinical interpretation and report

The main summary is now specific to the uploaded image: it lists that image’s
model predictions and segmented vessel area. Repetitive context has been moved
into an expandable section. The visible label is **Model prediction (%)**;
uncalibrated outputs are not renamed clinical risk. A short experimental-use
note remains visible. No percentages or model weights were changed.


The final section is generated from actual classifier outputs, explanation
availability, vessel segmentation, and available measurements. It includes:

1. Findings for each specialist and the selected Grad-CAM target.
2. U-Net vessel area and measurement context.
3. An exploratory width ratio when measurable.
4. Clinical correlation: measured blood pressure; eGFR and urine albumin testing
   when clinically indicated; appropriate clinical/brain-imaging evaluation for stroke.
5. Missing-stream and explanation limitations, with NIH/NIDDK references.

The supplied models are not validated to confirm or exclude these diseases.
The report therefore states **clinician review required**; it does not invent
normal/abnormal vessel cutoffs, clinical diagnoses, treatments, or five-year risks.

Use **Open printable report** to view a report containing all maps and the final
interpretation. Print / Save PDF uses your browser's print dialog. A JSON download
is also available. Both require the account that created the analysis.

## Login and private results

- Passwords are hashed with Werkzeug's scrypt implementation.
- Sessions expire after two hours and are revoked on logout or password reset.
- Form and upload POSTs require a CSRF token.
- Login attempts are rate-limited (10 attempts per client IP per 15 minutes).
- Each account can access only its own analysis images and reports.
- New uploads/results are outside public `static/`, under `instance/analyses/`.
- Legacy uploads/heatmaps from the old project are not served by the static route.
- Account database and session secret are created locally on first run.
  **No credentials, accounts, sessions or new user uploads are included in this ZIP.**

Create an additional account from your project terminal (password is prompted):

```bash
python -m flask --app app create-user reviewer
```

Reset an account password locally (also revokes its sessions):

```bash
python -m flask --app app reset-password your_username
```

Preserve `instance/` if upgrading an existing installation and you want to retain
accounts/results. This update contains no `instance/` folder to overwrite yours.
Remove saved analysis folders when no longer needed. For an HTTPS deployment,
set `COOKIE_SECURE=1`; hosting hardening is outside this localhost project update.
Do not expose first-time setup publicly before creating the initial account.

## Existing prediction fix retained

The hypertension checkpoint's training code used external RGB/255 scaling; the
old application did not. The corrected app reproduces the legacy input contract
for those weights. It restores varying outputs where the previous app returned
zero on every supplied test image. `reports/prediction_audit.csv` retains those results.

CKD/stroke checkpoint-specific training code and class mappings are still absent.
Their original raw-RGB inference contract is retained. Their numerical outputs
are not proof of disease accuracy. The older notebook's proxy targets do not
establish actual CKD/stroke outcome labels for the supplied later checkpoints.
The original gatekeeper is now mandatory before both streams. Its RGB/255 input
and class-0-as-fundus polarity were checked empirically on 9 retinal and 15
non-retinal/control examples; all were separated correctly in that limited check.
Original training metadata is still missing, so this is not a population accuracy
guarantee. See `GATEKEEPER_REPORT.md` and `reports/gatekeeper_audit.csv`.

## Verify the project

```bash
python -m pytest -q
python -m tools.diagnose_models
python -m tools.diagnose_gatekeeper
```

The supplied tests exercise real models, Grad-CAM direction/determinism,
actual U-Net inference, report generation, authentication, account isolation,
CSRF, rate limits, password reset/session revocation, private media access,
input rejection and incomplete-stream handling.

Evaluate classifiers with a patient-separated held-out CSV containing explicit
`filename,label` values and a threshold selected using a separate validation set:

```bash
python -m tools.evaluate_classifier --model hyp --csv heldout.csv --images dataset/images --threshold 0.5
```

Here 0.5 is an example, not a validated clinical cutoff. Filenames never supply labels.

Standalone U-Net utilities remain available:

```bash
python -m tools.unet_test test/normal_fundal_wikimedia.jpg
python -m tools.diagnose_hyp test/ckd_hyp_fundal.jpg
python -m tools.test --help
```

Optional corrected hypertension retraining (original dataset required):

```bash
python -m tools.retrain_hypertension --csv dataset/full_df.csv --images dataset/preprocessed_images --patient-column ID
```

Use the actual patient-ID column. This preserves the specialist architecture,
creates patient-separated splits, saves the best validation-AUC checkpoint, and
exports its preprocessing metadata. New training uses raw RGB because EfficientNet
already contains internal scaling. After validation, replace the checkpoint and
its preprocessing configuration together. Existing weights are not overwritten.
No full retraining was performed because training datasets are absent.

For original DRIVE U-Net training, set `DRIVE_TRAIN_DIR` or place data under
`drive_dataset/training/{images,1st_manual}`, then run `python -m tools.drive`.

## Main files

| File | Purpose |
|---|---|
| `app.py` | Protected routes, analysis orchestration and report serving |
| `auth.py` | Local accounts, sessions, CSRF and password commands |
| `inference.py` | Gated orchestration, CNN inference, corrected Grad-CAM and integrated U-Net |
| `gatekeeper.py`, `gatekeeper_config.json` | Mandatory fundus screening and its explicit empirical input contract |
| `vessel_pipeline.py` | Patch stitching, field estimation and exploratory width method |
| `clinical_report.py` | Findings-based interpretation and references |
| `templates/` and `static/` | Login, dashboard and printable report |
| `tests/` | Automated regression and real-model checks |
| `model_config.json` | Explicit checkpoint preprocessing contracts |
