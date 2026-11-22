# Fundus gatekeeper integration

The existing `fundus_gatekeeper_v2_massive.keras` now runs before both specialist
classification/Grad-CAM and U-Net. It is mandatory in `Predictor.analyze_complete`,
which is the web application's entry point. The trained weights are unchanged.

## Behavioural check and input convention

The checkpoint has a MobileNet-style network, a 224×224 RGB input, one sigmoid
output, and no internal pixel scaling. Its original training code and class-index
metadata were not included. Three input conventions were tested directly against
the unchanged checkpoint, and the retinal/non-retinal image types were visually inspected.

The configured convention is:

- RGB image, nearest-neighbour resize to 224×224, pixels divided by 255.
- Sigmoid output represents the non-fundus side in the tested samples.
- Fundus score = `1 - sigmoid_output`.
- Accept only when fundus score is **strictly greater than 0.50**; ties reject.
- No threshold tuning was performed: 0.50 is the binary decision midpoint.

This convention is empirically supported, not claimed to be recovered training
metadata. `gatekeeper_config.json` records that distinction explicitly.

| Sample group | Examples | Correct image-type decisions in this check |
|---|---:|---:|
| Original project retinal examples | 8 | 8 accepted |
| scikit-image retinal example | 1 | 1 accepted |
| scikit-image non-retinal photos/textures/silhouette/microscopy | 12 | 12 rejected |
| Synthetic noise, black and white controls | 3 | 3 rejected |

Raw 0–255 pixels, with the same output polarity, rejected 7 of the 9 retinal
examples. Both normalized conventions separated the ordinary photo examples,
but the -1…1 convention accepted the black control at the gate-model level.
The general file validator also rejects blank images before the gate runs.

The 24-example check is a small sanity check, not an independent representative
validation set. A classifier cannot guarantee rejection of every possible
non-fundus image. Gate acceptance is neither a clinical finding nor a validated
image-quality assessment. Training metadata and larger representative image-type
validation would still improve confidence in its contract.

## Stop behaviour

- A non-fundus decision returns HTTP **422**, `code=non_fundus_image`.
- Missing checkpoint/configuration, invalid output or a model failure returns
  HTTP **503**, `code=gatekeeper_unavailable`.
- Neither condition runs a specialist model, Grad-CAM or U-Net.
- Neither condition generates scores, an interpretation, a report or a database
  analysis record. Any temporary decoded image directory is removed.
- Successful results include the gate decision and the final clinical
  interpretation explains that an image-type check was passed.
- The interface clears previous results before a new upload; rejected images
  cannot leave a previous patient's scores on display.

The lower-level standalone specialist/segmentation diagnostic utilities remain
available for development. The full web pipeline is always gated; there is no
silent bypass or 'gate disabled' switch.

## Reproduce

Basic check using supplied retinal images and synthetic controls:

```bash
python -m tools.diagnose_gatekeeper
```

The complete audit also uses optional scikit-image reference images:

```bash
python -m pip install scikit-image==0.26.0
python -m tools.diagnose_gatekeeper --reference-images
```

The main app does not require scikit-image. Reference photographs are not bundled
in this ZIP. `reports/gatekeeper_audit.csv` includes all 24 observed cases and
all three preprocessing comparisons. Original model hashes are recorded in
`reports/model_sha256.json`.

Automated regression checks cover input scaling, class polarity, tie handling,
invalid outputs, missing gate configuration, real retinal acceptance, real
non-fundus rejection, no downstream model loading on rejection, cleanup, no
report/database row, and keeping the successful classifier/U-Net/login flow intact.

Latest test run: **46 tests passed**, including all earlier login, report, Grad-CAM
and U-Net regression checks and the new gatekeeper checks. Python compilation
and JavaScript syntax checking also passed.
