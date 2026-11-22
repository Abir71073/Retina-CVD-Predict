# Update the existing GitHub repository and Space

Run these commands in **Windows PowerShell** as the repository owner or a collaborator with write access. The same ZIP supplies the GitHub source and the Space build. The platforms have separate histories. No credentials belong in the ZIP or either repository.

## 1. Prepare a clean GitHub working copy

Install Git for Windows, Git LFS and 64-bit Python 3.12. Download the delivered ZIP into a new empty working folder, open PowerShell in that folder, then run:

```powershell
git --version
git lfs version
py -3.12 --version
git lfs install
Expand-Archive .\Retina-CVD-Predict-release.zip -DestinationPath .\unpacked
$release = (Resolve-Path .\unpacked\Retina-CVD-Predict).Path
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/Abir71073/Retina-CVD-Predict.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Set-Location .\Retina-CVD-Predict
git switch -c release/dual-stream-update
git rev-parse HEAD
```

Save the printed SHA as the previous GitHub revision. Skipping automatic LFS downloads during this clone avoids fetching the retired H5 weights. Use a fresh branch name if this branch already exists.

## 2. Apply the package and check the models

```powershell
Get-ChildItem -LiteralPath $release -Force | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination . -Recurse -Force }
git rm --ignore-unmatch -- retina_tri_risk_model.h5
py -3.12 -m tools.verify_models
```

All five checks must print OK. The removed H5 file belongs to the old version and remains in Git history. The current app loads five root-level Keras checkpoints. Hidden configuration files, the Dockerfile and README must sit next to app.py in the clone. Keep the ZIP outside the clone.

## 3. Test locally

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python app.py
```

Open http://127.0.0.1:5000, create a local account, and submit a photograph from test/. Confirm the gate accepts it, three scores appear, a vessel mask is shown, and the printable report opens. Stop the server with Ctrl+C. The instance/ directory is local, ignored by Git and excluded from the hosted deployment.

## 4. Push the new version to GitHub

```powershell
git add -A
git lfs ls-files
git status --short
git diff --cached --stat
git show :stroke_specialist.keras | Select-Object -First 3
```

The five .keras files should appear in the LFS listing. The staged stroke model should show a short LFS pointer beginning with `version https://git-lfs.github.com/spec/v1`. Check the file list before committing. If an account database, upload, secret or virtual environment appears, unstage it and correct the ignore rules before continuing.

```powershell
git commit -m "Update retinal classifiers, vessel analysis and reporting"
git push -u origin release/dual-stream-update
```

Open the existing GitHub repository. Choose **Compare & pull request**, set the base to **main**, review the changed paths, and merge. Then, in the same PowerShell window:

```powershell
git switch main
git pull --ff-only origin main
git lfs pull
py -3.12 -m tools.verify_models
$source = (git rev-parse HEAD).Trim()
```

If you want a versioned GitHub Release, create a new release targeting this merged commit and choose an unused tag. Your previous commits and releases remain accessible.

## 5. Configure the existing Hugging Face Space

Open the Space as its owner and go to **Settings → Variables and secrets**. Add these **before** uploading the release:

| Setting | Type | Value |
|---|---|---|
| RETINA_OWNER_USERNAME | Variable | Your chosen login, 3–40 lowercase letters, digits, dots, underscores or hyphens |
| RETINA_OWNER_PASSWORD | Secret | A unique password, 12–128 characters |
| APP_SECRET_KEY | Secret | A separate random value. Generate one locally with the command below. |
| RETINA_STORAGE_DIR | Optional variable | A writable, mounted persistent directory if configured for this Space. Otherwise leave unset. |

```powershell
py -3.12 -c "import secrets; print(secrets.token_hex(32))"
```

Paste the result into the Space secret field. The Docker entry point seeds the first account on an empty database before serving traffic. The owner password setting is used only at initialization. An existing account database keeps its stored password. With default ephemeral storage, a rebuild can clear accounts and reports; the configured owner is created again on an empty database. Use only demo images on a temporary deployment.

## 6. Preview and publish the Space update

From the **merged GitHub clone** install the Hugging Face client and log into the account with write access to the existing Space. Enter the write token at the login prompt.

```powershell
.\.venv\Scripts\python -m pip install -U huggingface_hub
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe auth whoami
.\.venv\Scripts\python -m tools.publish_space --source $source
```

The preview prints the current Space commit and every proposed upload and removal. Save that commit as the rollback point. It uploads the model bytes using the Hub client, preserves the Space’s own .gitattributes, and targets only the retired H5 and five named legacy heatmaps for removal if present. Unrelated Space files are preserved. When satisfied, copy the printed SHA into the next command:

```powershell
$spaceBefore = Read-Host "Paste the Space commit SHA printed by the preview"
.\.venv\Scripts\python -m tools.publish_space --source $source --parent $spaceBefore --publish
```

The helper submits one Space commit with the GitHub SHA in its message. If someone changes the Space after preview, rerun the preview and examine the new revision. Record the returned Space commit URL. Uploading to GitHub alone does not trigger this release procedure.

## 7. Verify the live Space

Watch **Logs → Build** and then the runtime logs until the Space is running. Use **Open in new tab** or the direct app link from the Space menu. The application restricts embedding, so the repository’s embedded preview can be blank even when the standalone app works. Sign in with the configured owner, use the same demo image that worked locally, and confirm the scores, Grad-CAM maps, vessel mask, and printable report appear. Sign out and check the report now requires login. The first prediction loads model weights and can take longer.

If the build reports a missing model or an LFS pointer, run `git lfs pull` and the checksum command in Step 4. If the first start reports missing owner credentials, add the settings in Step 5 and restart. For a full end-to-end check, verify both the GitHub commit/PR and the Space commit with a successful build.

## Rollback

Revert the merged pull request on GitHub using **Revert** on the PR, then merge that revert. The Space is separate. For a Space rollback, clone its Git history in another folder and revert the single deployment commit printed in Step 6. Use your Hugging Face username and write token when prompted. Reverting restores tracked files and triggers a rebuild; account records and runtime data require their own backup.

```powershell
git lfs install
git clone https://huggingface.co/spaces/abirkchakraborty/Retina-CVD-Predict space-rollback
Set-Location space-rollback
git log -5 --oneline
$deployment = Read-Host "Paste the deployment commit SHA to revert"
git revert --no-edit $deployment
git push origin main
```

## Links

- [Existing GitHub repository](https://github.com/Abir71073/Retina-CVD-Predict)
- [Existing Hugging Face Space](https://huggingface.co/spaces/abirkchakraborty/Retina-CVD-Predict)
- [GitHub LFS documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Hugging Face folder uploads](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)
