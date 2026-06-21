# GitHub upload guide for Windows PowerShell

## 1. Create an empty GitHub repository

On GitHub, create a repository such as:

```text
logistics-driver-matching-ml
```

Do not initialize it with a README, `.gitignore`, or license if you plan to
push this existing folder directly.

## 2. Open PowerShell in the project

```powershell
cd C:\Users\Acer\logistics-driver-matching-ml
```

## 3. Review upload contents

```powershell
git status
```

If this is not yet a Git repository:

```powershell
git init
git status
```

Check that `.venv`, generated CSV files, the SQLite database, model binaries,
temporary logs, and Python caches are not listed for commit.

## 4. Commit the project

```powershell
git add .
git status
git commit -m "Add logistics driver matching ML portfolio project"
```

## 5. Use the main branch

```powershell
git branch -M main
```

## 6. Connect the GitHub remote

Replace the placeholder with the HTTPS or SSH URL shown by GitHub:

```powershell
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
```

Confirm it:

```powershell
git remote -v
```

## 7. Push

```powershell
git push -u origin main
```

## If `origin` already exists

Inspect it:

```powershell
git remote -v
```

If it points to the wrong repository:

```powershell
git remote set-url origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

If GitHub rejects the push because the remote has an existing commit, inspect
the remote before combining histories:

```powershell
git pull origin main --allow-unrelated-histories
```

Resolve any conflicts carefully, commit the merge, and then push. Do not use a
force push unless you understand that it can overwrite remote history.

## Recommended repository settings

Repository description:

```text
Synthetic-data ML prototype for ranking feasible logistics drivers using scikit-learn, FastAPI, Streamlit, SQLite, and ranking metrics.
```

Recommended topics:

- `machine-learning`
- `recommendation-system`
- `ranking`
- `logistics`
- `scikit-learn`
- `fastapi`
- `streamlit`
- `sqlite`
- `python`
- `portfolio-project`

## Final GitHub checks

- README images render correctly.
- The repository contains no `.venv` folder or private `.env` files.
- Generated data and model binaries are absent unless intentionally added.
- Setup commands work from a clean clone.
- The description says “prototype” or “synthetic data.”
- The About section links to the Streamlit deployment only if one is actually
  deployed later.

## PowerShell script policy note

If Windows blocks the helper scripts, use a one-process bypass:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
```

Alternatively, enable locally created scripts for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
