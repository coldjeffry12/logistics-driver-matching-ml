# Streamlit Community Cloud Deployment Guide

This guide deploys the Streamlit interface for the synthetic-data portfolio
prototype. It does not deploy the FastAPI service and does not represent a
production system.

## Current public deployment

- Live app: https://logistics-driver-matching-ml-aldmp73kjkhhj9jbuhl4hx.streamlit.app
- GitHub repository: https://github.com/coldjeffry12/logistics-driver-matching-ml
- Branch: `main`
- Main file path: `src/streamlit_app.py`
- App visibility: Public
- Access check: Opens in an incognito/private browser without requiring login

The app is deployed from the GitHub repository
`coldjeffry12/logistics-driver-matching-ml`, branch `main`, using
`src/streamlit_app.py` as the main file.

## Deployment approach

Generated CSV files, the SQLite database, reports, and the `joblib` model are
intentionally excluded from Git. When the first recommendation is requested
on an empty Streamlit Cloud instance, `src/runtime_setup.py` deterministically:

1. Generates the synthetic shipment, driver, and historical-match data.
2. Creates the CSV and SQLite pipeline outputs.
3. Trains the existing Logistic Regression and Random Forest candidates.
4. Saves and loads the selected model.

The first recommendation may therefore take about one minute. Later
recommendations reuse Streamlit's cached recommender for that running app
instance. A cloud restart may regenerate the files because Community Cloud
storage is not permanent.

## Deploy the app

1. Push this project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with the GitHub account that can access the repository.
4. Select **Create app** or **New app**.
5. Choose repository `logistics-driver-matching-ml`.
6. Select branch `main`.
7. Enter main file path `src/streamlit_app.py`.
8. Open advanced settings if needed and select a supported Python 3.12
   runtime.
9. Deploy the app.
10. Submit one recommendation and allow the initial synthetic artifact
    generation and training to finish.
11. Copy the public app URL.
12. Replace the README live-demo placeholder and add the URL to the
    repository's GitHub About section.

No secrets or API keys are required.

## Expected configuration

The tracked `.streamlit/config.toml` configures headless execution and
disables usage-statistics collection. `.streamlit/secrets.toml` is ignored and
is not required by this project.

## Common errors

### Repository not connected

Streamlit cannot deploy a local-only folder. Push the repository to GitHub,
sign in to Streamlit with GitHub, and authorize access to the repository.

### Wrong application path

Use:

```text
src/streamlit_app.py
```

Do not use only `streamlit_app.py`.

### Missing requirements package

Confirm that `requirements.txt` is at the repository root. Review the
Community Cloud build log for the missing import, add the correct pip package
only if the application actually imports it, then push the correction.

### Missing model or data file

The application should automatically generate missing synthetic artifacts
when the first recommendation is submitted. Check the application logs for
the data-generation or training error. Do not upload private or real logistics
data as a workaround.

### Streamlit cannot import `src`

Confirm that the repository root contains `src/__init__.py` and that the main
file path is `src/streamlit_app.py`. The application supports both package and
direct-script imports.

### scikit-learn or joblib version mismatch

The cloud deployment trains and loads the model in the same environment, so a
serialization mismatch should not occur. Redeploy after clearing the app
cache if dependencies were changed. The compatible version ranges are listed
in `requirements.txt`.

### First recommendation appears slow

The first request on an empty instance runs deterministic synthetic data
generation and model training. Wait about one minute and monitor the app log.
This is a portfolio deployment compromise, not a production serving design.

### App is sleeping

Community Cloud may put an inactive app to sleep. Open the public URL and wait
for the app to wake before submitting a recommendation.

### App asks for sign-in

Open the app's **Settings**, select **Sharing**, and confirm that the app is
public. Verify the URL again in an incognito/private browser.

### Recommendation fails

Review the Streamlit Cloud logs for a missing package, failed synthetic
artifact generation, or a scikit-learn/joblib compatibility issue. The first
recommendation may take around one minute while artifacts are generated.

### Resource or timeout failure

Reboot the app once and inspect the build/runtime logs. If Community Cloud
resource limits prevent training, the fallback is to create a smaller,
deployment-specific synthetic artifact through a documented build process;
do not silently commit real data or change reported metrics.

## After deployment

Verify:

- The page loads without a missing-file error.
- A valid shipment returns five sorted recommendations.
- Scores are between zero and one.
- The synthetic-data and non-production disclaimers remain visible.
- The README and GitHub About section use the real public URL.
