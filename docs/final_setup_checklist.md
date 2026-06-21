# Final GitHub and Streamlit Setup Checklist

Status meanings:

- **Done** — verified locally in this preparation session.
- **Not done yet** — depends on a remote service that has not been configured.
- **Needs user action** — requires the repository owner to approve or supply
  information.

## A. Local setup

- **Done:** Project virtual environment exists.
- **Done:** Requirements install successfully.
- **Done:** Synthetic data generation completes.
- **Done:** Model training and evaluation complete with reproducible metrics.
- **Done:** Generated artifacts are excluded from Git.

## B. FastAPI test

- **Done:** FastAPI starts locally.
- **Done:** `/health` returns a ready status.
- **Done:** `/docs` opens.
- **Done:** `/recommend` returns five recommendations for a valid request.
- **Done:** The verification server is stopped afterward.

## C. Streamlit test

- **Done:** Streamlit starts locally.
- **Done:** The page and health endpoint respond.
- **Done:** A recommendation request returns the top five drivers.
- **Done:** No `SimpleImputer` or missing-file error occurs locally.
- **Done:** Missing cloud artifacts can be regenerated automatically.

## D. GitHub upload

- **Done:** Local Git repository initialized.
- **Done:** Upload exclusions and staged files reviewed.
- **Done:** Initial local commit created.
- **Not done yet:** Public GitHub repository created.
- **Needs user action:** Approve automatic repository creation or provide the
  empty repository URL.
- **Not done yet:** Remote `origin` added.
- **Not done yet:** Branch pushed to GitHub.

## E. GitHub Actions

- **Done:** Workflow is present and Linux-compatible.
- **Done:** Workflow regenerates synthetic artifacts and runs pytest.
- **Not done yet:** Workflow executed on GitHub.
- **Needs user action:** Push the repository, then confirm the Actions run.

## F. Streamlit Cloud deployment

- **Done:** `.streamlit/config.toml` added.
- **Done:** Main app path confirmed as `src/streamlit_app.py`.
- **Done:** No secrets, private files, or environment variables are required.
- **Not done yet:** App connected to the GitHub repository.
- **Not done yet:** App deployed on Streamlit Community Cloud.
- **Needs user action:** Deploy after the GitHub push and copy the public URL.

## G. README public links

- **Done:** Local FastAPI and Streamlit URLs documented.
- **Done:** Honest GitHub and live-demo placeholders documented.
- **Needs user action:** Replace placeholders after upload and deployment.

## H. Resume update

- **Done:** Resume-ready project wording exists in
  `docs/final_application_pack.md`.
- **Needs user action:** Add the final GitHub URL to the resume.
- **Needs user action:** Add the live-demo URL only after successful deployment.

## I. JobStreet update

- **Done:** JobStreet profile summary is prepared.
- **Needs user action:** Paste the summary and add the final GitHub URL.

## J. Recruiter sharing

- **Done:** Recruiter message and interview explanations are prepared.
- **Not done yet:** Public links are available.
- **Needs user action:** Share only after checking the public GitHub repository
  and Streamlit app while signed out.
