# Final GitHub and Streamlit Setup Checklist

Status meanings:

- **Done** — completed and verified during project setup.
- **User maintenance** — an item to recheck if a public URL or deployment
  configuration changes later.

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
- **Done:** Public GitHub repository created.
- **Done:** Remote `origin` points to
  `https://github.com/coldjeffry12/logistics-driver-matching-ml.git`.
- **Done:** Branch `main` pushed to GitHub.
- **Done:** README displays correctly on GitHub.
- **Done:** All five recruiter-facing screenshots display correctly.

## E. GitHub Actions

- **Done:** Workflow is present and Linux-compatible.
- **Done:** Workflow regenerates synthetic artifacts and runs pytest.
- **Done:** Initial GitHub Actions workflow completed successfully.

## F. Streamlit Cloud deployment

- **Done:** `.streamlit/config.toml` added.
- **Done:** Main app path confirmed as `src/streamlit_app.py`.
- **Done:** No secrets, private files, or environment variables are required.
- **Done:** App connected to the public GitHub repository.
- **Done:** App deployed on Streamlit Community Cloud.
- **Done:** App visibility set to public.
- **Done:** App opens in an incognito/private browser without login.
- **Done:** Public URL recorded:
  `https://logistics-driver-matching-ml-aldmp73kjkhhj9jbuhl4hx.streamlit.app`.

## G. README public links

- **Done:** Local FastAPI and Streamlit URLs documented.
- **Done:** Public GitHub repository URL added.
- **Done:** Public Streamlit URL added.
- **Done:** Deployment placeholders removed.

## H. Resume update

- **Done:** Resume-ready project wording exists in
  `docs/final_application_pack.md`.
- **Done:** Final GitHub and live-demo links prepared for resume use.
- **User maintenance:** Add the links to each submitted resume where space and
  application format permit.

## I. JobStreet update

- **Done:** JobStreet profile summary is prepared.
- **Done:** Public project links are prepared.
- **User maintenance:** Paste the selected summary and links into JobStreet.

## J. Recruiter sharing

- **Done:** Recruiter message and interview explanations are prepared.
- **Done:** Final recruiter message contains both public links.
- **Done:** Public GitHub repository and Streamlit demo are available.
- **Done:** Final documentation update committed and pushed as part of the
  public-link finalization.
- **User maintenance:** Recheck both links while signed out before a major
  application campaign.
