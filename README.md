# Employee Attrition Prediction

A Streamlit app that predicts employee attrition using a trained classification model.

## Project structure

- `streamlit_app/app.py` - Streamlit application entrypoint
- `model/` - saved model artifacts (`model.pkl`, `preprocessor.pkl`, `binary_mappings.pkl`)
- `notebook/attrition_prediction.ipynb` - notebook used to build and save the model
- `dataset/` - training and test CSV files
- `pyproject.toml` - project metadata and dependencies
- `uv.lock` - package lock file for `uv`

## Local setup

1. Open a terminal in the project root:
   ```powershell
   cd C:\Users\SHUBHAM\Downloads\Atrition\copy
   ```

2. Activate the venv if needed:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies (if not already installed):
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run streamlit_app/app.py
   ```

5. Open the browser at the URL shown by Streamlit (usually `http://localhost:8501`).

## Deployment to Streamlit Cloud

1. Push your repository to GitHub.
2. Go to https://share.streamlit.io.
3. Sign in with GitHub.
4. Choose the repository: `Shubham-Jaiswal5414/Employee_Attrition_Prediction`.
5. Use branch `main` and main file path `streamlit_app/app.py`.
6. Deploy the app.

## Notes

- The app requires the `model/` folder to contain the saved artifacts.
- If using Streamlit Cloud, ensure `requirements.txt` is present and up to date.
