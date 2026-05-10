# NBA Trade Acquisition Impact Tool

A Streamlit app that predicts the historical impact of NBA trades using machine learning and explains results via Claude AI.

## How it works

1. You type a free-form trade description (e.g. *"the Celtics acquiring Jrue Holiday"*)
2. Claude parses it into structured inputs (player, team, season)
3. A Gradient Boosting regressor and Random Forest classifier estimate the impact
4. Claude generates an analyst-style explanation and narrates the top historical comps

---

## File structure

All files must be in the **same directory** (repo root):

```
app.py
helpers.py
requirements.txt
final_trade_impact_regressor.pkl
final_trade_impact_classifier.pkl
final_model_features.pkl
final_modeling_dataset.csv
nba_transactions_complete.csv
```

> **Note:** If your pkl/csv files were downloaded with ` (1)` in the filename (e.g. `final_model_features (1).pkl`), rename them to remove the ` (1)` before running.

---

## Running locally

### 1. Create a virtual environment (recommended)

The pkl models were trained on **scikit-learn 1.6.1**. Using a different version will cause a runtime error. A virtual environment ensures the pinned version is used.

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

**macOS / Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-...
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Deploying to Streamlit Community Cloud

Streamlit Community Cloud deploys directly from a GitHub repository — no server setup needed.

### 1. Push to GitHub

Make sure all required files (see File structure above) are committed and pushed to a public or private GitHub repo.

```bash
git add app.py helpers.py requirements.txt \
        final_trade_impact_regressor.pkl \
        final_trade_impact_classifier.pkl \
        final_model_features.pkl \
        final_modeling_dataset.csv \
        nba_transactions_complete.csv
git commit -m "Add NBA trade impact app"
git push
```

> `nba_model_dataset.csv` (~19 MB raw modeling file) is **not needed** by the app — do not include it.

### 2. Connect the repo

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**
3. Select your repository, branch, and set **Main file path** to `app.py`
4. Click **Deploy**

### 3. Add the API key as a secret

1. After deployment, open your app's **Settings** → **Secrets**
2. Add the following:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

3. Save and reboot the app

The app reads the key via `st.secrets.get("ANTHROPIC_API_KEY", ...)` so this works automatically on Streamlit Cloud without any code changes.

---

## Model performance

| Model | Algorithm | Dataset | Key metrics |
|---|---|---|---|
| Regressor | Gradient Boosting | 3,302 trades | MAE 0.1005 (~±8 wins), R² 0.180 |
| Classifier | Random Forest | 5,745 acquisitions | AUC 0.614, F1 0.641, Accuracy 62.7% |

Train/test split is time-aware: trained pre-2020, tested on 2020+.

---

## Caveats

- Predictions reflect historical base rates, not guaranteed causal outcomes
- Draft picks and cash are not valued by the model
- Player stats are season-level profiles, not exact pre-trade snapshots
- The model does not account for injuries, chemistry, or coaching changes
