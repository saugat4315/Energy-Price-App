# ⚡ Norway Energy Analyzer

Streamlit app: day-ahead power prices in NO1–NO5 vs reservoirs, weather, demand/supply,
cross-border flows and river discharge, plus a statistical 7-day price forecast.

## Put it online (free, no Python needed on your PC)
1. Create a GitHub account → **New repository** (e.g. `norway-energy-app`, Public).
2. **Add file → Upload files**: drag in `app.py`, `data_sources.py`, `model.py`, `requirements.txt`,
   `README.md`, `.gitignore` and the `.streamlit` folder (`config.toml` only). **Do not upload any keys.**
3. Go to **share.streamlit.io** → sign in with GitHub → **Create app** → pick the repository, branch `main`, file `app.py`.
4. Before clicking Deploy, open **Advanced settings → Secrets** and paste:
   ```
   ENTSOE_TOKEN = "..."
   FROST_ID = "..."
   HYDAPI_KEY = "..."
   ```
5. **Deploy**. After a minute or two you get a link like `https://norway-energy-xxxx.streamlit.app`.

Every API call runs on Streamlit's server, so keys stay secret and nothing is blocked by the browser.
Data is cached for 1–24 hours depending on the source.

## Run on your own PC (optional)
```
pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   (then put your keys in it)
streamlit run app.py
```
