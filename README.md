# Football Predictions

Football Predictions is a football match analytics and probability estimation dashboard. It is **not a betting app**: the goal is to explore historical match results, bookmaker-implied probabilities, and simple baseline model estimates in a transparent data analytics workflow.

## Features

- Upload one or more historical football CSV files.
- Clean and standardize common `football-data.co.uk` columns.
- Analyze team performance, home/away strength, draw rates, goals scored and conceded, and recent form.
- Convert decimal bookmaker odds into normalized implied probabilities.
- Compare bookmaker implied probabilities with historical outcomes.
- Find historical matches with similar odds profiles.
- Visualize goals, result distributions, team form, and odds calibration.
- Train a simple baseline scikit-learn predictor using team form, goals, home/away context, and implied probabilities.
- Estimate home win, draw, and away win probabilities for a selected upcoming match.
- Flag analytical value signals where model probability exceeds bookmaker implied probability.

## Project structure

```text
app.py
src/
  data_loader.py
  preprocessing.py
  features.py
  odds.py
  predictor.py
  visualization.py
requirements.txt
README.md
```

## Supported input columns

The app is designed for common `football-data.co.uk` columns:

`Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `HTHG`, `HTAG`, `HTR`, `B365H`, `B365D`, `B365A`, `BWH`, `BWD`, `BWA`, `IWH`, `IWD`, `IWA`, `PSH`, `PSD`, `PSA`, `MaxH`, `MaxD`, `MaxA`, `AvgH`, `AvgD`, `AvgA`.

Rows without valid dates, teams, full-time goals, or full-time result are removed. Missing odds are allowed, but odds-based analysis and prediction need valid odds from the selected bookmaker source.

## Getting football-data.co.uk CSV files

1. Go to [football-data.co.uk](https://www.football-data.co.uk/data.php).
2. Choose a country and league, such as England Premier League.
3. Download CSV files for several seasons. The site usually links files by season and division, for example `mmz4281/2324/E0.csv`.
4. Keep the files as CSVs; no manual editing is required.
5. Upload one or more files in the Streamlit sidebar.

Using multiple seasons generally improves team-form summaries, odds calibration, and baseline model stability.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Model notes

The first version intentionally uses a simple, understandable baseline model rather than deep learning. Features include recent points per game, goals scored/conceded, home/away performance signals, and bookmaker implied probabilities. This makes the platform easier to validate and extend later with richer feature engineering, cross-validation, model comparison, and league-specific calibration.

## Responsible framing

This dashboard presents probabilities and historical analytics for education and research. A value signal is only an analytical comparison between the model estimate and bookmaker implied probability; it is not financial advice or a recommendation to wager.
