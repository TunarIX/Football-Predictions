# Football Predictions

Football Predictions is a football match analytics and probability estimation dashboard. It is **not a betting app**: the goal is to explore historical match results, bookmaker-implied probabilities, team strength signals, and baseline model estimates in a transparent data analytics workflow.

## Features

- Upload one or more historical football CSV files.
- Select a competition from a modular YAML configuration.
- Clean and standardize common `football-data.co.uk` columns.
- Load international match CSVs with national teams, scores, dates, tournaments, neutral-site flags, and optional odds.
- Analyze team performance, home/away strength, draw rates, goals scored and conceded, recent form, and Elo ratings.
- Convert decimal bookmaker odds into normalized implied probabilities.
- Compare bookmaker implied probabilities with historical outcomes.
- Find historical matches with similar odds profiles as supporting context.
- Visualize goals, result distributions, team form, and odds calibration.
- Train a modular baseline scikit-learn predictor using multiple football signals rather than odds matching alone.
- Estimate home win, draw, and away win probabilities for a selected upcoming match.
- Estimate a predicted score, confidence score, and explanation of the model drivers.
- Flag analytical value signals where model probability exceeds bookmaker implied probability.

## Project structure

```text
app.py
config/
  competitions.yml
src/
  config.py
  data_loader.py
  elo.py
  preprocessing.py
  features.py
  odds.py
  predictor.py
  visualization.py
requirements.txt
README.md
```

## Supported club input columns

The app is designed for common `football-data.co.uk` columns:

`Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `HTHG`, `HTAG`, `HTR`, `B365H`, `B365D`, `B365A`, `BWH`, `BWD`, `BWA`, `IWH`, `IWD`, `IWA`, `PSH`, `PSD`, `PSA`, `MaxH`, `MaxD`, `MaxA`, `AvgH`, `AvgD`, `AvgA`.

Rows without valid dates, teams, full-time goals, or full-time result are removed. Missing odds are allowed for descriptive analytics, but odds calibration and prediction need valid odds from the selected bookmaker source.

## Supported international input columns

International CSVs can use canonical names or common alternatives:

- Date: `Date` or `date`
- Teams: `HomeTeam`/`AwayTeam`, `home_team`/`away_team`, or `home`/`away`
- Scores: `FTHG`/`FTAG`, `home_score`/`away_score`, or `home_goals`/`away_goals`
- Optional fields: `Tournament`, `tournament`, `Neutral`, `neutral`, and any supported bookmaker odds columns

If `FTR` is missing or non-standard, the app infers the result from the full-time scores.

## Prediction logic

The model does **not** predict only by matching similar odds. Similar historical odds are shown as supporting context, while the baseline model combines:

- team recent form from last 5 matches
- team recent form from last 10 matches
- home-team home performance and away-team away performance
- goals scored and conceded trends
- head-to-head outcome and goal-difference history
- chronological Elo ratings with a home-advantage adjustment
- bookmaker implied probabilities as one feature group
- similar historical odds profiles for explanation/context

The current implementation uses `HistGradientBoostingClassifier` for match outcome probabilities and `RandomForestRegressor` models for home and away predicted goals. It is intentionally understandable and modular so future versions can add richer validation, league-specific models, player data, injuries, and advanced calibration.

## Competition configuration

Supported competitions are defined in `config/competitions.yml`. The initial configuration includes:

- Premier League (`E0`)
- La Liga (`SP1`)
- Serie A (`I1`)
- Bundesliga (`D1`)
- Ligue 1 (`F1`)
- FIFA World Cup
- International matches

Each competition entry includes:

```yaml
- name: Premier League
  country_or_type: England
  data_source: football-data.co.uk
  football_data_code: E0
  match_type: club
```

To add a new league later:

1. Open `config/competitions.yml`.
2. Add a new item under `competitions`.
3. Set `name`, `country_or_type`, `data_source`, `football_data_code` if available, and `match_type` (`club` or `international`).
4. Restart Streamlit so the sidebar reloads the configuration.
5. Upload CSV files that match either the football-data.co.uk schema or the international CSV schema.

## Getting football-data.co.uk CSV files

1. Go to [football-data.co.uk](https://www.football-data.co.uk/data.php).
2. Choose a country and league, such as England Premier League.
3. Download CSV files for several seasons. The site usually links files by season and division, for example `mmz4281/2324/E0.csv`.
4. Keep the files as CSVs; no manual editing is required.
5. Upload one or more files in the Streamlit sidebar.

Using multiple seasons generally improves form features, Elo ratings, head-to-head coverage, odds calibration, and baseline model stability.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Responsible framing

This dashboard presents probabilities and historical analytics for education and research. A value signal is only an analytical comparison between the model estimate and bookmaker implied probability; it is not financial advice or a recommendation to wager.
