# Football Predictions

Football Predictions is a football data analytics and probability estimation dashboard. It is **not a betting app**: the goal is to explore historical football data, bookmaker-implied probabilities, and transparent model estimates.

## Features

- Automatically download configured football-data.co.uk historical match and odds CSVs, while keeping manual uploads as a fallback.
- Use a modular competition configuration in `config/competitions.yml`.
- Support major football-data.co.uk club league CSVs and separate international CSV inputs.
- Clean and standardize common match, score, result, and bookmaker odds columns.
- Analyze last-5 and last-10 form, home/away performance, goals scored and conceded trends, draw rates, and team form.
- Calculate chronological Elo ratings for teams.
- Add head-to-head features for recent meetings.
- Convert decimal bookmaker odds into normalized implied probabilities.
- Train a calibrated multi-feature model that uses odds as only one input, alongside team form, venue performance, goals, H2H, Elo, rest, neutral-site, and tournament-context features.
- Estimate home win, draw, and away win probabilities for an upcoming match with probability calibration to reduce overconfident outputs.
- Download upcoming fixtures from football-data.co.uk when available and generate predictions for every fixture in the next 48 hours.
- Keep the selected odds source visible, preferring football-data.co.uk market averages (`AvgH`, `AvgD`, `AvgA`) and falling back to Bet365 (`B365H`, `B365D`, `B365A`) when needed.
- Estimate a likely score, confidence score, and explanation notes for each prediction.
- Highlight lightweight permutation feature importance so users can see which signals drove predictive power.
- Find similar historical matches using engineered football features as supporting context, not just similar odds.

## Project structure

```text
app.py
config/
  competitions.yml
src/
  competitions.py
  data_loader.py
  elo.py
  preprocessing.py
  features.py
  odds.py
  predictor.py
  visualization.py
scripts/
  data_sources.py
  update_historical_data.py
  update_upcoming_fixtures.py
  predict_next_48h.py
data/
  raw/
  processed/
  upcoming/
  predictions/
  reports/
requirements.txt
README.md
```

## Competition configuration

Supported competitions are configured in `config/competitions.yml`. Each entry contains:

```yaml
- name: Premier League
  country_or_type: England
  data_source: football-data.co.uk
  football_data_code: E0
  match_type: club
```

Current focus competitions are Premier League, La Liga, Serie A, Bundesliga, Ligue 1, FIFA World Cup, and general international matches. FIFA World Cup remains selectable, but it is a filtered view of the shared international national-team dataset rather than a separate World Cup data source.

### Add a new league later

1. Open `config/competitions.yml`.
2. Add a new item under `competitions`.
3. For football-data.co.uk leagues, set `data_source: football-data.co.uk` and add the site code when available, such as `E0`, `SP1`, `I1`, `D1`, or `F1`.
4. For national-team datasets, set `data_source: international_csv` and `match_type: international`.
5. Restart Streamlit and select the new competition in the sidebar.

No Python code change is required for a new competition that follows one of the existing loader formats.


## Automatic football-data.co.uk downloads

### Historical results and odds

Run:

```bash
python scripts/update_historical_data.py
```

The script reads `config/competitions.yml`, downloads configured football-data.co.uk league CSVs from the public season folders, stores the raw CSVs in `data/raw/`, cleans them with the same app preprocessing, removes duplicate matches, and writes `data/processed/historical_matches.csv`. Historical odds remain from the football-data.co.uk source family. Per football-data.co.uk notes, those odds are collected from Betbrain, Oddsportal, and individual bookmakers. The app does not silently mix in unrelated odds feeds.


### Historical backtesting and validation

Run a real-data chronological backtest after downloading historical data:

```bash
python scripts/backtest_model.py --train-until 2024-06-30 --test-from 2024-07-01
```

The backtest loads `data/processed/historical_matches.csv`, sorts matches by `Date`, trains only on rows on or before `--train-until`, and tests only on rows on or after `--test-from`. It intentionally avoids random train/test splits because football results are time-dependent and random splits can leak future team strength into past predictions.

Reports are written to `data/reports/`:

- `backtest_predictions.csv` contains match-level model probabilities, bookmaker implied probabilities, model/bookmaker picks, and correctness flags.
- `backtest_metrics.json` contains accuracy, log loss, Brier score, bookmaker favorite accuracy, and the model-vs-bookmaker accuracy edge.
- `backtest_calibration.csv` groups predictions by confidence bucket so predicted probabilities can be compared with observed hit rates.
- `backtest_confusion_matrix.csv` shows where the model confused home wins, draws, and away wins.
- `backtest_probability_comparison.csv` compares average model probabilities with average bookmaker-implied probabilities.

The Streamlit **Backtesting** page can run the same script against saved historical data and displays metrics, calibration, confusion matrix, model-vs-bookmaker probability comparison, and recent failed predictions.

Interpretation guide:

- **Accuracy** is the share of matches where the highest-probability pick matched `FTR`; it is easy to understand but ignores how confident the probabilities were.
- **Log loss** rewards assigning high probability to the actual outcome and strongly penalizes overconfident misses; lower is better.
- **Brier score** measures squared probability error across home/draw/away outcomes; lower is better and it is useful for probability quality.
- **Calibration** compares confidence buckets with actual correctness rates. If a 50–60% bucket wins around 50–60% of the time, those estimates are better calibrated.
- **Bookmaker baseline** converts decimal odds to normalized implied probabilities and uses the market favorite as a simple baseline pick. Model accuracy should be interpreted against this baseline, not as a guarantee of future outcomes.

Backtesting is for probability estimation and analytics only. It does not prove that future football outcomes can be guaranteed.

### Upcoming fixtures and odds

Run:

```bash
python scripts/update_upcoming_fixtures.py
```

The script uses an API-first fixture pipeline. The Odds API is the primary source for upcoming soccer fixtures and bookmaker odds, including `h2h` and `totals` markets. Set `ODDS_API_KEY` in `.env` before running it. Optional API-Football fixture support is available through `API_FOOTBALL_KEY` as a fallback/future provider when The Odds API returns no usable fixtures.

Club fixtures are written to `data/upcoming/upcoming_fixtures.csv`; international fixtures are written by `scripts/update_international_fixtures.py` to `data/upcoming/international_fixtures.csv`. Both outputs use these columns:

`Date`, `Time`, `Competition`, `HomeTeam`, `AwayTeam`, `HomeOdds`, `DrawOdds`, `AwayOdds`, `Over25Odds`, `Under25Odds`, `OddsSource`.

The `h2h` market is normalized to `HomeOdds`, `DrawOdds`, and `AwayOdds`. If a `totals` market has a 2.5-goals line, it is normalized to `Over25Odds` and `Under25Odds`. Missing markets or bookmaker prices are left blank and do not crash the app. The project never scrapes bookmaker websites directly. If the primary key is missing, scripts and the app show: `Set ODDS_API_KEY in .env or use manual CSV fallback.`

### Manual upcoming-fixture fallback

If automatic upcoming data is missing, upload or pass a manual CSV with equivalent columns. From the command line:

```bash
python scripts/update_upcoming_fixtures.py --manual-csv path/to/manual_upcoming.csv
```

In Streamlit, use the **Next 48 Hours Predictions** page and upload a manual upcoming fixtures CSV. The upload is normalized to the same upcoming schema, so a future real odds API can be added later without changing the prediction pipeline. This is especially useful for World Cup and international matches where public club-league fixture feeds may not contain coverage.

Manual CSV headers must be exactly:

```csv
Date,Time,Competition,HomeTeam,AwayTeam,HomeOdds,DrawOdds,AwayOdds,Over25Odds,Under25Odds,OddsSource
2026-06-25,20:00,Example League,Home FC,Away FC,2.10,3.30,3.50,1.90,1.95,Manual
```

If odds are unavailable, leave `HomeOdds`, `DrawOdds`, `AwayOdds`, `Over25Odds`, and `Under25Odds` blank and set `OddsSource` to `Unavailable`. The local scripts and Streamlit app will keep valid headers even when no automatic fixtures are available, so an empty fixture or prediction file should not crash the app.

### Next 48 hours predictions

Run:

```bash
python scripts/predict_next_48h.py
```

The script loads `data/processed/historical_matches.csv` and `data/upcoming/upcoming_fixtures.csv`, filters fixtures scheduled in the next 48 hours, and writes `data/predictions/next_48h_predictions.csv`. Each row includes home/draw/away probabilities, predicted score, confidence score, value signal, model explanation, similar historical matches, and visible odds source.

## Input formats

### football-data.co.uk club CSVs

The app recognizes common football-data.co.uk columns including:

`Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `HTHG`, `HTAG`, `HTR`, `B365H`, `B365D`, `B365A`, `BWH`, `BWD`, `BWA`, `IWH`, `IWD`, `IWA`, `PSH`, `PSD`, `PSA`, `MaxH`, `MaxD`, `MaxA`, `AvgH`, `AvgD`, `AvgA`.

### International CSVs

All national-team competitions share one historical file: `data/processed/international_matches.csv`. This includes FIFA World Cup, UEFA Euro, Nations League, World Cup qualifiers, international friendlies, and other national-team matches. Selecting **FIFA World Cup** filters this shared file to `Competition`/`Tournament = FIFA World Cup`; selecting **International matches** uses all rows. Do not maintain a duplicate World Cup historical CSV.

By default, `scripts/update_international_data.py` downloads international historical results from the public `martj42/international_results` GitHub dataset at `https://raw.githubusercontent.com/martj42/international_results/master/results.csv`. It saves the downloaded raw copy to `data/raw/international_matches.csv`, normalizes `date`, `tournament`, `home_team`, `away_team`, `home_score`, `away_score`, `neutral`, and `country`, and derives `FTR` from the final score.

Clicking **Update historical data** while **FIFA World Cup** or **International matches** is selected runs `scripts/update_international_data.py`. The updater validates that the processed file has rows, required columns, parsed dates, and `FTR` values of `H`, `D`, or `A` before writing, so an empty or invalid download will not overwrite a previously valid `data/processed/international_matches.csv`.

To override the public download, place your own CSV at `data/raw/international_matches.csv`. When that file exists, the updater uses it instead of downloading from GitHub. Manual files can include columns such as `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `competition`, `country`, `neutral`, and optional odds columns `home_odds`, `draw_odds`, `away_odds`. The loader maps these to the app's canonical format and derives `FTR` from the score when needed.

Manual CSV example for `data/raw/international_matches.csv`:

```csv
date,home_team,away_team,home_score,away_score,tournament,neutral,country
2022-11-20,Qatar,Ecuador,0,2,FIFA World Cup,True,Qatar
2022-09-22,France,Austria,2,0,UEFA Nations League,False,France
```

The processed output is normalized to `Date`, `Competition`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `Neutral`, `Country`, and `SourceFile`. World Cup matches should be included in the same raw international CSV with `tournament` or `competition` set to `FIFA World Cup`; the app filters those rows for the FIFA World Cup view.

### International upcoming fixtures

All national-team upcoming fixtures share one file: `data/upcoming/international_fixtures.csv`. FIFA World Cup fixtures are filtered from this file by `Competition`/`Tournament = FIFA World Cup`; broader **International matches** predictions use all rows. A separate `worldcup_fixtures.csv` is not required.

International fixtures CSV format:

```csv
Date,Time,Competition,HomeTeam,AwayTeam,HomeOdds,DrawOdds,AwayOdds,Over25Odds,Under25Odds,OddsSource
2030-06-13,20:00,FIFA World Cup,Spain,Brazil,,,,,,Unavailable
```

Odds may be unavailable for international fixtures. Leave `HomeOdds`, `DrawOdds`, and `AwayOdds` blank and use `OddsSource` such as `Unavailable`; predictions still run from Elo, weighted form, goals, H2H, and tournament-context features. If no international fixtures are present, the app reports: `No international fixtures available. Add data/upcoming/international_fixtures.csv or connect a fixture API.` `scripts/update_international_fixtures.py` now tries The Odds API first, optional API-Football fixture fallback second, and public international fixture feeds/manual CSVs after that.

## Local Ubuntu setup and workflow

This project can run entirely on a local Ubuntu machine. No deployment, container, or cloud service is required.

### Prerequisites

Install Python 3 with virtual-environment support:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

The project does not require secrets or API keys for the current local workflow. Optional local shell overrides are documented in `.env.example`.

### 1. Setup

Create a virtual environment and install Python dependencies with one command:

```bash
bash scripts/setup_local.sh
```

The script creates `.venv/`, upgrades `pip`, and installs `requirements.txt`. To use the environment manually after setup, run:

```bash
source .venv/bin/activate
```

### 2. Update data

Run the full local update workflow:

```bash
bash scripts/full_local_update.sh
```

This runs, in order:

1. `python scripts/update_historical_data.py --start-year 2018`
2. `python scripts/update_upcoming_fixtures.py`
3. `python scripts/predict_next_48h.py`

To include a manual upcoming-fixtures CSV fallback, pass the existing updater arguments through the workflow script:

```bash
bash scripts/full_local_update.sh --manual-csv path/to/manual_upcoming.csv
```

You can also run the data steps separately:

```bash
source .venv/bin/activate
python scripts/update_historical_data.py --start-year 2018
python scripts/update_upcoming_fixtures.py
```

### 3. Run dashboard

Start the Streamlit dashboard locally with one command:

```bash
bash scripts/run_local.sh
```

If `.venv/` is missing, this script runs `scripts/setup_local.sh` first. Then open the local Streamlit URL shown in your terminal, usually `http://localhost:8501`.

### 4. Run backtest

After historical data exists, run a chronological backtest:

```bash
source .venv/bin/activate
python scripts/backtest_model.py --train-until 2024-06-30 --test-from 2024-07-01
```

Backtest reports are written to `data/reports/`.

### 5. Generate next 48h predictions

After historical data and upcoming fixtures exist, generate the next 48 hours of predictions:

```bash
source .venv/bin/activate
python scripts/predict_next_48h.py
```

Predictions are written to `data/predictions/next_48h_predictions.csv`.

## Model notes

The baseline model is intentionally understandable and modular. It uses a calibrated gradient-boosting classifier over engineered pre-match features. Every training row is built chronologically from matches that occurred before the target fixture to avoid data leakage.

- last-5 and last-10 points per game;
- last-5 and last-10 goals scored/conceded and goal difference;
- scored-rate, clean-sheet rate, draw tendency, and longer-term 25-match strength;
- home-team home performance and away-team away performance;
- rest days and team match-count experience;
- neutral venue, World Cup, and knockout/high-pressure tournament context for international datasets;
- recent head-to-head history;
- chronological Elo ratings with neutral-site home advantage removed;
- bookmaker implied probabilities and market entropy;
- similar historical matches and permutation feature importance as context.

Odds are **not** the only driver of predictions. They are treated as market context alongside football performance features.


## Accuracy-focused improvements

This project deliberately avoids adding features that require post-match information. The highest-impact current improvements are:

1. **Leak-free football context**: features are still generated from prior matches only, now including goal difference, scoring reliability, clean sheets, draw tendency, rest days, experience, neutral venues, and tournament flags. Expected impact: better separation between teams with similar points totals but different underlying scoring profiles, and more realistic treatment of international matches played at neutral sites.
2. **Calibrated probabilities**: the model now calibrates the classifier output before presenting probabilities. Expected impact: fewer overconfident 70–90% predictions when the evidence is weak, improving probability reliability rather than only top-pick accuracy.
3. **Stronger match model**: gradient boosting replaces the previous random forest baseline because it can capture non-linear football interactions such as Elo strength plus market odds plus venue effects while staying lightweight. Expected impact: better use of interacting signals without adding heavy dependencies.
4. **Confidence scoring**: confidence now considers probability margin, entropy, team history depth, and agreement with the market. Expected impact: low-evidence international fixtures and evenly matched games receive lower confidence even when one class is marginally highest.
5. **Feature importance analysis**: permutation importance reports which features reduce log-loss most on recent training rows. Expected impact: users can identify whether the model is learning from meaningful football signals or mostly copying odds.

## Responsible framing

This dashboard presents probabilities and historical analytics for education and research. It is an analytics/probability tool, not a betting guarantee. A value signal is only an analytical comparison between model estimates and bookmaker implied probability; it is not financial advice or a recommendation to wager.
