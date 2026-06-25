# Football Predictions

Football Predictions is a football data analytics and probability estimation dashboard. It is **not a betting app**: the goal is to explore historical football data, bookmaker-implied probabilities, and transparent model estimates.

## Features

- Upload one or more historical match CSV files.
- Use a modular competition configuration in `config/competitions.yml`.
- Support major football-data.co.uk club league CSVs and separate international CSV inputs.
- Clean and standardize common match, score, result, and bookmaker odds columns.
- Analyze last-5 and last-10 form, home/away performance, goals scored and conceded trends, draw rates, and team form.
- Calculate chronological Elo ratings for teams.
- Add head-to-head features for recent meetings.
- Convert decimal bookmaker odds into normalized implied probabilities.
- Train a calibrated multi-feature model that uses odds as only one input, alongside team form, venue performance, goals, H2H, Elo, rest, neutral-site, and tournament-context features.
- Estimate home win, draw, and away win probabilities for an upcoming match with probability calibration to reduce overconfident outputs.
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

Current focus competitions are Premier League, La Liga, Serie A, Bundesliga, Ligue 1, FIFA World Cup, and general international matches.

### Add a new league later

1. Open `config/competitions.yml`.
2. Add a new item under `competitions`.
3. For football-data.co.uk leagues, set `data_source: football-data.co.uk` and add the site code when available, such as `E0`, `SP1`, `I1`, `D1`, or `F1`.
4. For national-team datasets, set `data_source: international_csv` and `match_type: international`.
5. Restart Streamlit and select the new competition in the sidebar.

No Python code change is required for a new competition that follows one of the existing loader formats.

## Input formats

### football-data.co.uk club CSVs

The app recognizes common football-data.co.uk columns including:

`Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `HTHG`, `HTAG`, `HTR`, `B365H`, `B365D`, `B365A`, `BWH`, `BWD`, `BWA`, `IWH`, `IWD`, `IWA`, `PSH`, `PSD`, `PSA`, `MaxH`, `MaxD`, `MaxA`, `AvgH`, `AvgD`, `AvgA`.

### International CSVs

International files can include columns such as `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `country`, `neutral`, and optional odds columns `home_odds`, `draw_odds`, `away_odds`. The loader maps these to the app's canonical format and derives `FTR` from the score when needed.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

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

This dashboard presents probabilities and historical analytics for education and research. A value signal is only an analytical comparison between model estimates and bookmaker implied probability; it is not financial advice or a recommendation to wager.
