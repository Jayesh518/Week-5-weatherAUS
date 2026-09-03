# Week 5 – WeatherAUS Comprehensive Data Science Project

## Project Overview
This repository consolidates the five-week WeatherAUS internship project into one reproducible end-to-end workflow:

- **Week 1:** Data acquisition, cleaning, preprocessing and exploratory analysis
- **Week 2:** Advanced visualization and visual storytelling
- **Week 3:** Statistical analysis and hypothesis testing
- **Week 4:** Machine learning model development and evaluation
- **Week 5:** Integrated findings, strategic recommendations, impact, limitations and future scope

## Dataset
`data/weatherAUS.csv` contains 142,193 observations and 24 columns covering Australian weather observations from 2007-11-01 to 2017-06-25 across 49 locations.

## Main Research/Business Question
Can historical weather observations provide useful information for identifying whether rainfall is likely to occur tomorrow, and how can those insights support practical outdoor and weather-sensitive decisions?

## Core Findings
- RainTomorrow is approximately 77.6% No and 22.4% Yes.
- Missingness is approximately 9.28% of all cells, concentrated in Sunshine, Evaporation and cloud variables.
- Winter has the highest observed next-day rain rate at approximately 26.1%.
- Mean Humidity3pm is 68.80% for RainTomorrow=Yes versus 46.51% for No.
- Welch t-test: t=182.608, p≈0; 95% CI for the mean difference = [22.05, 22.53] percentage points; Cohen's d=1.198.
- Logistic regression in Week 3: humidity coefficient=0.06533; odds ratio=1.0675 per one-percentage-point increase; pseudo-R²=0.2092.
- Week 4 Logistic Regression: accuracy=0.794, precision=0.528, recall=0.777, F1=0.628, ROC-AUC=0.873.
- Confusion matrix: TN=17,629, FP=4,435, FN=1,423, TP=4,952.

## Leakage Prevention
`RISK_MM` is excluded from RainTomorrow prediction because it represents future rainfall information and would create target leakage.

## Repository Structure
- `data/` – canonical dataset
- `weeks/week1/` … `weeks/week4/` – prior week scripts, notebooks, data artifacts and visualizations
- `src/` – integrated Week 5 pipeline
- `visualizations/` – consolidated charts
- `results/` – statistical/model result tables
- `reports/` – prior weekly DOCX reports
- `Week_5_Comprehensive_WeatherAUS_Report.docx` – final report

## Reproduction
```bash
pip install -r requirements.txt
python src/week5_integrated_pipeline.py
```

## Strategic Direction
The model is best treated as a decision-support baseline. Practical deployment should emphasize recall where missed rain is costly, combine predictions with current weather/API information, expose probability/confidence, develop location/season-specific models, compare stronger algorithms, retrain periodically, and monitor performance drift.
