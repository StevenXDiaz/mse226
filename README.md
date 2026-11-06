# Hotel cancellation prediction (MS&E 226 project)

## Data
Download `hotel_bookings.csv` from https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
(Antonio, Almeida & Nunes, Data in Brief 22, 2019) and place it in `data/`. The data is not included here.

## Setup
Python 3.12. `python -m venv .venv && .venv/bin/pip install -r requirements.txt`

## Run (Part 1)
1. `.venv/bin/python exploration.py`: Section 5 exploration on the development data; prints missing values, correlations with cancellation, correlations between covariates and cancellation rates by category.
2. `.venv/bin/python train.py`: Section 6 model comparison (5-fold CV, development data only); saves
   `outputs/model_comparison_<hotel>.csv`, calibration plots and the final pipelines
   `outputs/cancellation_model_<hotel>.joblib`. About 5 minutes.
3. `.venv/bin/python predict.py city 2016-02-10`: ranked bookings and expected cancellations for one day.

`evaluate_holdout.py` evaluates the saved pipelines on the holdout. It is for Part 2 only.

## Files
- `common.py`: loading, duplicate removal, the 80/20 holdout split (seed 42, by arrival date; saved to
  `outputs/split.csv`), cleaning rules, feature sets and the rare-agent grouping step.
- Random seed 42 throughout (split, CV folds, random forest, gradient boosting).
