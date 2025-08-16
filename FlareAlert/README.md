# FlareAlert

A solar flare prediction system using machine learning to forecast solar flares and geomagnetic storms.

## Features

- Real-time solar flare prediction
- Geomagnetic storm forecasting
- Hazard ensemble model combining logistic regression and XGBoost
- Operational smoothing for reduced false alarms
- Historical data analysis from NASA DONKI API

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Run training: `python scripts/hazard_ensemble_model.py`

## Model Performance

- **Precision**: 50.2%
- **False Positive Rate**: 4.8%
- **Ensemble AUC**: 0.781

## Architecture

Hybrid ensemble combining:
- Hazard model (logistic regression) for calibration
- XGBoost for pattern recognition
- Operational smoothing for stability

## License

MIT License
