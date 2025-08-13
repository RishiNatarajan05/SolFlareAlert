# Hazard Ensemble Approach for Solar Flare Prediction

## Overview

This document describes the new **Hazard Ensemble** approach for solar flare prediction, which addresses the limitations of pure XGBoost models when dealing with small positive class sizes (~99 events) and the need for well-calibrated probabilities in operational alert systems.

## Why Move Away from Pure XGBoost?

### Problems with XGBoost Alone

1. **Overfitting on Small Positive Class**: With only ~99 positive events, complex nonlinear models can overfit and give spiky, overconfident probabilities.

2. **Poor Calibration**: XGBoost probabilities are often not well-calibrated, making threshold selection unreliable for operational use.

3. **Operational Instability**: Pure tree models can be sensitive to small changes in input features, leading to unstable alert patterns.

4. **Lack of Interpretability**: Tree models are black boxes, making it difficult to understand why alerts are triggered.

## The Hazard Ensemble Solution

### Core Components

1. **Discrete-Time Hazard Model (60% weight)**
   - Logistic regression with L2 regularization
   - Models the event rate in the next 6 hours
   - Uses carefully selected features that work well for linear models
   - Provides well-calibrated probabilities and interpretability

2. **Calibrated XGBoost Model (40% weight)**
   - Optimized for PR-AUC + logloss
   - Uses isotonic calibration for better probability estimates
   - Captures nonlinear patterns the hazard model misses
   - Reduced complexity to prevent overfitting

3. **Operational Smoothing**
   - **Hysteresis**: Requires 2 consecutive hours over threshold
   - **Cooldown**: Suppresses repeat alerts for 6 hours
   - Reduces false alarms while maintaining recall

### Key Features

#### Cyclic Time Encodings
```python
# Instead of raw time features
hour_of_day, day_of_week, month, day_of_year

# Use cyclic encodings
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
day_of_week_sin = sin(2π * day_of_week / 7)
# etc.
```

#### Hazard Model Features
- Recent flare activity (6h, 24h counts)
- M/X class flare indicators
- CME activity and speed
- Geomagnetic activity (Kp index)
- Cyclic time features
- Lag features (7-day patterns)

#### XGBoost Features
- All available features
- Captures complex interactions
- Nonlinear patterns

## Implementation

### Training the Ensemble

```bash
# Train the hazard ensemble model
python scripts/hazard_ensemble_model.py
```

### Comparing with Current Approach

```bash
# Compare hazard ensemble vs current XGBoost
python scripts/compare_models.py
```

### Operational Usage

```bash
# Single prediction
python scripts/operational_predictor.py --mode single

# Continuous monitoring
python scripts/operational_predictor.py --mode continuous --interval 60

# Historical analysis
python scripts/operational_predictor.py --mode analyze --start-date 2024-01-01 --end-date 2024-01-31
```

## Expected Improvements

### 1. Better Calibration
- **Brier Score**: Lower (better) than pure XGBoost
- **Reliability**: Probabilities match actual event frequencies
- **Threshold Stability**: More reliable threshold selection

### 2. Higher Precision
- **Operational Smoothing**: Reduces false alarms by 30-50%
- **Ensemble Effect**: Combines strengths of both models
- **Feature Selection**: Hazard model uses only reliable features

### 3. Operational Benefits
- **Interpretability**: Hazard model coefficients show feature importance
- **Stability**: Less sensitive to small feature changes
- **Alert Quality**: Fewer false alarms, better precision

### 4. Maintained Recall
- **Target**: ~80% recall on validation set
- **Smoothing**: Minimal impact on true positive detection
- **Ensemble**: Combines complementary model strengths

## Model Architecture

```
Input Features
    ↓
┌─────────────────┐    ┌─────────────────┐
│   Hazard Model  │    │  XGBoost Model  │
│  (Logistic)     │    │  (Calibrated)   │
│  60% weight     │    │  40% weight     │
└─────────────────┘    └─────────────────┘
    ↓                        ↓
┌─────────────────────────────────────────┐
│         Ensemble Prediction             │
│  P(flare) = 0.6*P_hazard + 0.4*P_xgb   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│      Operational Smoothing             │
│  • Hysteresis (2h consecutive)         │
│  • Cooldown (6h between alerts)        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│         Final Alert Decision            │
└─────────────────────────────────────────┘
```

## Feature Engineering

### Hazard Model Features (Selected for Linear Performance)
```python
hazard_features = [
    # Recent activity
    'flare_count_6h', 'flare_count_24h', 
    'm_plus_count_6h', 'm_plus_count_24h',
    'hours_since_last_mx', 'max_flare_class',
    
    # CME activity
    'cme_count_24h', 'max_cme_speed',
    
    # Geomagnetic
    'current_kp', 'max_kp_24h',
    
    # Cyclic time
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    
    # Lag features
    'flare_count_7d', 'm_plus_count_7d'
]
```

### XGBoost Features (All Available)
- Uses all features including complex interactions
- Captures nonlinear patterns
- Handles feature interactions automatically

## Threshold Optimization

### Target Metrics
- **Recall**: ~80% (catch most flares)
- **Precision**: Maximize while maintaining recall
- **FPR**: Minimize false positive rate
- **F1**: Balance precision and recall

### Optimization Process
1. Test thresholds from 0.1 to 0.9
2. Find threshold giving ~80% recall
3. Among candidates, choose highest precision
4. Apply operational smoothing
5. Validate on held-out test set

## Operational Considerations

### Alert Rules
1. **Hysteresis**: Require 2 consecutive hours over threshold
2. **Cooldown**: Suppress alerts for 6 hours after last alert
3. **Risk Categories**:
   - LOW: < 40% probability
   - MEDIUM: 40-70% probability  
   - HIGH: > 70% probability

### Monitoring
- Continuous prediction every hour
- Real-time feature updates
- Alert history tracking
- Performance monitoring

### Maintenance
- Retrain every 6 months with new data
- Monitor calibration drift
- Update thresholds if needed
- Validate on recent data

## Performance Comparison

| Metric | Current XGBoost | Hazard Ensemble | Improvement |
|--------|----------------|-----------------|-------------|
| AUC | 0.XXX | 0.XXX | +X.X% |
| PR-AUC | 0.XXX | 0.XXX | +X.X% |
| Precision | 0.XXX | 0.XXX | +X.X% |
| Recall | 0.XXX | 0.XXX | +X.X% |
| F1-Score | 0.XXX | 0.XXX | +X.X% |
| Brier Score | 0.XXX | 0.XXX | -X.X% |

## When to Use This Approach

### ✅ Recommended For
- Small positive class sizes (< 1000 events)
- Need for calibrated probabilities
- Operational alert systems
- Interpretability requirements
- Time series with clustering

### ❌ Not Recommended For
- Large datasets (1000s of positives)
- Pure ranking tasks
- Real-time inference requirements
- Deep learning applications

## Future Enhancements

1. **Poisson Regression**: Model flare counts instead of binary events
2. **Time-Varying Coefficients**: Adapt to solar cycle changes
3. **Active Learning**: Retrain on new data automatically
4. **Ensemble Diversity**: Add more model types
5. **Feature Engineering**: Advanced solar physics features

## Conclusion

The Hazard Ensemble approach provides a robust, interpretable, and operationally stable solution for solar flare prediction. By combining the strengths of linear hazard modeling with calibrated tree-based learning, it addresses the key limitations of pure XGBoost while maintaining high predictive performance.

This approach is particularly well-suited for operational solar weather forecasting where reliability, interpretability, and calibrated probabilities are more important than raw ranking performance.
