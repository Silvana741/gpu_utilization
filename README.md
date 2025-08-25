# GPU Utilization and Power Draw Forecasting

This repository contains the analysis, benchmarking, and results of forecasting GPU utilization and power draw using statistical, machine learning, and deep learning models. The work leverages the [Nixtla forecasting libraries](https://github.com/Nixtla) — **StatsForecast**, **MLForecast**, and **NeuralForecast** — to provide a systematic comparison across classical, tree-based, and neural forecasting methods.

---

## 📖 Project Overview

Efficient resource management in GPU clusters requires accurate forecasts of utilization and power consumption. This project benchmarks a wide range of forecasting approaches on a dataset of 243 GPU jobs training Inception v4 network, sampled at 10-second intervals, originating from the [MIT Supercloud Dataset](https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset). The models are evaluated on short-horizon predictions (5–6 minutes ahead), both with and without exogenous variables such as GPU memory utilization, temperature, and power draw.  

The repository is organized to enable reproducibility of the full pipeline: preprocessing, feature engineering, model training, evaluation, and results visualization.

---

## ⚙️ Methods

The following families of models are benchmarked:

- **Classical Statistical Models** (from StatsForecast)
  - AutoARIMA  
  - AutoETS  
  - AutoTheta  

- **Machine Learning Models** (from MLForecast)
  - XGBoost (Chen & Guestrin, 2016)  
  - LightGBM (Ke et al., 2017)  
  - With configurations including:
    - Short, medium, and long lags  
    - Rolling statistics (mean, expanding std)  
    - Exogenous variables  

- **Deep Learning Models** (from NeuralForecast)
  - TSMixer / TSMixerX  
  - MLP-Multivariate  
  - N-HiTS  

Each family of models provides a complementary perspective: statistical models offer interpretability, tree-based methods capture nonlinear interactions, and neural networks excel at modeling long-term dependencies.

---

## 📊 Evaluation

Forecast accuracy is assessed using multiple error metrics:

- **MAE** (Mean Absolute Error)  
- **MSE** (Mean Squared Error)  
- **RMSE** (Root Mean Squared Error)  
- **sMAPE** (Symmetric Mean Absolute Percentage Error)  
- **wMAPE** (Weighted Mean Absolute Percentage Error)  

Formulas and descriptions are included in the paper/analysis for reproducibility.


---

## 🚀 Getting Started

### Requirements
- Python 3.9+  
- Nixtla libraries:
  - `statsforecast`
  - `mlforecast`
  - `neuralforecast`  
- Machine learning frameworks:
  - `xgboost`
  - `lightgbm`  
- Data manipulation and plotting:
  - `pandas`, `numpy`, `matplotlib`, `seaborn` , `plotly`, `polars`

## Results

### Key findings:

- Statistical models (AutoARIMA, AutoETS, AutoTheta) serve as strong baselines.

- Machine learning models benefit significantly from lag features and rolling transformations, with additional gains when exogenous variables are included.

- Neural models (TSMixer, N-HiTS) achieve the best performance on long-horizon forecasts, particularly when incorporating exogenous variables.

Detailed plots are available in the images/ directory. 




