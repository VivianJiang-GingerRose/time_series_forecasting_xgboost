### Forecasting quarterly CPI with ABS retail trade turnover data using GBM, turning a time series forecast problem into a regression problem.
---
Forecasted June quarter CPI: **1.13%** compare to previous quarter. Actual CPI: **1.8%** - not too bad ✌️

Next steps is to add more time series data that are typically available prior to cpi is published, such as petrol prices, logistic statistics, property sales/auction rate, stock market stats etc.

---
#### Algorithm details:

Source data (predictors): ABS retail turnover data prior to CPI being publised. <br>
Source data (prediction target): Quarterly CPI statistics published by ABS.

**Feature transformations:**<br>
1. Number of days, business days, holidays and weekends in the quarter.
2. Differences in time lags, calculated as current value minus previous period value
3. Rolling lag differences (periods = [1,3,6,9]) including: <br>
⋅ Rolling lag min, i.e. minimum value in the 1, 3, 6 or 9 period prior <br>
⋅ Rolling lag max, mean, standard deviation, mean divided by standard deviation, log value mean, log value minus log value mean. These features are similar to a time series decomposation process. <br>

**Predictions:**<br>
⋅ Training data: 1982 Q2 to 2022 Q1 <br>
⋅ Test data: 2022 Q2, target cpi value is left to be nan to prevent leakage <br>
⋅ Number of features: 250 <br>
⋅ XGBoost regression model built using training data without hyperparameter training (for now) <br>
⋅ No progressive prediction as we are only predicting one period ahead <br>
