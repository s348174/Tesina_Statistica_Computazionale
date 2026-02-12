import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
set_random_seed = 42
np.random.seed(set_random_seed)

# PREPARE MEAN ENERGY TIME SERIES
# Load data
df = pd.read_csv("ace_swics_ions_unified.csv", parse_dates=["time"])

# Sort by time and set it as the index for time series analysis
df = df.sort_values("time")
df = df.set_index("time")

# Define fundamental physical constants
amu = 1.66053906660e-27  # Atomic mass unit in kilograms
eV = 1.602176634e-19     # Electron volt in joules

# Define ion masses in kilograms using approximate atomic mass units
masses = {
    "He": 4 * amu,
    "C": 12 * amu,
    "O": 16 * amu,
    "Fe": 56 * amu
}

# Convert ion speeds from km/s to m/s for energy calculations
df["vHe_ms"] = df["vHe2"] * 1e3
df["vC_ms"]  = df["vC5"]  * 1e3
df["vO_ms"]  = df["vO6"]  * 1e3
df["vFe_ms"] = df["vFe10"]* 1e3

# Calculate mean kinetic energy for each ion species using E = 1/2 m v^2 and convert to eV
df["E_He"] = 0.5 * masses["He"] * df["vHe_ms"]**2 / eV
df["E_C"]  = 0.5 * masses["C"]  * df["vC_ms"]**2  / eV
df["E_O"]  = 0.5 * masses["O"]  * df["vO_ms"]**2  / eV
df["E_Fe"] = 0.5 * masses["Fe"] * df["vFe_ms"]**2 / eV

# Optional: Plot individual ion energies over time (currently commented out)
plt.figure(figsize=(10,5))
plt.plot(df.index, df["E_He"], label="He")
plt.plot(df.index, df["E_C"],  label="C")
plt.plot(df.index, df["E_O"],  label="O")
plt.plot(df.index, df["E_Fe"], label="Fe")

plt.ylabel("Mean kinetic energy [eV]")
plt.xlabel("Time")
plt.title("Mean Ion Energy vs Time (Solar Wind)")
plt.legend()
plt.tight_layout()
plt.show()

# Calculate composition-weighted mean energy based on ion abundance ratios
# Extract abundance ratios relative to oxygen
R_He = df["HetoO"]
R_C  = df["CtoO"]
R_Fe = df["FetoO"]

# Calculate normalization denominator for abundance weights
den = 1 + R_He + R_C + R_Fe
# Compute normalized weights for each ion species
w_He = R_He/den
w_C  = R_C/den
w_Fe = R_Fe/den
w_O  = 1/den

# Compute the abundance-weighted mean plasma energy
df["E_mean"] = (
    w_He*df["E_He"] +
    w_C*df["E_C"] +
    w_Fe*df["E_Fe"] +
    w_O*df["E_O"]
)

plt.figure(figsize=(10,5))
plt.plot(df.index, df["E_mean"])
plt.ylabel("Mean plasma energy [eV]")
plt.xlabel("Time")
plt.title("Composition-weighted Mean Ion Energy")
plt.show()

####################################################################################

"""# ARIMA FORECAST
from statsmodels.tsa.arima.model import ARIMA

# Apply a 3-point rolling average to smooth the mean energy series
series = df["E_mean"].rolling(3).mean().dropna()
# Split data into training and test sets
series_train = series.iloc[:-50]  # Use all but last 50 points for training
series_test = series.iloc[-50:]   # Use last 50 points for testing

model = ARIMA(series_train, order=(2,0,2))
res = model.fit()
forecast = res.forecast(steps=50)

# Plot observed vs forecast
plt.figure(figsize=(10,5))
plt.plot(series_train.index, series_train.values, label="Observed", color='blue')
plt.plot(series_test.index, series_test.values, label="Test", color='green')
# Create time index for the forecast period
future_index = pd.date_range(start=series_train.index[-1], periods=51, freq="D")[1:]
plt.plot(future_index, forecast.values, label="Forecast", color='red')
plt.xlabel("Time")
plt.ylabel("Mean plasma energy [eV]")
plt.title("ARIMA Forecast of Mean Ion Energy")
plt.legend()
plt.show()"""

########################################################################

# ROLLING ARIMA FORECAST
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. Preparing data
series = df["E_mean"].rolling(3, min_periods=1).mean()  # short rolling
series = series.interpolate(method='time')  # remove any NaN

# Check stationarity of the original series and the differenced series using ADF and KPSS tests
from statsmodels.tsa.stattools import adfuller, kpss

def check_stationarity(x):
    print("ADF Test")
    adf_result = adfuller(x, autolag="AIC")
    print(f"ADF statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    
    print("\nKPSS Test")
    kpss_result = kpss(x, regression="c", nlags="auto")
    print(f"KPSS statistic: {kpss_result[0]:.4f}")
    print(f"p-value: {kpss_result[1]:.4f}")

check_stationarity(series)

series_diff = series.diff().dropna()
check_stationarity(series_diff)

# Plot ACF and PACF of the differenced series to visually identify potential AR and MA orders
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(8,6))

plot_acf(series_diff, lags=30, ax=ax[0])
ax[0].set_title("ACF - Differenced Series")

plot_pacf(series_diff, lags=30, ax=ax[1], method="ywm")
ax[1].set_title("PACF - Differenced Series")

plt.tight_layout()
plt.show()

# Find optimal ARIMA parameters using AIC minimization
import warnings
import itertools
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

p_values = range(1, 2) # Based on ACF/PACF plots, we can limit p and q to small values to reduce computation time
q_values = range(0, 3) # We can try q=0,1,2 based on the ACF plot showing significant spikes at lags 1 and 2
d = 1

best_aic = float("inf")
best_order = None

reducing_factor = int(0.4 * len(series))  # Use only 40% of data for parameter selection to reduce computation time and avoid overfitting on the full series during this step
series_reduced = series.iloc[:reducing_factor]

for p, q in itertools.product(p_values, q_values):
    try:
        model = ARIMA(series_reduced, order=(p, d, q))
        result = model.fit()
        if result.aic < best_aic:
            best_aic = result.aic
            best_order = (p, d, q)
    except:
        continue

print(f"Best ARIMA{best_order} with AIC = {best_aic:.2f}")

# 2. Train/Test split
split = int(len(series) * 0.9) # 90% train, 10% test
X_train = series.iloc[:split].copy()
X_test  = series.iloc[split:].copy()

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")


# 3. Rolling forecast ARIMA
window_size = 324  # number of points used to train each ARIMA
(p, d, q) = best_order  # ARIMA parameters from best model selection

history = list(X_train[-window_size:])  # Initialize the window
predictions = []

# Initialize loop variables
t = 0
n = 3  # Number of steps to forecast at each iteration
k = len(X_test)  # Total number of test points to predict
# Rolling forecast loop: iteratively predict and update the window
while t < k:
    print(f"Rolling ARIMA: Forecasting steps {t} to {t+n} of {k}...")
    # Fit ARIMA model on current history window
    model = ARIMA(history, order=(p,d,q))
    model_fit = model.fit(method_kwargs={"maxiter": 30})  # Limit iterations for speed and stability
    # Forecast only remaining steps if less than n remain
    steps = min(n, k - t)
    yhat = model_fit.forecast(steps=steps)
    predictions.extend(yhat)
    
    # Update the rolling window with actual observed values
    history.extend(X_test.iloc[t:t+n])
    # Maintain fixed window size by removing oldest values
    if len(history) > window_size:
        history = history[-window_size:]
    t += n


# 4. Metrics
rmse = np.sqrt(mean_squared_error(X_test, predictions))
# Calculate mean absolute error
mae = mean_absolute_error(X_test, predictions)
print("\nRolling ARIMA performance:")
print("RMSE (eV):", rmse)
print("MAE (eV):", mae)

# Calculate relative error as a percentage
relative_error = np.mean(np.abs((X_test - predictions) / X_test)) * 100
print("Relative Error (%):", relative_error)


# 5. Plot
plt.figure(figsize=(12,6))
# Plot training data, test data, and rolling forecast
plt.plot(X_train.index, X_train, label="Train")
plt.plot(X_test.index, X_test, label="Test")
plt.plot(X_test.index, predictions, label="Rolling ARIMA Forecast", color='red')
plt.title("Rolling ARIMA Forecast of Mean Ion Energy")
plt.xlabel("Time")
plt.ylabel("Mean Plasma Energy [eV]")
# Add text box with model parameters and performance metrics
plt.text(0.02, 0.98, f'n={n}, k={k}, window_size={window_size}, RMSE={rmse:.2f} eV, MAE={mae:.2f} eV, RelErr={relative_error:.2f}%', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.legend()
plt.show()
