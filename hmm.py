import pandas as pd
import numpy as np

set_random_seed = 42
np.random.seed(set_random_seed)

# Load data
df = pd.read_csv("ace_swics_ions_unified.csv", parse_dates=["time"])

# Physical constants
amu = 1.66053906660e-27  # kg
eV = 1.602176634e-19     # J

# ========================
# COMPUTE MEAN ION ENERGY
# ========================
# Ion masses (approx atomic mass units)
masses = {
    "He": 4 * amu,
    "C": 12 * amu,
    "O": 16 * amu,
    "Fe": 56 * amu
}

# Convert speeds to m/s
df["vHe_ms"] = df["vHe2"] * 1e3
df["vC_ms"]  = df["vC5"]  * 1e3
df["vO_ms"]  = df["vO6"]  * 1e3
df["vFe_ms"] = df["vFe10"]* 1e3

# Mean kinetic energy E = 1/2 m v^2 (in eV)
df["E_He"] = 0.5 * masses["He"] * df["vHe_ms"]**2 / eV
df["E_C"]  = 0.5 * masses["C"]  * df["vC_ms"]**2  / eV
df["E_O"]  = 0.5 * masses["O"]  * df["vO_ms"]**2  / eV
df["E_Fe"] = 0.5 * masses["Fe"] * df["vFe_ms"]**2 / eV

# Single mean energy, weighted for abundance
R_He = df["HetoO"]
R_C  = df["CtoO"]
R_Fe = df["FetoO"]

den = 1 + R_He + R_C + R_Fe
w_He = R_He/den
w_C  = R_C/den
w_Fe = R_Fe/den
w_O  = 1/den

df["E_mean"] = (
    w_He*df["E_He"] +
    w_C*df["E_C"] +
    w_Fe*df["E_Fe"] +
    w_O*df["E_O"]
)

df = df.dropna(subset=["E_mean"])

# ========================================
# DEFINE HMM AND FIT TO ENERGY TIME SERIES
# ========================================
from hmmlearn.hmm import GaussianHMM

# HMM expects a 2D array
X = df["E_mean"].values.reshape(-1,1)

model = GaussianHMM(
    n_components=2,
    covariance_type="spherical",
    n_iter=200,
    random_state=42,
    algorithm="viterbi"
)
print(model.means_prior)
model.fit(X)

hidden_states = model.predict(X)
df["state"] = hidden_states

print("Means of each state (eV):")
for i in range(2):
    print(f"State {i}: μ = {model.means_[i][0]:.1f} eV")

print("\nTransition matrix:")
print(model.transmat_)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(df["time"], df["E_mean"], label="Mean Energy", alpha=0.6)
plt.scatter(df["time"], df["E_mean"], c=df["state"], cmap="coolwarm", s=8)
plt.xlabel("Time")
plt.ylabel("Mean Ion Energy [eV]")
plt.title("2-State HMM Regimes in Solar Wind Energy")
plt.show()

logprob, state_probs = model.score_samples(X)
df["P_active"] = state_probs[:, model.means_.argmax()]

time = df["time"]
E = df["E_mean"]
P = df["P_active"]

plt.figure(figsize=(14,5))

# Energy curve
plt.plot(time, E, color="black", linewidth=1, label="Mean ion energy")

# Shade high-probability regions
plt.fill_between(time, E.min(), E.max(),
                 where=P > 0.7,
                 color="red", alpha=0.15,
                 label="High-energy regime likely")

plt.xlabel("Time")
plt.ylabel("Mean ion energy [eV]")
plt.title("Regions of high probability of High-Energy Solar Wind Regimes (HMM)")
plt.legend()
plt.show()

# Change of regime
plt.figure(figsize=(14,2))

plt.scatter(df["time"], np.zeros(len(df)),
            c=df["state"], cmap="coolwarm", marker="|", s=200)

plt.yticks([])
plt.title("HMM-Detected Solar Wind Regimes Over Time")
plt.show()

# ==========================================================
# PREDICTIVE HMM (TRAIN ON FIRST HALF, TEST ON SECOND HALF)
# ==========================================================
# Split data into train/test
df = df.sort_values("time").reset_index(drop=True)

N = len(df)
split_idx = int(N / 1.25)  # Train on first 80%, test on last 20%

train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()

X_train = train["E_mean"].values.reshape(-1,1)
X_test  = test["E_mean"].values.reshape(-1,1)

# Fit HMM on training data
model = GaussianHMM(
    n_components=2,
    covariance_type="spherical",
    n_iter=300,
    random_state=42,
    algorithm="viterbi"
)
model.fit(X_train)

# Identify which state is "high-energy" and which is "low-energy"
high_state = model.means_.argmax()
low_state  = model.means_.argmin()

# Predict states on test set
test_states = model.predict(X_test)
test["state_pred"] = test_states
# Get posterior probabilities of being in high-energy state
logprob, posteriors = model.score_samples(X_test)
test["P_high"] = posteriors[:, high_state]

# Scores
# Log-likelihood per point is a common metric for HMMs, higher is better.
logL_test = model.score(X_test)
logL_train = model.score(X_train)

print("Log-likelihood train:", logL_train)
print("Log-likelihood test :", logL_test)
print("LL per punto (test):", logL_test / len(X_test))

# Predictive log-score (log p(x_t | x_{<t})) is another metric, higher is better.
log_scores = []

for t in range(len(X_test)-1):
    x_next = X_test[t+1].reshape(1,1)
    log_scores.append(model.score(x_next))

log_scores = np.array(log_scores)

print("Mean predictive log-score:", log_scores.mean())

# Baseline: predict next point is same as current point (random walk)
baseline_error = np.mean(np.abs(
    X_test[1:] - X_test[:-1]
))

hmm_error = np.mean(np.abs(
    X_test[1:] - model.means_[test_states[:-1]].reshape(-1,1)
))

print("Baseline MAE:", baseline_error) # MAE of naive random walk predictor
print("HMM regime MAE:", hmm_error) # MAE of HMM regime-based predictor (predict next point is mean of current regime)

def mean_state_duration(states):
    durations = []
    current = states[0]
    length = 1
    for s in states[1:]:
        if s == current:
            length += 1
        else:
            durations.append(length)
            current = s
            length = 1
    durations.append(length)
    return np.mean(durations)

print("Mean regime duration (test):",
      mean_state_duration(test_states))

# Forecast visualization
plt.figure(figsize=(14,5))

plt.plot(test["time"], test["E_mean"], color="black", label="Observed")
plt.fill_between(
    test["time"],
    test["E_mean"].min(),
    test["E_mean"].max(),
    where=test["P_high"] > 0.7,
    alpha=0.25,
    label="Forecast: high-energy regime"
)

plt.title("Out-of-sample Forecast of High-Energy Solar Wind")
plt.ylabel("Mean ion energy [eV]")
plt.legend()
plt.show()

# ===========================================================================
# VERIFY IF DURATION IS GEMETRICALLY DISTRIBUTED (A SIGN OF TRUE HMM REGIMES)
# ===========================================================================
def state_durations(states):
    durations = {0: [], 1: []}
    current = states[0]
    length = 1
    for s in states[1:]:
        if s == current:
            length += 1
        else:
            durations[current].append(length)
            current = s
            length = 1
    durations[current].append(length)
    return durations

dur = state_durations(hidden_states)

# Estimated parameters of geometric distribution (p = 1 - probability of staying in same state)
p_stay = np.diag(model.transmat_)
geom_means = 1 / (1 - p_stay)
print("Durata media HMM:", geom_means)

# Visualize duration distributions
import scipy.stats as stats

for i in [0, 1]:
    d = np.array(dur[i])
    p = p_stay[i]

    xs = np.arange(1, d.max()+1)
    geom_pmf = stats.geom(p=1-p).pmf(xs)

    plt.figure(figsize=(5,3))
    plt.hist(d, bins=100, density=True, alpha=0.6, label="Empirical")
    plt.plot(xs, geom_pmf, 'r-', lw=2, label="Geometric (HMM)")
    plt.title(f"State {i} duration distribution")
    plt.xlabel("Duration")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

# =========================================
# FIND P-VALUE OF HP TEST FOR GEOMETRICITY
# =========================================
def geometric_ks_pvalue(durations, n_boot=5000, random_state=0):
    # Implements KS test for geometric distribution, with bootstrap to get p-value
    rng = np.random.default_rng(random_state)
    durations = np.asarray(durations)

    # MLE per p: E[D] = 1 / (1 - p)
    mean_d = durations.mean()
    p_hat = 1 - 1 / mean_d

    # KS osservato
    ks_obs = stats.kstest(
        durations,
        lambda x: stats.geom(p=1-p_hat).cdf(x)
    ).statistic

    # Bootstrap
    ks_boot = []
    for _ in range(n_boot):
        sim = rng.geometric(p=1-p_hat, size=len(durations))
        p_sim = 1 - 1 / sim.mean()
        ks = stats.kstest(
            sim,
            lambda x: stats.geom(p=1-p_sim).cdf(x)
        ).statistic
        ks_boot.append(ks)

    ks_boot = np.array(ks_boot)
    p_value = np.mean(ks_boot >= ks_obs) # p-value = mean of sup|KS bootstrap - KS observed|

    return ks_obs, p_value, p_hat

# Sample latent state sequences from the fitted HMM
# (to account for posterior uncertainty in state durations)

from numpy.random import default_rng

def sample_state_sequences(model, X, n_samples=100, random_state=0):
    rng = np.random.RandomState(random_state)
    samples = []

    for _ in range(n_samples):
        # We only need the latent states, not the observations
        _, states = model.sample(len(X), random_state=rng)
        samples.append(states)

    return samples


# Draw multiple posterior-like state sequences
state_samples = sample_state_sequences(model, X, n_samples=20)


# Collect durations separately for each state
all_durations = {0: [], 1: []}

for states in state_samples:
    durs = state_durations(states)
    for s in [0, 1]:
        all_durations[s].extend(durs[s])


# Test geometric duration hypothesis for each state
for s in [0, 1]:
    ks, pval, p_hat = geometric_ks_pvalue(all_durations[s])

    print(f"State {s}")
    print(f"  p_hat = {p_hat:.3f}")
    print(f"  KS statistic = {ks:.3f}")
    print(f"  p-value = {pval:.4f}")
