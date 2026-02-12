import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

set_random_seed = 42
np.random.seed(set_random_seed)
scipy_random_state = np.random.RandomState(set_random_seed)

masses = np.array([4, 12, 16, 56])  # approximate atomic masses for He, C, O, Fe

def prepare_composition_data(file_path="ace_swics_ions_unified.csv"):
    data = pd.read_csv(file_path)
    he_abs = data["nHe2"].astype(float).values
    he_to_o = data["HetoO"].astype(float).values
    c_to_o = data["CtoO"].astype(float).values
    fe_to_o = data["FetoO"].astype(float).values

    # Convert ratios to absolute counts per row, then sum to totals.
    o_abs = np.divide(he_abs, he_to_o, out=np.zeros_like(he_abs), where=he_to_o != 0)
    c_abs = c_to_o * o_abs
    fe_abs = fe_to_o * o_abs

    counts = {
        "He": float(np.nansum(he_abs)),
        "C": float(np.nansum(c_abs)),
        "O": float(np.nansum(o_abs)),
        "Fe": float(np.nansum(fe_abs))
    }
    return data, counts

def bayesian_composition(counts, num_samples=10000):
    # Dirichlet prior (non-informative)
    alpha_prior = np.ones(4)

    # Posterior is also Dirichlet with parameters = observed counts + prior
    N = np.array([counts["He"], counts["C"], counts["O"], counts["Fe"]])
    alpha_posterior = alpha_prior + N

    # Sample from the posterior distribution
    samples = np.random.dirichlet(alpha_posterior, size=num_samples)

    return samples


# Test if abundances follow a power-law with mass, i.e. p_i ~ exp(-lambda * m_i)
def test_composition(file_path="ace_swics_ions_unified.csv"):
    _, counts = prepare_composition_data(file_path=file_path)

    samples = bayesian_composition(counts)
    mean_pi = samples.mean(axis=0)

    def loss(lmbda):
        p = np.exp(-lmbda*masses)
        p /= p.sum()
        return np.sum((p - mean_pi)**2)

    res = minimize(loss, x0=0.01)
    lambda_hat = res.x[0]

    p_model = np.exp(-lambda_hat*masses)
    p_model /= p_model.sum()

    dist = np.linalg.norm(samples - p_model, axis=1)
    p_value = np.mean(dist > np.linalg.norm(mean_pi - p_model))
    print("Lambda hat:", lambda_hat)
    print("P-value:", p_value)
    plt.figure(figsize=(8,5))
    plt.bar(["He", "C", "O", "Fe"], mean_pi, alpha=0.5, label="Posterior Mean")
    plt.plot(["He", "C", "O", "Fe"], p_model, marker='o', label="Power-law Fit")
    plt.ylabel("Relative Abundance")
    plt.title("Plasma Composition and Power-law Fit")
    plt.text(0.02, 0.98, f'λ={lambda_hat:.4f}, p-value={p_value:.4f}', transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.legend()
    plt.show()

test_composition()