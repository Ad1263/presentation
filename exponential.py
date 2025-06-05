import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# ------------------------------------------
# 1. Data: heights (cm) and three repeats
# ------------------------------------------
heights = np.array([0,  3,   6,   9,   12,   15,   18,   20], dtype=float)
intensity_data = np.array([
    [-25.3, -26.4, -23.3],
    [-12.1, -11.0, -14.2],
    [ -8.4,  -9.1,  -8.0],
    [ -3.1,  -3.5,  -2.7],
    [ -0.8,  -1.2,  -0.8],
    [  1.6,   1.6,   1.7],
    [  1.9,   2.1,   2.3],
    [  2.5,   2.7,   2.4]
])

# --------------------------------------------------
# 2. Compute the mean and standard deviation (σ_i)
# --------------------------------------------------
means = intensity_data.mean(axis=1)
stds  = intensity_data.std(axis=1, ddof=1)
# If any σ_i = 0 (unlikely here), replace with the average of the nonzero ones
stds[stds == 0] = np.mean(stds[stds != 0])

# ----------------------------------------------------------------
# 3. Define the exponential‐saturation model: I(h) = y0 + A*(1−e^(−k*h))
# ----------------------------------------------------------------
def exp_sat(h, y0, A, k):
    return y0 + A * (1 - np.exp(-k * h))

# ------------------------------------------
# 4. Initial guesses for [y0, A, k]
# ------------------------------------------
initial_guesses = [means[0], means[-1] - means[0], 0.1]

# ================================================================
# 5. Weighted fit (“chi‐squared minimization” using σ_i as weights)
# ================================================================
popt_w, pcov_w = curve_fit(
    exp_sat,
    heights,
    means,
    sigma=stds,
    absolute_sigma=True,
    p0=initial_guesses
)
y0_w, A_w, k_w = popt_w
perr_w        = np.sqrt(np.diag(pcov_w))  # 1σ uncertainties on [y0, A, k]

# Calculate χ², dof, reduced χ², and p‐value
fitted_w     = exp_sat(heights, *popt_w)
residuals_w  = means - fitted_w
chi2_val     = np.sum((residuals_w / stds) ** 2)
dof_w        = len(heights) - len(popt_w)    # 8 data points – 3 params = 5
chi2_red     = chi2_val / dof_w
p_value      = chi2.sf(chi2_val, dof_w)       # survival function of χ²

# ================================================================
# 6. Unweighted fit (Ordinary Least Squares, ignoring σ_i)
# ================================================================
popt_uw, pcov_uw = curve_fit(
    exp_sat,
    heights,
    means,
    p0=initial_guesses
)
y0_uw, A_uw, k_uw = popt_uw
perr_uw          = np.sqrt(np.diag(pcov_uw))  # 1σ uncertainties for OLS

# Calculate SSE for the unweighted fit
fitted_uw = exp_sat(heights, *popt_uw)
residuals_uw = means - fitted_uw
sse_uw = np.sum(residuals_uw ** 2)

# ================================================================
# 7. Print the results
# ================================================================
print("=== Weighted Least Squares Fit (Chi-Squared) ===")
print(f"  y0 = {y0_w:.4f} ± {perr_w[0]:.4f}")
print(f"  A  = {A_w:.4f} ± {perr_w[1]:.4f}")
print(f"  k  = {k_w:.4f} ± {perr_w[2]:.4f}")
print(f"Chi-square         = {chi2_val:.4f}")
print(f"Degrees of freedom = {dof_w}")
print(f"Reduced Chi-square = {chi2_red:.4f}")
print(f"P-value            = {p_value:.4e}\n")

print("=== Ordinary Least Squares Fit (Unweighted) ===")
print(f"  y0 = {y0_uw:.4f} ± {perr_uw[0]:.4f}")
print(f"  A  = {A_uw:.4f} ± {perr_uw[1]:.4f}")
print(f"  k  = {k_uw:.4f} ± {perr_uw[2]:.4f}")
print(f"SSE (sum of squared errors) = {sse_uw:.4f}")

# ================================================================
# 8. Plot both fits together for comparison
# ================================================================
h_fine = np.linspace(0, 20, 200)
fit_curve_w  = exp_sat(h_fine, *popt_w)
fit_curve_uw = exp_sat(h_fine, *popt_uw)

plt.figure(figsize=(7, 5))
# Data with error bars
plt.errorbar(
    heights,
    means,
    yerr=stds,
    fmt='o',
    capsize=4,
    label='Data (mean ± σ)',
    color='tab:blue'
)
# Weighted fit (solid red)
plt.plot(
    h_fine,
    fit_curve_w,
    label='Weighted Fit (χ²)',
    linestyle='-',
    color='tab:red'
)
# Unweighted fit (dashed green)
plt.plot(
    h_fine,
    fit_curve_uw,
    label='Unweighted Fit (OLS)',
    linestyle='--',
    color='tab:green'
)

plt.xlabel('Height (cm)')
plt.ylabel('Intensity (dBm)')
plt.title('Comparison: Weighted vs. Unweighted Fits')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
