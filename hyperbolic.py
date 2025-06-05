import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# --------------------------------------------------
# 1. Data: heights (cm) and three replicates (dBm)
# --------------------------------------------------
heights = np.array([0,  3,   6,   9,   12,   15,   18,   20], dtype=float)
intensity_data = np.array([
    [-25.3, -26.4, -23.3],
    [-12.1, -11.0, -11.2],
    [ -8.4,  -9.1,  -8.0],
    [ -3.1,  -3.5,  -2.7],
    [ -0.8,  -1.2,  -0.8],
    [  1.6,   1.6,   1.7],
    [  1.9,   2.1,   2.3],
    [  2.5,   2.7,   2.4]
])

# --------------------------------------------------
# 2. Compute means and standard deviations (σ_i)
# --------------------------------------------------
means = intensity_data.mean(axis=1)
stds  = intensity_data.std(axis=1, ddof=1)
# If any σ_i = 0 (edge‐case), replace with the average of nonzero σ’s
stds[stds == 0] = np.mean(stds[stds != 0])

# --------------------------------------------------
# 3. Define the hyperbolic‐saturation model:
#    I(h) = y0 + A * (h / (h + B))
# --------------------------------------------------
def hyper_model(h, y0, A, B):
    return y0 + A * (h / (h + B))

# --------------------------------------------------
# 4. Initial guesses for [y0, A, B]
#    - y0 ≈ baseline (means[0])
#    - A  ≈ total rise (means[-1] - means[0])
#    - B  ≈ a characteristic height (~6 cm)
# --------------------------------------------------
initial_guesses = [means[0], means[-1] - means[0], 6.0]

# ================================================================
# 5. Weighted Fit (Chi-Squared minimization) using σ_i as weights
# ================================================================
popt_w, pcov_w = curve_fit(
    hyper_model,
    heights,
    means,
    sigma=stds,
    absolute_sigma=True,
    p0=initial_guesses
)
y0_w, A_w, B_w = popt_w
perr_w         = np.sqrt(np.diag(pcov_w))  # 1σ uncertainties

# Compute chi-square, degrees of freedom, reduced chi-square, and p-value
fitted_w      = hyper_model(heights, *popt_w)
resid_w       = means - fitted_w
chi2_val      = np.sum((resid_w / stds) ** 2)
dof_w         = len(heights) - len(popt_w)  # 8 data points – 3 parameters = 5 dof
chi2_red_w    = chi2_val / dof_w
p_value_w     = chi2.sf(chi2_val, dof_w)    # survival function (upper tail)

# ================================================================
# 6. Unweighted Fit (Ordinary Least Squares, ignoring σ_i)
# ================================================================
popt_uw, pcov_uw = curve_fit(
    hyper_model,
    heights,
    means,
    p0=initial_guesses
)
y0_uw, A_uw, B_uw = popt_uw
perr_uw          = np.sqrt(np.diag(pcov_uw))

# Compute sum of squared errors (SSE) for the unweighted fit
fitted_uw     = hyper_model(heights, *popt_uw)
resid_uw      = means - fitted_uw
sse_uw        = np.sum(resid_uw ** 2)

# ================================================================
# 7. Print results and goodness‐of‐fit metrics
# ================================================================
print("=== Hyperbolic Model: Weighted Fit (Chi-Squared) ===")
print(f"  y0 = {y0_w:.4f} ± {perr_w[0]:.4f} dBm")
print(f"  A  = {A_w:.4f} ± {perr_w[1]:.4f} dBm")
print(f"  B  = {B_w:.4f} ± {perr_w[2]:.4f} cm")
print(f"Chi-square         = {chi2_val:.4f}")
print(f"Degrees of freedom = {dof_w}")
print(f"Reduced Chi-square = {chi2_red_w:.4f}")
print(f"P-value            = {p_value_w:.4e}\n")

print("=== Hyperbolic Model: Unweighted Fit (OLS) ===")
print(f"  y0 = {y0_uw:.4f} ± {perr_uw[0]:.4f} dBm")
print(f"  A  = {A_uw:.4f} ± {perr_uw[1]:.4f} dBm")
print(f"  B  = {B_uw:.4f} ± {perr_uw[2]:.4f} cm")
print(f"SSE (sum of squared errors) = {sse_uw:.4f} dBm²\n")

# ================================================================
# 8. Plot data + both hyperbolic‐fit curves
# ================================================================
h_fine = np.linspace(0, 20, 200)
curve_w   = hyper_model(h_fine, *popt_w)
curve_uw  = hyper_model(h_fine, *popt_uw)

plt.figure(figsize=(7, 5))
plt.errorbar(
    heights,
    means,
    yerr=stds,
    fmt='o',
    capsize=4,
    label='Data (mean ± σ)',
    color='tab:blue'
)
plt.plot(
    h_fine,
    curve_w,
    label='Hyperbolic Weighted Fit',
    linestyle='-',
    color='tab:red'
)
plt.plot(
    h_fine,
    curve_uw,
    label='Hyperbolic Unweighted Fit',
    linestyle='--',
    color='tab:green'
)
plt.xlabel('Height (cm)')
plt.ylabel('Intensity (dBm)')
plt.title('Hyperbolic Fit to Intensity vs. Height')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================================================
# 9. Plot residuals for both hyperbolic fits
# ================================================================
plt.figure(figsize=(7, 5))
# Weighted residuals (with error bars)
plt.errorbar(
    heights,
    resid_w,
    yerr=stds,
    fmt='o',
    capsize=4,
    label='Residuals (Weighted)',
    color='tab:red'
)
# Unweighted residuals (filled squares)
plt.scatter(
    heights,
    resid_uw,
    marker='s',
    label='Residuals (Unweighted)',
    color='tab:green'
)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Height (cm)')
plt.ylabel('Residual (dBm)')
plt.title('Hyperbolic Model Residuals')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
