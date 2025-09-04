import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio # Import for templates

# Set default template for plots
pio.templates.default = "plotly_white"

import os

# --- 1. Data Preparation (Based on Statements 1 and 4) ---

fractiles = [
    "0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-95%", "95-100%"
]
mpce_rural = np.array([
    1373, 1782, 2112, 2454, 2768, 3094, 3455, 3887, 4458, 5356, 6638, 10501
])
mpce_urban = np.array([
    2001, 2607, 3157, 3762, 4348, 4963, 5662, 6524, 7673, 9582, 12399, 20824
])

food_share_rural = 0.4638
food_share_urban = 0.3917

all_fractiles_labels = [f"Rural {f}" for f in fractiles] + [f"Urban {f}" for f in fractiles]
mpce_total = np.concatenate([mpce_rural, mpce_urban])

mpce_food_estimated = np.concatenate([
    mpce_rural * food_share_rural,
    mpce_urban * food_share_urban
])

n_fractiles = len(all_fractiles_labels)

# *** MODIFICATION: Use DIFFERENT Thresholds for Rural and Urban ***
RURAL_THRESHOLD = 1891 # Rs per person per month (Based on shortfall vs mpce.py)
URBAN_THRESHOLD = 2078 # Rs per person per month (Based on shortfall vs mpce.py / Urban Avg)

# Create the theta array with different values
theta = np.concatenate([
    np.full(len(fractiles), RURAL_THRESHOLD), # First 12 are rural
    np.full(len(fractiles), URBAN_THRESHOLD)  # Next 12 are urban
])
print(f"NOTE: Using DIFFERENT thresholds: Rural=Rs.{RURAL_THRESHOLD}, Urban=Rs.{URBAN_THRESHOLD}")


# *** ADD Household Sizes ***
# Using example from sub vs mpce.py, applied to both rural and urban
# Replace with more accurate data if available
base_h_i = np.array([5.8, 5.6, 5.4, 5.2, 5.0, 4.8, 4.6, 4.4, 4.2, 4.0, 3.8, 3.6])
household_sizes = np.concatenate([base_h_i, base_h_i])
print(f"NOTE: Using household sizes (Rural/Urban): {np.round(household_sizes, 1)}")
# --- 2. Model Parameters ---
# gamma = 1.0 # Previous parameter, now using squared shortfall
print(f"NOTE: Using SQUARED shortfall objective (minimizing sum w_i * E[max(0, theta - consumption)^2])")

n_simulations = 1000

# Use Rank-Based Vulnerability Weights
base_weights = np.array([2.0, 1.8, 1.6, 1.5, 1.3, 1.1, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4])
weights = np.concatenate([base_weights, base_weights])
print(f"NOTE: Using decreasing vulnerability weights (rank-based): {np.round(weights, 2)}")


# --- 3. Price Simulation ---
def simulate_prices(n_sims, n_groups):
    prices = np.random.normal(loc=1.0, scale=0.1, size=n_sims)
    prices[prices <= 0] = 0.01
    return prices

simulated_prices = simulate_prices(n_simulations, n_fractiles)

# --- 4. Shortfall Calculation Functions ---
# (Functions remain the same - calculate absolute shortfall)
def calculate_shortfall_squared(mpce_food, theta_val, subsidy, price, hh_size):
    """Calculates the squared monetary shortfall."""
    # Effective consumption = (Base Food Spending + Subsidy) / (Price * Household Size)
    # Ensure price * hh_size is not zero or too small
    denominator = np.maximum(1e-9, price * hh_size)
    effective_consumption = (mpce_food + subsidy) / denominator
    shortfall = np.maximum(0, theta_val - effective_consumption)
    return shortfall ** 2 # Return the square of the shortfall

def calculate_expected_squared_shortfall(subsidies, mpce_food_arr, theta_arr, prices_sim, weights_arr, hh_sizes_arr):
    total_expected_shortfall = 0
    num_groups = len(subsidies)
    num_sims = len(prices_sim)
    for i in range(num_groups):
        current_subsidy = np.maximum(0, subsidies[i])
        shortfalls_i = calculate_shortfall_squared(mpce_food_arr[i], theta_arr[i], current_subsidy, prices_sim, hh_sizes_arr[i])
        if np.any(np.isnan(shortfalls_i)) or np.any(np.isinf(shortfalls_i)):
             print(f"Warning: NaN/Inf in shortfall calc for group {i}")
             shortfalls_i = np.nan_to_num(shortfalls_i, nan=0.0, posinf=1e20, neginf=0.0)
        expected_shortfall_i = np.mean(shortfalls_i)
        total_expected_shortfall += weights_arr[i] * expected_shortfall_i # Apply weight
    if np.isnan(total_expected_shortfall) or np.isinf(total_expected_shortfall):
        print(f"Warning: Final total expected shortfall is NaN/Inf.")
        return 1e20
    return total_expected_shortfall

# --- 5. Optimization Setup ---
initial_subsidies = np.zeros(n_fractiles)

# Calculate initial *linear* shortfall in Rs terms (for budget setting) per fractile (UNWEIGHTED)
# Need a temporary function or direct calculation for linear shortfall
def calculate_shortfall_linear(mpce_food, theta_val, subsidy, price, hh_size):
    """Calculates the linear monetary shortfall per capita, considering household size."""
    # Effective consumption = (Base Food Spending + Subsidy) / (Price * Household Size)
    # Ensure price * hh_size is not zero or too small
    denominator = np.maximum(1e-9, price * hh_size)
    effective_consumption = (mpce_food + subsidy) / denominator
    shortfall = np.maximum(0, theta_val - effective_consumption)
    return shortfall

initial_shortfall_rs_unweighted = np.zeros(n_fractiles)
for i in range(n_fractiles):
     # theta[i] will now be RURAL_THRESHOLD or URBAN_THRESHOLD
     shortfalls_i_rs = calculate_shortfall_linear(mpce_food_estimated[i], theta[i], 0, simulated_prices, household_sizes[i])
     initial_shortfall_rs_unweighted[i] = np.mean(shortfalls_i_rs)
total_initial_rs_shortfall_sum = np.sum(initial_shortfall_rs_unweighted) # Cost estimate changes


# Set Budget based on cost to eliminate shortfall relative to DIFFERENT thresholds
BUDGET_BUFFER_FACTOR = 1.10 # Keep buffer
total_budget_B = total_initial_rs_shortfall_sum * BUDGET_BUFFER_FACTOR
print(f"\nNOTE: Setting budget based on cost to reach RURAL/URBAN thresholds.")

# Scaling factor uses the new initial objective value
initial_total_expected_squared_shortfall = calculate_expected_squared_shortfall(
    initial_subsidies, mpce_food_estimated, theta, simulated_prices, weights, household_sizes
)

scaling_factor = initial_total_expected_squared_shortfall if initial_total_expected_squared_shortfall > 1e-9 else 1.0

print(f"Initial Total Weighted Expected SQUARED Shortfall: {initial_total_expected_squared_shortfall:.6f}")
print(f"Total Initial Unweighted Rs Shortfall (estimated cost to reach thresholds): {total_initial_rs_shortfall_sum:.2f}")
print(f"Total Budget Allocated (B = {BUDGET_BUFFER_FACTOR*100:.0f}% of estimated cost): Rs. {total_budget_B:.2f}")
print(f"Using Objective Function Scaling Factor: {scaling_factor:.2f}")

# Objective function uses the weighted expected SQUARED shortfall
objective_func = lambda x: calculate_expected_squared_shortfall(
    x, mpce_food_estimated, theta, simulated_prices, weights, household_sizes
) / scaling_factor

budget_constraint = {'type': 'ineq', 'fun': lambda x: total_budget_B - np.sum(x)}
bounds = [(0, None) for _ in range(n_fractiles)]
initial_guess = np.zeros(n_fractiles)

# --- 6. Run Optimization ---
print("\nStarting optimization (minimizing weighted expected SQUARED shortfall)...")
result = minimize(
    objective_func,
    initial_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=[budget_constraint],
    options={'disp': True, 'maxiter': 1500, 'ftol': 1e-10}
)
print("Optimization finished.")

# --- Post-Processing and Results ---
# (Post-processing logic remains the same for finding final_subsidies)
final_subsidies = None
optimization_successful = False

if result.success:
    final_subsidies = result.x
    final_subsidies = np.maximum(0, final_subsidies)
    actual_spending = np.sum(final_subsidies)

    if actual_spending < total_budget_B * 0.999:
        print(f"\nNote: Optimizer used Rs. {actual_spending:.2f}, less than the allocated budget of Rs. {total_budget_B:.2f}.")
    else:
         print(f"\nNote: Optimizer used Rs. {actual_spending:.2f}, close to the allocated budget of Rs. {total_budget_B:.2f}.")
         if actual_spending > total_budget_B:
             print(f"  Adjusting subsidies slightly downwards to meet budget constraint.")
             if actual_spending > 1e-9:
                 final_subsidies = final_subsidies * (total_budget_B / actual_spending)
             else:
                 final_subsidies = np.zeros_like(final_subsidies)

    final_scaled_objective = result.fun
    final_total_expected_squared_shortfall = calculate_expected_squared_shortfall(
        final_subsidies, mpce_food_estimated, theta, simulated_prices, weights, household_sizes
    )
    optimization_successful = True

    print(f"\nOptimization Successful!")
    print(f"Optimal Subsidies Allocated (sum = Rs. {np.sum(final_subsidies):.2f}):")
    print(f"Final Total Weighted Expected SQUARED Shortfall (unscaled): {final_total_expected_squared_shortfall:.6f}")
    reduction = initial_total_expected_squared_shortfall - final_total_expected_squared_shortfall
    percent_reduction = (reduction / initial_total_expected_squared_shortfall) * 100 if initial_total_expected_squared_shortfall > 1e-9 else 0
    print(f"Reduction in Total Weighted Expected SQUARED Shortfall: {reduction:.6f} ({percent_reduction:.2f}%)")

else:
    # ... (failure handling remains the same) ...
    print("\nOptimization FAILED:")
    print(f"  Message: {result.message}")
    if hasattr(result, 'x') and result.x is not None and len(result.x) == n_fractiles:
         print("  Using subsidies from the last iteration before failure for analysis.")
         final_subsidies = np.maximum(0, result.x)
         current_sum = np.sum(final_subsidies)
         if current_sum > total_budget_B:
             if current_sum > 1e-9:
                 final_subsidies = final_subsidies * (total_budget_B / current_sum)
             else:
                 final_subsidies = np.zeros_like(final_subsidies)
    else:
         print("  Defaulting to initial subsidies (zeros) for analysis.")
         final_subsidies = initial_guess


# Calculate values for Percentage Reporting (using LINEAR shortfall, gamma=1)
subsidy_pct = (final_subsidies / total_budget_B * 100) if total_budget_B > 1e-9 else np.zeros(n_fractiles)

shortfall_before_pct = np.zeros(n_fractiles)
shortfall_after_pct = np.zeros(n_fractiles)
for i in range(n_fractiles):
    shortfalls_before_i_mean = np.mean(calculate_shortfall_linear(mpce_food_estimated[i], theta[i], 0, simulated_prices, household_sizes[i]))
    shortfalls_after_i_mean = np.mean(calculate_shortfall_linear(mpce_food_estimated[i], theta[i], final_subsidies[i], simulated_prices, household_sizes[i]))
    # theta[i] is now the specific rural/urban threshold
    shortfall_before_pct[i] = (shortfalls_before_i_mean / theta[i] * 100) if theta[i] > 1e-9 else 0
    shortfall_after_pct[i] = (shortfalls_after_i_mean / theta[i] * 100) if theta[i] > 1e-9 else 0


# --- 7. Visualization with Plotly ---

plot1_filename = "subsidy_allocation_pct_diff_theta.html"
plot2_filename = "shortfall_reduction_pct_diff_theta_sq_obj.html" # Indicate squared objective used
# Plot 1 title updated to reflect budget source
# *** MODIFICATION: Plot 1 - Enhanced Aesthetics ***
print("\nDisplaying Plot 1: Subsidy Allocation (% of Total Budget)...")
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=all_fractiles_labels,
    y=subsidy_pct,
    name='Subsidy Allocation (% of Budget)',
    marker_color=['#1f77b4']*len(fractiles) + ['#ff7f0e']*len(fractiles), # Distinct Blue/Orange
    hovertemplate = ('<b>%{x}</b><br>' + # Bold fractile label
                     'Subsidy Share: %{y:.2f}%<br>' + # Subsidy %
                     'Absolute Subsidy: ₹%{customdata:.2f}' + # Absolute Rs amount
                     '<extra></extra>'), # Remove trace name
    customdata = final_subsidies # Pass absolute subsidies for hover
))
fig1.update_layout(
    title=dict(text=f'<b>Optimized Subsidy Allocation (% of Total Budget = ₹{total_budget_B:.0f})</b><br>(Objective: Minimize Weighted Expected Squared Shortfall)', x=0.5), # Centered, bold title
    xaxis_title='MPCE Fractile Class (Rural/Urban)',
    yaxis_title='Allocated Subsidy (% of Total Budget)',
    yaxis_ticksuffix="%",
    xaxis_tickangle=-45,
    barmode='group',
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99), # Legend position
    margin=dict(l=60, r=30, t=50, b=100), # Adjust margins
    hovermode='x unified' # Show hover for all bars at a given x
)
fig1.show()

# Plot 2 title updated to reflect objective
# *** MODIFICATION: Plot 2 - Enhanced Aesthetics ***
print("Displaying Plot 2: Shortfall Comparison (% of Threshold)...")
fig2 = go.Figure()
# Add trace for BEFORE subsidy
fig2.add_trace(go.Bar(
    x=all_fractiles_labels,
    y=shortfall_before_pct,
    name='Initial Shortfall', # Simpler name
    marker_color='#fdbe85', # Lighter orange
    hovertemplate = ('<b>%{x}</b><br>' +
                     'Scenario: Initial<br>' +
                     'Shortfall: %{y:.2f}% of Threshold' +
                     '<extra></extra>'),
    customdata = theta # Pass threshold for potential use
))
# Add trace for AFTER subsidy
fig2.add_trace(go.Bar(
    x=all_fractiles_labels,
    y=shortfall_after_pct,
    name='Shortfall After Subsidy', # Simpler name
    marker_color='#bae4bc', # Lighter green
    hovertemplate = ('<b>%{x}</b><br>' +
                     'Scenario: After Subsidy<br>' +
                     'Shortfall: %{y:.2f}% of Threshold' +
                     '<extra></extra>'),
    customdata = theta
))
fig2.update_layout(
    title=dict(text=f'<b>Expected Linear Food Shortfall (% of Threshold) - Before vs. After Subsidy</b><br>(Subsidy Optimized for Squared Shortfall; Thresholds: R=₹{RURAL_THRESHOLD}, U=₹{URBAN_THRESHOLD})', x=0.5), # Centered, bold title with threshold info
    xaxis_title='MPCE Fractile Class (Rural/Urban)',
    yaxis_title='Expected Shortfall (% of Respective Threshold)', # Clarify y-axis
    yaxis_ticksuffix="%",
    xaxis_tickangle=-45,
    barmode='group',
    legend_title_text='Scenario', # Add title to legend
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    margin=dict(l=60, r=30, t=80, b=100), # Adjust margins (more top margin for longer title)
    hovermode='x unified'
)
# Optionally set y-axis range if needed, e.g., to zoom in on small 'after' values
# fig2.update_yaxes(range=[0, max(shortfall_before_pct)*1.1])

fig2.show()


# Detailed Allocation Table (reflecting squared objective)
print(f"\nDetailed Allocation (Thresholds: Rural=Rs.{RURAL_THRESHOLD}, Urban=Rs.{URBAN_THRESHOLD}):")
allocation_df = pd.DataFrame({
    'Fractile': all_fractiles_labels,
    'Weight (wi)': np.round(weights, 2),
    'Estimated Food MPCE (Rs.)': mpce_food_estimated,
    'Household Size (hi)': np.round(household_sizes, 1), # Add household size
    'Threshold (theta) (Rs.)': theta, # Shows the R/U threshold used
    'Final Subsidy (Rs.)': final_subsidies,
    'Subsidy (% of Budget)': subsidy_pct,
    'Initial Exp. Linear Shortfall (% of Threshold)': shortfall_before_pct, # Clarify linear
    'Final Exp. Linear Shortfall (% of Threshold)': shortfall_after_pct # Clarify linear
})
# Adjust formatting for percentages
pd.options.display.float_format = '{:.2f}'.format
allocation_df['Subsidy (% of Budget)'] = allocation_df['Subsidy (% of Budget)'].map('{:.2f}%'.format)
allocation_df['Initial Exp. Shortfall (% of Threshold)'] = allocation_df['Initial Exp. Shortfall (% of Threshold)'].map('{:.2f}%'.format)
allocation_df['Final Exp. Shortfall (% of Threshold)'] = allocation_df['Final Exp. Shortfall (% of Threshold)'].map('{:.2f}%'.format)

print(allocation_df.to_string())

print("\nNOTE on Plotting: fig.show() attempts to open plots in your default browser.")
