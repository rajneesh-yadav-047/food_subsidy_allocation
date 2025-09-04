import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

# --- 1. Data Preparation (Based on NSS Report Data) ---
fractiles = [
    "0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-95%", "95-100%"
]
mpce_rural = np.array([1373, 1782, 2112, 2454, 2768, 3094, 3455, 3887, 4458, 5356, 6638, 10501])
mpce_urban = np.array([2001, 2607, 3157, 3762, 4348, 4963, 5662, 6524, 7673, 9582, 12399, 20824])

food_share_rural = 0.4638
food_share_urban = 0.3917

all_fractiles_labels = [f"Rural {f}" for f in fractiles] + [f"Urban {f}" for f in fractiles]
mpce_total = np.concatenate([mpce_rural, mpce_urban])
mpce_food_estimated = np.concatenate([
    mpce_rural * food_share_rural,
    mpce_urban * food_share_urban
])

n_fractiles = len(all_fractiles_labels)

# Define Rural and Urban Thresholds (\u03b8)
theta_rural = 1891
theta_urban = 2078
theta = np.concatenate([
    np.full(len(fractiles), theta_rural),
    np.full(len(fractiles), theta_urban)
])

# Household sizes from HCES 2022–23
hh_size_rural = np.array([5.96, 5.60, 5.40, 5.13, 4.92, 4.74, 4.49, 4.29, 4.02, 3.76, 3.42, 2.88])
hh_size_urban = np.array([5.80, 5.42, 4.98, 4.65, 4.47, 4.24, 3.96, 3.67, 3.46, 3.11, 2.66, 2.08])
household_size_arr = np.concatenate([hh_size_rural, hh_size_urban])

# --- 2. Model Parameters ---
gamma = 1.0
n_simulations = 1000

# --- 3. Price Simulation ---
def simulate_prices(n_sims, n_groups):
    prices = np.random.normal(loc=1.0, scale=0.1, size=(n_sims, n_groups))
    prices[prices <= 0] = 0.01
    return prices

# Set random seed for reproducibility
np.random.seed(42)
simulated_prices = simulate_prices(n_simulations, n_fractiles)

# --- 4. Shortfall Calculation Functions ---
def calculate_shortfall(mpce_food_per_capita, theta_val, subsidy, price, gamma_val, hh_size):
    subsidy_per_capita = subsidy / hh_size
    effective_per_capita_consumption = (mpce_food_per_capita + subsidy_per_capita) / price
    shortfall = np.maximum(0, theta_val - effective_per_capita_consumption)
    return shortfall if gamma_val == 1.0 else shortfall ** gamma_val

def calculate_expected_shortfall(subsidies, mpce_food_arr, theta_arr, prices_sim, gamma_val, weights_arr, household_size_arr):
    total_expected_shortfall = 0
    for i in range(len(subsidies)):
        subsidy = np.maximum(0, subsidies[i])
        hh_size = household_size_arr[i]
        mpce_food_per_capita = mpce_food_arr[i]
        theta_val = theta_arr[i]
        shortfalls = calculate_shortfall(mpce_food_per_capita, theta_val, subsidy, prices_sim[:, i], gamma_val, hh_size)
        expected_shortfall = np.mean(shortfalls)
        total_expected_shortfall += weights_arr[i] * expected_shortfall
    return total_expected_shortfall if np.isfinite(total_expected_shortfall) else 1e20

# --- 5. Initial Shortfall and Budget Setup ---
initial_subsidies = np.zeros(n_fractiles)
initial_shortfall_rs_unweighted = np.zeros(n_fractiles)
initial_shortfall_per_capita = np.zeros(n_fractiles)

for i in range(n_fractiles):
    shortfalls = calculate_shortfall(mpce_food_estimated[i], theta[i], 0, simulated_prices[:, i], gamma, household_size_arr[i])
    initial_shortfall_per_capita[i] = np.mean(shortfalls)
    initial_shortfall_rs_unweighted[i] = initial_shortfall_per_capita[i] * household_size_arr[i]

total_initial_rs_shortfall_sum = np.sum(initial_shortfall_rs_unweighted)

# --- 6. Adjusted Weighting Based on Shortfall ---
weight_sum = np.sum(initial_shortfall_rs_unweighted)
weights = initial_shortfall_rs_unweighted / weight_sum if weight_sum > 0 else np.ones(n_fractiles) / n_fractiles

# Add a small offset to ensure all weights are meaningful, then re-normalize
WEIGHT_OFFSET = 0.01
weights = weights + WEIGHT_OFFSET
weights = weights / np.sum(weights)

# --- 7. Optimization Setup ---
initial_total_expected_shortfall = calculate_expected_shortfall(
    initial_subsidies, mpce_food_estimated, theta, simulated_prices, gamma, weights, household_size_arr
)

BUDGET_BUFFER_FACTOR = 1.50
total_budget_B = total_initial_rs_shortfall_sum * BUDGET_BUFFER_FACTOR
scaling_factor = initial_total_expected_shortfall if initial_total_expected_shortfall > 1e-9 else 1.0

objective_func = lambda x: calculate_expected_shortfall(
    x, mpce_food_estimated, theta, simulated_prices, gamma, weights, household_size_arr
) / scaling_factor

budget_constraint = {'type': 'ineq', 'fun': lambda x: total_budget_B - np.sum(x)}
bounds = [(0, None)] * n_fractiles
initial_guess = np.zeros(n_fractiles)

# --- 8. Run Optimization ---
result = minimize(
    objective_func,
    initial_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=[budget_constraint],
    options={'disp': False, 'maxiter': 1500, 'ftol': 1e-10}
)

# Final subsidies from optimization
final_subsidies = np.maximum(0, result.x)

# Ensure budget constraint is met
actual_spending = np.sum(final_subsidies)
if actual_spending > total_budget_B:
    final_subsidies *= total_budget_B / actual_spending

# --- 9. Visualization ---
# Convert subsidies to % of total budget
final_subsidies_percent = (np.array(final_subsidies) / total_budget_B) * 100

# Separate rural and urban
n = len(final_subsidies_percent) // 2
fractile_labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%", 
                   "50-60%", "60-70%", "70-80%", "80-90%", "90-95%", "95-100%"]
rural_labels = [f"R {f}" for f in fractile_labels]
urban_labels = [f"U {f}" for f in fractile_labels]
rural_subsidies = final_subsidies_percent[:n]
urban_subsidies = final_subsidies_percent[n:]

fig = go.Figure()

# Rural bar
fig.add_trace(go.Bar(
    x=rural_labels,
    y=rural_subsidies,
    name='Rural',
    marker=dict(color='rgba(55, 128, 191, 0.8)'),  # Blueish
    hovertemplate='R %{x}<br>%{y:.2f}% of Budget<extra></extra>'
))

# Urban bar
fig.add_trace(go.Bar(
    x=urban_labels,
    y=urban_subsidies,
    name='Urban',
    marker=dict(color='rgba(255, 153, 51, 0.8)'),  # Orange-ish
    hovertemplate='U %{x}<br>%{y:.2f}% of Budget<extra></extra>'
))

# Add a vertical dividing line between rural and urban
fig.update_layout(
    shapes=[
        dict(
            type='line',
            x0=n-0.5,  # Position the line at the center of the dividing point
            x1=n-0.5,
            y0=0,
            y1=1,
            xref='x',  # Use the x-axis scale for positioning
            yref='paper',  # Use the entire height of the plot
            line=dict(
                color='black',  # Line color
                width=2,  # Line width
                dash='dash'  # Optional: make it dashed
            )
        )
    ]
)

fig.update_layout(
    title=dict(
        text='Optimized Food Subsidy Allocation (% of Total Budget)',
        font=dict(size=22, family='Arial'),
        x=0.5
    ),
    xaxis=dict(
        title='Fractile Group',
        tickangle=45,
        tickfont=dict(size=11, family='Arial'),
        showgrid=False
    ),
    yaxis=dict(
        title='% of Budget',
        tickfont=dict(size=12, family='Arial'),
        gridcolor='rgba(200,200,200,0.3)',
        zeroline=False
    ),
    barmode='group',  # Optional: change to 'stack' if stacking is preferred
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(t=80, l=60, r=30, b=120),
    legend=dict(
        title='Region',
        font=dict(size=12),
        orientation='h',
        x=0.5,
        xanchor='center',
        y=1.1
    ),
    height=600,
    width=1000
)

fig.show()
