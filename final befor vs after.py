import numpy as np
import plotly.graph_objects as go

fractiles = [
    "0-5", "5-10", "10-20", "20-30", "30-40", "40-50",
    "50-60", "60-70", "70-80", "80-90", "90-95", "95-100"
]
labels_rural = [f"R {f}" for f in fractiles]
labels_urban = [f"U {f}" for f in fractiles]
labels_all = labels_rural + labels_urban

mpce_rural = np.array([1373, 1782, 2112, 2454, 2768, 3094, 3455, 3887, 4458, 5356, 6638, 10501])
mpce_urban = np.array([2001, 2607, 3157, 3762, 4348, 4963, 5662, 6524, 7673, 9582, 12399, 20824])
food_share_rural = 0.4638
food_share_urban = 0.3917

mpce_food_estimated = np.concatenate([
    mpce_rural * food_share_rural,
    mpce_urban * food_share_urban
])

theta_rural = 1891
theta_urban = 2078
theta = np.concatenate([
    np.full(len(fractiles), theta_rural),
    np.full(len(fractiles), theta_urban)
])

hh_size_rural = np.array([5.96, 5.60, 5.40, 5.13, 4.92, 4.74, 4.49, 4.29, 4.02, 3.76, 3.42, 2.88])
hh_size_urban = np.array([5.80, 5.42, 4.98, 4.65, 4.47, 4.24, 3.96, 3.67, 3.46, 3.11, 2.66, 2.08])
household_size_arr = np.concatenate([hh_size_rural, hh_size_urban])

gamma = 1.0
n_simulations = 1000
np.random.seed(42)

# --- Simulate Prices ---
def simulate_prices(n_sims, n_groups):
    prices = np.random.normal(loc=1.0, scale=0.1, size=(n_sims, n_groups))
    prices[prices <= 0] = 0.01
    return prices

simulated_prices = simulate_prices(n_simulations, len(labels_all))

# --- Shortfall Calculation ---
def calculate_shortfall(mpce_food_pc, theta_val, subsidy, price, gamma_val, hh_size):
    subsidy_pc = subsidy / hh_size
    eff_pc = (mpce_food_pc + subsidy_pc) / price
    shortfall = np.maximum(0, theta_val - eff_pc)
    return shortfall if gamma_val == 1.0 else shortfall ** gamma_val

# --- Budget & Subsidies ---
initial_shortfall_rs_unweighted = np.zeros(len(labels_all))

for i in range(len(labels_all)):
    shortfalls = calculate_shortfall(mpce_food_estimated[i], theta[i], 0, simulated_prices[:, i], gamma, household_size_arr[i])
    mean_shortfall = np.mean(shortfalls)
    initial_shortfall_rs_unweighted[i] = mean_shortfall * household_size_arr[i]

total_budget_B = np.sum(initial_shortfall_rs_unweighted) * 0.7

subsidy_pct = np.array([
    12.25, 10.53, 9.14, 7.71, 6.41, 4.94, 3.28, 1.21, 0.74, 0, 0, 0,
    12.62, 10.47, 8.54, 6.36, 4.12, 1.75, 0.34, 0.02, 0, 0, 0, 0
])
final_subsidies = (subsidy_pct / 100) * total_budget_B
actual_spending = np.sum(final_subsidies)
if actual_spending > total_budget_B:
    final_subsidies *= total_budget_B / actual_spending

# --- Compute Shortfalls ---
shortfall_before_pct = np.zeros(len(labels_all))
shortfall_after_pct = np.zeros(len(labels_all))

for i in range(len(labels_all)):
    hh_size = household_size_arr[i]
    shortfalls_before = calculate_shortfall(mpce_food_estimated[i], theta[i], 0, simulated_prices[:, i], gamma, hh_size)
    shortfalls_after = calculate_shortfall(mpce_food_estimated[i], theta[i], final_subsidies[i], simulated_prices[:, i], gamma, hh_size)
    shortfall_before_pct[i] = np.mean(shortfalls_before) / theta[i] * 100
    shortfall_after_pct[i] = np.mean(shortfalls_after) / theta[i] * 100

# --- Custom Colors ---
rural_before_color = '#4682B4'     # steel blue
rural_after_color = '#87CEFA'      # light sky blue
urban_before_color = '#DC143C'     # crimson
urban_after_color = '#FF7F7F'      # soft coral

before_colors = [rural_before_color] * 12 + [urban_before_color] * 12
after_colors = [rural_after_color] * 12 + [urban_after_color] * 12

# --- Plot ---
fig = go.Figure()

fig.add_trace(go.Bar(
    x=labels_all,
    y=shortfall_before_pct,
    name='Before (%)',
    marker_color=before_colors
))

fig.add_trace(go.Bar(
    x=labels_all,
    y=shortfall_after_pct,
    name='After (%)',
    marker_color=after_colors
))

fig.update_layout(
    title='Shortfall Before vs After Subsidy (as % of Consumption Threshold)',
    xaxis_title='MPCE Fractile',
    yaxis_title='Shortfall (% of Threshold)',
    xaxis_tickangle=-45,
    barmode='group',
    height=600,
    width=1200,
    plot_bgcolor='white',
    font=dict(size=14),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
)

# --- Add Dividing Line between Rural and Urban ---
fig.add_shape(
    type="line",
    x0=11.5, x1=11.5,
    y0=0, y1=max(max(shortfall_before_pct), max(shortfall_after_pct)),
    line=dict(color="black", width=2, dash="dot"),
    xref='x', yref='y'
)

fig.update_yaxes(showgrid=True, gridcolor='lightgray')

fig.show()
