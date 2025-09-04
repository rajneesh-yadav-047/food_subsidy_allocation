import pandas as pd

# --------------------------------------------------------------------------
# SCRIPT TO CALCULATE RURAL/URBAN FOOD ADEQUACY THRESHOLDS (Monetary Value)
# --------------------------------------------------------------------------
# This script calculates the monthly cost per person to purchase a defined
# food basket, using separate prices for rural and urban areas.
#
# !!! IMPORTANT !!!
# The quantities and prices below are HYPOTHETICAL examples.
# Replace them with accurate data based on:
#   1. Nutritional Norms (e.g., ICMR) -> Define Basket & Quantities
#   2. Market Price Data (e.g., Price Monitoring Cell, CPI) for 2022-23
# --------------------------------------------------------------------------

# --- 1. Define Food Basket, Quantities, and Units ---

# Reference Caloric Norm (for context, calculation uses defined quantities)
CALORIC_NORM_KCAL_PER_DAY = 2200 # Example value

# Representative Food Basket Items
# (Ensure these items match available price data)
basket_items = [
    "Rice", "Wheat (Atta)", "Arhar Dal (Tur)", "Edible Oil", "Sugar",
    "Potatoes", "Onions", "Milk", "Salt"
]

# Units for Quantities and Prices (Must be consistent)
basket_units = {
    "Rice": "kg", "Wheat (Atta)": "kg", "Arhar Dal (Tur)": "kg",
    "Edible Oil": "litre", "Sugar": "kg", "Potatoes": "kg",
    "Onions": "kg", "Milk": "litre", "Salt": "kg"
}

# Original base quantities (used to compute current totals)
base_quantities = {
    "Rice": 6.0,
    "Wheat (Atta)": 6.0,
    "Arhar Dal (Tur)": 1.5,
    "Edible Oil": 0.8,
    "Sugar": 1.0,
    "Potatoes": 3.0,
    "Onions": 1.5,
    "Milk": 5.0,
    "Salt": 0.5
}

# --- 2. Define Rural and Urban Prices (HYPOTHETICAL - Needs Replacement) ---
# Average Retail Prices (Rs per Unit) for the period (e.g., 2022-23)

rural_prices = {
    "Rice": 30.0,
    "Wheat (Atta)": 28.0,
    "Arhar Dal (Tur)": 100.0,
    "Edible Oil": 140.0, # Price per litre
    "Sugar": 40.0,
    "Potatoes": 20.0,
    "Onions": 25.0,
    "Milk": 50.0, # Price per litre
    "Salt": 15.0
}

urban_prices = {
    "Rice": 35.0,
    "Wheat (Atta)": 32.0,
    "Arhar Dal (Tur)": 115.0,
    "Edible Oil": 150.0, # Price per litre
    "Sugar": 42.0,
    "Potatoes": 25.0,
    "Onions": 30.0,
    "Milk": 55.0, # Price per litre
    "Salt": 18.0
}

# --- 3. Compute Scaling Factors ---
base_rural_total = sum(base_quantities[item] * rural_prices[item] for item in basket_items)
base_urban_total = sum(base_quantities[item] * urban_prices[item] for item in basket_items)

scale_rural = 1891 / base_rural_total    # ≈1.8821
scale_urban = 2078 / base_urban_total    # ≈1.8226

monthly_quantities_rural = {item: base_quantities[item] * scale_rural for item in basket_items}
monthly_quantities_urban = {item: base_quantities[item] * scale_urban for item in basket_items}

# --- 3. Perform Calculation ---

rural_threshold_calc = 0
urban_threshold_calc = 0
calculation_details = []

print("Calculating thresholds based on defined basket, quantities, and prices...\n")

# Verify all items are present in all dictionaries
missing_items = False
for item in basket_items:
    if item not in base_quantities:
        print(f"ERROR: Item '{item}' missing from base_quantities.")
        missing_items = True
    if item not in rural_prices:
        print(f"ERROR: Item '{item}' missing from rural_prices.")
        missing_items = True
    if item not in urban_prices:
        print(f"ERROR: Item '{item}' missing from urban_prices.")
        missing_items = True
    if item not in basket_units:
        print(f"ERROR: Item '{item}' missing from basket_units.")
        missing_items = True

if missing_items:
    print("\nPlease fix missing items before proceeding.")
else:
    # Proceed with calculation
    for item in basket_items:
        quantity_rural = monthly_quantities_rural[item]
        quantity_urban = monthly_quantities_urban[item]
        unit = basket_units[item]
        rural_price = rural_prices[item]
        urban_price = urban_prices[item]

        rural_cost_item = quantity_rural * rural_price
        urban_cost_item = quantity_urban * urban_price

        rural_threshold_calc += rural_cost_item
        urban_threshold_calc += urban_cost_item

        calculation_details.append({
            "Item": item,
            "Unit": unit,
            "Quantity (Rural)": quantity_rural,
            "Quantity (Urban)": quantity_urban,
            "Rural Price (Rs/Unit)": rural_price,
            "Urban Price (Rs/Unit)": urban_price,
            "Rural Cost (Rs)": rural_cost_item,
            "Urban Cost (Rs)": urban_cost_item
        })

    # --- 4. Display Results ---

    # Display detailed breakdown using pandas DataFrame
    details_df = pd.DataFrame(calculation_details)
    details_df = details_df[[ # Define column order
        "Item", "Unit", "Quantity (Rural)", "Quantity (Urban)", "Rural Price (Rs/Unit)",
        "Urban Price (Rs/Unit)", "Rural Cost (Rs)", "Urban Cost (Rs)"
    ]]
    print("--- Detailed Cost Breakdown (per person per month) ---")
    # Format floats for better display
    pd.options.display.float_format = '{:.2f}'.format
    print(details_df.to_string(index=False))
    print("-" * (details_df.to_string(index=False).find('\n'))) # Dynamic separator length

    # Display final calculated thresholds
    print(f"\nFINAL CALCULATED THRESHOLDS:")
    print(f"  Rural Food Adequacy Threshold: Rs. {rural_threshold_calc:.2f} per person per month")
    print(f"  Urban Food Adequacy Threshold: Rs. {urban_threshold_calc:.2f} per person per month")

    print("\n" + "="*60)
    print("REMINDER: These thresholds are based on the HYPOTHETICAL")
    print("          quantities and prices defined in this script.")
    print("          Replace them with accurate, validated data for")
    print("          your specific analysis and reference period.")
    print("="*60)

