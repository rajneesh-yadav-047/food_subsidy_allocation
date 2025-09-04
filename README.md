# Optimal Food Subsidy Allocation Model

[!License: MIT](https://opensource.org/licenses/MIT)

This project provides a Python-based framework for optimizing the allocation of a food subsidy budget. It aims to minimize food consumption shortfalls across different socio-economic groups by targeting aid more effectively than a uniform distribution. The model is designed for policy analysts, researchers, and economists working on food security and social welfare programs.

## About The Project

Distributing a limited food subsidy budget presents a significant challenge: how can we maximize the impact and ensure the aid reaches the most vulnerable populations? A flat, uniform subsidy is easy to implement but often inefficient, giving the same benefit to groups with vastly different needs.

This project tackles this problem using a two-part quantitative model:

1.  **Threshold Calculation (`calculate_thresholds.py`):** Establishes a monetary threshold for "food adequacy" based on the cost of a representative food basket. It calculates separate thresholds for rural and urban areas to account for price disparities.

2.  **Subsidy Optimization (`final subsidy vs mpce.py`):** Uses these thresholds in a stochastic optimization model. The model allocates a fixed budget across various population fractiles (grouped by expenditure) to minimize the total *weighted expected squared shortfall* in food consumption. This approach prioritizes more vulnerable groups and accounts for potential price volatility.

The project outputs a detailed, optimized allocation of subsidies and interactive visualizations showing the allocation and its impact on reducing food consumption shortfalls.

### Key Features

*   **Differentiated Thresholds:** Calculates separate food security thresholds for rural and urban populations.
*   **Targeted Allocation:** Uses `SciPy`'s optimization library to allocate subsidies based on need.
*   **Vulnerability Weighting:** Prioritizes lower-income fractiles by applying higher weights in the objective function.
*   **Stochastic Simulation:** Models price uncertainty to calculate the *expected* shortfall, making the allocation robust to market volatility.
*   **Rich Visualization:** Generates interactive plots with `Plotly` to clearly communicate the results.

### Built With

*   Python
*   NumPy
*   Pandas
*   SciPy
*   Plotly

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

You need Python 3 and pip installed on your system.

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/your_username/food_subsidy_allocation.git
    ```
2.  Navigate to the project directory:
    ```sh
    cd food_subsidy_allocation
    ```
3.  Install the required Python packages. It is recommended to use a virtual environment.
    ```sh
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    # Install packages
    pip install numpy pandas scipy plotly
    ```

## Usage

The model is run in two main steps.

### Step 1: Calculate Food Adequacy Thresholds

First, run the `calculate_thresholds.py` script to determine the monetary cost of a basic food basket.

> **IMPORTANT**: The prices and quantities in this script are **hypothetical**. You must replace them with accurate, validated data for your specific region and reference period.

```sh
python calculate_thresholds.py
```

This will print a detailed cost breakdown and the final rural and urban thresholds to the console.

```
--- Detailed Cost Breakdown (per person per month) ---
             Item   Unit  Quantity (Rural)  Quantity (Urban) ...
             Rice     kg              11.29             10.94 ...
...

FINAL CALCULATED THRESHOLDS:
  Rural Food Adequacy Threshold: Rs. 1891.00 per person per month
  Urban Food Adequacy Threshold: Rs. 2078.00 per person per month
```

### Step 2: Run the Subsidy Optimization

Once you have your thresholds, manually update them in `final subsidy vs mpce.py`. Then, run the script to perform the optimization and visualize the allocation.

```python
# In final subsidy vs mpce.py, update these values with the output from Step 1
RURAL_THRESHOLD = 1891
URBAN_THRESHOLD = 2078
```

Execute the script:
```sh
python "final subsidy vs mpce.py"
python "final befor vs after.py"
```

The script will run the optimization and automatically open two interactive HTML plots in your browser:

1.  **Subsidy Allocation:** Shows the percentage of the total budget allocated to each population fractile.
2.  **Shortfall Reduction:** Compares the expected food consumption shortfall before and after the optimized subsidy is applied.

A detailed table of the final allocation will also be printed to the console.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Rajneesh Avadhesh Yadav - rajy7020110306@gmail.com

Project Link: https://github.com/rajneesh-yadav-047/food_subsidy_allocation
