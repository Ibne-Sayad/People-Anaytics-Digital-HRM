import pandas as pd
from pulp import *
from io import StringIO

def optimize_teller_staffing(csv_file_path):
    # Load the CSV data
    data = pd.read_csv(csv_file_path, index_col=0)

    # Service level (customers per hour per teller)
    service_rate = 8

    # Calculate the number of tellers needed for each hour
    data['Tellers_Required'] = (data['Avg_Customer_Number'] / service_rate).apply(lambda x: -(-x // 1))  # Ceiling division

    # Decision variables: number of tellers for each shift
    tellers_shift1 = LpVariable("Tellers_Shift1", lowBound=0, cat='Integer')
    tellers_shift2 = LpVariable("Tellers_Shift2", lowBound=0, cat='Integer')

    # Create the optimization problem
    staffing_problem = LpProblem("Bank_Teller_Staffing_Optimization", LpMinimize)

    # Objective function: Minimize the total number of tellers
    staffing_problem += tellers_shift1 + tellers_shift2, "Total Tellers"

    # Constraints: The number of tellers must be sufficient for each hour
    for idx, hour in data.iterrows():
        if hour['Shift 1'] == 'X':
            staffing_problem += tellers_shift1 >= hour['Tellers_Required'], f"Shift1_{idx}"
        if hour['Shift 2'] == 'X':
            staffing_problem += tellers_shift2 >= hour['Tellers_Required'], f"Shift2_{idx}"

    # Solve the problem
    staffing_problem.solve()

    # Print the results
    print("Status:", LpStatus[staffing_problem.status])
    print("Objective value:", value(staffing_problem.objective))

    # Number of tellers needed for each shift
    print(f"Shift 1: Tellers Needed = {int(tellers_shift1.value())}")
    print(f"Shift 2: Tellers Needed = {int(tellers_shift2.value())}")

    # Print detailed solver information
    print(f"Total time (CPU seconds): {staffing_problem.solutionCpuTime:.2f} (Wallclock seconds): {staffing_problem.solutionCpuTime:.2f}")

if __name__ == "__main__":
    csv_file_path = "data/fau_bank_shifts.csv"
    optimize_teller_staffing(csv_file_path)
