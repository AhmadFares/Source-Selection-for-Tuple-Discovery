import sys
import pandas as pd
import time
from test_cases import TestCases
import cProfile
import pstats

# --- Coverage Functions ---
def compute_attr_coverage(T, UR, col):
    """
    Compute the coverage for a single attribute (column).
    Coverage = |values in T ∩ values in UR| / |values in UR|
    """
    t_values = set(T[col].dropna())
    ur_values = set(UR[col].dropna())
    
    if not ur_values:  # if UR is empty for this column, define coverage as 1.
        return 1
    return len(t_values.intersection(ur_values)) / len(ur_values)

def compute_overall_coverage(T, UR):
    """
    Compute the overall coverage as the average of the per-attribute coverages.
    Returns overall coverage and the list of per-attribute coverages.
    """
    coverages = []
    for col in UR.columns:
         if col == "Identifiant":
          continue
         cov = compute_attr_coverage(T, UR, col)
         coverages.append(cov)
    overall_cov = sum(coverages) / len(coverages) if coverages else 1
    return overall_cov, coverages

# --- Penalty Functions ---
def compute_attr_penalty(T, UR, col):
    """
    Compute the penalty for a single attribute (column).

    Penalty = |values in T that are NOT in UR| / |values in T|
    """
    t_values = set(T[col].dropna())
    if not t_values:
        return 0  # If T has no non-null values, penalty is 0.
    
    if col in UR.columns:
        ur_values = set(UR[col].dropna())
    else:
        ur_values = set()
        
        
    
    
    extra = t_values - ur_values
    return len(extra) / len(t_values)

def compute_overall_penalty(T, UR):
    """
    Compute the overall penalty as the average penalty across all columns in T.
    """
    penalties = []
    for col in T.columns:
        if col == "Identifiant":
          continue
        p = compute_attr_penalty(T, UR, col)
        penalties.append(p)
    overall_penalty = sum(penalties) / len(penalties) if penalties else 0
    return overall_penalty, penalties


# def optimize_selection(T, UR):
#     """
#     Post-optimization: Remove redundant rows from T while maintaining overall coverage.
#     For each row in T, we remove it temporarily and recompute the overall coverage.
#     If the coverage remains the same, we update T without that row.
#     """
#     orig_cov, _ = compute_overall_coverage(T, UR)
#     changed = True
#     while changed:
#         changed = False
#         # Iterate over a copy of T's index because we might modify T in the loop.
#         for idx in T.index.tolist():
#             T_sub = T.drop(index=idx).reset_index(drop=True)
#             sub_cov, _ = compute_overall_coverage(T_sub, UR)
#             # If removal doesn't change the overall coverage, accept the removal.
#             if sub_cov == orig_cov:
#                 T = T_sub.copy()
#                 changed = True
#                 # Restart the loop since T has changed.
#                 break
#     return T

# --- Row Selection Algorithm ---
def coverage_guided_row_selection(input_table, UR, theta):
    """
    Implements Coverage-Guided Row Selection with Penalty Optimization.
    
    Parameters:
        input_table: DataFrame representing the input table (𝒯).
        UR: DataFrame representing the User Request Table.
        theta: Coverage threshold.
    
    Returns:
        A DataFrame with the selected rows.
    """    
    # Ensure "identifiant" is retained in input_table
    print(input_table.columns)
    if "Identifiant" in input_table.columns:
        input_table = input_table[["Identifiant"] + UR.columns.tolist()]
    else:
        input_table = input_table[UR.columns]

    # Store selected rows in a list instead of using pd.concat
    selected_rows = []
    curr_coverage = 0
    curr_penalty = 0
    count = 0
    count_if = 0

    # Precompute unique values from UR to speed up coverage calculations
    UR_values = {col: set(UR[col].dropna().values) for col in UR.columns if col != "Identifiant"}

    for row in input_table.itertuples(index=False, name=None):
        # Convert tuple to dict for fast lookups
        row_dict = dict(zip(input_table.columns, row))

        # Simulate adding the row to selected_rows
        T_curr_values = {col: set(row_dict[col] for row_dict in selected_rows) for col in UR_values}
        T_curr_values = {col: vals | {row_dict[col]} for col, vals in T_curr_values.items()}  # Add new row values

        # Compute overall coverage using precomputed UR values
        coverages = [len(T_curr_values[col] & UR_values[col]) / len(UR_values[col]) if UR_values[col] else 1 for col in T_curr_values]
        cov = sum(coverages) / len(coverages) if coverages else 1

        # If the new table increases coverage (but still below theta), update selection
        if cov <= theta and cov > curr_coverage:
            count_if += 1
            selected_rows.append(row_dict)
            curr_coverage = cov
            curr_penalty, _ = compute_overall_penalty(pd.DataFrame(selected_rows), UR)
        else:
            if curr_coverage >= theta and curr_penalty!= 0:
                count += 1
                # Try replacing a row in selected_rows
                for idx in range(len(selected_rows)):
                    temp_rows = selected_rows[:idx] + selected_rows[idx + 1:]  # Remove one row
                    temp_rows.append(row_dict)  # Add new row

                    # Compute new coverage
                    T_sub_values = {col: set(temp_row[col] for temp_row in temp_rows) for col in UR_values}
                    sub_coverages = [len(T_sub_values[col] & UR_values[col]) / len(UR_values[col]) if UR_values[col] else 1 for col in T_sub_values]
                    sub_cov = sum(sub_coverages) / len(sub_coverages) if sub_coverages else 1

                    # Compute new penalty
                    sub_penalty, _ = compute_overall_penalty(pd.DataFrame(temp_rows), UR)

                    if sub_cov >= theta and curr_penalty > sub_penalty:
                        selected_rows = temp_rows.copy()
                        curr_coverage = sub_cov
                        curr_penalty = sub_penalty

    # Convert selected rows list to DataFrame once (avoids repeated pd.concat)
    T = pd.DataFrame(selected_rows)

    # Restore "Identifiant" efficiently
    if "Identifiant" in input_table.columns:
        T["Identifiant"] = input_table["Identifiant"].values[: len(T)]

    return T, count, count_if



if __name__ == '__main__':
    case_number = 1
    if len(sys.argv) > 1:
        try:
            case_number = int(sys.argv[1])
        except ValueError:
            print("Invalid case number provided. Defaulting to case 1.")
    
    test_cases = TestCases()
    T, UR = test_cases.get_case(case_number)
    
    print(f"Running test case {case_number}:\nT =\n{T}\n\nUR =\n{UR}\n")
    
    # Set a coverage threshold value
    theta = 1
    
    # Call the new algorithm function with the test case tables and threshold.
    # with cProfile.Profile() as pr:
    #     T_output, count, count_if = coverage_guided_row_selection(T, UR, theta)

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)  # Sort by time taken
    # stats.print_stats(20)
    start_time = time.time()
    T_output, count , count_if = coverage_guided_row_selection(T, UR, theta)
    # print('count:' , count)
    # print('count_if' , count_if)
    time_with_optimization = time.time() - start_time
    final_cov, _ = compute_overall_coverage(T_output, UR)
    final_penalty, attr = compute_overall_penalty(T_output, UR)
    print('time' , time_with_optimization)
    print("\nFinal output table T_output:")
    print(T_output)
    print('Overall Coverage: ', final_cov)
    print('Overall Penalty: ', final_penalty)
    print('Overall Penalty: ', attr)
    