import pandas as pd
from helpers.test_cases import TestCases
from Single_Source.Coverage_Guided_Row_Selection import algo_main, compute_overall_coverage, compute_overall_penalty, optimize_selection
from Multi_Source.Multi_Source import multi_source_algorithm 
from helpers.T_splitter_into_M import split_by_rows, split_by_diagonal, split_by_overlapping_rows 

def run_case(label, T_input, UR, use_split=False, splitter_fn=None):
    """Run a benchmark test and return coverage, penalty, and number of sources"""
    theta = 1

    if not use_split:
        T_output = algo_main(T_input, UR, theta)
        num_sources = 1  # Only one source used (full table)
    else:
        sources = splitter_fn(T_input)
        T_output, num_sources = multi_source_algorithm(sources, UR, theta)
        T_output, _ = optimize_selection(T_output, UR)

    cov, _ = compute_overall_coverage(T_output, UR)
    pen, _ = compute_overall_penalty(T_output, UR)

    return {
        "Case": label,
        "Coverage": round(cov, 4),
        "Penalty": round(pen, 4),
        "Sources Investigated": num_sources
    }


def main():
    test_cases = TestCases()
    results = []

    for case_number in range(1, 8):  # Loop from 1 to 7
        T_input, UR = test_cases.get_case(case_number)

        # --- Case X: No splitting
        results.append(run_case(f"Case {case_number} - No Split", T_input, UR))

        # --- Case X: With different split strategies
        results.append(run_case(f"Case {case_number} - Split by Rows", T_input, UR, use_split=True, splitter_fn=split_by_rows))
        results.append(run_case(f"Case {case_number} - Split by Diagonal", T_input, UR, use_split=True, splitter_fn=split_by_diagonal))
        results.append(run_case(f"Case {case_number} - Split by Overlapping Rows", T_input, UR, use_split=True,
                                splitter_fn=lambda df: split_by_overlapping_rows(df, overlap_size=5)))

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("benchmark_results.csv", index=False)


if __name__ == "__main__":
    main()
