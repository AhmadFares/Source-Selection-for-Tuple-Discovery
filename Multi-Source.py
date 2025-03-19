import pandas as pd
from Coverage_Guided_Row_Selection import algo_main, compute_overall_coverage
from T_splitter_into_M import split_by_columns, split_by_diagonal, split_by_hybrid, split_by_keywords, split_by_rows
from test_cases import TestCases

def get_next_M(sources, i):
    """ Get the next source M_i from the set of sources M.
    """
    if i < len(sources):
        return sources[i]
    return None  

def multi_source_algorithm(sources, UR, theta):
    """
    Implements the Multi-Source Algorithm to iteratively apply 
    Coverage-Guided Row Selection until the required coverage is reached.

    Returns:
        DataFrame: The completed table T.
    """
    T = pd.DataFrame()  # Start with an empty table
    i = 0
    terminate = False

    while not terminate:
        terminate = True  
        M_i = get_next_M(sources, i)
        if M_i is None:
            print("No more valid sources left.")
            return T  # Stop and return the last obtained table
        
        common_cols = [col for col in UR.columns if col in M_i.columns and col != "Identifiant"]
        
        if not common_cols:
            print(f"Skipping M_{i} as it has no common columns with UR.")
            i += 1
            continue
            
        # Apply the coverage-guided selection (blackbox function)
        new_T = algo_main(M_i, UR, theta)
        
        if T.empty:
            T = new_T
        else:
            T = T.merge(new_T, on="Identifiant", how="outer")
        
        # Compute current coverage
        final_cov, _ = compute_overall_coverage(T, UR)
        
        if final_cov >= theta:
            print("M" , i, "was the last source needed", "Coverage: ", final_cov)
            return T  # Stop if coverage requirement is met
        else:
            print("M" , i, "was not enough", "Coverage: ", final_cov)
            print(T)
            i += 1
            terminate = False  # Continue to the next source

    return T  # Return the last obtained table, even if coverage is not met


# --- Main Function ---
def main():
    """ Main function to split T and run the multi-source algorithm. """
    # Load the test case
    test_cases = TestCases()
    T_input, UR = test_cases.get_case(6)  # Load predefined test case 1

    theta = 1  # Example coverage threshold

    # --- Choose one split method ---
    #sources = split_by_rows(T_input)  

    #sources = split_by_columns(T_input)  
    #sources = split_by_hybrid(T_input)  
    #ssources = split_by_diagonal(T_input) 
    sources = split_by_keywords(T_input)

    T_output = multi_source_algorithm(sources, UR, theta)

    # Compute final coverage
    final_cov, _ = compute_overall_coverage(T_output, UR)

    # Display results
    print(f"Final Coverage: {final_cov:.4f}")
    print(f"Final Table:\n{T_output}\n")

if __name__ == "__main__":
    main()