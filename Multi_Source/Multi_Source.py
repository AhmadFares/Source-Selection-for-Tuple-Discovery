import pandas as pd

from Single_Source.Coverage_Guided_Row_Selection import algo_main, compute_overall_coverage, compute_overall_penalty, optimize_selection
from helpers.Source_Constructors import SourceConstructor
from helpers.Source_Constructors import dataframe_to_ur_dict
from helpers.T_splitter_into_M import split_by_columns, split_by_diagonal, split_by_hybrid, split_by_keywords, split_by_overlapping_rows, split_by_rows
from helpers.test_cases import TestCases

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
    chosen_order = []  # To keep track of the order of selected sources
    while not terminate:
        terminate = True  
        

        M_i = get_next_M(sources, i)
        if M_i is None:
            print("No more valid sources left.")
            return T, i + 1, chosen_order   # Stop and return the last obtained table
        
        common_cols = [col for col in UR.columns if col in M_i.columns and col != "Identifiant"]
        chosen_order.append(i+1)
        if not common_cols:
            #print(f"Skipping M_{i} as it has no common columns with UR.")
            i += 1
            terminate=False
            continue
            
        # Apply the coverage-guided selection (blackbox function)
        new_T = algo_main(M_i, UR, theta)
        
        if T.empty:
            T = new_T
        elif new_T.empty:
            print(f"Skipping M_{i} as it produced an empty table.")
        else:
            print(f"Columns in new_T: {new_T.columns.tolist()}")
            T = T.set_index("Identifiant").combine_first(new_T.set_index("Identifiant")).reset_index()
        
        # Compute current coverage
        final_cov, _ = compute_overall_coverage(T, UR)
        
        if final_cov >= theta:
            print("M" , i, "was the last source needed", "Coverage: ", final_cov)
            return T, i + 1   # Stop if coverage requirement is met
        else:
            print("M" , i, "was not enough", "Coverage: ", final_cov)
            print(T)
            i += 1
            terminate = False  # Continue to the next source

    return T, i + 1 , chosen_order  # Return the last obtained table, even if coverage is not met


# --- Main Function ---
def main():
    """ Main function to split T and run the multi-source algorithm. """
    # Load the test case
    test_cases = TestCases()
    T_input, UR = test_cases.get_case(20)  # Load predefined test case 1
    #UR = dataframe_to_ur_dict(UR)
    theta = 1  # Example coverage threshold
    constructor = SourceConstructor(T_input, UR)
    #sources = constructor.low_coverage_sources()
    #sources = constructor.group_by_sources()
    #sources = constructor.low_penalty_sources()
    sources = constructor.high_penalty_sources()
    
    #print(f"✅ Created {len(sources)} high-penalty sources.")

    # print(f"Generated {len(sources)} sources")
    # for i, src in enumerate(sources[:5]):
    #     print(f"Source {i} (first few rows):")
    #     print(src.head())

    # --- Choose one split method ---
    #Same schema:
    #sources = split_by_diagonal(T_input) 
    #sources = split_by_rows(T_input) 
    #sources = split_by_overlapping_rows(T_input, overlap_size=5) 

    #Different schema:
    #sources = split_by_columns(T_input)  
    #sources = split_by_hybrid(T_input)  
    #sources = split_by_keywords(T_input)
    
    
    T_output, _ = multi_source_algorithm(sources, UR, theta)
    T_output, _ = optimize_selection(T_output, UR)

    # Compute final coverage
    final_cov, _ = compute_overall_coverage(T_output, UR)
    final_pen, _ = compute_overall_penalty(T_output, UR)

    # Display results
    print(f"Final Coverage: {final_cov:.4f}")
    print(f"Final Penalty: {final_pen:.4f}")
    print(f"Final Table:\n{T_output}\n")

if __name__ == "__main__":
    main()