import pandas as pd
import time
from Coverage_Guided_Row_Selection import (
    coverage_guided_row_selection,
    compute_overall_coverage,
    compute_overall_penalty,
    penalty_optimization,
    optimize_selection,
)
from test_cases import TestCases


class BenchmarkTestCases:
    """
    Runs benchmarks on Coverage-Guided Row Selection across multiple test cases.
    It evaluates three algorithm variants and logs execution time, penalty, and coverage.
    """

    def __init__(self, test_cases, thetas):
        """
        Initializes the benchmarking process.
        
        Parameters:
        - test_cases: Instance of TestCases class containing (T, UR) pairs.
        - thetas: List of theta values to test.
        """
        self.test_cases = test_cases
        self.thetas = thetas
        self.results = []

    def run(self):
        """
        Executes benchmarks on test cases from 1 to 7 using three algorithm variants.
        """
        for case_number in range(1, 2):  # Include all test cases
            T_input, UR = self.test_cases.get_case(case_number)

            for theta in self.thetas:
                ### 🔹 VARIANT 1: One-pass (No optimization)
                start_time = time.time()
                T_output, i = coverage_guided_row_selection(T_input, UR, theta)
                time_taken = time.time() - start_time
                pen, _ = compute_overall_penalty(T_output, UR)
                cov, _ = compute_overall_coverage(T_output, UR)
                length = len(T_output)

                self.results.append(["UR" + str(case_number), "One_pass_variant (No opt.)", time_taken, pen, cov, theta, i, length])

                ### 🔹 VARIANT 2: Optimized with Penalty Optimization
                start_time = time.time()
                T_output, i = coverage_guided_row_selection(T_input, UR, theta)
                T_output, count = penalty_optimization(T_output, T_input, UR, i, theta)
                time_taken = time.time() - start_time
                pen, _ = compute_overall_penalty(T_output, UR)
                cov, _ = compute_overall_coverage(T_output, UR)
                length = len(T_output)

                self.results.append(["UR" + str(case_number), "Optimized_pass_Pen opt", time_taken, pen, cov, theta,count, length])

                ### 🔹 VARIANT 3: Optimized with Full Selection
                start_time = time.time()
                T_output, i = coverage_guided_row_selection(T_input, UR, theta)
                T_output, count = penalty_optimization(T_output, T_input, UR, i, theta)
                T_output, optcount = optimize_selection(T_output, UR)
                time_taken = time.time() - start_time
                pen, _ = compute_overall_penalty(T_output, UR)
                cov, _ = compute_overall_coverage(T_output, UR)
                length = len(T_output)

                self.results.append(["UR" + str(case_number), "Optimized_pass_overall", time_taken, pen, cov, theta, optcount,length])

    def get_results(self):
        """
        Returns the results as a pandas DataFrame.
        """
        return pd.DataFrame(
            self.results,
            columns=["Test Case", "Variant", "Time (s)", "Penalty", "Coverage", "Theta", "Count", "T length"],
        )

    def display_results(self):
        """
        Prints the results in a readable format.
        """
        df_results = self.get_results()
        print(df_results)

    def save_results(self, filename="benchmark_results.csv"):
        """
        Saves the results to a CSV file.
        """
        df_results = self.get_results()
        df_results.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    # Define theta values for testing
    theta_values = [1.0]

    # Initialize test cases
    test_cases = TestCases()

    # Run benchmark
    benchmark = BenchmarkTestCases(test_cases, theta_values)
    benchmark.run()

    # Display results
    benchmark.display_results()

    # Save results to a CSV file
    benchmark.save_results("benchmark_results.csv")
