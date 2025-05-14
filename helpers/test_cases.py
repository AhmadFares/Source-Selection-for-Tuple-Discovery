import pandas as pd
import sqlite3
import numpy as np


class TestCases:
    """
    This class contains test cases for the Coverage-Guided Row Selection algorithm.
    Each test case is defined as a tuple (T, UR) where:
      - T is the initial table (a pandas DataFrame).
      - UR is the User Request table (a pandas DataFrame) that specifies the required values for each column.
    """

    def __init__(self):
        self.cases = {}  # Dictionary to store test cases
        self.load_lisa_sheets()
        self.load_fixed_mathe_case()  # Load MATHE case

    def load_lisa_sheets(self):
        """
        Load the Lisa_Sheets table from the SQLite database and store it as T.
        """
        db_path = "data/Lisa_Tabular_Data.db"  # Database path
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM Lisa_Sheets;"
        T = pd.read_sql_query(query, conn)
        conn.close()

        # 🔹 Define User Requests (UR):
        """
        UR1: Answer in the Beggining of the table
        UR2: Answer in the End of Table
        UR3: Answer Distributed in the Table
        UR4: No Answers
        UR5: Partial Answers
        UR6: Multiple Answers Applicable -> To test importance of Penalty_Optimization
        UR7: Intermediate Answers -> To test importance of Optimize_Selection
        """
        user_requests = {
    1: {
        "Keyword1": ["venous approaches", "removal venous", "gestational hypertension", "pre eclampsia", "pregnancy methods"],
        "Keyword2": ["peripheral venous", "pregnancy hypertension", "haemorrhage", "lupus"]
    },
    2: {
        "Keyword1": ["mri lumbar", "sacroiliac tests", "spinal causes"],
        "Keyword2": ["spine mri", "spondylodiscitis pott", "severe undernutrition", "pain spinal"]
    },
    3: {
        "Keyword1": ["venous approaches", "sacroiliac tests", "pre eclampsia", "mri lumbar", "tumour stomach", "splenomegaly enlarged", "preventive cerclage", "rachis cervical"],
        "Keyword2": ["hyperplasia parathyroid", "oedematous syndrome", "schizophrenia following"]
    },
    4: {
        "Keyword1": ["aaaaa", "bbb", "cccc"],
        "Keyword2": ["dddd", "eeee", "ffff"]
    },
    5: {
        "Keyword1": ["venous approaches", "aaaaaa", "removal venous"],
        "Keyword2": ["bbbbbbb", "oedematous syndrome", "hyperplasia parathyroid"]
    },
    6: {
       #ID-FARES-Test||||||||||venous approaches|approach venous||
       "Keyword1": ["venous approaches"],
       "Keyword2": ["approach venous"]
    },
    7: {
       "Keyword1": ["cerebral mri", "limb trauma", "trendelebourg lameness", "complications pregnancy"],
       "Keyword2": ["stroke mri", "saluting trendelebourg", "maternal complications", "complications nerve"]
    }
}

        # 🔹 Convert all User Requests to properly formatted DataFrames
        for case_number, ur_data in user_requests.items():
            self.cases[case_number] = (T, self.create_flexible_dataframe(ur_data))

        # 🔹 Add additional test cases (without Lisa_Sheets)
        self.cases[10] = self.create_penalty_opt_case()
        self.cases[11] = self.create_optimized_selection_case()

    def create_flexible_dataframe(self, data_dict):
        """
        Convert a dictionary to a pandas DataFrame, handling columns with different lengths.
        Uses pd.Series to ensure misaligned columns are handled correctly.
        """
        return pd.DataFrame.from_dict({key: pd.Series(value, dtype=object) for key, value in data_dict.items()})

    def create_penalty_opt_case(self):
        """Returns a predefined penalty optimization test case."""
        T10 = pd.DataFrame({
            "A": ["v1", "v2", "x3", "x4", "v1", "v2"],
            "B": ["x1", "x2", "v3", "v4", "v3", "v4"]
        })
        UR10 = pd.DataFrame({
            "A": ["v1", "v2"],
            "B": ["v3", "v4"]
        })
        return T10, UR10

    def create_optimized_selection_case(self):
        """Returns a predefined optimized selection test case."""
        T11 = pd.DataFrame({
            "A": ["v1", "v2", "v1", "x3"],
            "B": ["x1", "x2", "v3", "v4"]
        })
        UR11 = pd.DataFrame({
            "A": ["v1", "v2"],
            "B": ["v3", "v4"]
        })
        return T11, UR11
    def build_random_ur(df, columns, k):
        ur = {}
        for col in columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0:
                selected = np.random.choice(unique_vals, size=min(k, len(unique_vals)), replace=False)
                ur[col] = list(selected)
        return ur
        
    def load_mathe_case(self, csv_path="./data/MATHE/output_table.csv", n_values_per_col=4):
        """
        Load MATHE and generate a random UR from the last 3 columns (for ad-hoc testing).
        """
        mathe_df = pd.read_csv(csv_path, delimiter=";")
        mathe_df.rename(columns={"id_assessment": "Identifiant"}, inplace=True)

        ur_columns = mathe_df.columns[-3:]
        ur_dict = self.build_random_ur(mathe_df, ur_columns, n_values_per_col)
        UR = self.create_flexible_dataframe(ur_dict)

        base_cols = list(ur_columns) + ["Identifiant"]
        T = mathe_df[base_cols].copy()

        self.cases[19] = (T, UR)

    def load_fixed_mathe_case(self, csv_path="./data/MATHE/output_table.csv"):
        """
        Load MATHE and use a fixed User Request (UR) across 3 columns.
        """
        mathe_df = pd.read_csv(csv_path, delimiter=";")
        mathe_df.rename(columns={"id_assessment": "Identifiant"}, inplace=True)

        fixed_ur_data = {
            "keyword_name": ["Two variables", "Orthogonality", "Three points rule", "Mean"],
            "topic_name": ["Linear Algebra", "Probability", "Optimization", "Discrete Mathematics"],
            "subtopic_name": [
                "Linear Transformations",
                "Vector Spaces",
                "Algebraic expressions, Equations, and Inequalities",
                "Triple Integration"
            ]
        }

        UR = self.create_flexible_dataframe(fixed_ur_data)
        base_cols = list(fixed_ur_data.keys()) + ["Identifiant"]
        T = mathe_df[base_cols].copy()

        self.cases[20] = (T, UR)

    def get_case(self, case_number=1):
        """
        Return the tuple (T, UR) for the specified case number.
        Defaults to case 1 if the given case is not found.
        """
        return self.cases.get(case_number, self.cases[1])



