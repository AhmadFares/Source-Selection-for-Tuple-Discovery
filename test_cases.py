import pandas as pd
import sqlite3


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

    def load_lisa_sheets(self):
        """
        Load the Lisa_Sheets table from the SQLite database and store it as T.
        """
        db_path = "Lisa_Tabular_Data.db"  # Database path
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

    def get_case(self, case_number=1):
        """
        Return the tuple (T, UR) for the specified case number.
        Defaults to case 1 if the given case is not found.
        """
        return self.cases.get(case_number, self.cases[1])



