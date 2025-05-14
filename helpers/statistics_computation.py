import numpy as np


def compute_UR_value_frequencies_in_sources(sources_list, UR_df):
    """
    UR_df: pandas DataFrame where columns = attributes, and values are desired values
    """
    # Build value_index from DataFrame
    value_index = {}
    idx = 0
    for col in UR_df.columns:
        for val in UR_df[col].dropna().unique():
            value_index[(col, val)] = idx
            idx += 1

    vector_length = len(value_index)
    source_vectors = {}

    for src_idx, df in enumerate(sources_list):
        vector = np.zeros(vector_length, dtype=np.float32)
        n_rows = len(df)
        if n_rows == 0:
            source_vectors[src_idx] = vector
            continue

        for (col, val), i in value_index.items():
            if col in df.columns:
                count = (df[col] == val).sum()
                vector[i] = count / n_rows

        source_vectors[src_idx] = vector

    return value_index, source_vectors
