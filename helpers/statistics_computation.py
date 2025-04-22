from collections import Counter, defaultdict

def compute_column_statistics(df):
    """
    Compute per-column normalized value frequency histograms.
    
    Returns:
        dict: {column_name: {value: normalized frequency}}
    """
    stats = {}
    for col in df.columns:
        if col == "Identifiant":
            continue  # Skip Identifiant

        value_counts = df[col].dropna().value_counts(normalize=True)
        stats[col] = value_counts.to_dict()
    return stats

def compute_all_source_statistics(sources_list):
    """
    Compute statistics for a list of sources.

    Returns:
        dict: {source_idx: column_statistics_dict}
    """
    all_stats = {}
    for i, source_df in enumerate(sources_list):
        all_stats[i] = compute_column_statistics(source_df)
    return all_stats
