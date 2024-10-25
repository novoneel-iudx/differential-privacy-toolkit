from typing import List, Union, Dict, Optional
import pandas as pd
import numpy as np

def clip(series: pd.Series, min_value: float, max_value: float) -> pd.Series:
    return series.clip(lower=min_value, upper=max_value)

def categorise(series: pd.Series, bins: Union[int, List[float]], labels: Optional[List[str]] = None) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels)

def hash_values(series: pd.Series, salt: str = '') -> pd.Series:
    return series.apply(lambda x: hash(str(x) + salt) if pd.notnull(x) else x)

def suppress(series: pd.Series, threshold: int = 5, replacement: Optional[Union[str, int, float]] = None) -> pd.Series:
    value_counts = series.value_counts()
    values_to_suppress = value_counts[value_counts < threshold].index
    return series.replace(values_to_suppress, replacement)

def sanitise_data(
    df: pd.DataFrame,
    columns_to_sanitise: List[str],
    sanitisation_rules: Dict[str, Dict[str, Union[str, float, int, List, Dict]]],
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Sanitise the input DataFrame based on specified rules.

    Args:
    df (pd.DataFrame): Input DataFrame to sanitise.
    columns_to_sanitise (List[str]): List of column names to apply sanitisation.
    sanitisation_rules (Dict[str, Dict[str, Union[str, float, int, List, Dict]]]): Rules for sanitisation.
        Format: {column_name: {'method': <method_name>, 'params': {...}}}
    drop_na (bool): Whether to drop rows with NA values after sanitisation.

    Returns:
    pd.DataFrame: Sanitised DataFrame.

    Available sanitisation methods and their parameters:

    1. 'clip': Clip values to a specified range.
       Parameters:
       - 'min_value' (float): Minimum value to clip to.
       - 'max_value' (float): Maximum value to clip to.

    2. 'categorise': Bin continuous data into categories.
       Parameters:
       - 'bins' (int or list): Number of bins or list of bin edges.
       - 'labels' (list, optional): Labels for the bins.

    3. 'hash': Apply a hash function to the data.
       Parameters:
       - 'salt' (str, optional): Salt to add to the hash function. Default is ''.

    4. 'suppress': Replace rare values to protect privacy.
       Parameters:
       - 'threshold' (int, optional): Minimum frequency of a value to avoid suppression. Default is 5.
       - 'replacement' (any, optional): Value to use for suppressed entries. Default is None.
    """
    df_sanitised = df.copy()

    for column in columns_to_sanitise:
        if column not in df_sanitised.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        rule = sanitisation_rules.get(column)
        if not rule:
            raise ValueError(f"No sanitisation rule specified for column '{column}'")

        method = rule['method']
        params = rule.get('params', {})

        if method == 'clip':
            df_sanitised[column] = clip(df_sanitised[column], params['min_value'], params['max_value'])
        elif method == 'categorise':
            df_sanitised[column] = categorise(df_sanitised[column], params['bins'], params.get('labels'))
        elif method == 'hash':
            df_sanitised[column] = hash_values(df_sanitised[column], params.get('salt', ''))
        elif method == 'suppress':
            df_sanitised[column] = suppress(df_sanitised[column], params.get('threshold', 5), params.get('replacement'))
        else:
            raise ValueError(f"Unknown sanitisation method '{method}' for column '{column}'")

    if drop_na:
        df_sanitised = df_sanitised.dropna(subset=columns_to_sanitise)

    return df_sanitised

# Example usage:
# df = pd.DataFrame({
#     'age': [25, 40, 35, 60, 18, 70, 22, 45, 50, 55],
#     'income': [50000, 80000, 65000, 120000, 20000, 90000, 55000, 75000, 85000, 95000],
#     'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
#     'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
# })

# sanitisation_rules = {
#     'age': {'method': 'clip', 'params': {'min_value': 18, 'max_value': 80}},
#     'income': {'method': 'categorise', 'params': {'bins': 3, 'labels': ['low', 'medium', 'high']}},
#     'name': {'method': 'hash', 'params': {'salt': 'my_secret_salt'}},
#     'city': {'method': 'suppress', 'params': {'threshold': 2, 'replacement': 'Other'}}
# }

# sanitised_df = sanitise_data(df, ['age', 'income', 'name', 'city'], sanitisation_rules)
# print(sanitised_df)

