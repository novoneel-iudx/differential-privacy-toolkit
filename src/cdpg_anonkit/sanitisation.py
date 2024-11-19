from typing import List, Union, Dict, Optional
import pandas as pd
import numpy as np
import hashlib

class SanitiseData:
    def clip(series: pd.Series, min_value: float, max_value: float) -> pd.Series:
        """
        Clip (limit) the values in a Series to a specified range.

        Parameters
        ----------
        series : pd.Series
            The input Series to be clipped.
        min_value : float
            The minimum value to clip to.
        max_value : float
            The maximum value to clip to.

        Returns
        -------
        pd.Series
            The clipped Series.
        """
        return series.clip(lower=min_value, upper=max_value)

    # TODO: remove common salt for all values and combine with unique id column 
    def hash_values(series: pd.Series, salt: str = '') -> pd.Series:
        """
        Hash the values in a Series using the SHA-256 algorithm.

        This can be used to pseudonymise values that need to be kept secret.
        The salt parameter can be used to add a common salt to all values.
        This can be useful if you want to combine the hashed values with other columns
        to create a unique identifier.

        Parameters
        ----------
        series : pd.Series
            The input Series to be hashed.
        salt : str, optional
            The salt to add to all values before hashing.
            Defaults to an empty string.

        Returns
        -------
        pd.Series
            The hashed Series.
        """
        return series.apply(lambda x: hashlib.sha256((str(x) + salt).encode('utf-8')).hexdigest() if pd.notnull(x) else x)
        ## uncomment to use the inbuilt python hashing function
        # return series.apply(lambda x: hash(str(x) + salt) if pd.notnull(x) else x)

    def suppress(series: pd.Series, threshold: int = 5, replacement: Optional[Union[str, int, float]] = None) -> pd.Series:
        """
        Suppress all values in a Series that occur less than a given threshold.
        
        Replace all values that occur less than the threshold with the replacement value.
        
        Parameters
        ----------
        series : pd.Series
            The input Series to be suppressed.
        threshold : int, optional
            The minimum number of occurrences for a value to be kept.
            Defaults to 5.
        replacement : Optional[Union[str, int, float]], optional
            The value to replace suppressed values with.
            Defaults to None, which means that the values will be replaced with NaN.
        
        Returns
        -------
        pd.Series
            The Series with suppressed values.
        """
        
        value_counts = series.value_counts()
        values_to_suppress = value_counts[value_counts < threshold].index
        return series.replace(values_to_suppress, replacement)

    def sanitise_data(
        df: pd.DataFrame,
        columns_to_sanitise: List[str],
        sanitisation_rules: Dict[str, Dict[str, Union[str, float, int, List, Dict]]],
        drop_na: bool = False
    ) -> pd.DataFrame:

        """
        Sanitise a DataFrame by applying different methods to each column.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to be sanitised.
        columns_to_sanitise : List[str]
            The columns in the DataFrame to be sanitised.
        sanitisation_rules : Dict[str, Dict[str, Union[str, float, int, List, Dict]]]
            A dictionary that maps each column in columns_to_sanitise to a dictionary
            that specifies the sanitisation method and parameters for that column.
            The dictionary should contain the following keys:
            * 'method': str, the sanitisation method to use
            * 'params': Dict[str, Union[str, float, int, List, Dict]], the parameters
              for the sanitisation method
        drop_na : bool, optional
            If True, drop all rows in the DataFrame that have any NaN values in the
            columns specified in columns_to_sanitise. Defaults to False.

        Returns
        -------
        pd.DataFrame
            The sanitised DataFrame.
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
                df_sanitised[column] = SanitiseData.clip(df_sanitised[column], params['min_value'], params['max_value'])
            elif method == 'hash':
                df_sanitised[column] = SanitiseData.hash_values(df_sanitised[column], params.get('salt', ''))
            elif method == 'suppress':
                df_sanitised[column] = SanitiseData.suppress(df_sanitised[column], params.get('threshold', 5), params.get('replacement'))
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
#     'name': {'method': 'hash', 'params': {'salt': 'my_secret_salt'}},
#     'city': {'method': 'suppress', 'params': {'threshold': 2, 'replacement': 'Other'}}
# }

# sanitised_df = sanitise_data(df, ['age', 'income', 'name', 'city'], sanitisation_rules)
# print(sanitised_df)

