import pytest
import pandas as pd

from src.cdpg_anonkit.generalisation import GeneraliseData

categorical_generaliser = GeneraliseData().CategoricalGeneraliser()

def test_categorical_generaliser_with_numeric_bins():
    """Test generalisation with a numeric value for bins"""
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = categorical_generaliser.generalise_categorical(data, bins=3)
    
    # Check if we get the expected number of categories
    assert len(result.unique()) == 3
    
    # Check if the values are properly distributed
    assert result.iloc[0] == result.iloc[1]  # 1 and 2 should be in the same bin
    assert result.iloc[8] == result.iloc[9]  # 9 and 10 should be in the same bin

def test_categorical_generaliser_with_custom_bins():
    """Test generalisation with custom bin edges"""
    data = pd.Series([1, 5, 10, 15, 20])
    bins = [0, 5, 10, 20]
    result = categorical_generaliser.generalise_categorical(data, bins=bins)
    
    # Check if binning is correct
    assert len(result.unique()) == 3
    assert result.iloc[0] == result.iloc[1]  # 1 and 5 should be in the same bin
    assert result.iloc[2] == result.iloc[2]  # 10 should be in its own bin
    assert result.iloc[3] == result.iloc[4]  # 15 and 20 should be in the same bin

def test_categorical_generaliser_with_custom_labels():
    """Test generalisation with custom labels"""
    data = pd.Series([1, 5, 10, 15, 20])
    bins = [0, 5, 10, 20]
    labels = ['Low', 'Medium', 'High']
    result = categorical_generaliser.generalise_categorical(data, bins=bins, labels=labels)
    
    # Check if labels are applied correctly
    assert set(result.unique()) == set(['Low', 'Medium', 'High'])
    assert result.iloc[0] == 'Low'  # 1 should be in 'Low'
    assert result.iloc[2] == 'Medium'  # 10 should be in 'Medium'
    assert result.iloc[4] == 'High'  # 20 should be in 'High'

def test_categorical_generaliser_with_invalid_bins():
    """Test handling of invalid bins input"""
    data = pd.Series([1, 2, 3, 4, 5])
    
    # Test with negative number of bins
    with pytest.raises(ValueError):
        categorical_generaliser.generalise_categorical(data, bins=-1)
    
def test_categorical_generaliser_with_mismatched_labels():
    """Test handling of mismatched number of labels and bins"""
    data = pd.Series([1, 2, 3, 4, 5])
    bins = [0, 2, 4, 5]
    labels = ['Low', 'Medium']  # One label missing
    
    with pytest.raises(ValueError):
        categorical_generaliser.generalise_categorical(data, bins=bins, labels=labels)

def test_categorical_generaliser_empty_series():
    """Test handling of empty input series"""
    data = pd.Series([])
    with pytest.raises(ValueError):
        result = categorical_generaliser.generalise_categorical(data, bins=3)
    
    # assert len(result) == 0
    # assert isinstance(result, pd.Series)

def test_categorical_generaliser_single_value():
    """Test handling of single value input"""
    data = pd.Series([1])
    result = categorical_generaliser.generalise_categorical(data, bins=3)
    
    assert len(result) == 1
    assert not result.iloc[0] is pd.NA