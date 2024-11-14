import pytest
import pandas as pd

from generalisation import GeneraliseData

temporal_generaliser = GeneraliseData().TemporalGeneraliser()

@pytest.fixture
def sample_timestamp_df():
    """Create a sample Series of timestamps in different formats for testing."""
    return pd.DataFrame({
        'timestamp': [
        "2023-11-14T10:01:56Z",
        "2023-11-14T12:34:56Z",
        "2023-11-14T16:59:56Z",
        "2023-11-14T12:30:56Z"]
        })

@pytest.fixture
def sample_test_resolution_handling():
    """Create a sample DataFrame for testing."""
    return pd.Series([
        "2024-01-01 10:30:00",
        "2024-01-01 11:29:00",
        "2024-01-01 08:59:00",
        "2024-01-01 06:01:00"      
        ])

def test_generalise_temporal_series_15(sample_test_resolution_handling):
    """Test temporal generalisation with Series input"""
    result = temporal_generaliser.generalise_temporal(data=sample_test_resolution_handling, temporal_resolution=15)
    
    assert isinstance(result, pd.Series)
    assert result.name == 'timeslot'
    assert len(result) == len(sample_test_resolution_handling)
    assert result.iloc[0] == '10_30'
    assert result.iloc[1] == '11_15'
    assert result.iloc[2] == '8_45'
    assert result.iloc[3] == '6_0'
    assert all(int(slot.split('_')[1]) % 15 == 0 for slot in result)

def test_generalise_temporal_series_30(sample_test_resolution_handling):
    """Test temporal generalisation with Series input"""
    result = temporal_generaliser.generalise_temporal(data=sample_test_resolution_handling, temporal_resolution=30)
    
    assert isinstance(result, pd.Series)
    assert result.name == 'timeslot'
    assert len(result) == len(sample_test_resolution_handling)
    assert result.iloc[0] == '10_30'
    assert result.iloc[1] == '11_0'
    assert result.iloc[2] == '8_30'
    assert result.iloc[3] == '6_0'
    assert all(int(slot.split('_')[1]) % 30 == 0 for slot in result)

def test_generalise_temporal_series_60(sample_test_resolution_handling):
    """Test temporal generalisation with Series input"""
    result = temporal_generaliser.generalise_temporal(data=sample_test_resolution_handling, temporal_resolution=60)
    
    assert isinstance(result, pd.Series)
    assert result.name == 'timeslot'
    assert len(result) == len(sample_test_resolution_handling)
    assert result.iloc[0] == '10_0'
    assert result.iloc[1] == '11_0'
    assert result.iloc[2] == '8_0'
    assert result.iloc[3] == '6_0'
    assert all(int(slot.split('_')[1]) % 60 == 0 for slot in result)

def test_generalise_temporal_dataframe(sample_timestamp_df):
    """Test temporal generalisation with DataFrame input"""
    result = temporal_generaliser.generalise_temporal(
        sample_timestamp_df, 
        timestamp_col='timestamp',
        temporal_resolution=30
    )
    
    assert isinstance(result, pd.Series)
    assert result.name == 'timeslot'
    assert len(result) == len(sample_timestamp_df)

@pytest.mark.parametrize("resolution", [15, 30, 60])
def test_generalise_temporal_resolutions(sample_test_resolution_handling, resolution):
    """Test different temporal resolutions"""
    result = temporal_generaliser.generalise_temporal(
        sample_test_resolution_handling,
        temporal_resolution=resolution
    )
    assert all(int(slot.split('_')[1]) % resolution == 0 for slot in result)

def test_generalise_temporal_invalid_resolution(sample_test_resolution_handling):
    """Test invalid temporal resolution"""
    with pytest.raises(AssertionError):
        temporal_generaliser.generalise_temporal(
            sample_test_resolution_handling,
            temporal_resolution=45
        )

def test_generalise_temporal_missing_column(sample_timestamp_df):
    """Test missing timestamp column in DataFrame"""
    with pytest.raises(ValueError):
        temporal_generaliser.generalise_temporal(sample_timestamp_df)

def test_generalise_temporal_invalid_input():
    """Test invalid input type"""
    with pytest.raises(TypeError):
        temporal_generaliser.generalise_temporal([1, 2, 3])
