import pytest
import pandas as pd

from generalisation import GeneraliseData

temporal_generaliser = GeneraliseData().TemporalGeneraliser()
def test_format_timestamp_valid(self, temporal_generaliser):
    """Test timestamp formatting with valid inputs"""
    timestamps = pd.Series([
        "2024-01-01 10:30:00",
        "2024-01-01T10:30:00",
        "01/01/2024 10:30:00"
    ])
    result = temporal_generaliser.format_timestamp(timestamps)
    assert all(isinstance(ts, pd.Timestamp) for ts in result)

def test_format_timestamp_invalid(self, temporal_generaliser):
    """Test timestamp formatting with invalid inputs"""
    invalid_timestamps = pd.Series(["invalid", "2024-13-01", "25:00:00"])
    result = GeneraliseData.TemporalGeneraliser.format_timestamp(invalid_timestamps)
    assert result.isna().all()

def test_generalise_temporal_series(self, temporal_generaliser, sample_timestamps):
    """Test temporal generalisation with Series input"""
    result = temporal_generaliser.generalise_temporal(sample_timestamps)
    
    assert isinstance(result, pd.Series)
    assert result.name == 'timeslot'
    assert len(result) == len(sample_timestamps)
    assert result.iloc[0] == '10_30'

def test_generalise_temporal_dataframe(self, temporal_generaliser, sample_timestamp_df):
    """Test temporal generalisation with DataFrame input"""
    result = temporal_generaliser.generalise_temporal(
        sample_timestamp_df, 
        timestamp_col='timestamp'
    )
    
    assert isinstance(result, pd.Series)
    assert result.name == 'timeslot'
    assert len(result) == len(sample_timestamp_df)

@pytest.mark.parametrize("resolution", [15, 30, 60])
def test_generalise_temporal_resolutions(self, temporal_generaliser, sample_timestamps, resolution):
    """Test different temporal resolutions"""
    result = temporal_generaliser.generalise_temporal(
        sample_timestamps,
        temporal_resolution=resolution
    )
    assert all(int(slot.split('_')[1]) % resolution == 0 for slot in result)

def test_generalise_temporal_invalid_resolution(self, temporal_generaliser, sample_timestamps):
    """Test invalid temporal resolution"""
    with pytest.raises(AssertionError):
        temporal_generaliser.generalise_temporal(
            sample_timestamps,
            temporal_resolution=45
        )

def test_generalise_temporal_missing_column(self, temporal_generaliser, sample_timestamp_df):
    """Test missing timestamp column in DataFrame"""
    with pytest.raises(ValueError):
        temporal_generaliser.generalise_temporal(sample_timestamp_df)

def test_generalise_temporal_invalid_input(self, temporal_generaliser):
    """Test invalid input type"""
    with pytest.raises(TypeError):
        temporal_generaliser.generalise_temporal([1, 2, 3])

@pytest.mark.parametrize("test_input,expected", [
    ("2024-01-01 10:30:00", "10_30"),
    ("2024-01-01 10:45:00", "10_30"),
    ("2024-01-01 11:00:00", "11_0"),
    ("2024-01-01 23:59:00", "23_30"),
])
def test_generalise_temporal_specific_times(self, temporal_generaliser, test_input, expected):
    """Test specific time values"""
    result = temporal_generaliser.generalise_temporal(
        pd.Series([test_input]),
        temporal_resolution=30
    )
    assert result.iloc[0] == expected