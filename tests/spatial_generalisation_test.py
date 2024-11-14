import pytest
import pandas as pd
import numpy as np
import h3
from datetime import datetime
from typing import Tuple, Literal, Union, get_args

from generalisation import GeneraliseData

spatial_generaliser = GeneraliseData().SpatialGeneraliser()

# TODO: Add pytest fixtures for common samples
# Test cases for format_coordinates
def test_format_coordinates_valid_input():
    """Test format_coordinates with valid input data"""
    input_data = pd.Series(['[40.7128, -74.0060]', '[51.5074, -0.1278]'])
    lat, lon = spatial_generaliser.format_coordinates(input_data)
    
    assert lat.iloc[0] == pytest.approx(40.7128)
    assert lon.iloc[0] == pytest.approx(-74.0060)
    assert lat.iloc[1] == pytest.approx(51.5074)
    assert lon.iloc[1] == pytest.approx(-0.1278)

def test_format_coordinates_whitespace():
    """Test format_coordinates with various whitespace patterns"""
    input_data = pd.Series(['[40.7128,   -74.0060]', '[  51.5074,-0.1278  ]'])
    lat, lon = spatial_generaliser.format_coordinates(input_data)
    
    assert lat.iloc[0] == pytest.approx(40.7128)
    assert lon.iloc[0] == pytest.approx(-74.0060)

def test_format_coordinates_scientific_notation():
    """Test format_coordinates with scientific notation"""
    input_data = pd.Series(['[1.23e-2, 4.56e1]'])
    lat, lon = spatial_generaliser.format_coordinates(input_data)
    
    assert lat.iloc[0] == pytest.approx(0.0123)
    assert lon.iloc[0] == pytest.approx(45.6)

def test_format_coordinates_empty_series():
    """Test format_coordinates with empty series"""
    input_data = pd.Series([], dtype=str)
    lat, lon = spatial_generaliser.format_coordinates(input_data)
    
    assert len(lat) == 0
    assert len(lon) == 0

@pytest.mark.parametrize("invalid_input", [
    '[invalid, 12.34]',
    '[12.34]',
    'null',
    '[]',
    '[12.34, 56.78, 90.12]'
])
def test_format_coordinates_invalid_input(invalid_input):
    """Test format_coordinates with various invalid inputs"""
    with pytest.raises((ValueError, IndexError)):
        spatial_generaliser.format_coordinates(pd.Series([invalid_input]))

# Test cases for generalise_spatial
def test_generalise_spatial_valid_input():
    """Test generalise_spatial with valid input"""
    lat = pd.Series([40.7128, 51.5074])
    lon = pd.Series([-74.0060, -0.1278])
    resolution = 7
    
    result = spatial_generaliser.generalise_spatial(lat, lon, resolution)

    assert len(result) == min(len(lat), len(lon))
    assert all(isinstance(x, str) for x in result)  # H3 indices are strings
    assert all(len(x) > 0 for x in result)
    assert all(h3.is_valid_cell(h) for h in result)

def test_generalise_spatial_bounds():
    """Test generalise_spatial with edge cases for latitude and longitude"""
    lat = pd.Series([90.0, -90.0])  # Maximum latitudes
    lon = pd.Series([180.0, -180.0])  # Maximum longitudes
    resolution = 7
    
    
    result = spatial_generaliser.generalise_spatial(lat, lon, resolution)

    assert len(result) == min(len(lat), len(lon))
    assert all(isinstance(x, str) for x in result)  # H3 indices are strings
    assert all(len(x) > 0 for x in result)
    assert all(h3.is_valid_cell(h) for h in result)

@pytest.mark.parametrize("invalid_resolution", [-1, 16, 100])
def test_generalise_spatial_invalid_resolution(invalid_resolution):
    """Test generalise_spatial with invalid resolution values"""
    lat = pd.Series([40.7128])
    lon = pd.Series([-74.0060])
    
    with pytest.raises(ValueError):
        spatial_generaliser.generalise_spatial(lat, lon, invalid_resolution)

def test_generalise_spatial_mismatched_series():
    """Test generalise_spatial with mismatched series lengths"""
    lat = pd.Series([40.7128, 51.5074])
    lon = pd.Series([-74.0060])
    resolution = 7
    
    # result = generaliser.generalise_spatial(lat, lon, resolution)

    # assert len(result) == min(len(lat), len(lon))
    with pytest.raises(Warning):
        spatial_generaliser.generalise_spatial(lat, lon, resolution)

# Additional helper functions for testing
def generate_random_coordinates(n: int) -> Tuple[pd.Series, pd.Series]:
    """Generate random valid coordinates for testing"""
    lat = pd.Series(np.random.uniform(-90, 90, n))
    lon = pd.Series(np.random.uniform(-180, 180, n))
    return lat, lon

def test_random_coordinates():
    """Test both functions with randomly generated coordinates"""
    n_points = 100
    lat, lon = generate_random_coordinates(n_points)
    resolution = 7
    
    # Convert to string format for format_coordinates
    coord_strings = pd.Series([f'[{lat[i]}, {lon[i]}]' for i in range(n_points)])
    
    # Test format_coordinates
    parsed_lat, parsed_lon = spatial_generaliser.format_coordinates(coord_strings)
    assert len(parsed_lat) == n_points
    assert len(parsed_lon) == n_points
    
    # Test generalise_spatial
    h3_indexes = spatial_generaliser.generalise_spatial(parsed_lat, parsed_lon, resolution)

    assert len(h3_indexes) == n_points
    assert len(h3_indexes) == min(len(lat), len(lon))
    assert all(isinstance(x, str) for x in h3_indexes)  # H3 indices are strings
    assert all(len(x) > 0 for x in h3_indexes)
    assert all(h3.is_valid_cell(h) for h in h3_indexes)
