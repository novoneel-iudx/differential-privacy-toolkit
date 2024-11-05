from typing import Tuple
import pandas as pd
import numpy as np
import h3

class GeneraliseData:
    # use if coordinates are formatted as [lat,lon] 
    def format_coordinates(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        def parse_coordinate(coordinate: str) -> Tuple[float, float]:
            try:
                lat, lon = coordinate.strip('[]').strip(' ').split(',')
                return float(lat), float(lon)
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid coordinate format: {coordinate}") from e
        
        coordinates = series.apply(parse_coordinate)
        latitude = coordinates.apply(lambda x: x[0])
        longitude = coordinates.apply(lambda x: x[1])
        return latitude, longitude

    # requires separate column entries for lat, lon
    def generalise_spatial(latitude: pd.Series, longitude: pd.Series, spatial_resolution: int) -> pd.Series:
        if not (0 <= spatial_resolution <= 15):
            raise ValueError("Spatial resolution must be between 0 and 15.")
        
        if not latitude.between(-90, 90).all():
            raise ValueError("Latitude values must be between -90 and 90.")
        
        if not longitude.between(-180, 180).all():
            raise ValueError("Longitude values must be between -180 and 180.")
        
        if len(latitude) != len(longitude):
            raise Warning("Latitude and longitude series are of unequal length! Extra values will be ignored.")

        h3_index = [
            h3.latlng_to_cell(lat, lon, spatial_resolution)
            for lat, lon in zip(latitude, longitude)
        ]
        
        return pd.Series(h3_index, name='h3_index')
    
