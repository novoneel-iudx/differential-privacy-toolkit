from typing import Tuple
import pandas as pd
import numpy as np
import h3
from typing import Literal, Union, get_args



class GeneraliseData:
    class SpatialGeneraliser:
        # TODO: rewrite so function can accept both dataframe with specified location column as well as series
        # helper function to clean coordinates attribute formatting

        @staticmethod
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

        @staticmethod
        def generalise_spatial(latitude: pd.Series, longitude: pd.Series, spatial_resolution: int) -> pd.Series:
            if not (0 <= spatial_resolution <= 15):
                raise ValueError("H3 Spatial resolution must be between 0 and 15.")
            
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

    class TemporalGeneraliser:

        @staticmethod     
        # TODO: Do we really need a wrapper around a standard pandas function?  
        # helper function to convert timestamp to pd.datetime object
        def format_timestamp(series: pd.Series) -> pd.Series:
            # Check if all timestamps in the series are of the same format
            return pd.to_datetime(series, format='mixed', errors='coerce') # coerce non-parseable values to NaT
        
        # def __init__(self):
        #     self.temporal_resolution_args = Literal[15, 30, 60]
        @staticmethod       
        def generalise_temporal(data: Union[pd.Series, pd.DataFrame],
                                timestamp_col: str = None,
                                temporal_resolution: int = 60
                            ) -> pd.Series:
            temporal_resolution_args = Literal[15, 30, 60]
            """
            Example:
            ### Using with a Series
            generalise_temporal(ts_series)
            
            ### Using with a DataFrame
            generalise_temporal(df, timestamp_col='timestamp')
            """
            # Validate temporal resolution
            options = list(get_args(temporal_resolution_args))
            assert temporal_resolution in options, (
                f"'{temporal_resolution}' is not in {options}, please choose a valid value"
            )
            
            # Extract the timestamp series based on input type
            if isinstance(data, pd.DataFrame):
                if timestamp_col is None:
                    raise ValueError(
                        "timestamp_col must be specified when input is a DataFrame"
                    )
                if timestamp_col not in data.columns:
                    raise ValueError(
                        f"Column '{timestamp_col}' not found in DataFrame. "
                        f"Available columns are: {list(data.columns)}"
                    )
                timestamp = data[timestamp_col]
            elif isinstance(data, pd.Series):
                timestamp = data
            else:
                raise TypeError(
                    f"Input must be pandas Series or DataFrame, not {type(data)}"
                )
            
            # Ensure timestamp data is datetime type
            try:
                timestamp = pd.to_datetime(timestamp)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert input to datetime: {str(e)}"
                )
            
            # Extract date and time components
            time = timestamp.dt.time
            
            # Create timeslots
            timeslot = time.apply(
                lambda x: f'{x.hour}_{((x.minute)//temporal_resolution)*temporal_resolution}'
            )
            
            return pd.Series(timeslot, name='timeslot')
