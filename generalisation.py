from typing import Tuple
import pandas as pd
import numpy as np
import h3
from typing import Literal, get_args



class GeneraliseData:
    class SpatialGeneralisation:
        
        # helper function to clean coordinates attribute formatting
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

        
        def generalise_spatial_h3(latitude: pd.Series, longitude: pd.Series, spatial_resolution: int) -> pd.Series:
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

    class TemporalGeneralisation:
               
        # helper function to clean date and time attribute formatting
        def format_timestamp(series: pd.Series) -> pd.Series:
            return series.apply(pd.to_datetime(errors='coerce')) # coerce non-parseable values to NaN
        
        temporal_resolution_args = Literal[15, 30, 60]
        def generalise_temporal(self, timestamp: pd.Series, temporal_resolution: temporal_resolution_args=60) -> pd.Series:
            
            options = list(get_args(self.temporal_resolution_args))
            assert temporal_resolution in options, f"'{temporal_resolution}' is not in {options}"

            timestamp.date = timestamp.dt.date
            timestamp.time = timestamp.dt.time
            timestamp.timeslot = timestamp.time.apply(lambda x: f'{x.hour}_{((x.minute)//temporal_resolution)*temporal_resolution}')          
            return pd.Series(timestamp, name='timestamp')   




        # def temporalGeneralization(dataframe, configFile):
        # configFile = configFile["temporal_generalize"]
        # temporalAttribute = configFile["temporal_attribute"]
        # TSR = configFile["timeslot_resolution"]
        # dataframe["Date"] = pd.to_datetime(dataframe[temporalAttribute]).dt.date
        # dataframe["Time"] = pd.to_datetime(dataframe[temporalAttribute]).dt.time
        # time = dataframe["Time"]
        # dataframe["Timeslot"] = time.apply(lambda x: f'{x.hour}_{((x.minute)//TSR)*TSR}')
        # dataframe.drop(columns = temporalAttribute, inplace = True)
        # return dataframe