"""
A toolkit for data anonymisation.
View the documentation for this project [here](https://novoneel-iudx.github.io/differential-privacy-toolkit/).
"""
__version__ = "0.1.2"


'''Functions to be exposed to user'''
from .sanitisation import *
from .generalisation import *

__all__ = [
    'SanitiseData',
    'GeneraliseData',
    'sanitise_data',
    'format_coordinates',
    'generalise_spatial',
    'generalise_temporal',
    'generalise_categorical'
]
