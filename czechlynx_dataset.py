#!/usr/bin/env python3
"""
Czech Lynx Dataset class for wildlife_datasets integration.

This module provides a single WildlifeDataset subclass for Czech Lynx data
that works with unified metadata.csv containing all clients and splits.
"""

import pandas as pd
from pathlib import Path
from wildlife_datasets import datasets
import os


class CzechLynxDataset(datasets.WildlifeDataset):
    def create_catalogue(self) -> pd.DataFrame:
        df = pd.read_csv('lynx_metadata.csv')
        return df
