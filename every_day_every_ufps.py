from data_processing import DataProcessor
from optimization import Optimizer
from visualization import Visualizer

import pandas as pd
import numpy as np
import time
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

from main import main
class Every_day_every_day_ufps:
    def filter_pairs(self,pair):
        return not any(pd.isna(value) for value in pair)

    def do_every(self):
        data_processor = DataProcessor()
        df = data_processor.load_and_prepare_data()

        every_day_every_ufps = pd.DataFrame()

        all_pairs = list(product(df['УФПС начала перевозки'].unique(), df['Дата начала перевозки'].unique()))
        filtered_pairs = list(filter(self.filter_pairs, all_pairs))

        for ufps,data_transport in filtered_pairs:
            vrem_df=main(ufps,data_transport)
            every_day_every_ufps = every_day_every_ufps.append(vrem_df) 

        every_day_every_ufps = every_day_every_ufps.reset_index(drop=True)

        return every_day_every_ufps




if __name__ == "__main__":
    every=Every_day_every_day_ufps()
    print (every.do_every())