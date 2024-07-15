from data_processing_3 import DataProcessor
from optimization_3 import Optimizer
from visualization_3 import Visualizer

import copy
import pandas as pd
import numpy as np
import time
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from pathlib import Path

class MainProcessor:
    def __init__(self):
        pass
    
    def run_main(self, ufps, data_transport, calculation):
        data_processor = DataProcessor()
        optimizer = Optimizer()
        visualizer = Visualizer()
        naym = 1
        all_car = 1

        df = data_processor.load_and_prepare_data()
        df_cleaned = data_processor.clean_and_transform_data(df, ufps, data_transport)
        table_cars_moscow, table_target_moscow_1day, table_cars_target_time_edit_m = data_processor.prepare_data(df, ufps, data_transport, naym, all_car, calculation)

        if naym == 1:
            if calculation=='mass':
                self.adjust_costs_for_naym(table_cars_moscow)

        table_cars_target_time_edit_m, vehicles, deliveries, times, table_target_moscow_1day_results, table_cars_moscow_results = data_processor.preprocess_tables(
            table_cars_target_time_edit_m, table_target_moscow_1day, table_cars_moscow, calculation
        )

        result_integer = optimizer.optimize_delivery_integer(vehicles, deliveries, times, calculation)
        optimizer.update_tables(result_integer, table_cars_moscow, table_target_moscow_1day, table_cars_moscow_results, table_target_moscow_1day_results, deliveries, vehicles)

        total_time_by_car_type = optimizer.calculate_total_time_by_car_type(result_integer, vehicles, deliveries, times)
        new_util = visualizer.prepare_all_util(total_time_by_car_type, table_cars_moscow_results)
        new_util_global = copy.deepcopy(new_util)
        new_util = visualizer.calculate_after_prep_util(new_util, table_cars_moscow_results)

        utilization_df = pd.DataFrame(new_util)
        utilization_df.drop('Потраченное время на маршруты', axis=1, inplace=True)

        time_by_vehicle_type = visualizer.calculate_time_by_vehicle_type(df_cleaned)
        time_by_vehicle_type = {k: time_by_vehicle_type[k] for k in sorted(time_by_vehicle_type)}

        count_by_model = df_cleaned['Модель ТС'].value_counts()
        count_by_model = count_by_model.sort_index()

        count_by_model_df = pd.DataFrame({'Модель ТС': count_by_model.index, 'Количество маршрутов': count_by_model.values})
        utilization_by_vehicle_type = visualizer.calculate_utilization_by_vehicle_type(time_by_vehicle_type, count_by_model_df)
        summary_df = visualizer.create_summary_df(utilization_by_vehicle_type, count_by_model, df_cleaned, ufps)

        merged_df = visualizer.merge_dataframes(summary_df, utilization_df)

        table_target_moscow_1day_results['УФПС'] = ufps
        table_target_moscow_1day_results['Даты начала перевозки по маршруту'] = pd.to_datetime(data_transport)

        return merged_df, table_target_moscow_1day_results, df_cleaned, new_util_global, table_cars_moscow_results

    def adjust_costs_for_naym(self, table_cars_moscow):
        print (table_cars_moscow)
        index = table_cars_moscow.index[table_cars_moscow['Модель ТС'] == 'Наемник - 7000'][0]
        if index > 0 and index < len(table_cars_moscow) - 1:
            prev_cost = table_cars_moscow.at[index - 1, 'затраты на км']
            next_cost = table_cars_moscow.at[index + 1, 'затраты на км']
            average_cost = (prev_cost + next_cost) / 2
            table_cars_moscow.at[index, 'затраты на км'] = average_cost

        index = table_cars_moscow.index[table_cars_moscow['Модель ТС'] == 'Наемник - 20000'][0]
        prev_cost = table_cars_moscow.at[index - 1, 'затраты на км']
        true_cost = table_cars_moscow.at[index, 'затраты на км']
        table_cars_moscow.at[index, 'затраты на км'] = ((prev_cost + true_cost) / 2) + 50

        index = table_cars_moscow.index[table_cars_moscow['Модель ТС'] == 'Наемник - 1500'][0]
        true_cost = table_cars_moscow.at[index, 'затраты на км']
        table_cars_moscow.at[index, 'затраты на км'] = true_cost + 41

        index = table_cars_moscow.index[table_cars_moscow['Модель ТС'] == 'Наемник - 2000'][0]
        true_cost = table_cars_moscow.at[index, 'затраты на км']
        table_cars_moscow.at[index, 'затраты на км'] = true_cost + 31

        index = table_cars_moscow.index[table_cars_moscow['Модель ТС'] == 'Наемник - 5000'][0]
        true_cost = table_cars_moscow.at[index, 'затраты на км']
        table_cars_moscow.at[index, 'затраты на км'] = true_cost + 36
def get_all_data():
    data_processor = DataProcessor()
    df = data_processor.load_and_prepare_data()

    df['Даты начала перевозки по маршруту'] = pd.to_datetime(df['Даты начала перевозки по маршруту'], dayfirst=True)
    df['Даты окончания перевозки по маршруту по плану'] = pd.to_datetime(df['Даты окончания перевозки по маршруту по плану'], dayfirst=True)

    all_times = list((df['Даты начала перевозки по маршруту'].dt.normalize()).unique())
    return all_times

if __name__ == "__main__":
    main_processor = MainProcessor()
    ufps = 'УФПС МОСКОВСКОЙ ОБЛ'
    root = 'files_data_new'
    calculation='mass' # mass volume
    pd.set_option('display.float_format', '{:.2f}'.format)

    columns = ['УФПС', 'Даты начала перевозки по маршруту', 'Стоимость перевозки оптимизация', 'Стоимость перевозки факт']
    effect_df = pd.DataFrame(columns=columns)

    all_data = get_all_data()
    for data in all_data:
        data_transport = data
        merged_df, table_target_moscow_1day_results, df_cleaned, new_util_global, table_cars_moscow_results = main_processor.run_main(ufps, data_transport,calculation)
        
        grouped_df = table_target_moscow_1day_results.groupby(['УФПС', 'Даты начала перевозки по маршруту'])['Стоимость'].sum().reset_index()
        grouped_df.rename(columns={'Стоимость': 'Стоимость перевозки оптимизация'}, inplace=True)
        grouped_df['Стоимость перевозки факт'] = df_cleaned['Стоимость перевозки факт (ТМС)'].sum()
        effect_df = pd.concat([effect_df, grouped_df], ignore_index=True)

    print(effect_df)