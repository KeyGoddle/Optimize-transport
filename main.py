from data_processing import DataProcessor
from optimization import Optimizer
from visualization import Visualizer

import pandas as pd
import numpy as np
import time
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    data_processor = DataProcessor()
    optimizer = Optimizer()
    visualizer = Visualizer()
    ufps = 'УФПС МОСКВЫ'
    data_transport = '2024-02-05T00:00:00.000000000'
    df = data_processor.load_and_prepare_data()
    #data_processor.clean_and_transform_data(df, ufps, data_transport)
    table_cars_moscow, table_target_moscow_1day, table_cars_target_time_edit_m = data_processor.prepare_data(df, ufps, data_transport)
    table_cars_target_time_edit_m, vehicles, deliveries, times, table_target_moscow_1day_results, table_cars_moscow_results = data_processor.preprocess_tables(table_cars_target_time_edit_m, table_target_moscow_1day, table_cars_moscow)
    

    result_integer = optimizer.optimize_delivery_integer(vehicles, deliveries, times)
    print("Результаты оптимизации:")
    print(f"Статус оптимизации: {result_integer['status']}")
    print(f"Общая стоимость доставки: {result_integer['objective_value']}")

    optimizer.update_tables(result_integer, table_cars_moscow, table_target_moscow_1day, table_cars_moscow_results, table_target_moscow_1day_results, deliveries, vehicles)

    total_time_by_car_type = optimizer.calculate_total_time_by_car_type(result_integer, vehicles, deliveries, times)

    utilization_data = visualizer.create_utilization_data(total_time_by_car_type, table_cars_moscow_results)
    utilization_df = pd.DataFrame(utilization_data)

    df_cleaned = visualizer.clean_and_transform_data(df, ufps, data_transport)
    time_by_vehicle_type = visualizer.calculate_time_by_vehicle_type(df_cleaned)
    count_by_model = df_cleaned['Модель ТС'].value_counts()
    count_by_model_df = pd.DataFrame({'Модель ТС': count_by_model.index, 'Количество маршрутов': count_by_model.values})

    utilization_by_vehicle_type = visualizer.calculate_utilization_by_vehicle_type(time_by_vehicle_type, count_by_model_df)
    summary_df = visualizer.create_summary_df(utilization_by_vehicle_type, count_by_model, df_cleaned)

    merged_df = visualizer.merge_dataframes(summary_df, utilization_df)
    print(merged_df)

if __name__ == "__main__":
    main()
