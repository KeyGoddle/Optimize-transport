import pandas as pd
import numpy as np
import time
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from pathlib import Path
class Visualizer:
    def create_utilization_data(self,total_time_by_car_type, table_cars_moscow_results):
        utilization_data = []

        for car_type, total_time in total_time_by_car_type.items():
            total_possible_time = 24 * table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == car_type]['Расчетный номер машины (из ТМС, 1С УАТ)'].sum()
            utilization_percentage = (total_time / total_possible_time) * 100
            used_cars_count = table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == car_type]['Количество рейсов'].sum()

            utilization_data.append({
                'Модель ТС': car_type,
                'Количество используемых машин (Оптимизация)': used_cars_count,
                'Процент утилизации(Оптимизация)': utilization_percentage
            })

        return utilization_data

    def clean_and_transform_data(self,df, ufps, data_transport):
        filtered_df = df[(df['УФПС начала перевозки'] == ufps) & (df['Дата начала перевозки'] == data_transport)]
        df_cleaned = filtered_df.dropna().copy()

        df_cleaned['Дата начала перевозки локальное время'] = pd.to_datetime(df_cleaned['Дата начала перевозки локальное время'])
        df_cleaned['Дата окончания перевозки (локальное время)'] = pd.to_datetime(df_cleaned['Дата окончания перевозки (локальное время)'])

        df_cleaned['Время маршрута'] = (df_cleaned['Дата окончания перевозки (локальное время)'] - df_cleaned['Дата начала перевозки локальное время']).dt.total_seconds() / 3600
        df_cleaned = df_cleaned.reset_index()
        return df_cleaned

    def update_tables(self,result_integer, table_cars_moscow, table_target_moscow_1day, table_cars_moscow_results, table_target_moscow_1day_results, deliveries, vehicles):
        for (i, j), value in result_integer['x_values'].items():
            if value > 0:
                car_index = table_cars_moscow[(table_cars_moscow['Модель ТС'] == vehicles[j]['тип'])].index[0] 
                delivers_index = table_target_moscow_1day[(table_target_moscow_1day['Наименование маршрута'] == deliveries[i]['точка'])].index[0] 
                
                table_cars_moscow_results.loc[car_index, 'Количество рейсов'] += value
                if isinstance(table_cars_moscow_results.loc[car_index, 'Наименование маршрута'], list):
                    table_cars_moscow_results.loc[car_index, 'Наименование маршрута'].append(deliveries[i]['точка'])
                else:
                    table_cars_moscow_results.loc[car_index, 'Наименование маршрута'] = [deliveries[i]['точка']]

                target_index = table_target_moscow_1day_results[table_target_moscow_1day_results['Наименование маршрута'] == deliveries[i]['точка']].index[0]

                table_target_moscow_1day_results.loc[target_index, 'Количество рейсов'] += value

                current_value = table_target_moscow_1day_results.loc[target_index, 'Модель ТС']

                if isinstance(current_value, list):
                    current_value.append(vehicles[j]['тип'])
                else:
                    table_target_moscow_1day_results.loc[target_index, 'Модель ТС'] = [vehicles[j]['тип']]
                
                table_target_moscow_1day_results.loc[target_index, 'Стоимость'] += table_cars_moscow.loc[car_index,'затраты на км'] * table_target_moscow_1day.loc[delivers_index,'Фактическое расстояние'] * value

        print(f"Время выполнения: {result_integer['execution_time']} секунд")

    def calculate_time_by_vehicle_type(self,df_cleaned):
        time_by_vehicle_type = {}

        for index, row in df_cleaned.iterrows():
            vehicle_type = row['Модель ТС']
            time_ = row['Время маршрута']

            if vehicle_type in time_by_vehicle_type:
                time_by_vehicle_type[vehicle_type] += time_
            else:
                time_by_vehicle_type[vehicle_type] = time_

        return time_by_vehicle_type

    def calculate_utilization_by_vehicle_type(self,time_by_vehicle_type, count_by_model_df):
        utilization_by_vehicle_type = {}

        for vehicle_type, total_time in time_by_vehicle_type.items():
            total_possible_time = 24 * count_by_model_df[count_by_model_df['Модель ТС'] == vehicle_type]['Количество маршрутов'].sum()
            utilization_percentage = (total_time / total_possible_time) * 100
            utilization_by_vehicle_type[vehicle_type] = utilization_percentage

        return utilization_by_vehicle_type

    def create_summary_df(self,utilization_by_vehicle_type, count_by_model, df_cleaned,ufps):
        summary_df = pd.DataFrame({
            'Модель ТС': utilization_by_vehicle_type.keys(),
            'Процент утилизации': utilization_by_vehicle_type.values(),
            'Количество маршрутов': count_by_model.values,
            'Дата начала перевозки': df_cleaned['Дата начала перевозки'][0],
            'УФПСИ':ufps
        })

        return summary_df

    def merge_dataframes(self,summary_df, utilization_df):
        merged_df = pd.merge(summary_df, utilization_df, on='Модель ТС', how='outer')
        return merged_df