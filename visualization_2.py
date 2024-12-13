import pandas as pd
import numpy as np
import time
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from pathlib import Path
class Visualizer:
    # def create_utilization_data(self,total_time_by_car_type, table_cars_moscow_results):
    #     utilization_data = []

    #     for car_type, total_time in total_time_by_car_type.items():
    #         total_possible_time = 24 * table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == car_type]['Номер ТС'].sum()
    #         utilization_percentage = (total_time / total_possible_time) * 100
    #         used_cars_count = table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == car_type]['Количество рейсов'].sum()
    #         print (total_time,'/(24*',table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == car_type]['Номер ТС'].sum(),')',table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == car_type])
    #         utilization_data.append({
    #             'Модель ТС': car_type,
    #             'Количество используемых машин (Оптимизация)': used_cars_count,
    #             'Процент утилизации(Оптимизация)': utilization_percentage
    #         })

    #     return utilization_data
    def prepare_all_util(self,total_time_by_car_type, table_cars_moscow_results):
        prep_utilization_data = []

        for car_type, total_time in total_time_by_car_type.items():
            used_cars_count = table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == car_type]['Количество рейсов'].sum()
           
            prep_utilization_data.append({
                    'Модель ТС': car_type,
                    'Количество используемых машин (Оптимизация)': used_cars_count,
                    'Потраченное время на маршруты': total_time
                    })

        return prep_utilization_data
    def calclulate_after_prep_util(self,prep_utilization_data,table_cars_moscow_results):
        for item in prep_utilization_data:
            total_possible_time = 24 * table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == item['Модель ТС']]['Номер ТС'].sum()
            utilization_percentage = (item['Потраченное время на маршруты'] / total_possible_time) * 100
            print ('Оптимизация',item['Потраченное время на маршруты'],'24 *', table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == item['Модель ТС']]['Номер ТС'].sum(),table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == item['Модель ТС']])
            item['Процент утилизации(Оптимизация)'] = utilization_percentage
            
        return prep_utilization_data


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
            print ('БЕз оптимизации',total_time, '/( 24*',count_by_model_df[count_by_model_df['Модель ТС'] == vehicle_type]['Количество маршрутов'].sum(),')',count_by_model_df[count_by_model_df['Модель ТС'] == vehicle_type])
            if total_possible_time == 0:  # Проверка на деление на ноль
                utilization_by_vehicle_type[vehicle_type] = 0
            else:

                utilization_percentage = (total_time / total_possible_time) * 100
                utilization_by_vehicle_type[vehicle_type] = utilization_percentage
        print (utilization_by_vehicle_type)
        return utilization_by_vehicle_type

    def create_summary_df(self,utilization_by_vehicle_type, count_by_model, df_cleaned,ufps):
        #print("Length of utilization_by_vehicle_type:", len(utilization_by_vehicle_type))
        #print("Length of count_by_model:", len(count_by_model))
        #print("Length of df_cleaned['Модель ТС']:", len(df_cleaned['Модель ТС']))
        
        
        summary_df = pd.DataFrame({
            'Модель ТС': utilization_by_vehicle_type.keys(),
            'Процент утилизации': utilization_by_vehicle_type.values(),
            'Количество маршрутов': count_by_model.values,
            'Даты начала перевозки по маршруту': df_cleaned['Даты начала перевозки по маршруту'].min(),
            'УФПС':ufps
        })

        return summary_df

    def merge_dataframes(self,summary_df, utilization_df):
        merged_df = pd.merge(summary_df, utilization_df, on='Модель ТС', how='outer')
        return merged_df