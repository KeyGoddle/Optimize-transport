from data_processing import DataProcessor
from optimization import Optimizer
from visualization import Visualizer
import copy

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

    def do_every(self,method):
        data_processor = DataProcessor()
        df = data_processor.load_and_prepare_data()

        every_day_every_ufps_util = pd.DataFrame()
        every_day_every_ufps_table_target = pd.DataFrame()
        every_df_cleaned = pd.DataFrame()
        table_cars_moscow_results=pd.DataFrame() #columns=['Модель ТС','Фактическая грузоподъемность ТС (в кг)','затраты на км','Расчетный номер машины (из ТМС, 1С УАТ)','Количество рейсов','Наименование маршрута']
        new_util_global =[]
        all_pairs = list(product(df['УФПС начала перевозки'].unique(), df['Дата начала перевозки'].unique()))
        filtered_pairs = list(filter(self.filter_pairs, all_pairs))


        if method=='УФПС МОСКВЫ':
            filtered_pairs = [pair for pair in filtered_pairs if "УФПС МОСКОВСКОЙ ОБЛ" not in pair]
        if method=='УФПС МОСКОВСКОЙ ОБЛ':
            filtered_pairs = [pair for pair in filtered_pairs if "УФПС МОСКВЫ" not in pair]


        model_dict = {}
        for ufps, data_transport in filtered_pairs:
            vrem_df_util, vrem_df_table_target, vrem_df_cleaned, vrem_util_global, vrem_table_cars_moscow_results = main(ufps, data_transport)
            
            for item in vrem_util_global:
                model = item['Модель ТС']
                if model in model_dict:
                    # Если модель уже есть в словаре, складываем данные
                    existing_item = model_dict[model]
                    existing_item['Количество используемых машин (Оптимизация)'] += item['Количество используемых машин (Оптимизация)']
                    existing_item['Потраченное время на маршруты'] += item['Потраченное время на маршруты']
                else:
                    # Если модели нет в словаре, добавляем ее
                    model_dict[model] = item.copy()  # Создаем копию элемента, чтобы не изменять исходный список
                    new_util_global.append(item)  # Добавляем копию элемента в новый список

            # Преобразуем словарь обратно в список значений
            new_util_global = list(model_dict.values())

            #print(vrem_util_global)
            
            if table_cars_moscow_results.empty:
                table_cars_moscow_results = vrem_table_cars_moscow_results.copy()
            else:
                # Выполняем слияние по заданным колонкам
                table_cars_moscow_results['Наименование маршрута'] = table_cars_moscow_results['Наименование маршрута'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                vrem_table_cars_moscow_results['Наименование маршрута'] = vrem_table_cars_moscow_results['Наименование маршрута'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                table_cars_moscow_results = pd.merge(table_cars_moscow_results, vrem_table_cars_moscow_results, on=['Модель ТС', 'Фактическая грузоподъемность ТС (в кг)', 'затраты на км', 'Расчетный номер машины (из ТМС, 1С УАТ)', 'Наименование маршрута'], how='outer', suffixes=('_left', '_right'))
                print (table_cars_moscow_results)
                # Группировка и агрегация
                table_cars_moscow_results = table_cars_moscow_results.groupby(['Модель ТС', 'Фактическая грузоподъемность ТС (в кг)', 'затраты на км', 'Расчетный номер машины (из ТМС, 1С УАТ)', 'Наименование маршрута']).agg({'Количество рейсов_left': 'sum', 'Количество рейсов_right': 'sum'}).reset_index()
                table_cars_moscow_results['Количество рейсов'] = table_cars_moscow_results['Количество рейсов_left'] + table_cars_moscow_results['Количество рейсов_right']
                #table_cars_moscow_results['Количество рейсов'] = table_cars_moscow_results['value_sum']
                
                table_cars_moscow_results.drop(['Количество рейсов_left', 'Количество рейсов_right'], axis=1, inplace=True)
                table_cars_moscow_results['Наименование маршрута'] = table_cars_moscow_results['Наименование маршрута'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)
            #=table_cars_moscow_results.append(vrem_table_cars_moscow_results)

            # Модель ТС  Фактическая грузоподъемность ТС (в кг)  затраты на км  Расчетный номер машины (из ТМС, 1С УАТ) Наименование маршрута

            every_df_cleaned = every_df_cleaned.append(vrem_df_cleaned) 
            every_day_every_ufps_util = every_day_every_ufps_util.append(vrem_df_util) 
            every_day_every_ufps_table_target = every_day_every_ufps_table_target.append(vrem_df_table_target) 

        table_cars_moscow_results = table_cars_moscow_results.reset_index(drop=True)
        every_day_every_ufps_util = every_day_every_ufps_util.reset_index(drop=True)
        every_day_every_ufps_table_target = every_day_every_ufps_table_target.reset_index(drop=True)
        every_df_cleaned = every_df_cleaned.reset_index(drop=True)
        #print (table_cars_moscow_results)
        return every_day_every_ufps_util,every_day_every_ufps_table_target,every_df_cleaned,new_util_global,table_cars_moscow_results

class Global_For_Unique_UFPS:
    def calculation_for_each_ups(self,every_day_every_ufps_table_target,every_df_cleaned,new_util_global,table_cars_moscow_results):
        util_dfs = self.calculate_week_util(every_df_cleaned,new_util_global,table_cars_moscow_results)
        return util_dfs

    def calculate_week_util(self,every_df_cleaned,new_util_global,table_cars_moscow_results):
        #df_before_util
        visualizer=Visualizer()
        time_by_vehicle_type = visualizer.calculate_time_by_vehicle_type(every_df_cleaned)
        count_by_model = every_df_cleaned['Модель ТС'].value_counts()
        count_by_model_df = pd.DataFrame({'Модель ТС': count_by_model.index, 'Количество маршрутов': count_by_model.values})
        model_list = count_by_model_df['Модель ТС'].tolist()  # Получаем список моделей транспортных средств из count_by_model_df
        time_by_vehicle_type_filtered = {key: value for key, value in time_by_vehicle_type.items() if key in model_list}  # Убираем из time_by_vehicle_type элементы, которых нет в model_list
        utilization_by_vehicle_type = visualizer.calculate_utilization_by_vehicle_type(time_by_vehicle_type_filtered, count_by_model_df)   
        summary_df = visualizer.create_summary_df(utilization_by_vehicle_type, count_by_model, every_df_cleaned,'УФПС МОСКОВСКОЙ ОБЛ')
        #print ('summary_df',summary_df)


        new_util=visualizer.calclulate_after_prep_util(new_util_global,table_cars_moscow_results)          #Добавлено новая утилизация    
        utilization_df=pd.DataFrame(new_util)                                                       #Добавлено новая утилизация 
        utilization_df.drop('Потраченное время на маршруты', axis= 1 , inplace= True ) 

        #print ('utilization_df',utilization_df)

        merged_df = visualizer.merge_dataframes(summary_df, utilization_df)
        return merged_df  #after


if __name__ == "__main__":
    every=Every_day_every_day_ufps()
    global_Ufps=Global_For_Unique_UFPS()
    visualizer=Visualizer()

    #вызов методов 
    #Если неодходимо вызвать весь код разом, тогда слудет задать method='all'
    #Если надо расчитать по отмедбльно ufps, тогда method='УФПС МОСКВЫ' или method ='УФПС МОСКОВСКОЙ ОБЛ'

    method='all'
    method='УФПС МОСКВЫ'
    method='УФПС МОСКОВСКОЙ ОБЛ'


    every_day_every_ufps_util,every_day_every_ufps_table_target,every_df_cleaned,new_util_global,table_cars_moscow_results=every.do_every(method)
    print (new_util_global)
    #print (every_day_every_ufps_util,every_day_every_ufps_table_target)
    #print (every_df_cleaned)
    utilization_global=global_Ufps.calculation_for_each_ups(every_day_every_ufps_table_target,every_df_cleaned,new_util_global,table_cars_moscow_results)
    print (utilization_global)