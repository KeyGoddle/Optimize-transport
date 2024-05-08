import pandas as pd
import numpy as np
import time
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from pathlib import Path
class DataProcessor:
    def load_and_prepare_data(self):
        filename = Path(__file__).resolve().parent / 'data/Исходная_выгрузка_ТМС_python_Excel.xlsx'
        xl = pd.ExcelFile(filename)
        df = xl.parse('Исходная выгрузка_ТМС_python')

        calculation_columns = ['Расчетный номер машины (из ТМС, 1С УАТ)', 'Наличие автомашины в собственном парке Почты',
                            'Сумма весов емкостей из Сортмастер, кг', 'Средняя скорость передвижения',
                            'Дата начала перевозки', 'Дата начала перевозки локальное время',
                            'Дата окончания перевозки (локальное время)', 'Фактическая грузоподъемность ТС (в кг)',
                            'УФПС начала перевозки', 'Наименование маршрута', 'Модель ТС', 'Фактическое расстояние',
                            'Стоимость расходной части перевозки', 'Расчет затрат (руб) на (тонна*км) по перевозкам и из ТМС, и из 1с уат']
        df = df[calculation_columns]

        df['затраты на км'] = df['Стоимость расходной части перевозки'] / df['Фактическое расстояние']

        df['Перевозка одним днем'] = np.where(
            df['Дата начала перевозки локальное время'].dt.day == df['Дата окончания перевозки (локальное время)'].dt.day, 1, 0)
        df = df[df['Перевозка одним днем'] == 1]

        return df


    def prepare_data(self,df, ufps, data_transport):
        df = df[(df['УФПС начала перевозки'] == ufps) & (df['Дата начала перевозки'] == data_transport)]

        df_car = df[['Расчетный номер машины (из ТМС, 1С УАТ)', 'Наличие автомашины в собственном парке Почты', \
            'Фактическая грузоподъемность ТС (в кг)', 'УФПС начала перевозки', 'Модель ТС', 'затраты на км']]

        df_car = df_car.dropna()

        dpt_cars_M = pd.pivot_table(df_car[(df_car['Наличие автомашины в собственном парке Почты']==1) & \
                                        (df_car['УФПС начала перевозки']=='УФПС МОСКВЫ')], \
                                        index=['Модель ТС'], values='Расчетный номер машины (из ТМС, 1С УАТ)', aggfunc='nunique')

        dpt_cars_M_attrib = pd.pivot_table(df_car[(df_car['Наличие автомашины в собственном парке Почты']==1)], \
                                            index=['Модель ТС'], values=['Фактическая грузоподъемность ТС (в кг)', 'затраты на км'], aggfunc='median')

        dpt_cars_M.reset_index(inplace=True)
        dpt_cars_M_attrib.reset_index(inplace=True)

        table_cars = dpt_cars_M_attrib.merge(dpt_cars_M)

        df_target = df[['Дата начала перевозки', 'УФПС начала перевозки', 'Наименование маршрута', 'Фактическое расстояние', \
                        'Сумма весов емкостей из Сортмастер, кг', 'Стоимость расходной части перевозки']]

        df_target = df_target.dropna()

        dpt_target_M_d_distance = pd.pivot_table(df_target , index='Наименование маршрута', \
                                        values='Фактическое расстояние', aggfunc='mean')

        dpt_target_M_d_kg = pd.pivot_table(df_target , index='Наименование маршрута', \
                                        values=['Сумма весов емкостей из Сортмастер, кг', 'Стоимость расходной части перевозки'], aggfunc='sum')

        dpt_target_M_d_distance.reset_index(inplace=True)
        dpt_target_M_d_kg.reset_index(inplace=True)

        table_target = dpt_target_M_d_distance.merge(dpt_target_M_d_kg)

        df_target_distance =df[df['Наличие автомашины в собственном парке Почты']==1][['Наименование маршрута', 'Фактическое расстояние']]

        df_target_distance =df_target_distance.dropna()

        dpt_target_distance = pd.pivot_table(df_target_distance, index='Наименование маршрута', values='Фактическое расстояние', \
                                            aggfunc='mean').reset_index()

        df_car_velocity = df[df['Наличие автомашины в собственном парке Почты']==1][['Модель ТС', 'Средняя скорость передвижения']]
        df_car_velocity = df_car_velocity.dropna()

        dpt_car_velocity = pd.pivot_table(df_car_velocity , index = 'Модель ТС', values='Средняя скорость передвижения', \
                                            aggfunc='mean').reset_index()

        dpt_car_velocity_real = dpt_cars_M_attrib.merge(dpt_car_velocity)

        df_combined = pd.merge(dpt_target_distance, dpt_car_velocity_real, how='cross')
        df_combined['время тс на маршруте'] = df_combined['Фактическое расстояние']/df_combined['Средняя скорость передвижения']

        return table_cars, table_target, df_combined

    def preprocess_tables(self,table_cars_target_time_edit_m, table_target_moscow_1day, table_cars_moscow):
        table_cars_target_time_edit_m = table_cars_target_time_edit_m[table_cars_target_time_edit_m['Наименование маршрута'].isin(table_target_moscow_1day['Наименование маршрута'])]
        table_cars_target_time_edit_m['время тс на маршруте'] = table_cars_target_time_edit_m['время тс на маршруте'].clip(upper=24)

        vehicles = [{'тип': row['Модель ТС'],
                    'Фактическая грузоподъемность ТС (в кг)': row['Фактическая грузоподъемность ТС (в кг)'],
                    'затраты_на_км': row['затраты на км'],
                    'Расчетный номер машины (из ТМС, 1С УАТ)': row['Расчетный номер машины (из ТМС, 1С УАТ)']} for _, row in table_cars_moscow.iterrows()]

        deliveries = [{'точка': row['Наименование маршрута'],
                    'Сумма весов емкостей из Сортмастер, кг': row['Сумма весов емкостей из Сортмастер, кг'],
                    'Фактическое расстояние': row['Фактическое расстояние']} for _, row in table_target_moscow_1day.iterrows()]
        
        times = {}
        for _, row in table_cars_target_time_edit_m.iterrows():
            key = (row['Наименование маршрута'], row['Модель ТС'])
            value = {'время': row['время тс на маршруте']}
            times[key] = value

        table_target_moscow_1day_results = table_target_moscow_1day.copy()
        table_target_moscow_1day_results['Количество рейсов'] = 0
        table_target_moscow_1day_results['Модель ТС'] = [None] * len(table_target_moscow_1day_results)
        table_target_moscow_1day_results['Стоимость'] = 0

        table_cars_moscow_results = table_cars_moscow.copy()
        table_cars_moscow_results['Количество рейсов'] = 0
        table_cars_moscow_results['Наименование маршрута'] = [None] * len(table_cars_moscow_results)

        return table_cars_target_time_edit_m, vehicles, deliveries, times, table_target_moscow_1day_results, table_cars_moscow_results
    def clean_and_transform_data(self,df, ufps, data_transport):
        filtered_df = df[(df['УФПС начала перевозки'] == ufps) & (df['Дата начала перевозки'] == data_transport)]
        df_cleaned = filtered_df.dropna().copy()

        df_cleaned['Дата начала перевозки локальное время'] = pd.to_datetime(df_cleaned['Дата начала перевозки локальное время'])
        df_cleaned['Дата окончания перевозки (локальное время)'] = pd.to_datetime(df_cleaned['Дата окончания перевозки (локальное время)'])

        df_cleaned['Время маршрута'] = (df_cleaned['Дата окончания перевозки (локальное время)'] - df_cleaned['Дата начала перевозки локальное время']).dt.total_seconds() / 3600
        df_cleaned = df_cleaned.reset_index()
        return df_cleaned