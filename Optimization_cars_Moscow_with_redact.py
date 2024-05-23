import pandas as pd
import numpy as np
import time
from ortools.linear_solver import pywraplp
import scipy
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
from inspect import currentframe, getframeinfo
from pathlib import Path
def load_and_prepare_data():
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


def prepare_data(df, ufps, data_transport):
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

def preprocess_tables(table_cars_target_time_edit_m, table_target_moscow_1day, table_cars_moscow):
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


def optimize_delivery_integer(vehicles, deliveries, times):
    solver = pywraplp.Solver.CreateSolver('SCIP')

    num_deliveries = len(deliveries)
    num_vehicles = len(vehicles)

    x = np.array([[solver.IntVar(0, solver.infinity(), f'x[{i},{j}]') for j in range(num_vehicles)] for i in range(num_deliveries)])

    # Create the cost matrix using the provided data
    cost_matrix = np.array([[vehicles[j]['затраты_на_км'] * delivery['Фактическое расстояние'] for j in range(num_vehicles)] for delivery in deliveries])

    solver.Minimize(solver.Sum(x[i, j] * cost_matrix[i, j] for i in range(num_deliveries) for j in range(num_vehicles)))

    for i in range(num_deliveries):
        solver.Add(solver.Sum(x[i, j] * vehicles[j]['Фактическая грузоподъемность ТС (в кг)'] for j in range(num_vehicles)) >= deliveries[i]['Сумма весов емкостей из Сортмастер, кг'])

    for j in range(num_vehicles):
        for i in range(num_deliveries):              #ставится коментарий тут для отключения считания с временем
            # Access the time for this vehicle and delivery point pair from the 'times' table
            time_value = times[(deliveries[i]['точка'], vehicles[j]['тип'])]['время']                      #ставится коментарий тут для отключения считания с временем
            solver.Add(x[i, j] * time_value <= 24)               #ставится коментарий тут для отключения считания с временем
            #print (time_value)

        solver.Add(solver.Sum(x[i, j] for i in range(num_deliveries)) <= vehicles[j]['Расчетный номер машины (из ТМС, 1С УАТ)'])

    start_time = time.time()  # Засекаем время начала оптимизации
    status = solver.Solve()
    execution_time = time.time() - start_time  # Вычисляем время выполнения оптимизации

    result = {
        'status': status,
        'objective_value': solver.Objective().Value(),
        'x_values': {(i, j): x[i, j].solution_value() for j in range(num_vehicles) for i in range(num_deliveries)},
        'execution_time': execution_time  # Добавляем время выполнения в результат
    }

    return result

def calculate_total_time_by_car_type(result_integer, vehicles, deliveries, times):
    total_time_by_car_type = {}
    for (i, j), value in result_integer['x_values'].items():
        if value > 0:
            car_type = vehicles[j]['тип']
            time_value = times[(deliveries[i]['точка'], car_type)]['время']

            if car_type in total_time_by_car_type:
                total_time_by_car_type[car_type] += time_value * value
            else:
                total_time_by_car_type[car_type] = time_value * value
    return total_time_by_car_type

def create_utilization_data(total_time_by_car_type, table_cars_moscow_results):
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

def clean_and_transform_data(df, ufps, data_transport):
    filtered_df = df[(df['УФПС начала перевозки'] == ufps) & (df['Дата начала перевозки'] == data_transport)]
    df_cleaned = filtered_df.dropna().copy()

    df_cleaned['Дата начала перевозки локальное время'] = pd.to_datetime(df_cleaned['Дата начала перевозки локальное время'])
    df_cleaned['Дата окончания перевозки (локальное время)'] = pd.to_datetime(df_cleaned['Дата окончания перевозки (локальное время)'])

    df_cleaned['Время маршрута'] = (df_cleaned['Дата окончания перевозки (локальное время)'] - df_cleaned['Дата начала перевозки локальное время']).dt.total_seconds() / 3600
    df_cleaned = df_cleaned.reset_index()
    return df_cleaned

def update_tables(result_integer, table_cars_moscow, table_target_moscow_1day, table_cars_moscow_results, table_target_moscow_1day_results, deliveries, vehicles):
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

def calculate_time_by_vehicle_type(df_cleaned):
    time_by_vehicle_type = {}

    for index, row in df_cleaned.iterrows():
        vehicle_type = row['Модель ТС']
        time_ = row['Время маршрута']

        if vehicle_type in time_by_vehicle_type:
            time_by_vehicle_type[vehicle_type] += time_
        else:
            time_by_vehicle_type[vehicle_type] = time_

    return time_by_vehicle_type

def calculate_utilization_by_vehicle_type(time_by_vehicle_type, count_by_model_df):
    utilization_by_vehicle_type = {}

    for vehicle_type, total_time in time_by_vehicle_type.items():
        total_possible_time = 24 * count_by_model_df[count_by_model_df['Модель ТС'] == vehicle_type]['Количество маршрутов'].sum()
        utilization_percentage = (total_time / total_possible_time) * 100
        utilization_by_vehicle_type[vehicle_type] = utilization_percentage

    return utilization_by_vehicle_type

def create_summary_df(utilization_by_vehicle_type, count_by_model, df_cleaned):
    summary_df = pd.DataFrame({
        'Модель ТС': utilization_by_vehicle_type.keys(),
        'Процент утилизации': utilization_by_vehicle_type.values(),
        'Количество маршрутов': count_by_model.values,
        'Дата начала перевозки': df_cleaned['Дата начала перевозки'][0]
    })

    return summary_df

def merge_dataframes(summary_df, utilization_df):
    merged_df = pd.merge(summary_df, utilization_df, on='Модель ТС', how='outer')
    return merged_df

def main():
    df = load_and_prepare_data()
    ufps = 'УФПС МОСКВЫ'
    data_transport = '2024-02-05T00:00:00.000000000'

    table_cars_moscow, table_target_moscow_1day, table_cars_target_time_edit_m = prepare_data(df, ufps, data_transport)
    table_cars_target_time_edit_m, vehicles, deliveries, times, table_target_moscow_1day_results, table_cars_moscow_results = preprocess_tables(table_cars_target_time_edit_m, table_target_moscow_1day, table_cars_moscow)

    result_integer = optimize_delivery_integer(vehicles, deliveries, times)
    print("Результаты оптимизации:")
    print(f"Статус оптимизации: {result_integer['status']}")
    print(f"Общая стоимость доставки: {result_integer['objective_value']}")

    update_tables(result_integer, table_cars_moscow, table_target_moscow_1day, table_cars_moscow_results, table_target_moscow_1day_results, deliveries, vehicles)

    total_time_by_car_type = calculate_total_time_by_car_type(result_integer, vehicles, deliveries, times)
    utilization_data = create_utilization_data(total_time_by_car_type, table_cars_moscow_results)
    utilization_df = pd.DataFrame(utilization_data)

    df_cleaned = clean_and_transform_data(df, ufps, data_transport)
    time_by_vehicle_type = calculate_time_by_vehicle_type(df_cleaned)
    count_by_model = df_cleaned['Модель ТС'].value_counts()
    count_by_model_df = pd.DataFrame({'Модель ТС': count_by_model.index, 'Количество маршрутов': count_by_model.values})

    utilization_by_vehicle_type = calculate_utilization_by_vehicle_type(time_by_vehicle_type, count_by_model_df)
    summary_df = create_summary_df(utilization_by_vehicle_type, count_by_model, df_cleaned)

    merged_df = merge_dataframes(summary_df, utilization_df)
    print(merged_df)

if __name__ == "__main__":
    main()


########-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#table_target_moscow_1day_results.columns

#table_target_moscow_1day_results['Стоимость расходной части перевозки'].sum()/table_target_moscow_1day_results['Стоимость'].sum()

#print (sum(table_target_moscow_1day_results['Стоимость'])



#dfsidfh[aosdghfpuagsduipfgaisdgfiagsu]