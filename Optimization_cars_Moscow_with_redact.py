import pandas as pd
import numpy as np
import time
from ortools.linear_solver import pywraplp
import scipy
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
from inspect import currentframe, getframeinfo
from pathlib import Path


filename = getframeinfo(currentframe()).filename
parent = Path(filename).resolve().parent
print (parent)
path = parent

# Загружаем ваш файл в переменную `file` / вместо 'example' укажите название свого файла из текущей директории
file = ('data/Исходная_выгрузка_ТМС_python_Excel.xlsx')
# Загружаем spreadsheet в объект pandas
xl = pd.ExcelFile(file)
df = xl.parse('Исходная выгрузка_ТМС_python')

df.columns

calulation_columns = ['Расчетный номер машины (из ТМС, 1С УАТ)', 'Наличие автомашины в собственном парке Почты', \
                      'Сумма весов емкостей из Сортмастер, кг', 'Средняя скорость передвижения', \
'Дата начала перевозки', 'Дата начала перевозки локальное время', 'Дата окончания перевозки (локальное время)','Фактическая грузоподъемность ТС (в кг)', 'УФПС начала перевозки', \
'Наименование маршрута', 'Модель ТС', 'Фактическое расстояние', 'Стоимость расходной части перевозки', \
'Расчет затрат (руб) на (тонна*км) по перевозкам и из ТМС, и из 1с уат']
df.dtypes

df['УФПС начала перевозки'].unique() #['УФПС МОСКОВСКОЙ ОБЛ', 'УФПС МОСКВЫ']

df = df[calulation_columns]
df['затраты на км'] = df['Стоимость расходной части перевозки']/df['Фактическое расстояние']

#оставим в выборке только те маршруты, которые покрываются за 24 часа, т.е. за 1 день.
df['Перевозка одним днем'] = np.where(
    df['Дата начала перевозки локальное время'].dt.day == df['Дата окончания перевозки (локальное время)'].dt.day, 1, 0)


#оценим, что мы вообще оптимизируем, какой процент в костах дадут оптимизируемые маршруты.
#1.маршруты 1 дня, сколько % от общего
#df[df['Перевозка одним днем']==1]['Стоимость расходной части перевозки'].sum()/df['Стоимость расходной части перевозки'].sum() #0.245 ?!!!!
#df[df['Дата начала перевозки локальное время'].isnull()]['Стоимость расходной части перевозки'].sum()/df['Стоимость расходной части перевозки'].sum()

df = df[df['Перевозка одним днем']==1]


# #что в фактическом расстоянии
# df['Фактическое расстояние'].hist(bins=500)
#
# #распредление ам по количеству маршрутов день
# df.groupby('Расчетный номер машины (из ТМС, 1С УАТ)')['Дата начала перевозки'].count().hist()
# #! одна и та же машина учавствует более одного раза, до 15


#распредление маршрутов и транспортных средств по УФПС, по дням. Какая фактическая размерность оптимизируемого вектора?!
# dpt = pd.pivot_table(df, index=['Дата начала перевозки', 'УФПС начала перевозки'], values='Наименование маршрута', aggfunc='count')
#! от 195 до 390 включая

#---------------------------------------------------------------------------------------------------
#vehicles = [
#    {'тип': 'Грузовик A', 'грузоподъемность': 1000, 'стоимость_на_1_км': 5, 'количество': 3},
#    {'тип': 'Грузовик B', 'грузоподъемность': 2000, 'стоимость_на_1_км': 7, 'количество': 2}


def prepare_data(df, ufps, data_transport):
    #условиие на формирование парка
    df = df[(df['УФПС начала перевозки'] == ufps) & (df['Дата начала перевозки'] == data_transport)]



    df_car = df[['Расчетный номер машины (из ТМС, 1С УАТ)', 'Наличие автомашины в собственном парке Почты', \
        'Фактическая грузоподъемность ТС (в кг)', 'УФПС начала перевозки', 'Модель ТС', 'затраты на км']]

    len(df_car) # 9342
    df_car = df_car.dropna()
    len(df_car) # 3401
    df_car.dtypes

    dpt_cars_M = pd.pivot_table(df_car[(df_car['Наличие автомашины в собственном парке Почты']==1) & \
                                       (df_car['УФПС начала перевозки']=='УФПС МОСКВЫ')], \
                                       index=['Модель ТС'], values='Расчетный номер машины (из ТМС, 1С УАТ)', aggfunc='nunique')



    dpt_cars_M_attrib = pd.pivot_table(df_car[(df_car['Наличие автомашины в собственном парке Почты']==1)], \
                                        index=['Модель ТС'], values=['Фактическая грузоподъемность ТС (в кг)', 'затраты на км'], aggfunc='median')

    dpt_cars_M.reset_index(inplace=True)
    dpt_cars_M_attrib.reset_index(inplace=True)

    table_cars = dpt_cars_M_attrib.merge(dpt_cars_M)
    #table_cars.to_pickle(path + 'table_cars_moscow.pkl')
    #---------------------------------------------------------------------------------------------------
    #deliveries = [
    #    {'точка': 'Точка 1', 'вес_груза': 1500, 'расстояние': 50},
    #    {'точка': 'Точка 2', 'вес_груза': 3000, 'расстояние': 70}
    #]

    #todo:добавляем стоимость перевозок (поскольку у нас агрегат из маршрутов)
    df_target = df[['Дата начала перевозки', 'УФПС начала перевозки', 'Наименование маршрута', 'Фактическое расстояние', \
                    'Сумма весов емкостей из Сортмастер, кг', 'Стоимость расходной части перевозки']]
    df_target['Дата начала перевозки'].unique()

    len(df_target) #9342
    df_target = df_target.dropna()
    len(df_target) #3401

    #для примера берем только Москву и 2024-02-05T00:00:00.000000000
    #df_target_M_d = df_target[(df_target['УФПС начала перевозки']=='УФПС МОСКВЫ') & \
    #(df_target['Дата начала перевозки']=='2024-02-05T00:00:00.000000000')] условие уже задано выше! для всего df


    #разброс фактических расстояний для одного маршрута в рамках одного дня
    dpt_target_M_d_distance = pd.pivot_table(df_target , index='Наименование маршрута', \
                                    values='Фактическое расстояние', aggfunc='mean') #aggfunc=['min','max']



    dpt_target_M_d_kg = pd.pivot_table(df_target , index='Наименование маршрута', \
                                    values=['Сумма весов емкостей из Сортмастер, кг', 'Стоимость расходной части перевозки'], aggfunc='sum')


    dpt_target_M_d_distance.reset_index(inplace=True)
    dpt_target_M_d_kg.reset_index(inplace=True)

    table_target = dpt_target_M_d_distance.merge(dpt_target_M_d_kg)

    #table_target.to_pickle(path + 'table_target_moscow_1day.pkl')
    #---------------------------------------------------------------------------------------------------

    df_target_distance =df[df['Наличие автомашины в собственном парке Почты']==1][['Наименование маршрута', 'Фактическое расстояние']]


    df_target_distance =df_target_distance.dropna()

    dpt_target_distance = pd.pivot_table(df_target_distance, index='Наименование маршрута', values='Фактическое расстояние', \
                                         aggfunc='mean').reset_index()

    len(dpt_target_distance) # 1197

    df_car_velocity = df[df['Наличие автомашины в собственном парке Почты']==1][['Модель ТС', 'Средняя скорость передвижения']]
    df_car_velocity = df_car_velocity.dropna()

    dpt_car_velocity = pd.pivot_table(df_car_velocity , index = 'Модель ТС', values='Средняя скорость передвижения', \
                                         aggfunc='mean').reset_index()


    dpt_car_velocity_real = dpt_cars_M_attrib.merge(dpt_car_velocity)
    len(dpt_car_velocity_real) #12


    df_combined = pd.merge(dpt_target_distance, dpt_car_velocity_real, how='cross')
    df_combined['время тс на маршруте'] = df_combined['Фактическое расстояние']/df_combined['Средняя скорость передвижения']

    #df_combined.to_pickle(path + 'table_cars_target_time_edit_m.pkl')

    return table_cars, table_target, df_combined

ufps = 'УФПС МОСКВЫ'
data_transport = '2024-02-05T00:00:00.000000000'


table_cars_moscow, table_target_moscow_1day, table_cars_target_time_edit_m = prepare_data(df, ufps, data_transport)


########-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
table_cars_target_time_edit_m = table_cars_target_time_edit_m[table_cars_target_time_edit_m['Наименование маршрута'].isin(table_target_moscow_1day['Наименование маршрута'])]

#смотрим, что со временем
#table_cars_target_time_edit_m['время тс на маршруте'].describe() #max 191...
table_cars_target_time_edit_m['время тс на маршруте'] = table_cars_target_time_edit_m['время тс на маршруте'].clip(upper=24)
#теперь максимум = 24

#марки машин
len(table_cars_moscow['Модель ТС'].unique()) #12
len(table_cars_target_time_edit_m['Модель ТС'].unique()) #12
#тут Ок!


### формирование словарей
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



#Подготовка данных (для графического представления)
########-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Фильтрация данных
filtered_df = df[(df['УФПС начала перевозки'] == ufps) & (df['Дата начала перевозки'] == data_transport)]
df_cleaned = filtered_df.dropna().copy()


# Преобразуем столбцы 'Дата начала перевозки локальное время' и 'Дата окончания перевозки локальное время' в формат datetime
df_cleaned['Дата начала перевозки локальное время'] = pd.to_datetime(df_cleaned['Дата начала перевозки локальное время'])
df_cleaned['Дата окончания перевозки (локальное время)'] = pd.to_datetime(df_cleaned['Дата окончания перевозки (локальное время)'])

# Вычисляем время маршрута как разницу между 'Дата окончания перевозки (локальное время)' и 'Дата начала перевозки локальное время'
df_cleaned['Время маршрута'] = (df_cleaned['Дата окончания перевозки (локальное время)'] - df_cleaned['Дата начала перевозки локальное время']).dt.total_seconds() / 3600  # переводим в часы
df_cleaned=df_cleaned.reset_index()
time_by_vehicle_type = {}
utilization_by_vehicle_type = {}

# Проходим по каждой строке датафрейма
for index, row in df_cleaned.iterrows():
    vehicle_type = row['Модель ТС']  # Получаем тип машины
    time_ = row['Время маршрута']  # Получаем время маршрута
    # Если тип машины уже есть в словаре, добавляем время к существующей сумме
    if vehicle_type in time_by_vehicle_type:
        time_by_vehicle_type[vehicle_type] += time_
    else:
        time_by_vehicle_type[vehicle_type] = time_  # Иначе создаем новую запись

count_by_model = df_cleaned['Модель ТС'].value_counts()
count_by_model_df = pd.DataFrame({'Модель ТС': count_by_model.index, 'Количество маршрутов': count_by_model.values})
# Рассчитываем процент утилизации для каждой модели машины
for vehicle_type, total_time in time_by_vehicle_type.items():
    total_possible_time = 24 * count_by_model_df[count_by_model_df['Модель ТС'] == vehicle_type]['Количество маршрутов'].sum()# Предположим, что каждая машина может быть использована в течение 24 часов
    utilization_percentage = (total_time / total_possible_time) * 100
    utilization_by_vehicle_type[vehicle_type] = utilization_percentage



# Создаем новый датафрейм на основе полученных данных
summary_df = pd.DataFrame({
    'Модель ТС': utilization_by_vehicle_type.keys(),
    'Процент утилизации': utilization_by_vehicle_type.values(),
    'Количество маршрутов': count_by_model.values,
    'Дата начала перевозки': df_cleaned['Дата начала перевозки'][0]
})

# Выводим результаты

#Оптимимзация главная функция
########-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


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


table_target_moscow_1day_results=table_target_moscow_1day
table_target_moscow_1day_results['Количество рейсов']=0
table_target_moscow_1day_results['Модель ТС']=[None] * len(table_target_moscow_1day_results)
table_target_moscow_1day_results['Стоимость']=0

table_cars_moscow_results=table_cars_moscow
table_cars_moscow_results['Количество рейсов']=0
table_cars_moscow_results['Наименование маршрута']=[None] * len(table_cars_moscow_results)

result_integer = optimize_delivery_integer(vehicles, deliveries, times)
print("Результаты оптимизации:")
print(f"Статус оптимизации: {result_integer['status']}")
print(f"Общая стоимость доставки: {result_integer['objective_value']}")
#summing_testing=0
for (i, j), value in result_integer['x_values'].items():
    if value>0:
        #print(f"Грузовик {vehicles[j]['тип']} отправляется на {deliveries[i]['точка']}, количество рейсов: {value}")
        car_index=table_cars_moscow[(table_cars_moscow['Модель ТС']== vehicles[j]['тип'] )].index[0] 
        delivers_index=table_target_moscow_1day[(table_target_moscow_1day['Наименование маршрута']==deliveries[i]['точка'])].index[0] 
        #summing_testing+=table_cars_moscow.loc[car_index,'затраты на км']*table_target_moscow_1day.loc[delivers_index,'Фактическое расстояние']*value

        table_cars_moscow_results.loc[car_index, 'Количество рейсов'] += value
        if isinstance(table_cars_moscow_results.loc[car_index, 'Наименование маршрута'], list):
            table_cars_moscow_results.loc[car_index, 'Наименование маршрута'].append(deliveries[i]['точка'])
        else:
            table_cars_moscow_results.loc[car_index, 'Наименование маршрута'] = [deliveries[i]['точка']]

        target_index = table_target_moscow_1day_results[
            table_target_moscow_1day_results['Наименование маршрута'] == deliveries[i]['точка']].index[0]

        table_target_moscow_1day_results.loc[target_index, 'Количество рейсов'] += value
        # Получаем текущее значение столбца 'Модель ТС'
        current_value = table_target_moscow_1day_results.loc[target_index, 'Модель ТС']

        # Проверяем, является ли текущее значение списком
        if isinstance(current_value, list):
            # Если значение уже список, добавляем элемент
            current_value.append(vehicles[j]['тип'])
        else:
            # Если значение не список (NoneType), создаем новый список с одним элементом
            table_target_moscow_1day_results.loc[target_index, 'Модель ТС'] = [vehicles[j]['тип']]
        
        table_target_moscow_1day_results.loc[target_index, 'Стоимость'] +=table_cars_moscow.loc[car_index,'затраты на км']*table_target_moscow_1day.loc[delivers_index,'Фактическое расстояние']*value
print(f"Время выполнения: {result_integer['execution_time']} секунд")

########-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Сначала создадим словарь для хранения суммарного времени маршрутов по типам машин
total_time_by_car_type = {}

# Перебираем результаты оптимизации
for (i, j), value in result_integer['x_values'].items():
    if value > 0:
        car_type = vehicles[j]['тип']  # Получаем тип машины
        time_value = times[(deliveries[i]['точка'], car_type)]['время']  # Получаем время маршрута

        # Если тип машины уже есть в словаре, добавляем время маршрута
        if car_type in total_time_by_car_type:
            total_time_by_car_type[car_type] += time_value * value
        else:  # Иначе, создаем новую запись
            total_time_by_car_type[car_type] = time_value * value



# Создаем список для хранения данных
utilization_data = []

# Заполняем список данными
for car_type, total_time in total_time_by_car_type.items():
    total_possible_time = 24 * table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == car_type]['Расчетный номер машины (из ТМС, 1С УАТ)'].sum()
    utilization_percentage = (total_time / total_possible_time) * 100
    used_cars_count = table_cars_moscow_results[table_cars_moscow_results['Модель ТС'] == car_type]['Количество рейсов'].sum()
    
    # Добавляем данные в список
    utilization_data.append({
        'Модель ТС': car_type,
        'Количество используемых машин (Оптимизация)': used_cars_count,
        'Процент утилизации(Оптимизация)': utilization_percentage
    })

# Создаем DataFrame из списка данных
utilization_df = pd.DataFrame(utilization_data)

# Выводим полученный DataFrame
########-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Сравнение двух датафреймов
merged_df = pd.merge(summary_df, utilization_df, on='Модель ТС', how='outer')
print (merged_df)

########-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
table_target_moscow_1day_results.columns

table_target_moscow_1day_results['Стоимость расходной части перевозки'].sum()/table_target_moscow_1day_results['Стоимость'].sum()

#print (sum(table_target_moscow_1day_results['Стоимость']))