import pandas as pd
import numpy as np
import time
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from pathlib import Path
class Optimizer:
    def optimize_delivery_integer(self,vehicles, deliveries, times):
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
    def calculate_total_time_by_car_type(self,result_integer, vehicles, deliveries, times):
        total_time_by_car_type = {}
        for (i, j), value in result_integer['x_values'].items():
            if value > 0:
                car_type = vehicles[j]['тип']
                time_value = times[(deliveries[i]['точка'], car_type)]['время']

                if car_type in total_time_by_car_type:
                    total_time_by_car_type[car_type] += time_value * value
                else:
                    total_time_by_car_type[car_type] = time_value * value
        #print (total_time_by_car_type)
        return total_time_by_car_type
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
    #def calculate_utilization_by_vehicle_type(self):
        #...