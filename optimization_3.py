import pandas as pd
import numpy as np
import time
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from pathlib import Path

class Optimizer:
    def optimize_delivery_integer(self, vehicles, deliveries, times, calculation):
        solver = pywraplp.Solver.CreateSolver('SCIP')

        num_deliveries = len(deliveries)
        num_vehicles = len(vehicles)

        x = np.array([[solver.IntVar(0, solver.infinity(), f'x[{i},{j}]') for j in range(num_vehicles)] for i in range(num_deliveries)])

        cost_matrix = np.array([[vehicles[j]['затраты_на_км'] * delivery['Расстояние по маршруту Факт,км'] for j in range(num_vehicles)] for delivery in deliveries])

        solver.Minimize(solver.Sum(x[i, j] * cost_matrix[i, j] for i in range(num_deliveries) for j in range(num_vehicles)))

        if calculation == 'mass':
            weight_col = 'Фактическая грузоподъемность, кг'
            sum_col = 'Суммарный вес по маршруту, кг'
        elif calculation == 'volume':
            weight_col = 'Фактический объем кузова, м3'
            sum_col = 'Суммарный объем грузовых ед.,м3'
        else:
            raise ValueError("Invalid calculation type. Use 'mass' or 'volume'.")

        for i in range(num_deliveries):
            solver.Add(solver.Sum(x[i, j] * vehicles[j][weight_col] for j in range(num_vehicles)) >= deliveries[i][sum_col])

        for j in range(num_vehicles):
            for i in range(num_deliveries):
                time_value = times[(deliveries[i]['точка'], vehicles[j]['тип'])]['время']
                solver.Add(x[i, j] * time_value <= 24)

            solver.Add(solver.Sum(x[i, j] for i in range(num_deliveries)) <= vehicles[j]['Номер ТС'])

        start_time = time.time()
        status = solver.Solve()
        execution_time = time.time() - start_time

        result = {
            'status': status,
            'objective_value': solver.Objective().Value(),
            'x_values': {(i, j): x[i, j].solution_value() for j in range(num_vehicles) for i in range(num_deliveries)},
            'execution_time': execution_time
        }

        return result

    def calculate_total_time_by_car_type(self, result_integer, vehicles, deliveries, times):
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

    def update_tables(self, result_integer, table_cars_moscow, table_target_moscow_1day, table_cars_moscow_results, table_target_moscow_1day_results, deliveries, vehicles):
        for (i, j), value in result_integer['x_values'].items():
            if value > 0:
                car_index = table_cars_moscow[(table_cars_moscow['Модель ТС'] == vehicles[j]['тип'])].index[0] 
                delivers_index = table_target_moscow_1day[(table_target_moscow_1day['Маршрут'] == deliveries[i]['точка'])].index[0] 
                
                table_cars_moscow_results.loc[car_index, 'Количество рейсов'] += value
                if isinstance(table_cars_moscow_results.loc[car_index, 'Маршрут'], list):
                    table_cars_moscow_results.loc[car_index, 'Маршрут'].append(deliveries[i]['точка'])
                else:
                    table_cars_moscow_results.loc[car_index, 'Маршрут'] = [deliveries[i]['точка']]

                target_index = table_target_moscow_1day_results[table_target_moscow_1day_results['Маршрут'] == deliveries[i]['точка']].index[0]

                table_target_moscow_1day_results.loc[target_index, 'Количество рейсов'] += value

                current_value = table_target_moscow_1day_results.loc[target_index, 'Модель ТС']

                if isinstance(current_value, list):
                    current_value.append(vehicles[j]['тип'])
                else:
                    table_target_moscow_1day_results.loc[target_index, 'Модель ТС'] = [vehicles[j]['тип']]
                
                table_target_moscow_1day_results.loc[target_index, 'Стоимость'] += table_cars_moscow.loc[car_index,'затраты на км'] * table_target_moscow_1day.loc[delivers_index,'Расстояние по маршруту Факт,км'] * value

        print(f"Время выполнения: {result_integer['execution_time']} секунд")

    # Additional methods can be added here
