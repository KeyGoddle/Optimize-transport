from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import pandas as pd

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    ufps='УФПС МОСКВОВСКОЙ ОБЛАСТИ'
    data["time_matrices"] = [
        [  # Vehicle 0
            [0, 6, 7],# 8, 7, 3, 6, 2, 3, 2, 6, 6, 4, 4, 5, 9, 7], # depot
            [6, 0, 7],# 3, 2, 6, 8, 4, 8, 8, 13, 7, 5, 8, 12, 10, 14], # 1
            [7, 6, 0],# 11, 10, 6, 3, 9, 5, 8, 4, 15, 14, 13, 9, 18, 9], # 2
        ],
        [  # Vehicle 1
            [0, 5, 8],# 7, 6, 2, 5, 1, 2, 1, 5, 5, 3, 3, 4, 8, 6], # depot
            [5, 0, 8],# 2, 1, 5, 7, 3, 7, 7, 12, 6, 4, 7, 11, 9, 13], # 1
            [8, 5, 0],# 10, 9, 5, 2, 8, 4, 7, 3, 14, 13, 12, 8, 17, 8], # 2
        ],
        [  # Vehicle 2
            [0, 7, 9], #9, 8, 4, 7, 3, 4, 3, 7, 7, 5, 5, 6, 10, 8], # depot
            [7, 0, 9],# 4, 3, 7, 9, 5, 9, 9, 14, 8, 6, 9, 13, 11, 15], # 1
            [9, 7, 0],# 12, 11, 7, 4, 10, 6, 9, 5, 16, 15, 14, 10, 19, 10], # 2
        ]
    ]
    data["distance_matrix"] = [
        [0, 4, 7],# 6, 5, 1, 4, 0, 1, 0, 4, 4, 2, 2, 3, 7, 5], # depot
        [4, 0, 7],# 1, 0, 4, 6, 2, 6, 6, 11, 5, 3, 6, 10, 8, 12], # 1
        [7, 4, 0],# 9, 8, 4, 1, 7, 3, 6, 2, 13, 12, 11, 7, 16, 7], # 2
    ]
    
    data["Summ_weight"] = [0, 6, 3]   #для depot, скока доставить в другие
    data["time_windows"] = [
        (3, 4),  # depot
        (5, 11),  # 1
        (16, 24),  # 2
    ]
    """
    РАзобраться почему routing.AddDimension(
            transit_callback_index,
            24,  #разрешенное время прибытия в узел раньше левой границы временного окна
            31,  #максимальное допустимое значение времени 
            False,  
            time,
        )
    """
    data["num_vehicles"] = 3
    data["vehicle_costs"] = [400, 300, 500]
    data["carrying_capacity"] = [2, 6, 3]
    data["depot"] = 0
    data["vehicle_names"] = ["LADA", "MERCEDES", "FORD"]  # Имена машин
    #data["route_names"] = ["Route A", "Route B", "Route C"]  # Именя маршрутов #


    time_data = []
    for vehicle_id, vehicle_matrix in enumerate(data["time_matrices"]):
        for from_node, times in enumerate(vehicle_matrix):
            for to_node, time in enumerate(times):
                time_data.append([data["vehicle_names"][vehicle_id], to_node, time])

    vehicle_df = pd.DataFrame(time_data, columns=["Vehicle", "Маршрут", "Время в часах на маршрут"])
    carrying_cap=pd.DataFrame({'Vehicle':data["vehicle_names"],
                               'Грузоподъемность': data["carrying_capacity"],
                               'Стоимость машины на километр':data["vehicle_costs"] })
    vehicle_df = vehicle_df.merge(carrying_cap, on='Vehicle', how='left')
    vehicle_df=vehicle_df[vehicle_df['Время в часах на маршрут']!=0]
    vehicle_df=vehicle_df[vehicle_df['Маршрут']!=0]
    vehicle_df=vehicle_df.drop_duplicates()
    #vehicle_df = vehicle_df[vehicle_df['From_Node'] != vehicle_df['To_Node']]
    #print (merged_df)
    #vehicle_df['Грузоподъемоность']= data["carrying_capacity"]

    # Создание DataFrame для матрицы расстояний
    distance_data = []
    for from_node, distances in enumerate(data["distance_matrix"]):
        for to_node, distance in enumerate(distances):
            distance_data.append([ to_node, distance])
    
    distance_df = pd.DataFrame(distance_data, columns=[ "Маршрут", "Расстояние"])

    marshrut_cap=pd.DataFrame({'Маршрут':[0,1,2],
                               'Сумма весов для доставки': data["Summ_weight"],
                                })
    distance_df = distance_df.merge(marshrut_cap, on='Маршрут', how='left')

    distance_df=distance_df[distance_df['Маршрут']!=0]
    distance_df=distance_df[distance_df['Расстояние']!=0]
    distance_df=distance_df.drop_duplicates()
    # Создание DataFrame для остальной информации
    additional_data = {
        "Маршрут": [0,1,2],
        "Временные окна": data["time_windows"],

    }

    additional_df = pd.DataFrame(additional_data)
    additional_df=additional_df[additional_df['Маршрут']!=0]
    print("Venicle DataFrame:")
    print(vehicle_df)
    print("\nМарршрут DataFrame:")
    print(distance_df)
    print("\n Временные окна")
    print(additional_df)
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    time_dimension = routing.GetDimensionOrDie("Time")
    distance_matrix = data["distance_matrix"]
    total_cost = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id} ({data['vehicle_names'][vehicle_id]} ):\n"
        route_time = 0
        route_distance = 0
        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            from_node = manager.IndexToNode(index)
            to_node = manager.IndexToNode(next_index)
            time_var = time_dimension.CumulVar(index)
            plan_output += (
                f"{manager.IndexToNode(index)}"
                f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
                " -> "
            )
            route_time = solution.Min(time_var)
            if to_node != 0:
                route_distance += distance_matrix[from_node][to_node]
            index = solution.Value(routing.NextVar(index))

        if " -> " in plan_output:
            plan_output = plan_output[:-4]  # Remove the last "->"
        time_var = time_dimension.CumulVar(index)
        if "->" in plan_output:
            route_time -= data["time_matrices"][vehicle_id][manager.IndexToNode(index)][data["depot"]]

        plan_output += (
            f"\nTime of the route: {route_time} часов\n"
            f"Cost of the route: {route_distance * data['vehicle_costs'][vehicle_id]}\n"
        )
        print(plan_output)
        total_cost += route_distance * data["vehicle_costs"][vehicle_id]
    print(f"Total cost of all routes: {total_cost}")
def print_solution_in_table(data, manager, routing, solution):
    """Prints solution as a DataFrame."""
    routes_data = {"Route": [], "Time": [], "Distance": [], "Cost": [], "Vehicle": [],  "Time Window": []}
    time_dimension = routing.GetDimensionOrDie("Time")
    distance_matrix = data["distance_matrix"]
    total_cost = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_info = {"nodes": [], "time": 0, "distance": 0, "load": 0, "time_window": ""}
        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            from_node = manager.IndexToNode(index)
            to_node = manager.IndexToNode(next_index)
            time_var = time_dimension.CumulVar(index)
            route_info["nodes"].append(from_node)
            route_info["time"] = solution.Min(time_var)
            if to_node != 0:
                route_info["distance"] += distance_matrix[from_node][to_node]
#                route_info["load"] += data["demands"][to_node]
            index = next_index
        route_cost = route_info["distance"] * data["vehicle_costs"][vehicle_id]
        if route_cost != 0:
            route_info["time"] -= data["time_matrices"][vehicle_id][route_info["nodes"][-1]][data["depot"]]
            route_info["time_window"] = f"({solution.Min(time_var)},{solution.Max(time_var)})"
            
            routes_data["Route"].append(route_info["nodes"][1])
            routes_data["Time"].append(route_info["time"])
            routes_data["Distance"].append(route_info["distance"])
            routes_data["Cost"].append(route_cost)
            routes_data["Vehicle"].append(data["vehicle_names"][vehicle_id])
#            routes_data["Load"].append(route_info["load"])
            routes_data["Time Window"].append(route_info["time_window"])
            total_cost += route_cost

    df_routes = pd.DataFrame(routes_data)
    
    print("Routes:")
    print(df_routes)
    print(f"Total cost of all routes: {total_cost}")
def main():
    """Solve the VRP with time windows."""
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(
        len(data["time_matrices"][0]), data["num_vehicles"], data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index, vehicle_id):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        return data["time_matrices"][vehicle_id][from_node][to_node]

    for vehicle_id in range(data["num_vehicles"]):
        transit_callback_index = routing.RegisterTransitCallback(
            lambda from_index, to_index, vehicle_id=vehicle_id: time_callback(from_index, to_index, vehicle_id)
        )
        routing.SetArcCostEvaluatorOfVehicle(transit_callback_index, vehicle_id)

        time = "Time"
        routing.AddDimension(
            transit_callback_index,
            24,  #разрешенное время прибытия в узел раньше левой границы временного окна
            31,  #максимальное допустимое значение времени 
            False,  
            time,
        )
        time_dimension = routing.GetDimensionOrDie(time)
        for location_idx, time_window in enumerate(data["time_windows"]):
            if location_idx == data["depot"]:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        depot_idx = data["depot"]
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data["time_windows"][depot_idx][0], data["time_windows"][depot_idx][1]
        )

    for vehicle_id in range(data["num_vehicles"]):
        vehicle_cost_per_distance = data["vehicle_costs"][vehicle_id]

        def vehicle_cost_callback(from_index, to_index, vehicle_cost_per_distance=vehicle_cost_per_distance):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            distance = data["distance_matrix"][from_node][to_node]
            return distance * vehicle_cost_per_distance
        cost_callback_index = routing.RegisterTransitCallback(vehicle_cost_callback)
        routing.SetArcCostEvaluatorOfVehicle(cost_callback_index, vehicle_id)
    
    

    def weight_callback(to_index):
        to_node = manager.IndexToNode(to_index)

        return data["Summ_weight"][to_node]  #[vehicle_id]  #при добавлении вес можно будет разделить, но тогда одна машина не сможет выполнить несколько маршрутов

    weight_callback_index = routing.RegisterUnaryTransitCallback(weight_callback)
    routing.AddDimensionWithVehicleCapacity(
        weight_callback_index,
        0,  
        data["carrying_capacity"],  
        True,  
        "Capacity"
    )

    # for i in range(data["num_vehicles"]):
    #     routing.AddVariableMinimizedByFinalizer(
    #         time_dimension.CumulVar(routing.Start(i))
    #     )
    #     routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print_solution(data, manager, routing, solution)
        print_solution_in_table(data, manager, routing, solution)
if __name__ == "__main__":
    main()

