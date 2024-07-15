import pandas as pd
import numpy as np
from pathlib import Path

class DataProcessor:
    def load_and_prepare_data(self):
        filename = Path(__file__).resolve().parent / 'data/new_data/Выгрузка_МР_Москва_05_02_11_02_2024_250624_+_расходы_без_АХО.xlsx'
        xl = pd.ExcelFile(filename)
        
        sheet_name = 'Отчет'
        
        df = xl.parse(sheet_name, skiprows=1)
        calculation_columns = ['Номер ТС', 'Исполнитель',
                               'Суммарный вес по маршруту, кг', 'Плановая средняя скорость передвижения',
                               'Даты начала перевозки по маршруту',
                               'Даты окончания перевозки по маршруту по плану', 'Фактическая грузоподъемность, кг',
                               'УФПС (Инициатор перевозки)', 'Маршрут', 'Модель ТС', 'Расстояние по маршруту Факт,км',
                               'Стоимость перевозки факт (ТМС)', 'Стоимость перевозки план (ТМС)',
                               'Фактический объем кузова, м3', 'Суммарный объем грузовых ед.,м3']
        df = df[calculation_columns]
        df['затраты на км'] = df['Стоимость перевозки факт (ТМС)'] / df['Расстояние по маршруту Факт,км']
        df['Даты начала перевозки по маршруту'] = pd.to_datetime(df['Даты начала перевозки по маршруту'], dayfirst=True)
        df['Даты окончания перевозки по маршруту по плану'] = pd.to_datetime(df['Даты окончания перевозки по маршруту по плану'], dayfirst=True)

        df['Перевозка одним днем'] = np.where(
            pd.to_datetime(df['Даты начала перевозки по маршруту']).dt.normalize() == pd.to_datetime(df['Даты окончания перевозки по маршруту по плану']).dt.normalize(), 1, 0)
        df = df[df['Перевозка одним днем'] == 1]
        
        return df

    def prepare_data(self, df, ufps, data_transport, naym, all_car, calculation):
        df = df[(df['УФПС (Инициатор перевозки)'] == ufps) & 
                (pd.to_datetime(df['Даты начала перевозки по маршруту']).dt.normalize() == pd.to_datetime(data_transport).normalize())]
        
        if calculation == 'mass':
            weight_col = 'Фактическая грузоподъемность, кг'
            sum_col = 'Суммарный вес по маршруту, кг'
        elif calculation == 'volume':
            weight_col = 'Фактический объем кузова, м3'
            sum_col = 'Суммарный объем грузовых ед.,м3'
        else:
            raise ValueError("Invalid calculation type. Use 'mass' or 'volume'.")

        df_car = df[['Номер ТС', 'Исполнитель', weight_col, 'УФПС (Инициатор перевозки)', 'Модель ТС', 'затраты на км']]
        print (df_car['Модель ТС'])
        df_car = df_car.dropna()

        dpt_cars_M = pd.pivot_table(df_car[(df_car['Исполнитель'] == 'Почта России')], 
                                    index=['Модель ТС'], values='Номер ТС', aggfunc='nunique')

        dpt_cars_M_attrib = pd.pivot_table(df_car[(df_car['Исполнитель'] == 'Почта России')], 
                                           index=['Модель ТС'], values=[weight_col, 'затраты на км'], aggfunc='median')

        dpt_cars_M.reset_index(inplace=True)
        dpt_cars_M_attrib.reset_index(inplace=True)

        table_cars = dpt_cars_M_attrib.merge(dpt_cars_M, on='Модель ТС')
        table_cars = table_cars[table_cars['Номер ТС'] > 1]

        if all_car == 1:
            def get_cars_repair(ufps):
                file = Path(__file__).resolve().parent / 'data/new_data/Cобсвтенный_автопарт_05_02_2024_без_АХО.xlsx'
                xl = pd.ExcelFile(file)
                df = xl.parse('TDSheet', skiprows=1)

                col = ['Организация', 'Гос. номер', 'Модель', 'Категория ТС',
                       'Сегмент грузоподъёмности ТС', 'Статус доступности']
                df = df[col]

                df['Организация'].replace('УФПС Московской области', 'УФПС МОСКОВСКОЙ ОБЛ', inplace=True)
                df['Организация'].replace('УФПС г. Москвы', 'УФПС МОСКВЫ', inplace=True)

                df = df[df['Организация'] == ufps]
                df = df[df['Категория ТС'] == 'Производственно почтовый']
                df = df[df['Статус доступности'] == 'Доступен']

                df_pt = pd.pivot_table(df, index='Модель', values='Гос. номер', aggfunc='nunique')
                df_pt.reset_index(inplace=True)
                df_pt['Модель'].replace('ГАЗ 2705', 'ГАЗ 27057', inplace=True)

                return df_pt

            df_cars_repair = get_cars_repair(ufps)
            table_cars = pd.merge(table_cars, df_cars_repair, how='left', left_on='Модель ТС', right_on='Модель')

            table_cars['Гос. номер'] = table_cars.apply(lambda col: col['Гос. номер'] if col['Гос. номер'] > col['Номер ТС'] else col['Номер ТС'], axis=1)
            table_cars.drop(['Номер ТС', 'Модель'], axis=1, inplace=True)
            table_cars.rename(columns={'Гос. номер': 'Номер ТС'}, inplace=True)

        if naym == 1:
            def get_cars_pension():
                file = Path(__file__).resolve().parent / 'data/new_data/Выгрузка_МР_Москва_05_02_11_02_2024_250624_+_расходы_без_АХО.xlsx'
                xl = pd.ExcelFile(file)
                sheet_name = 'Отчет'
        
                df = xl.parse(sheet_name, skiprows=1)
                col = ['Номер ТС', weight_col, 'Стоимость перевозки факт (ТМС)', 'Маршрут', 'Расстояние по маршруту Факт,км',
                       'Даты начала перевозки по маршруту', 'Даты окончания перевозки по маршруту по плану', 'Исполнитель']
                df = df[col]

                df = df[df['Исполнитель'] != 'Почта России']
                df = df[df['Номер ТС'].notna()]
                df = df[df['Стоимость перевозки факт (ТМС)'].notna()]

                df['Даты начала перевозки по маршруту'] = pd.to_datetime(df['Даты начала перевозки по маршруту'], dayfirst=True)
                df['Даты окончания перевозки по маршруту по плану'] = pd.to_datetime(df['Даты окончания перевозки по маршруту по плану'], dayfirst=True)

                df['затраты на км'] = df['Стоимость перевозки факт (ТМС)'] / df['Расстояние по маршруту Факт,км']
                df['Плановая средняя скорость передвижения'] = df[weight_col] / \
                    ((df['Даты окончания перевозки по маршруту по плану'] - df['Даты начала перевозки по маршруту']).dt.total_seconds() / 3600)

                col = ['Номер ТС', weight_col, 'затраты на км', 'Плановая средняя скорость передвижения']
                df = df[col]
                df.dropna(inplace=True)

                dpt_car_pension = pd.pivot_table(df, index=weight_col, values=['затраты на км', 'Плановая средняя скорость передвижения'], aggfunc='mean')
                dpt_car_pension.reset_index(inplace=True)
                dpt_car_pension['Модель ТС'] = dpt_car_pension[weight_col].apply(lambda row: 'Наемник - ' + str(round(row)))

                dpt_car_pension.rename(columns={weight_col: weight_col}, inplace=True)
                dpt_car_pension['Номер ТС'] = 100

                return dpt_car_pension

            table_cars_naym = get_cars_pension()
            table_cars = pd.concat([table_cars, table_cars_naym], ignore_index=True)

        df_target = df[['Даты начала перевозки по маршруту', 'УФПС (Инициатор перевозки)', 'Маршрут', 'Расстояние по маршруту Факт,км', 
                        sum_col, 'Стоимость перевозки факт (ТМС)']]
        df_target = df_target.dropna()

        dpt_target_M_d_distance = pd.pivot_table(df_target, index='Маршрут', values='Расстояние по маршруту Факт,км', aggfunc='mean')
        dpt_target_M_d_kg = pd.pivot_table(df_target, index='Маршрут', values=[sum_col, 'Стоимость перевозки факт (ТМС)'], aggfunc='sum')

        dpt_target_M_d_distance.reset_index(inplace=True)
        dpt_target_M_d_kg.reset_index(inplace=True)

        table_target = dpt_target_M_d_distance.merge(dpt_target_M_d_kg, on='Маршрут')

        if naym == 1:
            def get_target_pension(ufps, data_transport):
                file = Path(__file__).resolve().parent / 'data/new_data/Выгрузка_МР_Москва_05_02_11_02_2024_250624_+_расходы_без_АХО.xlsx'
                xl = pd.ExcelFile(file)
                sheet_name = 'Отчет'
        
                df = xl.parse(sheet_name, skiprows=1)
                col = ['Номер ТС', 'УФПС (Инициатор перевозки)', 'Даты начала перевозки по маршруту', 
                       'Даты окончания перевозки по маршруту по плану', 'Расстояние по маршруту Факт,км', 'Маршрут', 
                       'Стоимость перевозки факт (ТМС)', sum_col]

                df = df[df['УФПС (Инициатор перевозки)'] == ufps]
                df = df[pd.to_datetime(df['Даты начала перевозки по маршруту']).dt.date == pd.to_datetime(data_transport).date()]
                df = df[col]
                df = df[~df['Стоимость перевозки факт (ТМС)'].isna()]
                df = df[df['Стоимость перевозки факт (ТМС)'] > 0]
                df = df[~df[sum_col].isna()]

                df['Перевозка одним днем'] = np.where(
                    pd.to_datetime(df['Даты начала перевозки по маршруту']).dt.day == pd.to_datetime(df['Даты окончания перевозки по маршруту по плану']).dt.day, 1, 0)

                df = df[df['Перевозка одним днем'] == 1]

                col_table_target_moscow_1day = ['Маршрут', 'Расстояние по маршруту Факт,км', 'Стоимость перевозки факт (ТМС)', sum_col]
                df = df[col_table_target_moscow_1day]

                return df

            table_target_naym = get_target_pension(ufps, data_transport)
            table_target = pd.concat([table_target, table_target_naym], ignore_index=True)

        df_target_distance = df[df['Исполнитель'] == 'Почта России'][['Маршрут', 'Расстояние по маршруту Факт,км']]
        df_target_distance = df_target_distance.dropna()

        dpt_target_distance = pd.pivot_table(df_target_distance, index='Маршрут', values='Расстояние по маршруту Факт,км', aggfunc='mean').reset_index()

        df_car_velocity = df[df['Исполнитель'] == 'Почта России'][['Модель ТС', 'Плановая средняя скорость передвижения']]
        df_car_velocity = df_car_velocity.dropna()

        dpt_car_velocity = pd.pivot_table(df_car_velocity, index='Модель ТС', values='Плановая средняя скорость передвижения', aggfunc='mean').reset_index()
        dpt_car_velocity_real = dpt_cars_M_attrib.merge(dpt_car_velocity, on='Модель ТС')

        if naym == 1:
            dpt_target_distance = table_target[['Маршрут', 'Расстояние по маршруту Факт,км']]
            table_cars_naym_average = get_cars_pension()
            table_cars_naym_average = table_cars_naym_average[['Модель ТС', weight_col, 'затраты на км', 'Плановая средняя скорость передвижения']]
            dpt_car_velocity_real = pd.concat([dpt_car_velocity_real, table_cars_naym_average], ignore_index=True)

        df_combined = pd.merge(dpt_target_distance, dpt_car_velocity_real, how='cross')
        df_combined['время тс на маршруте'] = df_combined['Расстояние по маршруту Факт,км'] / df_combined['Плановая средняя скорость передвижения']
        table_cars = table_cars.loc[table_cars['затраты на км'] != 0]
        return table_cars, table_target, df_combined

    def preprocess_tables(self, table_cars_target_time_edit_m, table_target_moscow_1day, table_cars_moscow, calculation):
        if calculation == 'mass':
            sum_col = 'Суммарный вес по маршруту, кг'
            weight_col = 'Фактическая грузоподъемность, кг'
        elif calculation == 'volume':
            sum_col = 'Суммарный объем грузовых ед.,м3'
            weight_col = 'Фактический объем кузова, м3'
        else:
            raise ValueError("Invalid calculation type. Use 'mass' or 'volume'.")

        table_cars_target_time_edit_m = table_cars_target_time_edit_m[table_cars_target_time_edit_m['Маршрут'].isin(table_target_moscow_1day['Маршрут'])]
        table_cars_target_time_edit_m['время тс на маршруте'] = table_cars_target_time_edit_m['время тс на маршруте'].clip(upper=24)

        vehicles = [{'тип': row['Модель ТС'],
                     weight_col: row[weight_col],
                     'затраты_на_км': row['затраты на км'],
                     'Номер ТС': row['Номер ТС']} for _, row in table_cars_moscow.iterrows()]

        deliveries = [{'точка': row['Маршрут'],
                       sum_col: row[sum_col],
                       'Расстояние по маршруту Факт,км': row['Расстояние по маршруту Факт,км']} for _, row in table_target_moscow_1day.iterrows()]

        times = {}
        for _, row in table_cars_target_time_edit_m.iterrows():
            key = (row['Маршрут'], row['Модель ТС'])
            value = {'время': row['время тс на маршруте']}
            times[key] = value

        table_target_moscow_1day_results = table_target_moscow_1day.copy()
        table_target_moscow_1day_results['Количество рейсов'] = 0
        table_target_moscow_1day_results['Модель ТС'] = [None] * len(table_target_moscow_1day_results)
        table_target_moscow_1day_results['Стоимость'] = 0

        table_cars_moscow_results = table_cars_moscow.copy()
        table_cars_moscow_results['Количество рейсов'] = 0
        table_cars_moscow_results['Маршрут'] = [None] * len(table_cars_moscow_results)

        return table_cars_target_time_edit_m, vehicles, deliveries, times, table_target_moscow_1day_results, table_cars_moscow_results


    def clean_and_transform_data(self, df, ufps, data_transport):
        filtered_df = df[(df['УФПС (Инициатор перевозки)'] == ufps) & 
                         (pd.to_datetime(df['Даты начала перевозки по маршруту']).dt.normalize() == pd.to_datetime(data_transport).normalize())]

        df_cleaned = filtered_df.dropna().copy()

        df_cleaned['Даты начала перевозки по маршруту'] = pd.to_datetime(df_cleaned['Даты начала перевозки по маршруту'])
        df_cleaned['Даты окончания перевозки по маршруту по плану'] = pd.to_datetime(df_cleaned['Даты окончания перевозки по маршруту по плану'])

        df_cleaned['Время маршрута'] = (df_cleaned['Даты окончания перевозки по маршруту по плану'] - df_cleaned['Даты начала перевозки по маршруту']).dt.total_seconds() / 3600
        df_cleaned = df_cleaned.reset_index()
        
        return df_cleaned


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    ufps = 'УФПС МОСКВЫ'
    data_transport = '2024-02-11'
    calculation ='mass' #''  # or volume
    df = processor.load_and_prepare_data()
    table_cars, table_target, df_combined = processor.prepare_data(df, ufps, data_transport, naym=1, all_car=1, calculation=calculation)

    table_cars = table_cars.loc[table_cars['затраты на км'] != 0]
    table_cars_target_time_edit_m, vehicles, deliveries, times, table_target_moscow_1day_results, table_cars_moscow_results = processor.preprocess_tables(df_combined, table_target, table_cars,calculation=calculation)

    df_cleaned = processor.clean_and_transform_data(df, ufps, data_transport)
    print(table_cars)
