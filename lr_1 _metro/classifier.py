import numpy as np
import pandas as pd

class K_NN():
    def __init__(self, dict_stations, rev_dict_stations, coffee, tea, y, k=3):
        self.st = dict_stations.copy()
        self.st_keys = rev_dict_stations.copy()
        self.coffee = coffee
        self.tea = tea
        self.y = y
        self.k = k

    def guess_class(self, x: int, k: int) -> str:
        nearest_k = self.y.sort_values(by=str(x)).head(k)
        nearest_k['coffee_weights'] = nearest_k['coffee'] * 1 / (nearest_k[str(x)] ** 2)
        nearest_k['tea_weights'] = nearest_k['tea'] * 1 / (nearest_k[str(x)] ** 2)
        coffee_coef = nearest_k['coffee_weights'].sum()
        tea_coef = nearest_k['tea_weights'].sum()
        
        if coffee_coef > tea_coef:
            return 'coffee'
        else:
            return 'tea'
    
    def on_station(self, station: str) -> str:
        temp = self.st[station]
        if temp in self.y.all(1):
            if self.y.loc[temp, :]['coffee'] > self.y.loc[temp, :]['tea']:
                str_drink = 'coffee'
            else:
                str_drink = 'tea'
        else:
            str_drink = self.guess_class(temp, self.k)
            
        return 'on "{}" drink {}'.format(self.st_keys[temp], str_drink)

class Station_classifier:
    def __init__(self, adjustment_matrix_file: str, station_names_file: str):
        df_dist = pd.read_csv(adjustment_matrix_file)
        df_names = pd.read_csv(station_names_file)
        stations = np.unique(df_names['name'])

        self.dict_stations = dict(zip(stations, range(stations.shape[0]))) 
        self.rev_dict_stations = dict(zip(range(stations.shape[0]), stations))
        df_dist['coffee'] = df_names['coffee']
        df_dist['tea'] = df_names['tea']
        self.df_tea = df_dist[df_dist['tea'] != 0]
        self.df_coffee = df_dist[df_dist['coffee'] != 0]
        self.df_y = df_dist[(df_dist['tea'] != 0) | (df_dist['coffee'] != 0)]

    def classify(self, station: str, n_neighbors: int = 3) -> str:
        knn = K_NN(self.dict_stations, self.rev_dict_stations, self.df_coffee, self.df_tea, self.df_y, n_neighbors)
        return knn.on_station(' ' + station)


# if __name__ == '__main__':
#     classifier = Station_classifier('distance_data.csv', 'station_list.csv')
#     print(classifier.classify('Замоскворецкая линия Алма-Атинская'))