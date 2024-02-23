import re
import numpy as np
import xgboost as xgb
from util.config import Config
from data_processing.common import Common
from sampling.map_elites import MapElites


class Sail(object):

    def __init__(self) -> None:
        # Common().load_data('HiPAcc')
        self.rounds_acquisition = 25
        self.rounds_map_elites = 80
        self.X: np.ndarray = None
        self.y: np.ndarray = None

    def get_init_samples(self) -> list[Config]:
        prefix = './Data/InitSamples/'
        subfix = '.dimacs.samples'
        path = prefix + Common().sys_name + subfix

        configs = []

        with open(path, 'r') as f:
            line = f.readline()
            while line:
                tokens = re.sub('\s+', '', line).split(';')
                config_list = []
                for option in tokens:
                    o = True if option == '1' else False
                    config_list.append(o)
                config = Config(config_list)

                try:
                    config.get_real_performance()
                    configs.append(config)
                except:
                    config_names = ''
                    for i in range(len(config_list)):
                        if config_list[i]:
                            config_names += Common().id_to_option[i + 1]
                            config_names += ','
                    config_names = config_names[:-1]
                    print('Error when getting performance of config: ', config_names)

                line = f.readline()
        
        return configs

    def init_model(self) -> xgb.Booster:
        configs = self.get_init_samples()
        self.X = np.array([config.config_options for config in configs])
        self.y = np.array([config.get_real_performance() for config in configs])
        return self.train_model()

    def search_optimal_config(self) -> Config:

        print('Sail: Initializing prediction model......')
        model = self.init_model()

        # 通过xgboost的save_model方法，可以实现incremental training
        # 但是xgboost似乎不适合incremental training，后续可以再找找别的模型看看
        
        best: Config = None

        # predict phase
        print('Sail: Start prediction phase......')
        for i in range(self.rounds_acquisition):
            print('Sail: {}th MAP-Elites......'.format(i))
            map_elites = MapElites(model)
            try:
                config = map_elites.search_configs(self.rounds_map_elites)
                performance = config.get_real_performance()
                if best == None or performance > best.get_real_performance():
                    best = config
                self.X = np.append(self.X, [config.config_options], axis=0)
                self.y = np.append(self.y, performance)
                model = self.train_model()
            except:
                print('Error evaluating performance, config: {}'.format(None if config == None else config.get_selected_options_names()))

        # last iteration
        print('Sail: Start optimization phase')
        try:
            config = map_elites.search_configs(self.rounds_map_elites)
            performance = config.get_real_performance()
            if best == None or performance > best.get_real_performance():
                best = config
        except:
            print('Error evaluating performance, config: {}'.format(None if best == None else best.get_selected_options_names()))
        
        return best

    def train_model(self) -> xgb.Booster:
        print('Sail: Training model......')
        dtrain = xgb.DMatrix(self.X, label=self.y)
        num_round = 10
        param = {
            'max_depth': 2,
            'eta': 1,
            'objective': 'reg:squarederror'
        }
        model = xgb.train(param, dtrain, num_round)
        return model
