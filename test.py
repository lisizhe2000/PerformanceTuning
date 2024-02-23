from functools import reduce

import xgboost as xgb
from testing.test_ml import TestML
from util.config import Config
from data_processing.common import Common
from util.distance_util import DistanceUtil
from util.expr_running_util import ExprRunningUtil
from sampling.init_samples import InitSampling
from sampling.map_elites import Feature, MapElites
from data_processing.measurement_tree import MeasurementTree
from sampling.sail import Sail
from util.indicators_util import IndicatorsUtil


class Test():

    def __init__(self):
        self.data = Common()
        # self.data.load_data('HSMGP', 'AverageTimePerIteration')
        # self.data.load_xml('Dune', 'Performance')
        self.data.load_csv('JavaGC_num', chunk_reading=True)
    
    def test_parse_dimacs(self):
        print('numOptions: ', self.data.num_options)
        print('numConstraint: ', self.data.num_constraints)
        print('option_to_id: ', self.data.option_to_id)
        print('id_to_option: ', self.data.id_to_option)
        print('constraints: ', self.data.constraints)
       
    def test_parse_measurements(self):
        options = [1,2,3,5,9,26,10,18,27,13,24]
        options_b = [False] * self.data.num_options
        for option in options:
            options_b[option - 1] = True      
        performance = self.data.measurement_tree.get_performance(options_b)  
        print(performance)

    def test_make_node(self):
        tree = MeasurementTree()
        tree.make_measurement_node([False, True], 1.02)
        tree.make_measurement_node([True, True], 3.02)
        tree.make_measurement_node([True, False], 4.02)
        tree.make_measurement_node([False, False], 7.02)
        print(tree.get_performance([False, False]))
        print(tree.get_performance([False, True]))
        print(tree.get_performance([True, False]))
        print(tree.get_performance([True, True]))

    def test_get_init_samples(self):
        sail = Sail()
        configs = sail.get_init_samples()
        for i in range(len(configs)):
            print(configs[i].get_selected_options_names(), configs[i].get_real_performance())

    def test_get_feature_index(self):
        num_options = Common().num_options
        feature_num_selected = Feature(0, num_options + 1, num_options + 1, lambda configs: reduce(lambda x, y: x+y, configs.config_options))
        for num_selected in range(num_options + 1):
            conf_options = [False] * num_options
            for i in range(num_selected):
                conf_options[i] = True
            config = Config(conf_options)
            index = feature_num_selected.get_index(config)
            print('num_selected: {}, feature_index:{}'.format(num_selected, index))

    def test_crossover(self):
        # TODO
        pass

    def test_update_archive(self):
        # TODO
        pass

    def test_gen_random_config(self):
        for i in range(5):
            print(Config.gen_random_config().to_bin_str())

    def test_gen_next_sat_config(self):
        for _ in range(5):
            print(Config.get_next_sat_config().to_bin_str())

    def test_init_model(self) -> xgb.Booster:
        sail = Sail()
        model = sail.init_model()
        # print(sail.X)
        # print(sail.y)
        # print(sail.X.shape)
        # print(sail.y.shape)
        y_pred = model.predict(xgb.DMatrix(sail.X))
        # print(y_pred)
        # print('diff: ', sail.y - y_pred)
        # print('error rate: ', abs(sail.y - y_pred) / sail.y)
        return model

    def test_map_init(self, model):
        map_elites = MapElites(model)
        map_elites.init()
        print(map_elites.archive)

    def test_map_elites(self, model):
        mapelits = MapElites(model)
        config = mapelits.search_configs(model, 1000)
        print('config = {}, performance = {}'.format(config.to_bin_str, config.get_or_eval_real_performance()))

    def test_sail(self):
        sail = Sail()
        config = sail.search_optimal_config()
        print(f'Rank: {IndicatorsUtil.get_rank(config)}, config: {config.get_selected_options_names()}, performance: {config.get_real_performance()}, num of all config: {len(Common().all_performances)}')
        with open('./res', 'a') as f:
            f.write(f'Rank: {IndicatorsUtil.get_rank(config)}, config: {config.get_selected_options_names()}, performance: {config.get_real_performance()}\n')

    def test_load_csv(self):
        self.data.load_csv('Apache')
        print(f'len configs pool: {len(Common().configs_pool)}')
        print([config.to_bin_str() for config in Common().configs_pool])
        print([config.get_real_performance() for config in Common().configs_pool])

    def test_hamming_distance(self):
        config_a = Common().configs_pool[0]
        config_b = Common().configs_pool[1]
        print(config_a.to_bin_str())
        print(config_b.to_bin_str())
        distance = DistanceUtil.hamming_distance(config_a, config_b)
        print(distance)
        assert distance == 2

    def test_fscs(self):
        samples = InitSampling.fscs(5)
        print([config.to_bin_str() for config in samples])

    def test_get_rank(self):
        rank = ExprRunningUtil.get_performance_rank(13.0827868231411)
        print(f'rank: {rank}')
        
    def test_read_csv(self):
        print(f'len configs pool: {len(Common().configs_pool)}')
        print([config.to_bin_str() for config in Common().configs_pool[-10:]])
        print([config.get_real_performance() for config in Common().configs_pool[-10:]])
        

if __name__ == '__main__':
    test = Test()

    # test.test_parse_dimacs()
    # test.test_parse_measurements()
    # test.test_get_init_samples()
    # test.test_get_feature_index()
    # test.test_gen_random_config()
    # test.test_gen_next_sat_config()
    
    # print('rank:{}, num of all config:{}'.format(len(Common().all_performances), ExprUtil.get_rank(37938.39285714286)))
    # model = test.test_init_model()
    # test.test_map_init(model)
    # test.test_map_elites(model)

    # test.test_sail()
    # test.test_load_csv()
    # test.test_hamming_distance()
    # test.test_fscs()
    # test.test_get_rank()
    # test.test_read_csv()


    # test_ml = TestML()
    # test_ml.test_cart()    
