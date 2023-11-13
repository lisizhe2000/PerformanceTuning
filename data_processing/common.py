import re
import xml.etree.ElementTree as ET
import pandas as pd
from data_processing.measurement_tree import MeasurementTree
from pysat.formula import CNF
from pysat.solvers import Minisat22


def singleton(class_):
    instances = {}
    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return get_instance


@singleton
class Common(object):

    # 在第一次实例化时，需要加载对应数据
    def __init__(self) -> None:       
        self.sys_name = None

        self.num_options = 0
        self.all_performances = []

        # dimacs way
        self.num_constraints = 0
        self.option_to_id = {}
        self.id_to_option = {}
        self.constraints = []

        # using SAT solver
        self.cnf = None
        self.solver = None
        self.sat_config_iter = None

        # for xml
        self.performance_col_name = None
        self.measurement_tree = MeasurementTree()

        # for configs pool
        self.configs_pool = None


    def __parse_dimacs(self) -> None:
        prefix = './Data/FM/'
        subfix = '.dimacs'
        path = prefix + self.sys_name + subfix
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                tokens = line.split()
                if tokens[0] == 'c':
                    id = int(tokens[1])
                    name = tokens[2]
                    self.option_to_id[name] = id
                    self.id_to_option[id] = name
                elif tokens[0] == 'p':
                    self.num_options = int(tokens[2])
                    self.num_constraints = int(tokens[3])
                else:
                    self.constraints.append(tokens[:-1])
                line = f.readline()             


    def __parse_measurements(self) -> None:
        prefix = './Data/AllMeasurements/'
        subfix = '_raw_bin.xml'
        path = prefix + self.sys_name + subfix
        tree = ET.parse(path)
        results = tree.getroot()

        for row in results:

            config_options_list = [False] * self.num_options
            config_options_list[0] = True   # 第一个节点为root，必选，但Measurements文件中没有标识出来
            performance = 0

            for data in row:
                if data.attrib['column'] == 'Configuration':
                    config = data.text
                    config_options = re.sub('\s+', '', config).split(',')
                    for option in config_options:
                        id = self.option_to_id[option] - 1
                        config_options_list[id] = True
                elif data.attrib['column'] == self.performance_col_name:
                    performance = float(data.text)
                    self.all_performances.append(performance)

            self.measurement_tree.make_measurement_node(config_options_list, performance)


    def load_xml(self, sys_name: str, performance_col_name: str) -> None:
        self.sys_name = sys_name
        self.performance_col_name = performance_col_name
        self.__parse_dimacs()
        self.__parse_measurements()
        dimacs_path = './Data/FM/' + self.sys_name + '.dimacs'
        self.cnf = CNF(from_file=dimacs_path)
        self.solver = Minisat22(bootstrap_with=self.cnf)
        self.sat_config_iter = self.solver.enum_models()

    
    def load_csv(self, sys_name: str) -> None:
        self.sys_name = sys_name
        from util.config import Config
        self.configs_pool: list[Config] = []
        self.all_performances = []
        csv_path = './Data/CsvMeasurements/' + sys_name + '.csv'
        df = pd.read_csv(csv_path)
        
        self.num_options = len(df.columns) - 1

        for _, row in df.iterrows():
            row = row.tolist()
            # config_options = [True if option == 1 else False for option in row[:-1]]
            config_options = row[:-1]
            performance = row[-1]
            config = Config(config_options, performance=performance)
            self.all_performances.append(performance)
            self.configs_pool.append(config)
