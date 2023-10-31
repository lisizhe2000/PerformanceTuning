class Node(object):
    def __init__(self) -> None:
        pass




class NonLeafNode(Node):


    def __init__(self) -> None:
        self.l = None   # option selected
        self.r = None   # option not selected


    def get_child(self, option_selected: bool) -> Node:
        return self.l if not option_selected else self.r




class LeafNode(Node):
    def __init__(self) -> None:
        self.performance = 0




class MeasurementTree(object):


    def __init__(self) -> None:
        self.root = NonLeafNode()


    def make_measurement_node(self, config_options: list, performance: float) -> None:
        node = self.root
        for i in range(len(config_options)):
            option_selected = config_options[i]
            child = node.get_child(option_selected)
            if child == None:
                child = NonLeafNode() if i < len(config_options) - 1 else LeafNode() 
                if not option_selected:
                    node.l = child
                else:
                    node.r = child
            node = child
        node.performance = performance


    def get_performance(self, config_options: list) -> float:
        node = self.root
        for option in config_options:
            node = node.get_child(option)
        return node.performance
