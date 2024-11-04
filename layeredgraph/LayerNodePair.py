from layeredgraph import LayerNode

class LayerNodePair:
    def __init__(self, source: LayerNode, destination: LayerNode):
        self._source = source
        self._destination = destination

    def to_string(self):
        return f"{self._source.to_string()}->{self._destination.to_string()}"
    
    def get_source(self):
        return self._source
    
    def get_destination(self):
        return self._destination
    
    def is_same_layer(self):
        return self._source.is_same_layer(self._destination)
    
    def is_same_node(self):
        return self._source.is_same_node(self._destination)
    
    def is_same_layer_node(self):
        return self._source.is_same_layer_node(self._destination)
    
    def __hash__(self):
        return hash(self.to_string())
    
    def __str__(self):
        return self.to_string()
    
    def __repr__(self):
        return self.to_string()

    def __eq__(self, other):
        return self.to_string() == other.to_string()

    def __ne__(self, other):
        return not(self == other)
    
    def __lt__(self, other):
        return self.to_string() < other.to_string()