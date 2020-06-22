class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.adjacency_matrix = None

    # Modify topology
    def insert_edge(self, source, target):
        pass

    def erase_edge(self, edgeid):
        pass

    def erase_edges(self, node0, node1):
        pass

    # Node and Edge properties
    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_external_nodes(self):
        pass

    @property
    def external_nodes(self):
        pass

    def is_external_node(self, node):
        pass

    def adjacent_edges(self, node):
        pass

    def adjacent_nodes(self, edge):
        pass

    def adjacency(self, source, target):
        pass

    # Topology properties
    @property
    def n_loops(self):
        return self.n_edges - self.n_nodes - 1

    @property
    def is_connected(self):
        pass

    @property
    def is_one_particle_irreducible(self):
        pass

    @property
    def has_tadpoles(self):
        pass

    @property
    def has_self_energies(self):
        pass

    @property
    def is_on_shell(self):
        pass

    # Output information
    def __str__(self):
        pass

    def save(self, filename):
        pass

    # Private helper functions
