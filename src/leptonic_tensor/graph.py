import numpy as np
# import pylatex
import itertools


class Graph:
    def __init__(self, graph=None):
        if graph is None:
            graph = {}
        self.graph = graph

    @property
    def nodes(self):
        return list(self.graph.keys())

    @property
    def external_nodes(self):
        external = []
        for node in list(self.graph.keys()):
            if self.degree(node) == 1:
                external.append(node)
        return external

    @property
    def edges(self):
        edges = []
        for node in self.graph:
            for neighbor in self.graph[node]:
                if (neighbor, node) not in edges:
                    edges.append((node, neighbor))
        return edges

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def n_external(self):
        return len(self.external_nodes)

    @property
    def adjacency_matrix(self):
        raise NotImplementedError

    def degree(self, node):
        adjacent_nodes = self.graph[node]
        degree = len(adjacent_nodes) + adjacent_nodes.count(node)
        return degree

    @property
    def degree_sequence(self):
        sequence = []
        for node in self.graph:
            sequence.append(self.degree(node))
        sequence.sort(reverse=True)
        return sequence

    @staticmethod
    def is_degree_sequence(sequence):
        return all(x >= y for x, y in zip(sequence, sequence[1:]))

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def add_edge(self, edge):
        edge = set(edge)
        (node1, node2) = tuple(edge)
        if node1 in self.graph:
            self.graph[node1].append(node2)
        else:
            self.graph[node1] = [node2]
        if node2 in self.graph:
            self.graph[node2].append(node1)
        else:
            self.graph[node2] = [node1]

    def is_connected(self, nodes_encountered=None, start_node=None):
        if nodes_encountered is None:
            nodes_encountered = set()
        graph = self.graph
        nodes = list(graph.keys())
        if not start_node:
            start_node = nodes[0]
        nodes_encountered.add(start_node)
        if len(nodes_encountered) != len(nodes):
            for node in graph[node]:
                if node not in nodes_encountered:
                    if self.is_connected(nodes_encountered, node):
                        return True
        else:
            return True
        return False

    def __str__(self):
        result = 'Nodes: {}\n'.format(self.nodes)
        result += 'Edges: {}'.format(self.edges)
        return result

    def __repr__(self):
        return f'Graph({self.graph})'

#    def draw(self, doc):
#        doc.append(pylatex.NoEscape('\\feynmandiagram[large]{\n'))
#        for edge in self.edges:
#            doc.append(pylatex.NoEscape(f'a{edge[0]} -- a{edge[1]},\n'))
#        doc.append(pylatex.NoEscape('};\n'))


def PruferToTree(a):
    n = len(a)
    g = {}
    for i in range(n+2):
        g[i] = []
    g = Graph(g)
    degree = [1]*(n+2)
    for node in a:
        degree[node] += 1
    for node1 in a:
        for node2 in g.nodes:
            if degree[node2] == 1:
                g.add_edge([node1, node2])
                degree[node1] -= 1
                degree[node2] -= 1
                break
    u = v = 0
    for node in g.nodes:
        if degree[node] == 1:
            if u == 0:
                u = node
            else:
                v = node
                break
    g.add_edge([u, v])
    return g


def check_tree(g):
    for (_, degree) in graph.degree():
        if degree == 2 or degree > 4:
            return False
    return True


def generate_trees(order):
    """ Based on the WROM algorithm """
    # Create tree rooted at primary root
    layout = list(range(order // 2 + 1)) + list(range(1, (order + 1) // 2))

    while layout is not None:
        layout = _get_next(layout)
        if layout is not None:
            allowed, graph = _check_allowed(layout)
            if allowed:
                yield graph
        layout = _successor(layout)


def _successor(layout, p=None):
    if p is None:
        p = len(layout) - 1
        while layout[p] == 1:
            p -= 1
    if p == 0:
        return None

    q = p - 1
    while layout[q] != layout[p] - 1:
        q -= 1
    next_layout = list(layout)
    for i in range(p, len(next_layout)):
        next_layout[i] = next_layout[i - p + q]
    return next_layout


def _get_next(layout):
    # Use WROM algorithm
    # Condition A always true, so start with condition B
    left, right = _split(layout)
    mleft = max(left)
    mright = max(right)
    valid = mleft <= mright

    if valid and mleft == mright:
        if len(left) > len(right):
            valid = False
        elif len(left) == len(right) and left > right:
            valid = False

    if valid:
        return layout
    p = len(left)
    next_layout = _successor(layout, p)
    if layout[p] > 2:
        left, right = _split(next_layout)
        left_height = max(left)
        suffix = range(1, left_height + 2)
        next_layout[-len(suffix):] = suffix
    return next_layout


def _split(layout):
    found = False
    m = None
    for i in range(len(layout)):
        if layout[i] == 1:
            if found:
                m = i
                break
            else:
                found = True

    if m is None:
        m = len(layout)

    left = [layout[i] - 1 for i in range(1, m)]
    right = [0] + [layout[i] for i in range(m, len(layout))]
    return left, right


def _check_allowed(layout):
    graph = to_graph(layout)
    for degree in graph.degree_sequence:
        if degree == 2 or degree > 4:
            return False, None
    return True, graph


def to_graph(layout):
    result = [[0]*len(layout) for i in range(len(layout))]
    graph = Graph()
    stack = []
    for i in range(len(layout)):
        graph.add_node(i)
        i_level = layout[i]
        if stack:
            j = stack[-1]
            j_level = layout[j]
            while j_level >= i_level:
                stack.pop()
                j = stack[-1]
                j_level = layout[j]
            graph.add_edge((i, j))
        stack.append(i)
    return graph


if __name__ == '__main__':
    # doc = pylatex.Document('basic')
    # doc.packages.append(pylatex.Package(pylatex.NoEscape('tikz-feynman')))
    n_external = 6
    orders = []
    for n3 in range(0, n_external-1):
        for n4 in range(0, n_external//2):
            if n3+2*n4 == n_external-2:
                orders.append(n3+n4+n_external)
    print(set(orders))
    ngraphs = 0
    for order in set(orders):
        graphs = generate_trees(order)
        for graph in graphs:
            if n_external == graph.n_external:
                print(graph)
                print(f'Degree Sequence: {graph.degree_sequence}')
                ngraphs += 1
                # graph.draw(doc)
                # doc.append(pylatex.NoEscape('\n'))

    print(ngraphs)
    # doc.generate_pdf(clean_tex=False)
    # doc.generate_tex()
