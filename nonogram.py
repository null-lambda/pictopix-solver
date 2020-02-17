from itertools import accumulate, chain, groupby
import time


class MultisetNode:
    def __init__(self, data, next_nodes):
        self.index = None
        self.data = data
        self.next_nodes = next_nodes
        self.parent_nodes = set()

    def __repr__(self):
        return f'Node({self.data}, {len(self.next_nodes)})'

    def connected(self):
        return bool(self.next_nodes) or bool(self.parent_nodes)

    def biconnected(self):
        return bool(self.next_nodes) and bool(self.parent_nodes)

    def reversed_paths(self):
        # print(self)
        if not self.next_nodes:
            yield [self]
            return
        for node in self.next_nodes:
            for subpath in node.reversed_paths():
                subpath.append(self)
                yield subpath

    def paths(self):
        for path in self.reversed_paths():
            yield path[::-1]


# generate non-cyclic directed graph of k-multiset from {0, ..., n}
# with start node and end node
class MultisetGraph:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.end_node = MultisetNode(n, set())

        def layers_reversed(k):
            if k == 0:
                return []
            if k == 1:
                return [[MultisetNode(i, set([self.end_node])) for i in range(n + 1)]]
            subgraph = layers_reversed(k - 1)
            layer = [MultisetNode(i, set(subgraph[-1][i: n + 1]))
                     for i in range(n + 1)]
            subgraph.append(layer)
            return subgraph
        layers = layers_reversed(k)[::-1]
        if layers:
            self.start_node = MultisetNode(0, set(layers[0]))
        else:
            self.start_node = MultisetNode(0, set([self.end_node]))

        self.start_node.index = -1
        for i, layer in enumerate(layers):
            for node in layer:
                node.index = i
        self.end_node.index = n
        self.nodes = list(chain.from_iterable(layers))

        for node in chain([self.start_node], self.nodes):
            for subnode in node.next_nodes:
                subnode.parent_nodes.add(node)

    def paths(self):
        return self.start_node.paths()

    @property
    def edges(self):
        for node in chain([self.start_node], self.nodes):
            for subnode in node.next_nodes:
                yield (node, subnode)

    def dissolve_node(self, node):
        s = set([node])
        while s:
            n = s.pop()
            if n == self.start_node or n == self.end_node:
                continue
            for subnode in n.next_nodes:
                subnode.parent_nodes.remove(n)
                if not subnode.biconnected():
                    s.add(subnode)
            for subnode in n.parent_nodes:
                subnode.next_nodes.remove(n)
                if not subnode.biconnected():
                    s.add(subnode)
            n.parent_nodes = set()
            n.next_nodes = set()

    def dissolve_edge(self, node_parent, node_child):
        node_parent.next_nodes.discard(node_child)
        node_child.parent_nodes.discard(node_parent)
        if not node_parent.biconnected():
            self.dissolve_node(node_parent)
        if not node_child.biconnected():
            self.dissolve_node(node_child)

    def clean_dissolved_nodes(self):
        self.nodes = [node for node in self.nodes if node.connected()]


class Nonogram:
    def __init__(self, clues, *, callback=None):
        self.clues = clues
        self.callback = callback

    @staticmethod
    def _transpose(grid):
        return list(zip(*grid))

    @staticmethod
    def _gen_line_candidates(clue, line_length):
        n, k = line_length, len(clue)
        n_filled = sum(clue)
        n_gap = k - 1
        n_free = n - n_filled - n_gap

        idx_base = [0] + list(accumulate(c + 1 for c in clue[:-1]))
        graph = MultisetGraph(n_free, k)
        for node in graph.nodes:
            idx_shift = node.data
            idx_start = idx_base[node.index] + idx_shift
            idx_end = idx_start + clue[node.index]
            node.data = (idx_start, idx_end)  # (start i)
        graph.start_node.data = (0, 0)
        graph.end_node.data = (n, n)

        return graph

    @staticmethod
    def _zip_graph(cands, line_length):
        base_path = next(cands.paths())
        base = [0] * line_length
        for node in base_path:
            (idx_start, idx_end) = node.data
            for j in range(idx_start, idx_end):
                base[j] = 1

        # search '1' cells
        for node in cands.nodes:
            (idx_start, idx_end) = node.data
            for j in range(idx_start, idx_end):
                if base[j] == 0:
                    base[j] = -1
        # search '0' cells
        for node1, node2 in cands.edges:
            (_, idx_start) = node1.data
            (idx_end, _) = node2.data
            for j in range(idx_start, idx_end):
                if base[j] == 1:
                    base[j] = -1

        return base

    @staticmethod
    def _filter_cands(cands, line):
        # filter '1' nodes
        for node in cands.nodes:
            (idx_start, idx_end) = node.data
            if any(line[j] == 0 for j in range(idx_start, idx_end)):
                cands.dissolve_node(node)
        # search '0' nodes
        for node1, node2 in list(cands.edges):
            if not node1.connected():
                continue
            (_, idx_start) = node1.data
            (idx_end, _) = node2.data
            if any(line[j] == 1 for j in range(idx_start, idx_end)):
                cands.dissolve_edge(node1, node2)
        cands.clean_dissolved_nodes()

    def solve(self):
        (clues_v, clues_h) = self.clues
        c, r = len(clues_v), len(clues_h)

        grid = [-1 for _ in range(r * c)]
        rows = [[j * c + i for i in range(c)] for j in range(r)]
        cols = [[j * c + i for j in range(r)] for i in range(c)]
        lines = rows + cols
        clues = clues_h + clues_v
        cands = []
        for clue, line in zip(clues, lines):
            graph = self._gen_line_candidates(clue, len(line))
            cands.append((graph, line))
        for graph, line in cands:
            for idx, cell_value in zip(line, self._zip_graph(graph, len(line))):
                grid[idx] = cell_value

        fixed, grid_prev = False, None
        while not fixed:
            for graph, line in cands:
                cells = [grid[idx] for idx in line]
                self._filter_cands(graph, cells)
                cells_updated = self._zip_graph(graph, len(line))
                for idx, cell_value in zip(line, cells_updated):
                    grid[idx] = cell_value
                if self.callback:
                    self.callback(call_location='line_update',
                                  grid=grid, rows=rows, cols=cols, line=line)
            if self.callback:
                self.callback(call_location='grid_update', grid=grid, rows=rows, cols=cols)
            fixed = grid_prev == grid
            grid_prev = grid[:]
        finished = all(cell != -1 for cell in grid)
        if not finished:
            print('undetermined solution')
            return [[grid[idx] for idx in row] for row in rows]

        valid = True
        for clue, line in zip(clues, lines):
            segments = []
            for k, g in groupby(enumerate(line), key=lambda i: grid[i[1]]):
                if k == 1:
                    g = list(g)
                    start, end = g[0][0], g[-1][0]
                    segments.append(end - start + 1)
            if len(segments) == 0:
                segments = [0]
            valid = valid and (clue == tuple(segments))
            print(clue, segments)
        if not valid:
            print('invalid solution')
            return None
        return [[grid[idx] for idx in row] for row in rows]


def print_grid(grid):
    cell_to_char = {-1: '_ ', 0: '0 ', 1: '1 '}
    for row in grid:
        print(''.join(cell_to_char[i] for i in row))
    print()


def test():
    def test_cases():
        v_clues = ((1, 1, 3), (3, 2, 1, 3), (2, 2), (3, 6, 3),
                   (3, 8, 2), (15,), (8, 5), (15,),
                   (7, 1, 4, 2), (7, 9,), (6, 4, 2,), (2, 1, 5, 4),
                   (6, 4), (2, 6), (2, 5), (5, 2, 1),
                   (6, 1), (3, 1), (1, 4, 2, 1), (2, 2, 2, 2))
        h_clues = ((2, 1, 1), (3, 4, 2), (4, 4, 2), (8, 3),
                   (7, 2, 2), (7, 5), (9, 4), (8, 2, 3),
                   (7, 1, 1), (6, 2), (5, 3), (3, 6, 3),
                   (2, 9, 2), (1, 8), (1, 6, 1), (3, 1, 6),
                   (5, 5), (1, 3, 8), (1, 2, 6, 1), (1, 1, 1, 3, 2))
        yield (v_clues, h_clues)
        v_clues = ((2, 2), (1, 1, 2, 1), (1, 2, 1, 1, 1), (1, 2, 1, 1, 2),
                   (2, 1, 2, 3), (2, 1, 3, 2), (1, 3, 1, 1, 1, 1),
                   (1, 1, 1, 2, 2, 1), (2, 1, 5, 2, 1), (2, 1, 1, 2),
                   (1, 1), (2, 2), (1, 1), (1, 1), (1, 1))
        h_clues = ((1, 1, 2), (1, 1, 1, 1, 2), (2, 3, 1, 1), (2, 2, 1), (1, 1, 2),
                   (1, 2, 1, 4), (1, 4), (5,), (1, 5), (2, 2, 1, 4), (1, 2, 2),
                   (1, 1, 1, 2), (2, 1, 2, 1), (2, 1, 1, 1), (1, 2))
        yield (v_clues, h_clues)

    def callback(**kwargs):
        if kwargs['call_location'] == 'grid_update':
            grid, rows = kwargs['grid'], kwargs['rows']
            print_grid([[grid[idx] for idx in row] for row in rows])
            time.sleep(0.1)

    for args in test_cases():
        grid = Nonogram(args, callback=callback).solve()
        print_grid(grid)


if __name__ == "__main__":
    test()
