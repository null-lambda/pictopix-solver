from functools import reduce
from itertools import accumulate, chain
from typing import List, Tuple
from collections import deque

Line = List[int]
Grid = List[Line]
Clue = Tuple[int]

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
        #print(self)
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
            layer = [MultisetNode(i, set(subgraph[-1][i: n + 1])) for i in range(n + 1)]
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
    def __init__(self, clues: Tuple[List[Clue], List[Clue]]):
        self.clues = clues 

    @staticmethod 
    def _transpose(grid):
        return list(zip(*grid))

    @staticmethod
    def _gen_line_candidates(clue: Clue, line: Line) -> [Line]:
        n, k = len(line), len(clue)
        n_filled = sum(clue)
        n_gap = k - 1
        n_free = n - n_filled - n_gap

        idx_base = [0] + list(accumulate(c + 1 for c in clue[:-1]))
        graph = MultisetGraph(n_free, k)
        for node in graph.nodes:
            idx_shift = node.data
            idx_start = idx_base[node.index] + idx_shift
            idx_end =  idx_start + clue[node.index]
            node.data = (idx_start, idx_end) # (start i)
        graph.start_node.data = (0, 0)
        graph.end_node.data = (n, n)

        return graph

    @staticmethod 
    def _zip_cands(cands, line_length):
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

        """ def merge_ranges(intervals):
            if not intervals:
                return 
            intervals = sorted(intervals)
            current_start, current_end = intervals[0]
            for (start, end) in intervals[1:]:
                if start < current_end:
                    current_end = max(end, current_end)
                else:
                    yield (current_start, current_end)
                    (current_start, current_end) = (start, end)
            yield (current_start, current_end)

        # search '1' cells
        interval_1s = []
        for node in cands.nodes:
            (idx_start, idx_end) = node.data 
            interval_1s.append((idx_start, idx_end))
        for idx_start, idx_end in merge_ranges(interval_1s):
            for j in range(idx_start, idx_end):
                if base[j] == 0:
                    base[j] = -1

        # search '0' cells
        interval_0s = []
        for node1, node2 in cands.edges:
            (_, idx_start) = node1.data
            (idx_end, _) = node2.data  
            interval_0s.append((idx_start, idx_end))
        for idx_start, idx_end in merge_ranges(interval_0s):
            for j in range(idx_start, idx_end):
                if base[j] == 1:
                    base[j] = -1 """
        
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

        return cands

    @staticmethod 
    def _finished(grid: [Line]) -> bool:
        return all(x != -1 for row in grid for x in row)

    def solve(self):
        (clues_v, clues_h) = self.clues 
        c, r = len(clues_v), len(clues_h)
        grid_t = [[-1 for i in range(r)] for j in range(c)]

        cands_v = [self._gen_line_candidates(clue, line) for clue, line in zip(clues_v, grid_t)]
        grid_t = [self._zip_cands(cands, r) for cands in cands_v]
        grid = self._transpose(grid_t)

        cands_h = [self._gen_line_candidates(clue, line) for clue, line in zip(clues_h, grid)]
        #print_grid(grid)

        while not self._finished(grid):
            cands_h = [self._filter_cands(cand, line) for cand, line in zip(cands_h, grid)]
            grid = [self._zip_cands(cands, c) for cands in cands_h]
            #print_grid(grid)

            grid_t = self._transpose(grid)
            cands_v = [self._filter_cands(cand, line) for cand, line in zip(cands_v, grid_t)]
            grid_t = [self._zip_cands(cands, r) for cands in cands_v]
            grid = self._transpose(grid_t)
            #print_grid(grid)
        
        return tuple(tuple(row) for row in grid)

def print_grid(grid):
    cell_to_char = {-1: '_ ', 0: '0 ', 1: '1 '}
    for row in grid: 
        print(''.join(cell_to_char[i] for i in row))
    print()
        
def solve(clues, width, height):
    # print(width, height, clues)
    return Nonogram(clues).solve()
    

def test():
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
    args = ((v_clues, h_clues), 20, 20)

    grid = solve(*args)
    print_grid(grid)

    v_clues = ((2, 2), (1, 1, 2, 1), (1, 2, 1, 1, 1), (1, 2, 1, 1, 2), 
           (2, 1, 2, 3), (2, 1, 3, 2), (1, 3, 1, 1, 1, 1), 
           (1, 1, 1, 2, 2, 1), (2, 1, 5, 2, 1), (2, 1, 1, 2), 
           (1, 1), (2, 2), (1, 1), (1, 1), (1, 1))
    h_clues = ((1, 1, 2), (1, 1, 1, 1, 2), (2, 3, 1, 1), (2, 2, 1), (1, 1, 2), 
               (1, 2, 1, 4), (1, 4), (5,), (1, 5), (2, 2, 1, 4), (1, 2, 2), 
               (1, 1, 1, 2), (2, 1, 2, 1), (2, 1, 1, 1), (1, 2))
    args = ((v_clues, h_clues), 15, 15)

    grid = solve(*args)
    print_grid(grid)
    

if __name__ == "__main__":
    test()
