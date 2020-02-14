from functools import reduce
from typing import List, Tuple

Line = List[int]
Grid = List[Line]
Clue = Tuple[int]

def partition(n, k):
    if k == 0:
        yield tuple()
        return
    for x in range(0, n + 1):
        for y in partition(n - x, k - 1):
            yield y + (x,)

class Nonogram:
    def __init__(self, clues: Tuple[List[Clue], List[Clue]]):
        self.clues = clues 

    @staticmethod 
    def _contraint_prop(clue: Clue, line: Line) -> [Line]:
        n, k = len(line), len(clue)
        n_filled = sum(clue)
        n_gap = k - 1
        n_free = n - n_filled - n_gap
        
        idx = 0
        for l in clue: 
            idx_end = idx + l
            if l - n_free > 0:
                idx_start = idx_end - (l - n_free)
                for j in range(idx_start, idx_end):
                    line[j] = 1
            idx += l + 1
        return line

    @staticmethod
    def _gen_line_candidates(clue: Clue, line: Line) -> [Line]:
        n, k = len(line), len(clue)
        n_filled = sum(clue)
        n_gap = k - 1
        n_free = n - n_filled - n_gap
        cands = []
    
        for idx_shifts in partition(n_free, k):
            cand = [0] * n
            idx = 0
            for l, idx_shift in zip(clue, idx_shifts[::-1]):
                idx += idx_shift
                for _ in range(l):
                    cand[idx] = 1
                    idx += 1
                idx += 1
            cands.append(cand)  
        return cands

    @staticmethod 
    def _transpose(grid: Grid) -> Grid:
        # r, c = len(grid), len(grid[0])
        # return [[grid[j][i] for j in range(r)] for i in range(c)]
        return list(map(list, zip(*grid)))
    
    @staticmethod
    def _valid_cand(cand, line):
        return all(x == -1 or c == -1 or x == c for x, c in zip(cand, line))

    @staticmethod 
    def _filter_cands(cands: [Line], line: Line) -> [Line]:
        return [cand for cand in cands if Nonogram._valid_cand(cand, line)]

    @staticmethod 
    def _zip_lines(cands: [Line]) -> Line:
        def compare_cells(x, y):
            return x if x == y and x != -1 else -1
        return [reduce(compare_cells, cells) for cells in zip(*cands)]

    @staticmethod 
    def _finished(grid: [Line]) -> bool:
        return all(x != -1 for row in grid for x in row)

    def solve(self):
        (clues_v, clues_h) = self.clues 
        c, r = len(clues_v), len(clues_h)
        grid = [[-1 for i in range(c)] for j in range(r)]
        grid = [self._contraint_prop(clue, line) for clue, line in zip(clues_h, grid)]
        grid_t = self._transpose(grid)
        grid_t = [self._contraint_prop(clue, line) for clue, line in zip(clues_v, grid_t)]
        grid = self._transpose(grid_t)  

        cands_h = [self._gen_line_candidates(clue, line) for clue, line in zip(clues_h, grid)]
        grid = [self._zip_lines(cands) for cands in cands_h]
        grid_t = self._transpose(grid)

        cands_v = [self._gen_line_candidates(clue, line) for clue, line in zip(clues_v, grid_t)]
        grid_t = [self._zip_lines(cands) for cands in cands_v]
        grid = self._transpose(grid_t)
        
        while not self._finished(grid):
            cands_h = [self._filter_cands(cand, line) for cand, line in zip(cands_h, grid)]
            grid = [self._zip_lines(cands) for cands in cands_h]
            grid_t = self._transpose(grid)

            cands_v = [self._filter_cands(cand, line) for cand, line in zip(cands_v, grid_t)]
            grid_t = [self._zip_lines(cands) for cands in cands_v]
            grid = self._transpose(grid_t)
        

        return tuple(tuple(row) for row in grid)

def print_grid(grid):
    cell_to_char = {-1: '_ ', 0: '0 ', 1: '1 '}
    for row in grid: 
        print(''.join(cell_to_char[i] for i in row))
        
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
