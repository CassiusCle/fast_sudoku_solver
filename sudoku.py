import numpy as np
from typing import Union, Optional, List
from sudoku_utils import print_sudoku_puzzle

class Sudoku:
    """Represents a Sudoku puzzle.

    Attributes:
        puzzle: A string representing the initial Sudoku puzzle.
        solved: A boolean indicating whether the puzzle is solved.
        candidate_solution: A string representing a candidate solution.
        solution: A string representing the correct solution.
    """

    # Class-level constants for validating solutions
    _ROW_INDICES: list = [np.arange(i*9, i*9+9) for i in range(8)]
    _COL_INDICES: list = [np.arange(i, 81, 9) for i in range(9)]
    _SUBSQ_INDICES = [np.concatenate([np.arange(start, stop) for start, stop in ranges]) 
                    for ranges in [[(i*9+k*3, i*9+3+k*3) for i in range(3)] for k in [0, 1, 9, 10]]]
    IDX_LIST_ITEMS: list = _ROW_INDICES + _COL_INDICES + _SUBSQ_INDICES
    ALL_DIGITS: np.array = np.arange(1, 10)

    def __init__(self, puzzle: str, 
                 solution: Optional[str] = None
                ):
        """Initializes the Sudoku puzzle with the given puzzle and solution."""
        self.puzzle: str = puzzle
        self.solution: str = solution
        self.solved: bool = False

    def print_sudoku(self, solve_status: str = 'Unsolved', candidate_solution: str = None):
        """Prints the Sudoku puzzle based on its solve status.

        Args:
            solve_status: The status of the puzzle solution ('Unsolved', 'Candidate', 'Solution').
        """
        if solve_status == 'Unsolved':
            print('Sudoku puzzle (Unsolved):')
            print_sudoku_puzzle(self.puzzle)
        elif solve_status == 'Candidate':
            if candidate_solution and candidate_solution == self.solution:
                print('Sudoku puzzle (Solved correctly):')
            else:
                print('Sudoku puzzle (Candidate solution):')
            print_sudoku_puzzle(self.puzzle, candidate_solution)
            if not candidate_solution:
                print('N.B.: No Candidate solution provided')
        elif solve_status == 'Solution':
            print('Sudoku puzzle (Correct solution):')
            print_sudoku_puzzle(self.puzzle, self.solution)
            if not self.solution:
                print('N.B.: No solution available for this puzzle.')
    
    def validate_solution(self, candidate_solution: str) -> bool:
        """Validates if the solution is correct for a Sudoku puzzle.

        This method checks the minimum number of rows, columns and 3x3 subgrids to ensure that 
        each contains all digits from 1 to 9 without repetition. The approach for
        solution validation is based on the one outlined in a MathOverflow post
        (https://mathoverflow.net/questions/129143/verifying-the-correctness-of-a-sudoku-solution).

        Note: IDX_LIST_ITEMS and ALL_DIGITS are defined as globals within the global
        scope of the Sudoku class for performance reasons.

        Returns:
            bool: True if the solution is valid, False otherwise.
        """
        if len(candidate_solution) != 81:
            raise ValueError('Candidate solution does not contain enough elements.')
        elif type(candidate_solution) is not str:
            raise TypeError(f'Candidate solution is not of type str: {type(candidate_solution)}.')
        
        solution_array = np.array(list(candidate_solution), dtype=int)
        for idx in Sudoku.IDX_LIST_ITEMS:
            if not np.array_equal(np.sort(np.unique(solution_array[idx])), Sudoku.ALL_DIGITS):
                self.solved = False
                return False
        self.solved = True
        
        # TODO: Remove later, just checking if validation is actually same as correct solution 
        if self.solution:
            self.validation_correct = candidate_solution == self.solution
        else:
            self.solution = candidate_solution
        return True
    
