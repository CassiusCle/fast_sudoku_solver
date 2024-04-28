from typing import Optional
from sudoku_utils import print_sudoku_puzzle

class Sudoku:
    """Represents a Sudoku puzzle.

    Attributes:
        puzzle: A string representing the initial Sudoku puzzle.
        solved: A boolean indicating whether the puzzle is solved.
        candidate_solution: A string representing a candidate solution.
        solution: A string representing the correct solution.
    """

    def __init__(self, puzzle: str, solution: Optional[str] = None):
        """Initializes the Sudoku puzzle with the given puzzle and solution."""
        self.puzzle = puzzle
        self.solved = False
        self.candidate_solution = None
        self.solution = solution

    def print_sudoku(self, solve_status: str = 'Unsolved'):
        """Prints the Sudoku puzzle based on its solve status.

        Args:
            solve_status: The status of the puzzle solution ('Unsolved', 'Candidate', 'Solution').
        """
        if solve_status == 'Unsolved':
            print('Sudoku puzzle (Unsolved):')
            print_sudoku_puzzle(self.puzzle)
        elif solve_status == 'Candidate':
            if self.solved and self.candidate_solution:
                print('Sudoku puzzle (Solved correctly):')
            else:
                print('Sudoku puzzle (Candidate solution):')
            print_sudoku_puzzle(self.puzzle, self.candidate_solution)
            if not self.candidate_solution:
                print('N.B.: No Candidate solution has been computed yet')
        elif solve_status == 'Solution':
            print('Sudoku puzzle (Correct solution):')
            print_sudoku_puzzle(self.puzzle, self.solution)
            if not self.solution:
                print('N.B.: The correct solution is not known for this puzzle')
