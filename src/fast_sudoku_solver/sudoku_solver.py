"""  
This module provides a SudokuSolver class with methods to solve Sudoku puzzles using  
constraint propagation and backtracking search. It includes functionality to validate  
solutions, solve puzzles, and compute the number of possible value combinations  
for an unsolved Sudoku puzzle.  

The SudokuSolver class relies on NumPy for numerical computations and employs  
constraint propagation combined with a backtracking search to find valid solutions.  

Typical usage example:  
    sudoku_solver = SudokuSolver()  
    solution = sudoku_solver.solve(unsolved_sudoku="...puzzle_string...")  
    is_valid = sudoku_solver.validate(candidate_solution=solution)  
"""

import logging
from typing import List, Optional, Union

import numpy as np

from fast_sudoku_solver.techniques import ConstraintPropagation, Backtracking
from fast_sudoku_solver.services import SudokuFormatter, SudokuValidator


class SudokuSolver:
    """A class for solving 9x9 Sudoku puzzles."""
    
    def __init__(self) -> None:
        """Initializes the SudokuSolver class."""  
        self.setup_logging() 
    
    def setup_logging(self, log_level: int = logging.INFO) -> None:  
        """Configures the logging.  
  
        Args:  
            log_level: The level of logging detail. Defaults to logging.INFO.  
        """ 
        logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")  
        self.logger = logging.getLogger(__name__)  
    
    @staticmethod
    def validate(
        candidate_solution: Union[str, List[Union[int, str]], np.ndarray]
    ) -> bool:
        """
        Checks if a candidate solution for the Sudoku puzzle is valid.

        This method first checks the type and converts candidate solution if
        provided as a string or list into a 3D NumPy array, then validates the
        converted solution using the `validate_solution` method.

        Args:
            candidate_solution: The candidate solution in string, list, or 3D np.ndarray format.

        Returns:
            True if the solution is valid, False otherwise.

        Raises:
            TypeError: If candidate_solution is not one of the expected types.  
            ValueError: If the conversion to a 3D NumPy array fails or if the validation fails.
        """
        if not isinstance(candidate_solution, (str, list, np.ndarray)):
            raise TypeError("Candidate solution must be a string, list or np.ndarray.")

        if isinstance(candidate_solution, (str, list)):
            _, candidate_solution = SudokuFormatter.convert_to_numpy(candidate_solution)

        if candidate_solution.shape != (9, 9, 9):
            raise ValueError("Candidate solution must be a 9x9x9 3D array.")

        if not np.all(candidate_solution.sum(axis=2) == 1):
            raise ValueError("Candidate solution does not contain exactly one digit on each field.")

        return SudokuValidator.validate_3d_solution(candidate_solution)
    
    @staticmethod
    def solve(
        unsolved_sudoku: Union[str, List[Union[int,str]]], max_iterations: int = 10_000_000
    ) -> Optional[str]:
        """
        Solves a Sudoku puzzle.

        Solves the Sudoku puzzle by pruning candidates based on filled values until no further
        reduction is possible. If only one combination is left, it returns the solution.
        Otherwise, it attempts to find a solution using backtracking.

        Args:
            unsolved_sudoku: The unsolved Sudoku puzzle in string or list format.
            max_iterations: The maximum number of iterations to attempt before aborting.

        Returns:
            The solved Sudoku puzzle in string format or None if a solution cannot be found.

        Raises:
            ValueError: If the provided Sudoku string is not valid.
        """
        puzzle_2d, options_3d = SudokuFormatter.convert_to_numpy(unsolved_sudoku)

        is_solved, puzzle_2d, options_3d = ConstraintPropagation.apply(puzzle_2d, options_3d)

        if is_solved:
            return SudokuFormatter.convert_to_string(options_3d)

        is_solved, _, options_3d = Backtracking.apply(puzzle_2d, options_3d, max_iterations)
        
        if is_solved:
            return SudokuFormatter.convert_to_string(options_3d)
        
        return None
        
    @staticmethod
    def compute_possibilities(unsolved_sudoku: str) -> int:
        """
        Compute the total number of possible value combinations for an unsolved Sudoku puzzle.

        Args:  
            unsolved_sudoku: A string or list representation of the unsolved Sudoku puzzle.  
  
        Returns:  
            The total number of possible value combinations for the given unsolved Sudoku puzzle.  
        """

        # Convert the input string to NumPy arrays for the puzzle state and possible values
        _, options_3d = SudokuFormatter.convert_to_numpy(unsolved_sudoku)

        # Compute the product of the sums of possible values for each cell
        num_possibilities: int = options_3d.sum(axis=2, dtype=np.longdouble).prod()

        return num_possibilities