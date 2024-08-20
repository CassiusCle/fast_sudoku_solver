"""  
This module provides a Sudoku class with methods to solve Sudoku puzzles using  
constraint propagation techniques. The class includes functionalities to validate  
solutions, solve puzzles, and compute the number of possible value combinations  
for an unsolved Sudoku puzzle.

The Sudoku solver relies on NumPy for numerical computations and pandas for  
processing CSV files containing Sudoku puzzles and their solutions. The solver  
employs a combination of constraint propagation and brute force search to find  
valid solutions, with an option to limit the number of iterations for the brute  
force search.  

Typical usage example:  
    sudoku_solver = Sudoku()  
    solution = sudoku_solver.solve(unsolved_sudoku="...puzzle_string...")  
    is_valid = sudoku_solver.validate_solution(candidate_solution=solution)  
"""  

import logging
from typing import List, Optional, Union

import numpy as np

from fast_sudoku_solver.techniques import ConstraintPropagation, Backtracking
from src.fast_sudoku_solver.services import SudokuFormatter, SudokuValidator


class SudokuSolver:
    """A class for solving 9x9 Sudoku puzzles."""
    
    def __init__(self) -> None:
        """Initializes the SudokuSolver class."""
        pass
    
    def setup_logging(self, log_level: int = logging.INFO) -> None:
        """ TODO: Write docstring"""
        self.logger = logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    @staticmethod
    def validate(
        candidate_solution: Union[str, List[str], np.ndarray]
    ) -> bool:
        """
        Checks if a candidate solution for the Sudoku puzzle is valid.

        This method first checks the type and converts candidate solution if
        provided as a string or list into a 3D NumPy array, then validates the
        converted solution using the `validate_solution` method.

        Args:
            candidate_solution (str, List[str], np.ndarray]): The candidate
                solution in string, list or 3D np.ndarray format.

        Returns:
            bool: True if the solution is valid, False otherwise.

        Raises:
            TypeError: If candidate_solution is not of type str or List[str].
            ValueError: If the conversion to a 3D NumPy array fails or if the validation fails.

        TODO: Add case for 2D Puzzle
        """
        if not isinstance(candidate_solution, (str, list, np.ndarray)):
            raise TypeError("Candidate solution must be a string, list or np.ndarray.")

        if isinstance(candidate_solution, (str, list)):
            _, candidate_solution = SudokuFormatter.convert_to_numpy(candidate_solution)

        elif isinstance(candidate_solution, np.ndarray):
            if candidate_solution.shape != (9, 9, 9):
                raise ValueError("Candidate solution must be a 9x9x9 3D array.")
            if not np.all(candidate_solution.sum(axis=2) == 1):
                raise ValueError(
                    "Candidate solution does not contain exactly one digit on each field."
                )
        else:
            raise TypeError(
                f"Expected str, list or numpy.ndarray, but got {type(candidate_solution).__name__}."
            )

        return SudokuValidator.validate_3d_solution(candidate_solution=candidate_solution)
    
    @staticmethod
    def solve(
        unsolved_sudoku: Union[str, List[str]], max_iterations: int = 10_000_000
    ) -> Optional[str]:
        """
        Solves a Sudoku puzzle.

        Solves the Sudoku puzzle by pruning candidates based on filled values until no further
        reduction is possible. If only one combination is left, it returns the solution.
        Otherwise, it attempts to brute force solutions.

        Args:
            unsolved_sudoku (str): The unsolved Sudoku puzzle in string or list format
            max_iterations (int, optional): The maximum number of iterations to attempt before
                                            aborting. Defaults to 10,000,000.

        Returns:
            str:    The solved Sudoku puzzle in string format, or None if a solution cannot be found
                    within the maximum number of iterations.

        Raises:
            ValueError: If the provided Sudoku string is not valid.
        """
        puzzle_2d, options_3d = SudokuFormatter.convert_to_numpy(unsolved_sudoku)

        is_solved, puzzle_2d, options_3d = ConstraintPropagation.apply(puzzle_2d, options_3d)

        if is_solved:
            return SudokuFormatter.convert_to_string(options_3d)

        is_solved, _, options_3d = Backtracking.apply(puzzle_2d, options_3d, max_iterations)
        
        if is_solved:
            return SudokuFormatter.np_puzzle_to_string(options_3d)
        
    @staticmethod
    def compute_possibilities(unsolved_sudoku: str) -> int:
        # TODO: Make it also take list of sudoku's
        """
        Compute the total number of possible value combinations for an unsolved Sudoku puzzle.

        This static method takes an unsolved Sudoku puzzle in string format, converts it to a
        2D NumPy array representing the puzzle state, and a 3D NumPy array representing the possible
        values for each cell. It then applies constraint propagation to reduce the number of
        possibilities and computes the product of the sums of possible values for each cell, which
        represents the total number of combinations.

        Args:
            unsolved_sudoku:    A string representation of the unsolved Sudoku puzzle, where each
                                character represents a cell value (1-9) or a placeholder for an
                                unknown value (typically '0' or '.').

        Returns:
            The total number of possible value combinations for the given unsolved Sudoku puzzle as
            an integer.
        """

        # Convert the input string to NumPy arrays for the puzzle state and possible values
        _, options_3d = SudokuFormatter.convert_to_numpy(unsolved_sudoku)

        # Compute the product of the sums of possible values for each cell
        num_possibilities: int = options_3d.sum(axis=2, dtype=np.longdouble).prod()

        return num_possibilities