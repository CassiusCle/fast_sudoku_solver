import itertools
import logging
from typing import Generator, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

from utils import (validate_3D_solution, iter_to_np_puzzle, np_puzzle_to_string, 
                   generate_cell_index_updates)
from techniques import apply_constraint_propagation

# Set up logging configuration  
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s') 

class Sudoku:
    """A class for solving 9x9 Sudoku puzzles."""
    
    def __init__(self) -> None:
        """Initializes the SudokuSolver class."""
        pass
    
    @staticmethod
    def validate_solution(candidate_solution: Union[str, List[str], np.ndarray]) -> bool:
        """ Checks if a candidate solution for the Sudoku puzzle is valid.

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
            raise TypeError('Candidate solution must be a string or a list.')
        
        if isinstance(candidate_solution, (str, list)):
            _, candidate_solution = iter_to_np_puzzle(candidate_solution)

        elif isinstance(candidate_solution, np.ndarray):
            if candidate_solution.shape != (9, 9, 9):
                raise ValueError('Candidate solution must be a 9x9x9 3D array.')
            if not np.all(candidate_solution.sum(axis=2) == 1):
                raise ValueError('Candidate solution does not contain exactly one digit on each field.')
        else:
            raise TypeError(f'Expected str, list or numpy.ndarray, but got {type(candidate_solution).__name__}.')

        return validate_3D_solution(candidate_solution=candidate_solution)

    @staticmethod
    def solve(unsolved_sudoku: Iterable, max_iterations: int = 10_000_000) -> Optional[str]:
        """Solves a Sudoku puzzle.

        Solves the Sudoku puzzle by pruning candidates based on filled values until no further
        reduction is possible. If only one combination is left, it returns the solution.
        Otherwise, it attempts to brute force solutions.

        Args:
            unsolved_sudoku (str): The unsolved Sudoku puzzle in string format.
            max_iterations (int, optional): The maximum number of iterations to attempt before
                                             aborting. Defaults to 10,000,000.

        Returns:
            str: The solved Sudoku puzzle in string format, or None if a solution cannot be found
                 within the maximum number of iterations.

        Raises:
            ValueError: If the provided Sudoku string is not valid.
        """
        puzzle_2d, options_3d = iter_to_np_puzzle(unsolved_sudoku)

        is_solved, puzzle_2d, options_3d = apply_constraint_propagation(puzzle_2d, options_3d)
    
        if is_solved:
            return np_puzzle_to_string(options_3d)

        num_possibilities: int = options_3d.sum(axis=2, dtype=np.longdouble).prod()

        # Return answer if only one possibility left
        if num_possibilities == 1:
            return np_puzzle_to_string(options_3d)
        # elif num_possibilities > 1_000_000:
        #     print(f'Many possibilities:\n\t{num_possibilities}\n')

        if num_possibilities >= max_iterations or num_possibilities < 0:
            logging.info(f'More than {max_iterations:_} combinations to check, aborting...')
            return None

        it = np.nditer(puzzle_2d, flags=['multi_index'])
        options_idx = [[(*it.multi_index, int(d)) for d in np.nditer(np.where(options_3d[*it.multi_index,:] == 1)[0])] for v in it if v == 0]
        options_idx

        # Create the generator
        generator = generate_cell_index_updates(*options_idx)

        # Set-up first option
        for idx in next(generator): 
            options_3d[*idx[1][:2], :] = 0
            options_3d[*idx[1]] = 1
        
        # Return first option if valid
        if validate_3D_solution(options_3d):
            return np_puzzle_to_string(options_3d)

        # Iterate over other options
        for changes in generator:
            for idx in changes:
                options_3d[*idx[0]] = 0
                options_3d[*idx[1]] = 1
            
            if validate_3D_solution(options_3d):
                return np_puzzle_to_string(options_3d)
        
        return None
    
    @staticmethod
    def dev_compute_possibilities(unsolved_sudoku: str) -> int:
        """
        Compute the total number of possible value combinations for an unsolved Sudoku puzzle.

        This static method takes an unsolved Sudoku puzzle in string format, converts it to a 2D NumPy array
        representing the puzzle state, and a 3D NumPy array representing the possible values for each cell.
        It then applies constraint propagation to reduce the number of possibilities and computes the product
        of the sums of possible values for each cell, which represents the total number of combinations.

        Args:
            unsolved_sudoku: A string representation of the unsolved Sudoku puzzle, where each character
                             represents a cell value (1-9) or a placeholder for an unknown value (typically '0' or '.').

        Returns:
            The total number of possible value combinations for the given unsolved Sudoku puzzle as an integer.
        """
        
        # Convert the input string to NumPy arrays for the puzzle state and possible values
        puzzle_2d, options_3d = iter_to_np_puzzle(unsolved_sudoku)

        # Apply constraint propagation to reduce the number of possibilities
        _, puzzle_2d, options_3d = apply_constraint_propagation(puzzle_2d, options_3d)
    
        # Compute the product of the sums of possible values for each cell
        num_possibilities: int = options_3d.sum(axis=2, dtype=np.longdouble).prod()

        return num_possibilities


def main():
    directory = '../data'
    # file_name = 'sudokus_100k_sub_1mio.csv'
    file_name = 'sudokus_difficult_sub_1mio.csv'
    df = pd.read_csv(os.path.join(directory, file_name))

    sudoku_solver = Sudoku()

    test_new = False

    def process_row(row):
        max_it = 5_000_000
        start_time = time.perf_counter()
        solution = sudoku_solver.solve(unsolved_sudoku=row['quizzes'], max_iterations=max_it)
        solve_time = time.perf_counter() - start_time if solution else None
        valid = solution == row['solutions']

        if test_new:
            start_time = time.perf_counter()
            solution_new = sudoku_solver.solve_new(unsolved_sudoku=row['quizzes'], max_iterations=max_it)
            solve_time_new = time.perf_counter() - start_time if solution else None
            valid_new = solution == row['solutions']
            return pd.Series([solution, solve_time, valid, solution_new, solve_time_new, valid_new])
        else:
            return pd.Series([solution, solve_time, valid])

    # Apply the function to each row with a progress bar
    tqdm.pandas()

    start_full_data = time.perf_counter()

    if test_new:
        df[['solution', 'solve_time', 'valid', 'solution_new', 'solve_time_new', 'valid_new']] = df.progress_apply(process_row, axis=1)
        # df[['solution', 'solve_time', 'valid', 'solution_new', 'solve_time_new', 'valid_new']] = df.apply(process_row, axis=1)
        print(f"Valid alt. solutions:\t{all(df[df['solution_alt'].notna()]['valid_alt'])}")
    else:
        df[['solution', 'solve_time', 'valid']] = df.progress_apply(process_row, axis=1)
        # df[['solution', 'solve_time', 'valid']] = df.apply(process_row, axis=1)
        print(f"Valid solutions:\t{all(df[df['solution'].notna()]['valid'])}")

    solve_time_full_data =  time.perf_counter() - start_full_data 

    print(f'\nTiming results:')
    print(f'\nTotal time: {solve_time_full_data:.4}')

    if test_new:
        print(df[['solve_time', 'solve_time_alt']].describe())
    else:
        print(df[['solve_time']].describe())

if __name__ == "__main__":
    import os
    import pandas as pd
    import time
    from tqdm import tqdm

    # import argparse
    # parser = argparse.ArgumentParser(description="A simple script that greets the user.")
    # parser.add_argument("--data_size", default="10k", help="The name of the person to greet.")
    # args = parser.parse_args()

    # NOTE: Profiling: python -m cProfile -o output.pstats -s time sudoku.py  
    # python -m pstats output.pstats
    main()
    # main(args)
