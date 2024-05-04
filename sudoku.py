import itertools
import logging
from typing import Generator, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

# Set up logging configuration  
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s') 

class SudokuSolver:
    """A class for solving 9x9 Sudoku puzzles."""
    
    # Class-level constants
    PUZZLE_SIZE: int = 9
    PUZZLE_DEPTH: int = 9
    SHAPE_2D: Tuple[int, int] = (PUZZLE_SIZE, PUZZLE_SIZE)
    SHAPE_3D: Tuple[int, int, int] = (PUZZLE_SIZE, PUZZLE_SIZE, PUZZLE_DEPTH)
    
    def __init__(self) -> None:
        """Initializes the SudokuSolver class."""
        pass

    def print_puzzle(self, puzzle: Union[str, int], solution: Optional[str] = None) -> None:
        """Prints a sudoku puzzle and its solution in a formatted way.

        Args:
            puzzle: A string or integer representing the initial sudoku puzzle.
            solution: An optional string representing the solution to the puzzle.
        """
        
        # Convert puzzle numbers to letters for readability to distinguish from solution values later
        alphabet = 'abcdefghi'
        puzzle = ''.join(
            [alphabet[int(c) - 1] if c not in ['.', '0'] else c for c in str(puzzle)]
        )
        
        # Overlay solution onto puzzle if provided
        if solution:
            puzzle = ''.join(
                [c1 if c1.isalpha() else c2 for c1, c2 in zip(puzzle, solution)]
            )

        # Helper function to divide a string into equal-sized chunks
        def chunk_string(string: str, chunk_size: int) -> list[str]:
            """Divides a string into chunks of equal size."""
            return [string[i:i + chunk_size] for i in range(0, len(string), chunk_size)]

        # Break the puzzle string into lines and 3x3 blocks
        digits_per_line: list = chunk_string(puzzle, 9)
        digits_per_line: list = [chunk_string(line, 3) for line in digits_per_line]
        
        # Define the horizontal and vertical lines for the sudoku grid
        hz_line = '─' * 9
        top_line = f'┌{hz_line}┬{hz_line}┬{hz_line}┐'
        mid_line = f'├{hz_line}┼{hz_line}┼{hz_line}┤'
        bottom_line = f'└{hz_line}┴{hz_line}┴{hz_line}┘'

        # Assemble the top line of the sudoku grid
        output = [top_line]
        for i, digits in enumerate(digits_per_line):
            # Join the 3x3 blocks with vertical lines
            output.append('│' + '│'.join(''.join(chunk) for chunk in digits) + '│')
            # Add middle lines after every third line to form grid
            if i in [2, 5]:
                output.append(mid_line)
        # Add the bottom line to complete the grid
        output.append(bottom_line)    

        # Helper function to replace characters with formatted numbers
        def replace_chars(chars: str) -> str:
            """Replaces characters in the puzzle output with formatted numbers."""
            return ''.join(
                f'({alphabet.index(c) + 1})' if c.isalpha() else ' . ' if c in ['.', '0']
                else f' {c} ' if c.isdigit() else c for c in chars
            )
        
        # Print the final formatted sudoku grid
        print('\n'.join(replace_chars(line) for line in output))
    

    def _validate_solution(self, candidate_solution: np.ndarray) -> bool:
        """Check if a Sudoku solution is valid.

        Validates a 9x9x9 3D array representing a Sudoku puzzle solution. Each layer in the third dimension 
        corresponds to the positions of (n+1)s in the solution. The validation approach is inspired by a 
        MathOverflow post (https://mathoverflow.net/questions/129143/verifying-the-correctness-of-a-sudoku-solution).

        Args:
            candidate_solution (np.ndarray): A 3D array representing a proposed Sudoku solution.

        Raises:
            TypeError: If candidate_solution is not a numpy.ndarray.
            ValueError: If candidate_solution does not have the correct shape (9x9x9).
            ValueError: If candidate_solution does not contain exactly one digit for each field.

        Returns:
            bool: True if the solution is valid, False otherwise.
        """
        if not isinstance(candidate_solution, np.ndarray):
            raise TypeError(f'Expected numpy.ndarray, got {type(candidate_solution).__name__}.')

        if candidate_solution.shape != (9, 9, 9):
            raise ValueError('Candidate solution must be a 9x9x9 3D array.')
        
        if not np.all(candidate_solution.sum(axis=2) == 1):
            raise ValueError('Candidate solution does not contain exactly one digit on each field.')
        
        # Check if all rows and columns contain each digit only once
        if not (np.all(candidate_solution.sum(axis=1) == 1) and
                np.all(candidate_solution.sum(axis=0) == 1)):
            return False
        
        # Check if all cols contain each digit only once (sum is over rows)
        if not np.all(candidate_solution.sum(axis=0) == 1):
            return False

        # Check if boxes are valid (just the top left 4 need to be checked)
        for i in range(0, 6, 3):
            for j in range(0, 6, 3):
                if not np.all(candidate_solution[i:i+3, j:j+3].sum(axis=(0, 1)) == 1):
                    return False
    
        return True
    
    def validate_solution(self, candidate_solution: Union[str, List[str]]) -> bool:
        """Converts a candidate solution from string or list format and validates it.

        This method first converts a candidate solution provided as a string or list into a 3D NumPy array, 
        then validates the converted solution using the `validate_solution` method.

        Args:
            candidate_solution (Union[str, List[str]]): The candidate solution in string or list format.

        Returns:
            bool: True if the solution is valid, False otherwise.

        Raises:
            TypeError: If candidate_solution is not of type str or List[str].
            ValueError: If the conversion to a 3D NumPy array fails or if the validation fails.
        """
        if not isinstance(candidate_solution, (str, list)):
            raise TypeError('Candidate solution must be a string or a list.')
        
        _, candidate_3d = self._string_to_np_puzzle(candidate_solution)
        return self._validate_solution(candidate_solution=candidate_3d)
    
    def _string_to_np_puzzle(self, sudoku: str) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a string representing a sudoku puzzle into 2D and 3D numpy array representations.

        The 2D array represents the puzzle itself, and the 3D array represents the possibilities for each cell.

        Args:
            sudoku (str): A string representing a sudoku puzzle.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the 2D puzzle array and the 3D possibilities array.
        """
        # Convert the string to a 2D numpy array
        puzzle_2d: np.ndarray = np.reshape(np.array(list(sudoku), dtype=np.uint8), 
                                           newshape=SudokuSolver.SHAPE_2D)
        options_3d: np.ndarray = np.zeros(SudokuSolver.SHAPE_3D, dtype=np.uint8) # [row][column][depth]
        
        # Update options_3d based on the non-zero values in puzzle_2d
        nonzero_indices = np.nonzero(puzzle_2d)        
        values = puzzle_2d[nonzero_indices] - 1
        options_3d[nonzero_indices[0], nonzero_indices[1], values] = 1
        
        # Set all possibilities to 1 for cells with zero value in puzzle_2d
        zero_indices = np.where(puzzle_2d == 0)
        options_3d[zero_indices[0], zero_indices[1]] = 1
        
        return puzzle_2d, options_3d

    def _np_puzzle_to_string(self, np_puzzle: np.ndarray) -> str:
        """Converts a 3D NumPy array representing a Sudoku puzzle into a string.

        This method takes a 3D NumPy array where each 2D slice along the third axis represents
        the possibilities for each cell in the Sudoku grid. It converts this array into a string
        representation of the puzzle by taking the argmax along the third axis, adding one to
        shift from zero-based to one-based indexing, and then flattening the result into a 1D
        array. This array is then joined into a single string.

        Args:
            np_puzzle (np.ndarray): A 3D NumPy array representing the possibilities of a Sudoku puzzle.

        Returns:
            str: A string representation of the Sudoku puzzle, with numbers representing the filled cells
                    and zeros for the empty cells.

        Raises:
            ValueError: If `np_puzzle` is not a 3D NumPy array or if its shape does not conform to
                        the expected Sudoku puzzle shape.
        """
        if not isinstance(np_puzzle, np.ndarray) or len(np_puzzle.shape) != 3:
            raise ValueError('The input must be a 3D NumPy array representing a Sudoku puzzle.')
        if np_puzzle.shape != SudokuSolver.SHAPE_3D:
            raise ValueError(f'Expected puzzle shape {SudokuSolver.SHAPE_3D}, but got {np_puzzle.shape}.')

        # Convert the 3D possibilities array into a 1D string representation
        puzzle_string: str = ''.join(map(str, (np_puzzle.argmax(axis=2) + 1).flatten()))
        return puzzle_string
         
    def _generate_cell_index_updates(self, *iterables: Iterable[int]) -> Generator[Tuple[Tuple[None, int], ...], None, None]:
        """Yields unique combinations of cell indices for updating a Sudoku puzzle's possibilities.

        This generator function yields the indices of the cells in the Sudoku possibilities cube
        that need to be updated. It provides a tuple with the indices of the cells that need to be
        set to 1 and the indices of cells from the previous update that must be set to 0.

        Args:
            *iterables: Variable length iterable list of integers.

        Yields:
            Generator[Tuple[Tuple[None, int], ...], None, None]: A generator of tuples containing
            the indices to be updated in the possibilities cube.
        """
        combinations: Iterator[Tuple[int, ...]] = itertools.product(*iterables)
        prev_comb: Tuple[int, ...] = None  # Initialize the previous combination

        comb = next(combinations)
        yield tuple((None, c) for c in comb) # Yield indices for the first combination
        prev_comb = comb

        for comb in combinations:
            # Yield indices that are different from the previous combination
            yield tuple((p, c) for c, p in zip(comb, prev_comb) if c != p)
            prev_comb = comb
      
    def solve(self, unsolved_sudoku: str, max_iterations: int = 10_000_000) -> Optional[str]:
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
        puzzle_2d, options_3d = self._string_to_np_puzzle(unsolved_sudoku)

        while True:
            prev_puzzle_2d = puzzle_2d
            known_cells = np.argwhere(puzzle_2d)
            
            rows, cols = known_cells[:, 0], known_cells[:, 1]
            values = puzzle_2d[rows, cols] - 1

            # Set column, row, and box to zero for all known cells
            options_3d[rows, :, values] = 0
            options_3d[:, cols, values] = 0
            box_start_rows, box_start_cols = 3 * (rows // 3), 3 * (cols // 3)
            for box_start_row, box_start_col, value in zip(box_start_rows, box_start_cols, values):
                options_3d[box_start_row:box_start_row+3, box_start_col:box_start_col+3, value] = 0

            # Set known cells back to one
            options_3d[rows, cols, values] = 1

            puzzle_2d = options_3d.argmax(axis=2) + 1
            puzzle_2d[options_3d.sum(axis=2) != 1] = 0

            if np.array_equal(puzzle_2d, prev_puzzle_2d):
                break

            if puzzle_2d.sum() == 405:
                return self._np_puzzle_to_string(options_3d)
        
        num_possibilities: int = options_3d.sum(axis=2).prod()

        # Return answer if only one possibility left
        if num_possibilities == 1:
            return self._np_puzzle_to_string(options_3d)

        if num_possibilities >= max_iterations or num_possibilities < 0:
            logging.info(f'More than {max_iterations:_} combinations to check, aborting...')
            return None

        it = np.nditer(puzzle_2d, flags=['multi_index'])
        options_idx = [[(*it.multi_index, int(d)) for d in np.nditer(np.where(options_3d[*it.multi_index,:] == 1)[0])] for v in it if v == 0]
        options_idx

        # Create the generator
        generator = self._generate_cell_index_updates(*options_idx)

        # Set-up first option
        for idx in next(generator): 
            options_3d[*idx[1][:2], :] = 0
            options_3d[*idx[1]] = 1
        
        # Return first option if valid
        if self._validate_solution(options_3d):
            return self._np_puzzle_to_string(options_3d)

        # Iterate over other options
        for changes in generator:
            for idx in changes:
                options_3d[*idx[0]] = 0
                options_3d[*idx[1]] = 1
            
            if self._validate_solution(options_3d):
                return self._np_puzzle_to_string(options_3d)
        
        return None

def main():
    directory = 'data'
    file_name = 'sudokus_100k_sub_1mio.csv'
    df = pd.read_csv(os.path.join(directory, file_name))

    sudoku_solver = SudokuSolver()

    def process_row(row):
        max_it = 5_000_000
        start_time = time.perf_counter()
        solution = sudoku_solver.solve(unsolved_sudoku=row['quizzes'], max_iterations=max_it)
        solve_time = time.perf_counter() - start_time if solution else None
        valid = solution == row['solutions']

        # start_time = time.perf_counter()
        # solution_alt = sudoku_solver.solve_2progress(unsolved_sudoku=row['quizzes'], max_iterations=max_it)
        # solve_time_alt = time.perf_counter() - start_time if solution else None
        # valid_alt = solution == row['solutions']

        return pd.Series([solution, solve_time, valid])#, solution_str, solve_time_str, valid_str])
        # return pd.Series([solution, solve_time, valid, solution_alt, solve_time_alt, valid_alt])

    # Apply the function to each row with a progress bar
    tqdm.pandas()
    # df[['solution', 'solve_time', 'valid', 'solution_alt', 'solve_time_alt', 'valid_alt']] = df.progress_apply(process_row, axis=1)
    df[['solution', 'solve_time', 'valid']] = df.progress_apply(process_row, axis=1)

    print(f"Valid solutions:\t{all(df[df['solution'].notna()]['valid'])}")
    # print(f"Valid alt. solutions:\t{all(df[df['solution_alt'].notna()]['valid_alt'])}")

    print(f'\nTiming results:')
    print(df[['solve_time']].describe())
    # print(df[['solve_time', 'solve_time_alt']].describe())

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
