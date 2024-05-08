import itertools
import logging
from typing import Generator, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

# Set up logging configuration  
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s') 

class Sudoku:
    """A class for solving 9x9 Sudoku puzzles."""
    
    # Class-level constants
    PUZZLE_SIZE: int = 9
    PUZZLE_DEPTH: int = 9
    SHAPE_2D: Tuple[int, int] = (PUZZLE_SIZE, PUZZLE_SIZE)
    SHAPE_3D: Tuple[int, int, int] = (PUZZLE_SIZE, PUZZLE_SIZE, PUZZLE_DEPTH)
    
    def __init__(self) -> None:
        """Initializes the SudokuSolver class."""
        pass

    @staticmethod
    def print_puzzle(puzzle: Union[str, int], solution: Optional[str] = None) -> None:
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
    
    @staticmethod
    def _validate_solution(candidate_solution: np.ndarray) -> bool:
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
    
    @staticmethod
    def validate_solution(candidate_solution: Union[str, List[str]]) -> bool:
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
        
        _, candidate_3d = Sudoku._string_to_np_puzzle(candidate_solution)
        return Sudoku._validate_solution(candidate_solution=candidate_3d)
    
    @staticmethod
    def _string_to_np_puzzle(sudoku: str) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a string representing a sudoku puzzle into 2D and 3D numpy array representations.

        The 2D array represents the puzzle itself, and the 3D array represents the possibilities for each cell.

        Args:
            sudoku (str): A string representing a sudoku puzzle.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the 2D puzzle array and the 3D possibilities array.
        """
        if '.' in sudoku:
            sudoku = sudoku.replace('.', '0')

        # Convert the string to a 2D numpy array
        puzzle_2d: np.ndarray = np.reshape(np.fromiter(sudoku, dtype='l'), 
                                           newshape=Sudoku.SHAPE_2D)
        options_3d: np.ndarray = np.zeros(Sudoku.SHAPE_3D, dtype='l') # [row][column][depth]
        
        # Update options_3d based on the non-zero values in puzzle_2d
        nonzero_indices = np.nonzero(puzzle_2d)        
        values = puzzle_2d[nonzero_indices] - 1
        options_3d[nonzero_indices[0], nonzero_indices[1], values] = 1
        
        # Set all possibilities to 1 for cells with zero value in puzzle_2d
        zero_indices = np.where(puzzle_2d == 0)
        options_3d[zero_indices[0], zero_indices[1]] = 1
        
        return puzzle_2d, options_3d

    @staticmethod
    def _np_puzzle_to_string(np_puzzle: np.ndarray) -> str:
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
        if np_puzzle.shape != Sudoku.SHAPE_3D:
            raise ValueError(f'Expected puzzle shape {Sudoku.SHAPE_3D}, but got {np_puzzle.shape}.')

        # Convert the 3D possibilities array into a 1D string representation
        puzzle_string: str = ''.join(map(str, (np_puzzle.argmax(axis=2) + 1).flatten()))
        return puzzle_string
    
    @staticmethod
    def _generate_cell_index_updates(*iterables: Iterable[int]) -> Generator[Tuple[Tuple[None, int], ...], None, None]:
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
    
    @staticmethod
    def _apply_elimination(puzzle_2d: np.ndarray, 
                           options_3d: np.ndarray
                           ) -> Tuple[bool, bool, np.ndarray, np.ndarray]:
        """
        Apply basic elimination rules to the Sudoku puzzle until no further progress is made.

        This method iteratively applies Sudoku elimination rules to the given puzzle.
        It updates the puzzle state and the options cube until the puzzle is solved
        or no more progress can be made.

        Args:
            puzzle_2d: A 2D NumPy array representing the current state of the Sudoku puzzle.
            options_3d: A 3D NumPy array representing the possible values for each cell.

        Returns:
            A tuple containing:
            - has_progress: A boolean indicating if progress was made in the last iteration.
            - is_solved: A boolean indicating if the puzzle is solved.
            - puzzle_2d: The updated 2D puzzle state.
            - options_3d: The updated 3D options cube.
        """
        
        i = 0
        has_progress = False 
        is_solved = False

        while True:
            # Store the previous state of the puzzle to detect changes
            prev_puzzle_2d = puzzle_2d
            # prev_options_3d = options_3d.copy()
            # Find the indices of cells with known values
            known_cells = np.argwhere(puzzle_2d)
            
            # Extract row and column indices, and adjust values for 0-indexing
            rows, cols = known_cells[:, 0], known_cells[:, 1]
            values = puzzle_2d[rows, cols] - 1

            # Eliminate options based on known cell values
            options_3d[rows, :, values] = 0
            options_3d[:, cols, values] = 0
            box_start_rows, box_start_cols = 3 * (rows // 3), 3 * (cols // 3)
            for box_start_row, box_start_col, value in zip(box_start_rows, box_start_cols, values):
                options_3d[box_start_row:box_start_row+3, box_start_col:box_start_col+3, value] = 0

            # Set known cells back to one
            options_3d[rows, cols, values] = 1

            # Update the puzzle state based on the options cube
            puzzle_2d = options_3d.argmax(axis=2) + 1
            # Reset cells with multiple options to zero
            puzzle_2d[options_3d.sum(axis=2) != 1] = 0

            i += 1

            # Check for changes in the puzzle state to determine progress
            # if np.array_equal(options_3d, prev_options_3d): # TODO: Change to 3D?  (better flow then with "solved")
            if np.array_equal(puzzle_2d, prev_puzzle_2d):  # TODO: Change to 3D? (better flow then with "solved")
                if i > 1: 
                    has_progress = True
                else:
                    has_progress = False
                break
            
            # Check if the puzzle is solved
            if puzzle_2d.sum() == 405:
                is_solved = True
                break
        
        return has_progress, is_solved, puzzle_2d, options_3d
    
    @staticmethod
    def _apply_hidden_singles(puzzle_2d: np.ndarray, 
                             options_3d: np.ndarray
                           ) -> Tuple[bool, bool, np.ndarray, np.ndarray]:
        """
        Apply the 'hidden singles' rule to the Sudoku puzzle until no further progress is made.

        # This method iteratively applies Sudoku elimination rules to the given puzzle.
        # It updates the puzzle state and the options cube until the puzzle is solved
        # or no more progress can be made.

        Args:
            puzzle_2d: A 2D NumPy array representing the current state of the Sudoku puzzle.
            options_3d: A 3D NumPy array representing the possible values for each cell.

        Returns:
            A tuple containing:
            - has_progress: A boolean indicating if progress was made in the last iteration.
            - is_solved: A boolean indicating if the puzzle is solved.
            - puzzle_2d: The updated 2D puzzle state.
            - options_3d: The updated 3D options cube.
        """
        # Store the previous state of the puzzle to detect changes
        # prev_options_3d = options_3d.copy()
        prev_puzzle_2d = puzzle_2d

        # Compute columns with singles
        cols_w_singles = np.argwhere((options_3d.sum(axis=0) == 1)) # 3d
        # Compute the row on which the single is found
        row_indices = options_3d.argmax(axis=0)
        # Singles from columns
        singles = {(row_indices[c,v], c, v) for c, v in cols_w_singles}
        
        # Compute rows with singles
        rows_w_singles = np.argwhere((options_3d.sum(axis=1) == 1)) # 3d
        # Compute the colum on which the single is found
        col_indices = options_3d.argmax(axis=1)
        # Singles from rows
        singles.update({(r, col_indices[r,v], v) for r, v in rows_w_singles})
        

        for x in range(0, 9, 3):
            for y in range(0, 9, 3):
                # Compute row on which single is found
                row_idx = np.argmax(options_3d[y:y+3, x:x+3, :].sum(axis=1), axis=0)

                # Compute column on which single is found
                column_agg = options_3d[y:y+3, x:x+3, :].sum(axis=0)
                col_idx = np.argmax(column_agg, axis=0)

                # Compute depth on which single is found
                depth_idx = np.argwhere(column_agg.sum(axis=0) == 1)

                # Singles from this subsquare
                singles.update({(r[0]+y, c[0]+x, v[0]) for r, c, v in zip(row_idx[depth_idx], col_idx[depth_idx], depth_idx)})

        # Compute already known values
        known_values = {(*kv, puzzle_2d[*kv]-1) for kv in np.argwhere(puzzle_2d)}
        
        # Compute hidden singles
        hidden_singles: set = singles - known_values

        # Return in case there are no hidden singles
        if not hidden_singles:
            has_progress = False
            is_solved = False
            return has_progress, is_solved, puzzle_2d, options_3d

        rows, cols, values = zip(*hidden_singles)            
        rows = np.array(rows)
        cols = np.array(cols)
        values = np.array(values)

        # Set column, row, and box to zero for all known cells
        options_3d[rows, :, values] = 0
        options_3d[:, cols, values] = 0
        options_3d[rows, cols, :] = 0
        box_start_rows, box_start_cols = 3 * (rows // 3), 3 * (cols // 3)
        for box_start_row, box_start_col, value in zip(box_start_rows, box_start_cols, values):
            options_3d[box_start_row:box_start_row+3, box_start_col:box_start_col+3, value] = 0

        # Set known cells back to one
        options_3d[rows, cols, values] = 1

        puzzle_2d = options_3d.argmax(axis=2) + 1
        puzzle_2d[options_3d.sum(axis=2) != 1] = 0

        # Check for changes in the puzzle state to determine progress
        # if np.array_equal(options_3d, prev_options_3d): # TODO: Change to 3D?  (better flow then with "solved")
        if np.array_equal(puzzle_2d, prev_puzzle_2d): # TODO: Change to 3D?  (better flow then with "solved")
            has_progress = False
        else:
            has_progress = True
                
        # Check if the puzzle is solved
        if puzzle_2d.sum() == 405:
            is_solved = True
        else:
            is_solved = False
        
        return has_progress, is_solved, puzzle_2d, options_3d
    
    @staticmethod
    def solve(unsolved_sudoku: str, max_iterations: int = 10_000_000) -> Optional[str]:
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
        puzzle_2d, options_3d = Sudoku._string_to_np_puzzle(unsolved_sudoku)

        i = 0
        while True:
            i += 1
            has_progress, is_solved, puzzle_2d, options_3d = Sudoku._apply_elimination(puzzle_2d, options_3d)
            if is_solved:
                break
            elif not has_progress and i > 1:
                break
            # has_progress is true --> down
            # is_solved --> break
            # not has progress --> break


            has_progress, is_solved, puzzle_2d, options_3d = Sudoku._apply_hidden_singles(puzzle_2d, options_3d)
            if is_solved:
                break
            elif has_progress:
                continue # Begin again up


        if is_solved:
            return Sudoku._np_puzzle_to_string(options_3d)

        num_possibilities: int = options_3d.sum(axis=2, dtype=np.longdouble).prod()

        # Return answer if only one possibility left
        if num_possibilities == 1:
            return Sudoku._np_puzzle_to_string(options_3d)
        # elif num_possibilities > 1_000_000:
        #     print(f'Many possibilities:\n\t{num_possibilities}\n')

        if num_possibilities >= max_iterations or num_possibilities < 0:
            logging.info(f'More than {max_iterations:_} combinations to check, aborting...')
            return None

        it = np.nditer(puzzle_2d, flags=['multi_index'])
        options_idx = [[(*it.multi_index, int(d)) for d in np.nditer(np.where(options_3d[*it.multi_index,:] == 1)[0])] for v in it if v == 0]
        options_idx

        # Create the generator
        generator = Sudoku._generate_cell_index_updates(*options_idx)

        # Set-up first option
        for idx in next(generator): 
            options_3d[*idx[1][:2], :] = 0
            options_3d[*idx[1]] = 1
        
        # Return first option if valid
        if Sudoku._validate_solution(options_3d):
            return Sudoku._np_puzzle_to_string(options_3d)

        # Iterate over other options
        for changes in generator:
            for idx in changes:
                options_3d[*idx[0]] = 0
                options_3d[*idx[1]] = 1
            
            if Sudoku._validate_solution(options_3d):
                return Sudoku._np_puzzle_to_string(options_3d)
        
        return None
    

    @staticmethod
    def dev_compute_possibilities(unsolved_sudoku: str) -> int:
        """
        Compute possibilities 
        """
        puzzle_2d, options_3d = Sudoku._string_to_np_puzzle(unsolved_sudoku)

        i = 0
        while True:
            i += 1
            has_progress, is_solved, puzzle_2d, options_3d = Sudoku._apply_elimination(puzzle_2d, options_3d)
            if is_solved:
                break
            elif not has_progress and i > 1:
                break
            # has_progress is true --> down
            # is_solved --> break
            # not has progress --> break


            has_progress, is_solved, puzzle_2d, options_3d = Sudoku._apply_hidden_singles(puzzle_2d, options_3d)
            if is_solved:
                break
            elif has_progress:
                continue # Begin again up

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
