"""  
This module implements various Sudoku solving techniques used within the Sudoku solver.
These techniques include finding singles, hidden singles, and applying elimination strategies
to reduce the options for each cell in the puzzle. The functions within this module
work in tandem with the Sudoku class in the `sudoku.py` module to solve puzzles.  

The techniques are built upon the representation of the Sudoku grid as a 3D NumPy array,   
where the third dimension represents the possible numbers that can occupy a cell. The   
module provides a higher-level function `apply_constraint_propagation` which encapsulates   
the application of all the implemented techniques in a sequence that facilitates the solving   
of the puzzle.  

The solving strategy is based on iteratively applying these techniques to prune the   
set of possible values for each cell until the puzzle is solved or no further progress can be made.  
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Set, Generator, Iterable, Iterator, Tuple
from itertools import product

import numpy as np

from src.fast_sudoku_solver.services import SudokuValidator

class SolvingStrategy(ABC):
    # Abstract class, TODO: implement correctly
        
    @classmethod
    @abstractmethod
    def apply(cls, *args, **kwargs):
        # TODO: ADD docstring if necessary
        pass
    
    
class ConstraintPropagation(SolvingStrategy):


    @classmethod
    def apply(cls, puzzle_2d: np.ndarray, options_3d: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Apply constraint propagation techniques to a Sudoku puzzle.

        This function applies two constraint propagation techniques: elimination and hidden singles.
        It iterates over these techniques to progressively reduce the number of possible values for each
        cell in the puzzle until the puzzle is solved or no further progress can be made.

        Args:
            puzzle_2d:  A 2D NumPy array representing the current state of the Sudoku puzzle.
                        Each cell contains the value of the puzzle (1-9) or 0 if the value is unknown.
            options_3d: A 3D NumPy array representing the possible values for each cell.
                        The first two dimensions correspond to the puzzle grid, and the third dimension
                        contains a binary indicator for the possible values (1-9).

        Returns:
            A tuple containing:
            - is_solved: A boolean indicating if the puzzle is solved.
            - puzzle_2d: The updated 2D puzzle state.
            - options_3d: The updated 3D options cube.

        Raises:
            ValueError: If the input arrays do not meet the required shapes or value constraints.
        """

        # Validate the input arrays
        if not (isinstance(puzzle_2d, np.ndarray) and puzzle_2d.ndim == 2):
            raise ValueError("puzzle_2d must be a 2D NumPy array.")
        if not (isinstance(options_3d, np.ndarray) and options_3d.ndim == 3):
            raise ValueError("options_3d must be a 3D NumPy array.")

        iteration_count: int = 0
        while True:
            iteration_count += 1
            # Apply elimination technique
            has_progress, is_solved, puzzle_2d, options_3d = cls._apply_elimination(
                puzzle_2d, options_3d
            )
            if is_solved:
                break

            if not has_progress and iteration_count > 1:
                break

            # Apply hidden singles technique
            has_progress, is_solved, puzzle_2d, options_3d = cls._apply_hidden_singles(
                puzzle_2d, options_3d
            )
            if is_solved:
                break

            if has_progress:
                continue  # Restart the loop to apply elimination again

        return is_solved, puzzle_2d, options_3d
    
    @staticmethod
    def _update_puzzle(
        rows: np.array, cols: np.array, values: np.array, options_3d: np.ndarray
    ) -> np.ndarray:
        """
        Update the puzzle state based on the known values and options cube.

        This function updates the options cube by setting the possibilities for known values
        to zero in their respective rows, columns, and blocks. It then updates the puzzle state
        based on the options cube, setting cells with multiple options back to zero.

        Args:
            rows: A NumPy array of row indices for known values.
            cols: A NumPy array of column indices for known values.
            values: A NumPy array of values corresponding to the indices in rows and cols.
            options_3d: A 3D NumPy array representing the possible values for each cell.

        Returns:
            A tuple containing the updated 2D puzzle state and the updated 3D options cube.

        """
        # Set column, row, and block to zero for all known cells
        options_3d[rows, :, values] = 0
        options_3d[:, cols, values] = 0
        options_3d[rows, cols, :] = 0

        # Calculate the starting indices of the 3x3 blocks
        block_start_rows = 3 * (rows // 3)
        block_start_cols = 3 * (cols // 3)

        # Set the possibilities in the 3x3 blocks to zero
        for block_start_row, block_start_col, value in zip(
            block_start_rows, block_start_cols, values
        ):
            options_3d[
                block_start_row : block_start_row + 3,
                block_start_col : block_start_col + 3,
                value,
            ] = 0

        # Set known cells back to one
        options_3d[rows, cols, values] = 1

        # Update the puzzle state based on the options cube
        puzzle_2d = options_3d.argmax(axis=2) + 1
        # Reset cells with multiple options to zero
        puzzle_2d[options_3d.sum(axis=2) != 1] = 0

        return puzzle_2d, options_3d
    
    @staticmethod
    def _find_singles(options_3d: np.ndarray) -> Set[Tuple[int, int, int]]:
        """
        Find all unique options for digits in rows, columns, or subsquares of a Sudoku puzzle.

        This function identifies all cells within the options cube that represent the only possible
        choice for a digit in its respective row, column, or 3x3 subsquare. It is a part of the
        constraint propagation process in solving a Sudoku puzzle.

        Args:
            options_3d: A 3D NumPy array representing the possible values for each cell in the Sudoku
                        puzzle. The first two dimensions correspond to the puzzle grid, and the third
                        dimension contains a binary indicator for the possible values (1-9).

        Returns:
            A set of tuples, where each tuple contains the row index, column index, and value index
            of a cell that represents a unique option for that digit in its row, column, or subsquare.

        """
        singles = set()

        # Compute singles in columns
        cols_w_singles = np.argwhere(options_3d.sum(axis=0) == 1)
        row_indices = options_3d.argmax(axis=0)
        singles.update({(row_indices[c, v], c, v) for c, v in cols_w_singles})

        # Compute singles in rows
        rows_w_singles = np.argwhere(options_3d.sum(axis=1) == 1)
        col_indices = options_3d.argmax(axis=1)
        singles.update({(r, col_indices[r, v], v) for r, v in rows_w_singles})

        # Compute singles in 3x3 subsquares
        for x in range(0, 9, 3):
            for y in range(0, 9, 3):
                subsquare = options_3d[y:y+3, x:x+3, :]
                row_idx = np.argmax(subsquare.sum(axis=1), axis=0)
                col_idx = np.argmax(subsquare.sum(axis=0), axis=0)
                depth_idx = np.argwhere(subsquare.sum(axis=(0, 1)) == 1)

                singles.update({
                    (r[0] + y, c[0] + x, v)
                    for r, c, v in zip(row_idx[depth_idx], col_idx[depth_idx], depth_idx.flatten())
                })

        return singles

    @staticmethod
    def _compute_hidden_singles(
        singles: Set[Tuple[int, int, int]], puzzle_2d: np.ndarray
    ) -> Set[Tuple[int, int, int]]:
        """
        Compute the hidden singles for a Sudoku puzzle.

        A hidden single occurs when a cell is the only one in a row, column, or block
        that can accommodate a certain number. This function calculates the hidden singles
        by subtracting the known values from the set of all singles.

        Args:
            singles:    A set of tuples representing cells that are the only option for a digit
                        in their row, column, or subsquare.
            puzzle_2d:  A 2D NumPy array representing the current state of the Sudoku puzzle.

        Returns:
            A set of tuples representing the hidden singles in the puzzle.

        """
        # Compute already known values
        known_values = {(*kv, puzzle_2d[*kv] - 1) for kv in np.argwhere(puzzle_2d > 0)}

        # Return the hidden singles
        return singles - known_values
    
    @classmethod
    def _apply_elimination(
        cls, puzzle_2d: np.ndarray, options_3d: np.ndarray
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
        is_solved: bool = False

        iteration_count: int = 0
        while True:
            iteration_count += 1

            # Store the previous state of the puzzle to detect changes
            prev_puzzle_2d = puzzle_2d
            # prev_options_3d = options_3d.copy()
            # Find the indices of cells with known values
            known_cells = np.argwhere(puzzle_2d)

            # Extract row and column indices, and adjust values for 0-indexing
            rows, cols = known_cells[:, 0], known_cells[:, 1]
            values = puzzle_2d[rows, cols] - 1

            # Update puzzle based on known values
            puzzle_2d, options_3d = cls._update_puzzle(rows, cols, values, options_3d)

            # Check for changes in the puzzle state and break out if no progress was made this iteration
            has_progress = not np.array_equal(puzzle_2d, prev_puzzle_2d)
            if not has_progress:
                has_progress = iteration_count > 1
                break

            # Check if the puzzle is solved and breakout if it is
            is_solved = np.all(puzzle_2d > 0)
            if is_solved:
                break

        return has_progress, is_solved, puzzle_2d, options_3d
    
    @classmethod
    def _apply_hidden_singles(
        cls, puzzle_2d: np.ndarray, options_3d: np.ndarray
    ) -> Tuple[bool, bool, np.ndarray, np.ndarray]:
        """
        Apply the 'hidden singles' rule to the Sudoku puzzle.

        This function identifies 'hidden singles' in the puzzle, which are cells that are the only ones
        in a row, column, or block that can accommodate a certain number. It updates the puzzle state
        and options cube accordingly.

        Args:
            puzzle_2d:  A 2D NumPy array representing the current state of the Sudoku puzzle.
                        Each cell contains the value of the puzzle (1-9) or 0 if the value is unknown.
            options_3d: A 3D NumPy array representing the possible values for each cell.
                        The first two dimensions correspond to the puzzle grid, and the third dimension
                        contains a binary indicator for the possible values (1-9).

        Returns:
            A tuple containing:
            - has_progress: A boolean indicating if progress was made in the last iteration.
            - is_solved: A boolean indicating if the puzzle is solved.
            - puzzle_2d: The updated 2D puzzle state.
            - options_3d: The updated 3D options cube.

        """
        prev_puzzle_2d = puzzle_2d.copy()
        singles = cls._find_singles(options_3d)
        hidden_singles = cls._compute_hidden_singles(singles, puzzle_2d)

        if not hidden_singles:
            return False, False, puzzle_2d, options_3d

        # Update puzzle using newly discovered known values
        puzzle_2d, options_3d = cls._update_puzzle(*map(np.array, zip(*hidden_singles)), options_3d)

        has_progress: bool = not np.array_equal(puzzle_2d, prev_puzzle_2d)
        is_solved: bool = np.all(puzzle_2d > 0)

        return has_progress, is_solved, puzzle_2d, options_3d
    


class Backtracking(SolvingStrategy):
    
    def apply(cls, 
              puzzle_2d: np.ndarray, 
              options_3d: np.ndarray,
              max_iterations: int = 10_000_000
              ) -> Tuple[bool, np.ndarray, np.ndarray]:
        # TODO: Write doctring in style of other classes.
    
        num_possibilities: int = options_3d.sum(axis=2, dtype=np.longdouble).prod()

        # Return answer if only one possibility left
        if num_possibilities == 1:
            return True, puzzle_2d, options_3d
        
        
        if num_possibilities >= max_iterations or num_possibilities < 0:
            logging.info(
                f"More than {max_iterations:_} combinations to check, aborting..."
            )
            return False, puzzle_2d, options_3d
       
        # List options for each cell
        it = np.nditer(puzzle_2d, flags=["multi_index"])
        options_idx = [
            [
                (*it.multi_index, int(d))
                for d in np.nditer(np.where(options_3d[*it.multi_index, :] == 1)[0])
            ]
            for v in it
            if v == 0
        ]

        # Create the generator that iterates over the possible cell index updates
        generator = cls.generate_cell_index_updates(*options_idx)

        # Set-up first option
        for idx in next(generator):
            options_3d[*idx[1][:2], :] = 0
            options_3d[*idx[1]] = 1

        # Return first option if valid
        if SudokuValidator.validate_3d_solution(options_3d):
            return True, puzzle_2d, options_3d

        # Iterate over other options
        for changes in generator:
            for idx in changes:
                options_3d[*idx[0]] = 0
                options_3d[*idx[1]] = 1

            if SudokuValidator.validate_3d_solution(options_3d):
                return True, puzzle_2d, options_3d
            
        logging.info(f"No solution found after {max_iterations:_} combinations checked. Closing of...")
        
        return False, puzzle_2d, options_3d

    @staticmethod
    def generate_cell_index_updates(
        *iterables: Iterable[int],
    ) -> Generator[Tuple[Tuple[None, int], ...], None, None]:
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
        combinations: Iterator[Tuple[int, ...]] = product(*iterables)
        prev_comb: Tuple[int, ...] = None  # Initialize the previous combination

        comb = next(combinations)
        yield tuple((None, c) for c in comb)  # Yield indices for the first combination
        prev_comb = comb

        for comb in combinations:
            # Yield indices that are different from the previous combination
            yield tuple((p, c) for c, p in zip(comb, prev_comb) if c != p)
            prev_comb = comb
    
