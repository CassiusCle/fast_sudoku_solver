"""  
The `utils` module contains utility functions that assist in the solving of Sudoku puzzles.   
These functions include conversions between different representations of Sudoku puzzles,   
validation of solutions, and printing of puzzles for visualization.  

The module provides essential functionalities such as `iter_to_np_puzzle` which transforms   
a string representation of a Sudoku puzzle into a 2D array (the puzzle itself) and a 3D array   
(the potential values for each cell). The `np_puzzle_to_string` function performs the inverse   
operation, converting the 3D array representation back into a string format. The `validate_3d_solution`   
function ensures that a given solution adheres to the rules of Sudoku. The `generate_cell_index_updates`   
function creates a generator for efficiently iterating through the possible updates to a Sudoku puzzle during solving.  

Together, these utilities support the primary solving mechanism by providing data transformation   
and validation capabilities.  
"""  


import itertools
from typing import Generator, Iterable, Iterator, Union, Tuple, Optional

import numpy as np

PUZZLE_SIZE: int = 9
PUZZLE_DEPTH: int = 9
SHAPE_2D: Tuple[int, int] = (PUZZLE_SIZE, PUZZLE_SIZE)
SHAPE_3D: Tuple[int, int, int] = (PUZZLE_SIZE, PUZZLE_SIZE, PUZZLE_DEPTH)


def print_puzzle(puzzle: Union[str, int], solution: Optional[str] = None) -> None:
    """Prints a sudoku puzzle and its solution in a formatted way.

    Args:
        puzzle (str): A string or integer representing the initial sudoku puzzle.
        solution (str): An optional string representing the solution to the puzzle.
    """

    # Convert puzzle numbers to letters for readability to distinguish from solution values later
    alphabet = "abcdefghi"
    puzzle = "".join(
        [alphabet[int(c) - 1] if c not in [".", "0"] else c for c in str(puzzle)]
    )

    # Overlay solution onto puzzle if provided
    if solution:
        puzzle = "".join(
            [c1 if c1.isalpha() else c2 for c1, c2 in zip(puzzle, solution)]
        )

    # Helper function to divide a string into equal-sized chunks
    def chunk_string(string: str, chunk_size: int) -> list[str]:
        """Divides a string into chunks of equal size."""
        return [string[i : i + chunk_size] for i in range(0, len(string), chunk_size)]

    # Break the puzzle string into lines and 3x3 blocks
    digits_per_line: list = chunk_string(puzzle, 9)
    digits_per_line: list = [chunk_string(line, 3) for line in digits_per_line]

    # Define the horizontal and vertical lines for the sudoku grid
    hz_line = "─" * 9
    top_line = f"┌{hz_line}┬{hz_line}┬{hz_line}┐"
    mid_line = f"├{hz_line}┼{hz_line}┼{hz_line}┤"
    bottom_line = f"└{hz_line}┴{hz_line}┴{hz_line}┘"

    # Assemble the top line of the sudoku grid
    output = [top_line]
    for i, digits in enumerate(digits_per_line):
        # Join the 3x3 blocks with vertical lines
        output.append("│" + "│".join("".join(chunk) for chunk in digits) + "│")
        # Add middle lines after every third line to form grid
        if i in [2, 5]:
            output.append(mid_line)
    # Add the bottom line to complete the grid
    output.append(bottom_line)

    # Helper function to replace characters with formatted numbers
    def replace_chars(chars: str) -> str:
        """Replaces characters in the puzzle output with formatted numbers."""
        return "".join(
            (
                f"({alphabet.index(c) + 1})"
                if c.isalpha()
                else " . " if c in [".", "0"] else f" {c} " if c.isdigit() else c
            )
            for c in chars
        )

    # Print the final formatted sudoku grid
    print("\n".join(replace_chars(line) for line in output))


def validate_3d_solution(candidate_solution: np.ndarray) -> bool:
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
    # Check if all rows and columns contain each digit only once
    if not (
        np.all(candidate_solution.sum(axis=1) == 1)
        and np.all(candidate_solution.sum(axis=0) == 1)
    ):
        return False

    # Check if all cols contain each digit only once (sum is over rows)
    if not np.all(candidate_solution.sum(axis=0) == 1):
        return False

    # Check if boxes are valid (just the top left 4 need to be checked)
    for i in range(0, 6, 3):
        for j in range(0, 6, 3):
            if not np.all(
                candidate_solution[i : i + 3, j : j + 3].sum(axis=(0, 1)) == 1
            ):
                return False

    return True


def iter_to_np_puzzle(sudoku: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an iterable representing a sudoku puzzle into 2D and 3D NumPy array representations.

    The 2D NumPy array represents the current state of the Sudoku puzzle.
        Each cell contains the value of the puzzle (1-9) or 0 if the value is unknown.
    The 3D NumPy array representing the possible values for each cell.
        The first two dimensions correspond to the puzzle grid, and the third dimension
        contains a binary indicator for the possible values (1-9).

    Args:
        sudoku (str): A string representing a sudoku puzzle.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the 2D puzzle array and the 3D possibilities array.
    """
    if "." in sudoku:
        sudoku = sudoku.replace(".", "0")

    # Convert the string to a 2D numpy array
    puzzle_2d: np.ndarray = np.reshape(
        np.fromiter(sudoku, dtype="l"), newshape=SHAPE_2D
    )
    options_3d: np.ndarray = np.zeros(SHAPE_3D, dtype="l")  # [row][column][depth]

    # Update options_3d based on the non-zero values in puzzle_2d
    nonzero_indices = np.nonzero(puzzle_2d)
    values = puzzle_2d[nonzero_indices] - 1
    options_3d[nonzero_indices[0], nonzero_indices[1], values] = 1

    # Set all possibilities to 1 for cells with zero value in puzzle_2d
    zero_indices = np.where(puzzle_2d == 0)
    options_3d[zero_indices[0], zero_indices[1]] = 1

    return puzzle_2d, options_3d


def np_puzzle_to_string(np_puzzle: np.ndarray) -> str:
    """Converts a 3D NumPy array representing a Sudoku puzzle into a string.

    This method takes a 3D NumPy array where each 2D slice along the third axis represents
    the possibilities for each cell in the Sudoku grid. It converts this array into a string
    representation of the puzzle by taking the argmax along the third axis, adding one to
    shift from zero-based to one-based indexing, and then flattening the result into a 1D
    array. This array is then joined into a single string.

    Args:
        np_puzzle (np.ndarray): A 3D NumPy array representing the possibilities of a Sudoku puzzle.

    Returns:
        str:    A string representation of the Sudoku puzzle, with numbers representing the filled
                cells and zeros for the empty cells.

    Raises:
        ValueError: If `np_puzzle` is not a 3D NumPy array or if its shape does not conform to
                    the expected Sudoku puzzle shape.
    """
    if not isinstance(np_puzzle, np.ndarray) or len(np_puzzle.shape) != 3:
        raise ValueError(
            "The input must be a 3D NumPy array representing a Sudoku puzzle."
        )
    if np_puzzle.shape != SHAPE_3D:
        raise ValueError(
            f"Expected puzzle shape {SHAPE_3D}, but got {np_puzzle.shape}."
        )

    # Convert the 3D possibilities array into a 1D string representation
    puzzle_string: str = "".join(map(str, (np_puzzle.argmax(axis=2) + 1).flatten()))
    return puzzle_string


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
    combinations: Iterator[Tuple[int, ...]] = itertools.product(*iterables)
    prev_comb: Tuple[int, ...] = None  # Initialize the previous combination

    comb = next(combinations)
    yield tuple((None, c) for c in comb)  # Yield indices for the first combination
    prev_comb = comb

    for comb in combinations:
        # Yield indices that are different from the previous combination
        yield tuple((p, c) for c, p in zip(comb, prev_comb) if c != p)
        prev_comb = comb
