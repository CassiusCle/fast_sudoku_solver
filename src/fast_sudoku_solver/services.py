"""  
The `services` module provides classes and utility functions that assist in the solving of Sudoku puzzles.  
These classes include validation, conversion between different puzzle representations, and pretty-printing  
of Sudoku puzzles. The module ensures that solutions adhere to the rules of Sudoku, and supports the primary  
solving mechanism by providing data transformation and validation capabilities.  
"""

from typing import Tuple, Union, List

import numpy as np

class SudokuValidator:
    """Class that provides validation methods for Sudoku puzzles."""  

    @staticmethod
    def validate_3d_solution(candidate_solution: np.ndarray) -> bool:
        """Check if a Sudoku solution is valid.

        Validates a 9x9x9 3D array representing a Sudoku puzzle solution. Each layer in the third dimension
        corresponds to the positions of (n+1)s in the solution. The validation ensures that each number appears  
        only once per row, column, and 3x3 box, as per Sudoku rules. Validation strategy was inspired by this
        MathOverflow post (https://mathoverflow.net/questions/129143/verifying-the-correctness-of-a-sudoku-solution).

        Args:
            candidate_solution: A 3D array representing a proposed Sudoku solution.

        Raises:
            TypeError: If candidate_solution is not a numpy.ndarray.
            ValueError: If candidate_solution does not have the correct shape (9x9x9).
            ValueError: If candidate_solution does not contain exactly one digit for each field.

        Returns:
            True if the solution is valid, False otherwise.
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

class SudokuFormatter:
    """Class that provides formatting methods for Sudoku puzzles."""
    
    PUZZLE_SIZE: int = 9
    PUZZLE_DEPTH: int = 9
    SHAPE_2D: Tuple[int, int] = (PUZZLE_SIZE, PUZZLE_SIZE)
    SHAPE_3D: Tuple[int, int, int] = (PUZZLE_SIZE, PUZZLE_SIZE, PUZZLE_DEPTH)
    
    @classmethod
    def convert_to_numpy(cls, sudoku: str) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a string representing a sudoku puzzle into 2D and 3D NumPy array representations.

        The 2D NumPy array represents the current state of the Sudoku puzzle.
            Each cell contains the value of the puzzle (1-9) or 0 if the value is unknown.
        The 3D NumPy array representing the possible values for each cell.
            The first two dimensions correspond to the puzzle grid, and the third dimension
            contains a binary indicator for the possible values (1-9).

        Args:
            sudoku: A string representing a sudoku puzzle.

        Returns:
            A tuple containing the 2D puzzle array and the 3D possibilities array.
        """
        sudoku = sudoku.replace(".", "0")  

        # Convert the string to a 2D numpy array
        puzzle_2d: np.ndarray = np.reshape(np.fromiter(sudoku, dtype="l"), newshape=cls.SHAPE_2D)
        options_3d: np.ndarray = np.zeros(cls.SHAPE_3D, dtype="l")  # [row][column][depth]

        # Update options_3d based on the non-zero values in puzzle_2d
        nonzero_indices = np.nonzero(puzzle_2d)
        values = puzzle_2d[nonzero_indices] - 1
        options_3d[nonzero_indices[0], nonzero_indices[1], values] = 1

        # Set all possibilities to 1 for cells with zero value in puzzle_2d
        zero_indices = np.where(puzzle_2d == 0)
        options_3d[zero_indices[0], zero_indices[1]] = 1

        return puzzle_2d, options_3d
    
    @classmethod
    def convert_to_string(cls, np_puzzle: np.ndarray) -> str:
        """Converts a 3D NumPy array representing a Sudoku puzzle into a string.

        This method takes a 3D NumPy array where each 2D slice along the third axis represents
        the possibilities for each cell in the Sudoku grid. It converts this array into a string
        representation of the puzzle by taking the argmax along the third axis, adding one to
        shift from zero-based to one-based indexing, and then flattening the result into a 1D
        array. This array is then joined into a single string.

        Args:
            np_puzzle: A 3D NumPy array representing the possibilities of a Sudoku puzzle.

        Raises:
            ValueError: If `np_puzzle` is not a 3D NumPy array or if its shape does not conform to
                        the expected Sudoku puzzle shape.
        
        Returns:
            A string representation of the Sudoku puzzle.
        """
        if not isinstance(np_puzzle, np.ndarray) or len(np_puzzle.shape) != 3:
            raise ValueError(
                "The input must be a 3D NumPy array representing a Sudoku puzzle."
            )
        if np_puzzle.shape != cls.SHAPE_3D:
            raise ValueError(
                f"Expected puzzle shape {cls.SHAPE_3D}, but got {np_puzzle.shape}."
            )

        # Convert the 3D possibilities array into a 1D string representation
        puzzle_string: str = "".join(map(str, (np_puzzle.argmax(axis=2) + 1).flatten()))
        return puzzle_string
    
    @staticmethod
    def _chunk_string(string: str, chunk_size: int) -> List[str]:
        """Divides a string into chunks of equal size.  
  
        Args:  
            string: The string to be divided.  
            chunk_size: The size of each chunk.  
  
        Returns:  
            A list of string chunks.  
        """
        return [string[i : i + chunk_size] for i in range(0, len(string), chunk_size)]
    
    @staticmethod    
    def _replace_chars(chars: str, solution: str = None, alphabet: str = "abcdefghi") -> str:
        """Replaces characters in the puzzle output with formatted numbers or placeholders.  
  
        Args:  
            chars: The string containing characters to replace.  
            solution: The string representing the solution to overlay on the puzzle.  
            alphabet: A string representing the alphabet used for replacement.  
  
        Returns:  
            The formatted string with characters replaced.  
        """
        
        formatted_chars = []
        for c in chars:
            if c.isalpha():
                if solution is None:
                    formatted_chars.append(f" {alphabet.index(c) + 1} ")
                else:    
                    formatted_chars.append(f"({alphabet.index(c) + 1})")
            elif c in [".", "0"]:
                formatted_chars.append(" . ")
            elif c.isdigit():
                formatted_chars.append(f" {c} ")
            else:
                formatted_chars.append(c)
        return "".join(formatted_chars)
    
    @classmethod
    def print(cls, puzzle: Union[str, int] = None, solution: str = None) -> None:
        """Prints a sudoku puzzle and its solution in a formatted way.  
  
        Args:  
            puzzle: A string or integer representing the initial sudoku puzzle.  
            solution: An optional string representing the solution to the puzzle.  
        """

        # Convert puzzle numbers to letters for readability to distinguish from solution values later
        alphabet = "abcdefghi"
        if puzzle is not None:    
            puzzle = "".join(
                [alphabet[int(c) - 1] if c not in [".", "0"] else c for c in str(puzzle)]
            )
        else:
            puzzle = "." * 81
        
        # Overlay solution onto puzzle if provided
        if solution is not None:
            puzzle = "".join(
                [c1 if c1.isalpha() else c2 for c1, c2 in zip(puzzle, solution)]
            )

        # Break the puzzle string into lines and 3x3 blocks
        digits_per_line: list = cls._chunk_string(puzzle, 9)
        digits_per_line: list = [cls._chunk_string(line, 3) for line in digits_per_line]

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

        # Print the final formatted sudoku grid
        print("\n".join(cls._replace_chars(line, solution, alphabet) for line in output))