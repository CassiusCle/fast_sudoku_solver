import numpy as np
import itertools
from typing import Optional, Tuple, Union
from functools import reduce
import logging

# Set up logging configuration  
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') 

class SudokuSolver:
    """A class for solving 9x9 Sudoku puzzles."""

    # Class-level constants for validating solutions
    ROW_INDICES: list = [(f'r{i+1}', np.arange(i*9, i*9+9)) for i in range(9)]
    COL_INDICES: list = [(f'c{i+1}', np.arange(i, 81, 9)) for i in range(9)]
    SUBSQ_INDICES: list = [(f's{k+1}', np.concatenate([np.arange(start, stop) for start, stop in ranges])) 
                     for k, ranges in enumerate([[(i*9+j*3, i*9+3+j*3) for i in range(3)] for j in [0, 1, 2, 9, 10, 11, 18, 19, 20]])]
    _dt: np.dtype = np.dtype([('name', np.unicode_, 16), ('values', np.int32, (9,))])   
    VALIDATION_IDX: np.array = np.array(ROW_INDICES[:8] + COL_INDICES + SUBSQ_INDICES[0:2] + SUBSQ_INDICES[3:5], dtype=_dt)
    ALL_DIGITS: np.array = np.arange(1, 10)

    def __init__(self):
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
    
    def validate_solution(self, candidate_solution: str | list) -> Tuple[Union[str, None], bool]:
        """Validates if the solution is correct for a Sudoku puzzle.

        This method checks the minimum number of rows, columns and 3x3 subgrids to ensure that 
        each contains all digits from 1 to 9 without repetition. The approach for
        solution validation is based on the one outlined in a MathOverflow post
        (https://mathoverflow.net/questions/129143/verifying-the-correctness-of-a-sudoku-solution).

        Note: VALIDATION_IDX and ALL_DIGITS are defined as globals within the global
        scope of the Sudoku class for performance reasons.

        Returns:
            bool: True if the solution is valid, False otherwise.
        """
        if len(candidate_solution) != 81:
            raise ValueError('Candidate solution does not contain enough elements.')
        elif type(candidate_solution) not in (str, list):
            logging.warning(str(candidate_solution))
            raise TypeError(f'Candidate solution is not of type str or list: {type(candidate_solution)}.')
        
        solution_array = np.array(list(candidate_solution), dtype=int)
        for grouping, idx in SudokuSolver.VALIDATION_IDX:
            if not np.array_equal(np.sort(np.unique(solution_array[idx])), SudokuSolver.ALL_DIGITS):
                self.solved = False
                return (grouping, False)
        return (None, True)
    
    def _prune_filled_values(self, candidates_per_field: list[list[int]], idx: list[int]) -> list[list[int]]:
        """Prune filled values from Sudoku fields within a specified grouping.

        This function removes candidates from the fields that aren't complete yet in cases where 
        that candidate is already present in a grouping that it is part of. 

        Args:
            combinatorial_elements (List[List[int]]): Current state of the Sudoku board.
            idx (List[int]): Indices of the board fields that belong to a certain grouping
                            (e.g. a row, column or subsquare).

        Returns:
            List[List[int]]: The original combinatorial_elements but with the group_candidates 
                            replaced by the pruned_candidates.
        """
        group_candidates = [candidates_per_field[i] for i in idx]
        filled_values = [v[0] for v in group_candidates if len(v) == 1]
        pruned_candidates = [
            [option for option in field_options if option not in filled_values] 
            if len(field_options) != 1 else field_options 
            for field_options in group_candidates
        ]

        for i, candidate in zip(idx, pruned_candidates):
            candidates_per_field[i] = candidate
        
        return candidates_per_field

    def _count_combinations(self, candidates_per_field: list[list[int]]) -> int:
        """Calculate the total number of combinations.

        Args:
            candidates_per_field (List[List[int]]): A list of lists, where each inner list contains the possible candidates for a field.

        Returns:
            int: The total number of combinations.
        """
        return reduce(lambda x, y: x * len(y), candidates_per_field, 1)

    def solve(self, unsolved_sudoku: str, verbose: bool = False) -> str:
        """Solve the Sudoku puzzle.

        This method solves the Sudoku puzzle by first pruning the candidates based on filled values until no further reduction is possible.
        If only one combination is left, it returns the solution. Otherwise, it brute forces solutions.

        Args:
            unsolved_sudoku (str): The unsolved Sudoku puzzle in string format.
            verbose (bool, optional): If True, print the iteration at which the solution was found. Defaults to False.
            fill_from_top (bool, optional): If True, fill the Sudoku from top. Defaults to True.

        Returns:
            str: The solved Sudoku puzzle in string format.
        """    
        candidates_per_field = [list(range(1,10)) if i == '0' else [int(i)] for i in list(unsolved_sudoku)]

        # Prune the candidates based on filled values until no further reduction is possible
        current_combinations = self._count_combinations(candidates_per_field)
        while True:
            previous_combinations = current_combinations
            for _, idx in SudokuSolver.ROW_INDICES+SudokuSolver.COL_INDICES+SudokuSolver.SUBSQ_INDICES:
                candidates_per_field = self._prune_filled_values(idx=idx, candidates_per_field=candidates_per_field)
            
            current_combinations = self._count_combinations(candidates_per_field)
            
            if current_combinations >= previous_combinations or current_combinations == 1:
                break
        
        # Return solution in case only one combination is left (=solved)
        if current_combinations == 1:
            return ''.join([str(c[0]) for c in candidates_per_field])

        if current_combinations >= 10_000_000:
            logging.warning('More than 10,000,000 combinations to check, aborting...')
            return None
        
        # Brute force solutions        
        combinations = itertools.product(*candidates_per_field)
        i = 0
        for combination in combinations:
            i += 1
            if self.validate_solution(candidate_solution=list(combination))[1]:
                break
        if verbose: print(f'Solution found at iteration: {i} of {current_combinations}')
        return ''.join([str(i) for i in combination])

        
        


    
