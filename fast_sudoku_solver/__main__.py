"""  
This module serves as the entry point for the sudoku_solver package.  
 
It allows users to solve Sudoku puzzles provided as a command-line argument.  
The puzzle should be a flattened string of 81 characters where each character  
represents a cell in the puzzle (1-9 for filled cells, '0' or '.' for empty cells).  

Example usage:  
    python -m sudoku_solver "................................................................................."  
"""

import sys 

from fast_sudoku_solver.sudoku import Sudoku
from fast_sudoku_solver.utils import print_puzzle

def main() -> None:
    """  
    The main function of the script. It parses the command line argument and attempts to solve the provided Sudoku puzzle.  
    Exits the program with an appropriate message if an error occurs or if the puzzle is solved successfully.  
    """  
    if len(sys.argv) != 2:
        print("Usage: python -m sudoku_solver <flattened_puzzle>")
        sys.exit(1)
        
    unsolved_puzzle = sys.argv[1]
    if len(unsolved_puzzle) != 81 or not all(c.isdigit() or c == '.' for c in unsolved_puzzle):
        print("Error: The puzzle must be a string of 81 characters, containing digits 1-9 and '0' or '.' for empty fields.")
        sys.exit(1)
        
    try:
        solution = Sudoku.solve(unsolved_sudoku=unsolved_puzzle)
        if solution:  
            print("Solved Sudoku:")  
            print_puzzle(puzzle=unsolved_puzzle, solution=solution)
            print(f'Flattened solution: {solution}')  
        else:  
            print("No solution has been found for the provided Sudoku puzzle.")  
            sys.exit(1)
    except ValueError as e:
        print(f"An error occured while solving the sudoku: : {e}")
        sys.exit(1)    

if __name__ == "__main__":
    main()