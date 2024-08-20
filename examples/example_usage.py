"""Example usage script for the fast_sudoku_solver package.  
  
This script demonstrates how to use the SudokuSolver class to solve Sudoku puzzles provided  
as strings. It imports example puzzles from a separate module and utilizes the SudokuFormatter  
for printing the unsolved and solved puzzles.  
  
Example output:  
> Solving puzzle no. 124  
> Unsolved puzzle:  
> ┌─────────┬─────────┬─────────┐  
> ...  
> Solved puzzle:  
> ┌─────────┬─────────┬─────────┐  
> ...  
"""  

from fast_sudoku_solver.sudoku_solver import SudokuSolver
from fast_sudoku_solver.services import SudokuFormatter
from examples.sudoku_puzzles import example_sudokus

def solve_and_print(puzzle_number: int, unsolved_puzzle: str) -> None:  
    """Solves a Sudoku puzzle and prints both the unsolved and solved versions.  
  
    Args:  
        puzzle_number: The identifier for the puzzle.  
        unsolved_puzzle: The string representation of the unsolved Sudoku puzzle.  
    """
    print(f"Solving puzzle no. {puzzle_number}\n")
    print("Unsolved puzzle:")
    SudokuFormatter.print(puzzle=unsolved_puzzle)

    try:
        solution = SudokuSolver.solve(unsolved_puzzle)
        print("Solved puzzle:")
        SudokuFormatter.print(puzzle=unsolved_puzzle, solution=solution)
    except Exception as e:
        print(f"An error occurred while solving  and printing the puzzle: {e}")

def main() -> None:
    """ Main function to load and solve example Sudoku puzzles. """
    # Select and solve the first example Sudoku puzzle  
    puzzle_number, unsolved_puzzle = next(iter(example_sudokus.items()))  
    solve_and_print(puzzle_number, unsolved_puzzle)  

if __name__ == "__main__":
    main()