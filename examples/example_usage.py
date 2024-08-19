from fast_sudoku_solver.sudoku import Sudoku
from fast_sudoku_solver.utils import print_puzzle
from examples.sudoku_puzzles import example_sudokus

# Function to solve and print a Sudoku puzzle
def solve_and_print(puzzle_number: str, puzzle: str) -> None:
    """ Solves a Sudoku puzzle and prints the unsolved and solved versions. """
    print(f"Solving puzzle no. {puzzle_number}", end="\n")
    print("Unsolved puzzle:")
    print_puzzle(puzzle=puzzle)

    try:
        solution = Sudoku.solve(puzzle)
        print("Solved puzzle:")
        print_puzzle(puzzle=puzzle, solution=solution)
    except Exception as e:
        print(f"An error occurred while solving  and printing the puzzle: {e}")

def main() -> None:
    """ Main function to load and solve example Sudoku puzzles. """
    # Select the first example sudoku puzzle
    unsolved_puzzle = list(example_sudokus.items())[0]

    # Solve and print the puzzle
    solve_and_print(*unsolved_puzzle)

if __name__ == "__main__":
    main()

# Example output:
# > Solving puzzle no. 124
# > Unsolved puzzle:
# > ┌─────────┬─────────┬─────────┐
# > │ .  .  7 │ .  .  . │ .  .  . │
# > │ .  .  5 │ .  4  . │ .  7  . │
# > │ .  6  9 │ 5  .  . │ .  3  1 │
# > ├─────────┼─────────┼─────────┤
# > │ .  .  . │ 4  .  5 │ 8  .  2 │
# > │ .  5  . │ .  2  . │ .  4  . │
# > │ 6  .  2 │ 3  .  1 │ .  .  . │
# > ├─────────┼─────────┼─────────┤
# > │ 2  9  . │ .  .  3 │ 5  8  . │
# > │ .  3  . │ .  1  . │ 2  .  . │
# > │ .  .  . │ .  .  . │ 3  .  . │
# > └─────────┴─────────┴─────────┘
# > Solved puzzle:
# > ┌─────────┬─────────┬─────────┐
# > │ 4  1 (7)│ 9  3  8 │ 6  2  5 │
# > │ 3  2 (5)│ 1 (4) 6 │ 9 (7) 8 │
# > │ 8 (6)(9)│(5) 7  2 │ 4 (3)(1)│
# > ├─────────┼─────────┼─────────┤
# > │ 1  7  3 │(4) 9 (5)│(8) 6 (2)│
# > │ 9 (5) 8 │ 6 (2) 7 │ 1 (4) 3 │
# > │(6) 4 (2)│(3) 8 (1)│ 7  5  9 │
# > ├─────────┼─────────┼─────────┤
# > │(2)(9) 1 │ 7  6 (3)│(5)(8) 4 │
# > │ 5 (3) 6 │ 8 (1) 4 │(2) 9  7 │
# > │ 7  8  4 │ 2  5  9 │(3) 1  6 │
# > └─────────┴─────────┴─────────┘