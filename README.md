# fast_sudoku_solver
  
## Description  
  
This is a Python-project for solving 9x9 Sudoku puzzles using constraint propagation techniques combined with brute force search when necessary. The solver leverages the power of NumPy for numerical computations, making it highly performant for a Python based solver. 
It includes functionality to validate solutions, solve puzzles, and compute the number of possible value combinations for an unsolved Sudoku puzzle.
  
## Installation  
To run the Sudoku Solver, you will need Python 3.9 or higher. You can install it in one of two ways:

### Option 1: Install from PyPI using pip
```bash 
pip install fast_sudoku_solver
```

### Option 2: Clone the repository
Clone the repository to your local machine:

```bash
git clone https://github.com/CassiusCle/fast_sudoku_solver
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

When running Sudoku Solver from the cloned repository, it is advised to install it as a package in "editable" mode using:
```bash
pip install -e .
```
 
## Usage
There are two main ways of using the Sudoku Solver, through the command-line or within Python.

N.B.: The solver was designed to take in Sudoku puzzles in the form of strings. Here the characters in the string represent the flattened Sudoku grid and empty cells are denoted by either "0" or ".".

### Command line
To run the Sudoku Solver from the command-line, simply run a command like below with your unsolved Sudoku:
```bash
python -m fast_sudoku_solver ..7........5.4..7..695...31...4.58.2.5..2..4.6.23.1...29...358..3..1.2........3..
```

```bash
> Solved Sudoku:
> ┌─────────┬─────────┬─────────┐
> │ 4  1 (7)│ 9  3  8 │ 6  2  5 │
> │ 3  2 (5)│ 1 (4) 6 │ 9 (7) 8 │
> │ 8 (6)(9)│(5) 7  2 │ 4 (3)(1)│
> ├─────────┼─────────┼─────────┤
> │ 1  7  3 │(4) 9 (5)│(8) 6 (2)│
> │ 9 (5) 8 │ 6 (2) 7 │ 1 (4) 3 │
> │(6) 4 (2)│(3) 8 (1)│ 7  5  9 │
> ├─────────┼─────────┼─────────┤
> │(2)(9) 1 │ 7  6 (3)│(5)(8) 4 │
> │ 5 (3) 6 │ 8 (1) 4 │(2) 9  7 │
> │ 7  8  4 │ 2  5  9 │(3) 1  6 │
> └─────────┴─────────┴─────────┘
> Flattened solution: 417938625325146978869572431173495862958627143642381759291763584536814297784259316
```

### Python
The code examples below show a few of the functionalities of the package. Please also see the `examples/example_usage.py` script and the various Sudoku examples that are included in the repository.

#### Solving a Sudoku
```python 
from fast_sudoku_solver.sudoku_solver import SudokuSolver  
 
unsolved_puzzle = "..7........5.4..7..695...31...4.58.2.5..2..4.6.23.1...29...358..3..1.2........3.."
solution: str = SudokuSolver.solve(unsolved_puzzle) 

# Print solution as string
print("Solved Puzzle:", solution) 
```
 
#### Validating a solution:
```python 
is_valid: bool = SudokuSolver.validate(solution)

print("Is the solution valid?", is_valid)  
```

#### Printing a Sudoku in a formatted grid
```python 
from fast_sudoku_solver.services import SudokuFormatter

# Pretty print orignal (unsolved) puzzle
SudokuFormatter.print(puzzle=unsolved_puzzle)

# Pretty print solution only
SudokuFormatter.print(solution=solution)

# Pretty print original puzzle overlaid with solution
SudokuFormatter.print(puzzle=unsolved_puzzle, solution=solution) 
```

## Testing

To run the tests for the Sudoku Solver, navigate to the project root and execute:

```python
python pytest 
```

N.B.: Testing is yet to be implemented in a later version.

## Contributing

If you'd like to contribute to the Sudoku Solver project, please feel free to make a pull request.

## License
 
This project is licensed under the MIT License - see the LICENSE file for details.