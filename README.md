# Sudoku Solver  
  
## Description  
  
This Sudoku Solver is a Python program designed to solve 9x9 Sudoku puzzles using constraint propagation techniques combined with brute force search when necessary. The solver leverages the power of NumPy for numerical computations, making it highly performant for a Python based solver.
  
## Installation  
To run the Sudoku Solver, you will need Python 3.11 or higher. You can install it in one of two ways:

### Option 1: Install from PyPI using pip
```bash 
pip install fast_sudoku_solver
```

### Option 2: Clone the repository
Clone the repository to your local machine:

```bash
git clone https://github.com/CassiusCle/fast_sudoku_solver
cd sudoku_solver
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

When running Sudoku Solver from the cloned repository, it is advised to still install it as a package in "editable" mode using:
```bash
pip install -e .
```
 
## Usage
There are two main ways of using the Sudoku Solver, through the command-line or within Python.

N.B.: The solver was designed to take in Sudoku puzzles in the form of strings. Here the characters in the string represent the flattened Sudoku grid and empty cells are denoted by either "0" or ".".

### Command line
To run the Sudoku Solver from the command-line, simply run a command like below with your unsolved Sudoku:
```bash
python -m sudoku_solver ..7........5.4..7..695...31...4.58.2.5..2..4.6.23.1...29...358..3..1.2........3..
```

### Python
The code examples below show a few of the functionalities of the package. Please also see the `examples/example_usage.py` script and the example puzzles that are included in this repository.

#### Solving a Sudoku
```python 
from sudoku_solver.sudoku import Sudoku  
 
unsolved_puzzle = "..7........5.4..7..695...31...4.58.2.5..2..4.6.23.1...29...358..3..1.2........3.."
solution: str = sudoku_solver.solve(puzzle) 

print("Solved Puzzle:", solution)  
```
 
#### Validating a solution:
```python 
is_valid: bool = Sudoku.validate_solution(solution)

print("Is the solution valid?", is_valid)  
```

#### Printing a Sudoku in a formatted grid
```python 
from sudoku_solver.utils import print_puzzle

print_puzzle(puzzle=unsolved_puzzle)

print_puzzle(solution=unsolved_puzzle)

# Print original puzzle overlaid with solution
print_puzzle(puzzle=puzzle, solution=solution) 
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