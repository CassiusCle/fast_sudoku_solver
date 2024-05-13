# Sudoku Solver  
  
## Description  
  
This Sudoku Solver is a Python program designed to solve 9x9 Sudoku puzzles using constraint propagation techniques combined with brute force search when necessary. The solver leverages the power of NumPy for numerical computations and provides a command-line interface for easy interaction.  
  
## Installation  
  
To run the Sudoku Solver, you will need Python 3.6 or higher. Clone the repository to your local machine:  
  
```bash  
git clone https://github.com/cassiuscle/Sudoku_Solver.git  
cd Sudoku_Solver  
```

Install the required dependencies:

```bash  
pip install -r requirements.txt  
```
 
## Usage
You can use the Sudoku Solver by importing the Sudoku class from sudoku_solver.sudoku and creating an instance:

```python 
from sudoku_solver.sudoku import Sudoku  
  
sudoku_solver = Sudoku()  
puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"  
solution = sudoku_solver.solve(puzzle)  
print("Solved Puzzle:", solution)  
```
 
To validate a solution:

```python 
is_valid = sudoku_solver.validate_solution(solution)  
print("Is the solution valid?", is_valid)  
```
 
## Testing

To run the tests for the Sudoku Solver, navigate to the project root and execute:

```python
python -m unittest discover tests  
```

## Contributing

If you'd like to contribute to the Sudoku Solver project, please feel free to make a pull request.

## License
 
This project is licensed under the MIT License - see the LICENSE file for details.