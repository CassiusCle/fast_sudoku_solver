from typing import Union, Optional, List

def print_sudoku_puzzle(puzzle: Union[str, int], solution: Optional[str] = None) -> None:
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
    def chunk_string(string: str, chunk_size: int) -> List[str]:
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