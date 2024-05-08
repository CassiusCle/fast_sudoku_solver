from typing import Tuple

import numpy as np

def apply_elimination(puzzle_2d: np.ndarray, 
                        options_3d: np.ndarray
                        ) -> Tuple[bool, bool, np.ndarray, np.ndarray]:
    """
    Apply basic elimination rules to the Sudoku puzzle until no further progress is made.

    This method iteratively applies Sudoku elimination rules to the given puzzle.
    It updates the puzzle state and the options cube until the puzzle is solved
    or no more progress can be made.

    Args:
        puzzle_2d: A 2D NumPy array representing the current state of the Sudoku puzzle.
        options_3d: A 3D NumPy array representing the possible values for each cell.

    Returns:
        A tuple containing:
        - has_progress: A boolean indicating if progress was made in the last iteration.
        - is_solved: A boolean indicating if the puzzle is solved.
        - puzzle_2d: The updated 2D puzzle state.
        - options_3d: The updated 3D options cube.
    """
    
    i = 0
    has_progress = False 
    is_solved = False

    while True:
        # Store the previous state of the puzzle to detect changes
        prev_puzzle_2d = puzzle_2d
        # prev_options_3d = options_3d.copy()
        # Find the indices of cells with known values
        known_cells = np.argwhere(puzzle_2d)
        
        # Extract row and column indices, and adjust values for 0-indexing
        rows, cols = known_cells[:, 0], known_cells[:, 1]
        values = puzzle_2d[rows, cols] - 1

        # Eliminate options based on known cell values
        options_3d[rows, :, values] = 0
        options_3d[:, cols, values] = 0
        box_start_rows, box_start_cols = 3 * (rows // 3), 3 * (cols // 3)
        for box_start_row, box_start_col, value in zip(box_start_rows, box_start_cols, values):
            options_3d[box_start_row:box_start_row+3, box_start_col:box_start_col+3, value] = 0

        # Set known cells back to one
        options_3d[rows, cols, values] = 1

        # Update the puzzle state based on the options cube
        puzzle_2d = options_3d.argmax(axis=2) + 1
        # Reset cells with multiple options to zero
        puzzle_2d[options_3d.sum(axis=2) != 1] = 0

        i += 1

        # Check for changes in the puzzle state to determine progress
        # if np.array_equal(options_3d, prev_options_3d): # TODO: Change to 3D?  (better flow then with "solved")
        if np.array_equal(puzzle_2d, prev_puzzle_2d):  # TODO: Change to 3D? (better flow then with "solved")
            if i > 1: 
                has_progress = True
            else:
                has_progress = False
            break
        
        # Check if the puzzle is solved
        if puzzle_2d.sum() == 405:
            is_solved = True
            break
    
    return has_progress, is_solved, puzzle_2d, options_3d

def apply_hidden_singles(puzzle_2d: np.ndarray, 
                            options_3d: np.ndarray
                        ) -> Tuple[bool, bool, np.ndarray, np.ndarray]:
    """
    Apply the 'hidden singles' rule to the Sudoku puzzle until no further progress is made.

    # This method iteratively applies Sudoku elimination rules to the given puzzle.
    # It updates the puzzle state and the options cube until the puzzle is solved
    # or no more progress can be made.

    Args:
        puzzle_2d: A 2D NumPy array representing the current state of the Sudoku puzzle.
        options_3d: A 3D NumPy array representing the possible values for each cell.

    Returns:
        A tuple containing:
        - has_progress: A boolean indicating if progress was made in the last iteration.
        - is_solved: A boolean indicating if the puzzle is solved.
        - puzzle_2d: The updated 2D puzzle state.
        - options_3d: The updated 3D options cube.
    """
    # Store the previous state of the puzzle to detect changes
    # prev_options_3d = options_3d.copy()
    prev_puzzle_2d = puzzle_2d

    # Compute columns with singles
    cols_w_singles = np.argwhere((options_3d.sum(axis=0) == 1)) # 3d
    # Compute the row on which the single is found
    row_indices = options_3d.argmax(axis=0)
    # Singles from columns
    singles = {(row_indices[c,v], c, v) for c, v in cols_w_singles}
    
    # Compute rows with singles
    rows_w_singles = np.argwhere((options_3d.sum(axis=1) == 1)) # 3d
    # Compute the colum on which the single is found
    col_indices = options_3d.argmax(axis=1)
    # Singles from rows
    singles.update({(r, col_indices[r,v], v) for r, v in rows_w_singles})
    

    for x in range(0, 9, 3):
        for y in range(0, 9, 3):
            # Compute row on which single is found
            row_idx = np.argmax(options_3d[y:y+3, x:x+3, :].sum(axis=1), axis=0)

            # Compute column on which single is found
            column_agg = options_3d[y:y+3, x:x+3, :].sum(axis=0)
            col_idx = np.argmax(column_agg, axis=0)

            # Compute depth on which single is found
            depth_idx = np.argwhere(column_agg.sum(axis=0) == 1)

            # Singles from this subsquare
            singles.update({(r[0]+y, c[0]+x, v[0]) for r, c, v in zip(row_idx[depth_idx], col_idx[depth_idx], depth_idx)})

    # Compute already known values
    known_values = {(*kv, puzzle_2d[*kv]-1) for kv in np.argwhere(puzzle_2d)}
    
    # Compute hidden singles
    hidden_singles: set = singles - known_values

    # Return in case there are no hidden singles
    if not hidden_singles:
        has_progress = False
        is_solved = False
        return has_progress, is_solved, puzzle_2d, options_3d

    rows, cols, values = zip(*hidden_singles)            
    rows = np.array(rows)
    cols = np.array(cols)
    values = np.array(values)

    # Set column, row, and box to zero for all known cells
    options_3d[rows, :, values] = 0
    options_3d[:, cols, values] = 0
    options_3d[rows, cols, :] = 0
    box_start_rows, box_start_cols = 3 * (rows // 3), 3 * (cols // 3)
    for box_start_row, box_start_col, value in zip(box_start_rows, box_start_cols, values):
        options_3d[box_start_row:box_start_row+3, box_start_col:box_start_col+3, value] = 0

    # Set known cells back to one
    options_3d[rows, cols, values] = 1

    puzzle_2d = options_3d.argmax(axis=2) + 1
    puzzle_2d[options_3d.sum(axis=2) != 1] = 0

    # Check for changes in the puzzle state to determine progress
    # if np.array_equal(options_3d, prev_options_3d): # TODO: Change to 3D?  (better flow then with "solved")
    if np.array_equal(puzzle_2d, prev_puzzle_2d): # TODO: Change to 3D?  (better flow then with "solved")
        has_progress = False
    else:
        has_progress = True
            
    # Check if the puzzle is solved
    if puzzle_2d.sum() == 405:
        is_solved = True
    else:
        is_solved = False
    
    return has_progress, is_solved, puzzle_2d, options_3d