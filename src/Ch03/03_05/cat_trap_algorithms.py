"""
Cat Trap Algorithms

This is the relevant code for the LinkedIn Learning Course 
AI Algorithms for Game Design with Python, by Eduardo Corpeño.

For the GUI, this code uses the Cat Trap UI VSCode extension
included in the extensions folder.
"""

import random
import copy
import time
import numpy as np

# Constants
CAT_TILE = 6
BLOCKED_TILE = 1
EMPTY_TILE = 0
LAST_CALL_MS = 0.5
VERBOSE = True

class CatTrapGame:
    """
    Represents a Cat Trap game state. Includes methods for initializing the game board, 
    managing game state, and selecting moves for the cat using different algorithms.
    """

    def __init__(self, size):
        self.cat = [size // 2] * 2
        self.size = size
        self.hexgrid = np.full((size, size), EMPTY_TILE)
        self.hexgrid[tuple(self.cat)] = CAT_TILE
        self.deadline = 0
        self.terminated = False
        self.start_time = time.time()
        self.reached_max_depth = False
        self.max_depth = float('inf')

    def initialize_random_hexgrid(self):
        """Randomly initialize blocked hexgrid."""
        num_blocks = random.randint(round(0.067 * (self.size**2)), round(0.13 * (self.size**2)))
        count = 0
        self.hexgrid[tuple(self.cat)] = CAT_TILE

        while count < num_blocks:
            r = random.randint(0, self.size - 1)
            c = random.randint(0, self.size - 1)
            if self.hexgrid[r, c] == EMPTY_TILE:
                self.hexgrid[r, c] = BLOCKED_TILE
                count += 1    
        if VERBOSE:
            print('\n======= NEW GAME =======')
            self.pretty_print_hexgrid()

    def set_hexgrid(self, hexgrid):
        """Copy incoming hexgrid."""
        self.hexgrid = hexgrid
        self.cat = list(np.argwhere(self.hexgrid == CAT_TILE)[0])  # Find the cat position  
        if VERBOSE:
            print('\n======= NEW GAME =======')
            self.pretty_print_hexgrid()
   
    def block_tile(self, coord):
        self.hexgrid[tuple(coord)] = BLOCKED_TILE

    def unblock_tile(self, coord):
        self.hexgrid[tuple(coord)] = EMPTY_TILE

    def place_cat(self, coord):
        self.hexgrid[tuple(coord)] = CAT_TILE
        self.cat = coord

    def move_cat(self, coord):
        self.hexgrid[tuple(self.cat)] = EMPTY_TILE  # Clear previous cat position
        self.place_cat(coord)
    
    def get_cat_moves(self):
        """
        Get a list of valid moves for the cat.
        """
        hexgrid = self.hexgrid
        r, c = self.cat
        n = self.size
        col_offset = r % 2  # Offset for columns based on row parity
        moves = []

        # Directions with column adjustments
        directions = {
            'E': (0, 1),
            'W': (0, -1),
            'NE': (-1, col_offset),
            'NW': (-1, -1 + col_offset),
            'SE': (1, col_offset),
            'SW': (1, -1 + col_offset),
        }

        for dr, dc in directions.values():
            tr, tc = r + dr, c + dc  # Calculate target row and column
            if 0 <= tr < n and 0 <= tc < n and hexgrid[tr, tc] == EMPTY_TILE:
                moves.append([tr, tc])

        return moves

    def apply_move(self, move, cat_turn):
        """
        Apply a move to the game state.
        """
        action_str = "move cat to" if cat_turn else "block"
        if self.hexgrid[move[0], move[1]] != EMPTY_TILE:
            self.pretty_print_hexgrid()
            print('\n=====================================')
            print(f'Attempting to {action_str} {move} = {self.hexgrid[move[0], move[1]]}')
            print('Invalid Move! Check your code.')
            print('=====================================\n')

        if cat_turn:
            self.hexgrid[move[0], move[1]] = CAT_TILE  # Place the cat
            self.hexgrid[tuple(self.cat)] = EMPTY_TILE  # Remove the old cat
            self.cat = move
        else:
            self.hexgrid[move[0], move[1]] = BLOCKED_TILE

    def time_left(self):
        """
        Calculate the time remaining before the deadline.
        """
        return (self.deadline - time.time()) * 1000

    def print_hexgrid(self):
        """
        Print the current state of the game board.
        """
        for r in range(0, self.size, 2):
            print(self.hexgrid[r])
            if r + 1 < self.size:
                print('', self.hexgrid[r + 1])
        print()
        return
    
    def pretty_print_hexgrid(self):
        """
        Print the current state of the game board using custom characters.
        """
        # Create a mapping for tile values to characters.
        # These are emojis, so they may not render properly in some settings.
        # Note that these are strings with a space preceding the tiles, but
        # not the cat. For regular ASCII characters, change to single characters 
        # like the alternatives shown in the comments.
        tile_map = {
            EMPTY_TILE: ' ⬡',   # Alternative: '-'
            BLOCKED_TILE: ' ⬢', # Alternative: 'X'
            CAT_TILE: '🐈'      # Alternative: 'C'
        }

        for r in range(self.size):
            # Add a leading space for odd rows for staggered effect
            prefix = ' ' if r % 2 != 0 else ''
            # Convert each row using the tile map
            row_display = ' '.join(tile_map[cell] for cell in self.hexgrid[r])
            print(prefix + row_display)

        return

    def utility(self, num_moves, cat_turn):
        """
        Calculate the utility of the current game state.
        """
        # Terminal cases
        r, c = self.cat
        n = self.size
        if r == 0 or r == n - 1 or c == 0 or c == n - 1:
            return float(100)
        
        # Only the cat can run out of moves
        if num_moves == 0: 
            return float(-100)

        # Use the evaluation function
        # Evaluation function options: 'moves', 'straight_exit', 'custom'
        evaluation_function = 'straight_exit'

        if evaluation_function == 'moves':
            return self.score_moves(cat_turn)
        elif evaluation_function == 'straight_exit':
            return self.score_straight_exit(cat_turn)
        elif evaluation_function == 'custom':
            return self.score_custom(cat_turn)
        return 0

    def score_moves(self, cat_turn):
        """
        Evaluate based on the number of valid moves available for the cat.
        """
        cat_moves = self.get_cat_moves()
        return len(cat_moves) if cat_turn else len(cat_moves) - 1

    def get_target_position(self, scout, direction):
        """
        Get the target position based on the specified direction.
        """
        r, c = scout
        col_offset = r % 2  # Offset for columns based on row parity
        
        if direction == 'E':
            return [r, c + 1]
        elif direction == 'W':
            return [r, c - 1]
        elif direction == 'NE':
            return [r - 1, c + col_offset]
        elif direction == 'NW':
            return [r - 1, c - 1 + col_offset]
        elif direction == 'SE':
            return [r + 1, c + col_offset]
        elif direction == 'SW':
            return [r + 1, c - 1 + col_offset]
        return [r, c]

    def score_straight_exit(self, cat_turn):
        """
        Evaluate the cat's access to the nearest board edge along straight paths.
        """
        distances = []
        directions =  ['E', 'W', 'NE', 'NW', 'SE', 'SW']
        n = self.size
        for dir in directions:
            distance = 0
            r, c = self.cat
            while not (r < 0 or r >= n or c < 0 or c >= n):
                if self.hexgrid[r, c] == BLOCKED_TILE:
                    distance += n # Increase cost for blocked paths
                    break
                distance += 1
                r, c = self.get_target_position([r, c], dir)
            distances.append(distance)

        distances.sort()
        return self.size - (distances[0] if cat_turn else distances[1])

    def score_custom(self, cat_turn):
        """
        Placeholder for a custom evaluation function.
        """
        return 2 if cat_turn else 1

    # ===================== Intelligent Agents =====================
    """
    Intelligent Agents for the Cat Trap game. These agents take the game state and the
    cat's position as inputs and return the new position of the cat or indicate a failure.

    Available options:
      - random_cat: A random move for the cat.
      - alpha_beta: Use Alpha-Beta Pruning.
      - depth_limited: Use Depth-Limited Search with a specified maximum depth.
      - iterative_deepening: Use Iterative Deepening with an allotted time.
      - use_minimax: Use the Minimax algorithm.

    If none of these options are selected, no intelligent behavior is applied.
    """

    def select_cat_move(self, random_cat, alpha_beta, depth_limited, minimax, max_depth, iterative_deepening, allotted_time):
        """Select a move for the cat based on the chosen algorithm."""
        self.reached_max_depth = False 
        self.start_time = time.time()
        self.deadline = self.start_time + allotted_time 
        self.terminated = False
        self.max_depth = float('inf') 

        if VERBOSE:
            print('\n======= NEW MOVE =======')

        if random_cat:
            move = self.random_cat_move() 
        elif minimax:
            # Select a move using the Minimax algorithm.
            move, _ = self.alpha_beta() if alpha_beta else self.minimax()   
        elif depth_limited:
            # Select a move using Depth-Limited Search with optional Alpha-Beta pruning.
            self.max_depth = max_depth
            move, _ = self.alpha_beta() if alpha_beta else self.minimax()
        elif iterative_deepening:
            # Select a move using the Iterative Deepening algorithm.
            move, _ = self.iterative_deepening(use_alpha_beta = alpha_beta)
        else:
            move = None

        elapsed_time = (time.time() - self.start_time) * 1000
        if VERBOSE:
            print(f'Elapsed time: {elapsed_time:.3f}ms ')
            print(f'New cat coordinates: {move}')
            temp = copy.deepcopy(self)
            if move != [-1, -1]:
                temp.move_cat(move)
            temp.pretty_print_hexgrid()
        return move

    def random_cat_move(self):
        """Randomly select a move for the cat."""
        moves = self.get_cat_moves()
        if moves:
            return random.choice(moves)
        return self.cat

    def max_value(self, game, depth):
        """
        Calculate the maximum value for the current game state in the minimax algorithm.
        """
        if self.time_left() < LAST_CALL_MS:
            self.terminated = True
            return [-1, -1], 0
        
        legal_moves = game.get_cat_moves()  # Available directions: E, W, NE, NW, SE, SW
        if not legal_moves or depth == self.max_depth:
            if depth == self.max_depth:
                self.reached_max_depth = True  
            return self.cat, (game.size**2 - depth) * game.utility(len(legal_moves), cat_turn = True)
        
        best_value = float('-inf')
        best_move = legal_moves[0]
        for move in legal_moves:
            next_game = copy.deepcopy(game)
            next_game.apply_move(move, cat_turn = True)
            value = self.min_value(next_game, depth + 1)

            if self.terminated:
                return [-1, -1], 0
            
            if value > best_value:
                best_value = value
                best_move = move
  
        return best_move, best_value

    def min_value(self, game, depth):
        """
        Calculate the minimum value for the current game state in the minimax algorithm.

        Unlike max_value, min_value does not iterate over specific directions ('E', 'W', etc.).
        Instead, it examines every possible free tile on the board. This simplifies implementation
        for moves like blocking hexgrid, where legal positions are any unoccupied hexgrid, not directional.
        """
        if self.time_left() < LAST_CALL_MS:
            self.terminated = True
            return 0

        # Check if terminal state or depth limit is reached
        r, c = game.cat
        n = game.size
        if depth == self.max_depth or (
            r == 0 or r == n - 1 or 
            c == 0 or c == n - 1
        ):
            if depth == self.max_depth:
                self.reached_max_depth = True
            
            return (n**2 - depth) * game.utility(1, cat_turn = False)
        
        best_value = float('inf')

        # Iterate through all legal moves for the player (empty tiles)
        legal_moves = [list(coord) for coord in np.argwhere(game.hexgrid == EMPTY_TILE)]
        for move in legal_moves:
            next_game = copy.deepcopy(game)
            next_game.apply_move(move, cat_turn = False)
            _, value = self.max_value(next_game, depth + 1)
            best_value = min(best_value, value)

            if self.terminated:
                return 0
        
        return best_value

    def minimax(self):
        """
        Perform the Minimax algorithm to determine the best move.
        """
        return self.max_value(self, depth = 0)

    def alpha_beta_max_value(self, game, alpha, beta, depth):
        """
        Calculate the maximum value for the current game state using Alpha-Beta pruning.
        """
        if self.time_left() < LAST_CALL_MS:
            self.terminated = True
            return [-1, -1], 0
        
        legal_moves = game.get_cat_moves()  # Available directions: E, W, NE, NW, SE, SW
        if not legal_moves or depth == self.max_depth:
            if depth == self.max_depth:
                self.reached_max_depth = True
            return self.cat, (game.size**2 - depth) * game.utility(len(legal_moves), cat_turn = True)
        
        best_value = float('-inf')
        best_move = legal_moves[0]
        for move in legal_moves:
            next_game = copy.deepcopy(game)
            next_game.apply_move(move, cat_turn = True)
            value = self.alpha_beta_min_value(next_game, alpha, beta, depth + 1)

            if self.terminated:
                return [-1, -1], 0
            
            if value > best_value:
                best_value = value
                best_move = move

            if best_value >= beta: # Pruning
                return best_move, best_value
            alpha = max(alpha, best_value)

        return best_move, best_value

    def alpha_beta_min_value(self, game, alpha, beta, depth):
        """
        Calculate the minimum value for the current game state using Alpha-Beta pruning.

        Unlike max_value, min_value does not iterate over specific directions ('E', 'W', etc.).
        Instead, it examines every possible free tile on the board. This simplifies implementation
        for moves like blocking hexgrid, where legal positions are any unoccupied hexgrid, not directional.
        """
        if self.time_left() < LAST_CALL_MS:
            self.terminated = True
            return 0

        # Check if terminal state or depth limit is reached
        r, c = game.cat
        n = game.size
        if depth == self.max_depth or (
            r == 0 or r == n - 1 or 
            c == 0 or c == n - 1
        ):
            if depth == self.max_depth:
                self.reached_max_depth = True
            return (n**2 - depth) * game.utility(1, cat_turn = False)
        
        best_value = float('inf')

        # Iterate through all legal moves for the player (empty tiles)
        legal_moves = [list(coord) for coord in np.argwhere(game.hexgrid == EMPTY_TILE)]
        for move in legal_moves:
            next_game = copy.deepcopy(game)
            next_game.apply_move(move, cat_turn = False)
            _, value = self.alpha_beta_max_value(next_game, alpha, beta, depth + 1)
            best_value = min(best_value, value)

            if self.terminated:
                return 0
            
            if best_value <= alpha: # Pruning
                return best_value
            beta = min(beta, best_value)

        return best_value

    def alpha_beta(self, alpha = float('-inf'), beta = float('inf')):
        """
        Perform the Alpha-Beta pruning algorithm to determine the best move.
        """
        return self.alpha_beta_max_value(self, alpha, beta, depth = 0)

    def iterative_deepening(self, use_alpha_beta):
        """
        Perform iterative deepening search with an option to use Alpha-Beta pruning.
        """
        self.placeholder_warning()
        return self.random_cat_move(), 0

    def placeholder_warning(self):
        signs = '⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️'
        print(f'{signs} {signs}')
        print('                WARNING')
        print('This is a temporary implementation using')
        print("the random algorithm. You're supposed to")
        print('write code to solve a challenge.')
        print('Did you run the wrong version of main.py?')
        print('Double-check its path.')
        print(f'{signs} {signs}')

if __name__ == '__main__':
    signs = '⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️'
    print(f'\n{signs} {signs}')
    print('               WARNING')
    print('You ran cat_trap_algorithms.py')
    print('This file contains the AI algorithms')
    print('and classes for the intelligent cat.')
    print('Did you mean to run main.py?')
    print(f'{signs} {signs}\n')
