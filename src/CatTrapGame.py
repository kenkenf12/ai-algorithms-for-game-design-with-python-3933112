"""
This is the relevant code for the LinkedIn Learning Course 
AI Algorithms for Game Design with Python, by Eduardo Corpeño

For the GUI, this code uses the Cat Trap UI VSCode Extension.
"""

import random
import copy
import time
import numpy as np

# Constants
CAT_TILE = 6
BLOCK_TILE = 1
EMPTY_TILE = 0
LAST_CALL_MS = 5

class InvalidMove(ValueError):
    pass

class CatTrapGame:
    """
    Represents a Cat Trap game state. Includes methods for initializing the game board, 
    managing game state, and selecting moves for the cat using different algorithms.
    """

    def __init__(self, size):
        self.cat_i = size // 2
        self.cat_j = size // 2
        self.size = size
        self.tiles = np.full((size, size), EMPTY_TILE)
        self.deadline = 0
        self.terminated = False
        self.start_time = time.time()
        self.eval_function = CatEvaluationFunction()
        self.reached_max_depth = False 

    def initialize_random_blocks(self, cat):
        """Randomly initialize blocked tiles."""
        num_blocks = random.randint(round(0.067 * (self.size**2)), round(0.13 * (self.size**2)))
        count = 0
        self.cat_i, self.cat_j = cat
        self.tiles[self.cat_i, self.cat_j] = CAT_TILE

        while count < num_blocks:
            i = random.randint(0, self.size - 1)
            j = random.randint(0, self.size - 1)
            if self.tiles[i, j] == EMPTY_TILE:
                self.tiles[i, j] = BLOCK_TILE
                count += 1    

    def initialize_blocks(self, blocks, cat):
        """Initialize the game with specific blocked tiles and cat position."""
        i, j = cat
        self.tiles[i, j] = CAT_TILE

        for block in blocks:
            if block != [i, j]:
                self.tiles[block[0], block[1]] = BLOCK_TILE

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

    def select_cat_move(self, random_cat, alpha_beta, depth_limited, use_minimax, max_depth, iterative_deepening, allotted_time):
        """Select a move for the cat based on the chosen algorithm."""
        self.reached_max_depth = False 
        self.start_time = time.time()
        self.deadline = self.start_time + allotted_time 

        if random_cat:
            result = self.random_cat_move()    
        elif depth_limited:
            result = self.depth_limited_cat_move(max_depth=max_depth, alpha_beta=alpha_beta)
        elif iterative_deepening:
            self.deadline = self.start_time + allotted_time
            result = self.iterative_deepening_cat_move(alpha_beta=alpha_beta)
        elif use_minimax:
            result = self.minimax_cat_move()
        elif alpha_beta:
            result = self.alpha_beta_cat_move()
        else:
            result = None

        elapsed_time = (time.time() - self.start_time) * 1000
        print('Elapsed time: %.3fms ' % elapsed_time)
        return result

    def random_cat_move(self):
        """Randomly select a move for the cat."""
        moves = self.get_valid_moves()
        print(moves)  # Available directions for the next move: 'E', 'W', 'SE', 'SW', 'NE', 'NW'
        if moves:
            direction = random.choice(moves)
            return self.get_target_position(self.cat_i, self.cat_j, direction)
        return [self.cat_i, self.cat_j]

    def minimax_cat_move(self):
        """Select a move using the Minimax algorithm."""
        move, _ = self.minimax()
        return move

    def alpha_beta_cat_move(self):
        """Select a move using the Alpha-Beta Pruning algorithm."""
        move, _ = self.alpha_beta()
        return move

    def depth_limited_cat_move(self, max_depth, alpha_beta):
        """Select a move using Depth-Limited Search with optional Alpha-Beta pruning."""
        move, _ = self.alpha_beta(max_depth=max_depth) if alpha_beta else self.minimax(max_depth=max_depth)
        return move

    def iterative_deepening_cat_move(self, alpha_beta):
        """Select a move using the Iterative Deepening algorithm."""
        move, _ = self.iterative_deepening(alpha_beta=alpha_beta)
        return move
    
    def get_valid_moves(self):
        """
        Get a list of valid moves for the cat.
        """
        tiles, cat_i, cat_j = self.tiles, self.cat_i, self.cat_j
        size = self.size
        moves = []
        # Check possible directions for the next move: 'E', 'W', 'SE', 'SW', 'NE', 'NW'
        if cat_j < size - 1 and tiles[cat_i][cat_j + 1] == EMPTY_TILE:
            moves.append('E')
        if cat_j > 0 and tiles[cat_i][cat_j - 1] == EMPTY_TILE:
            moves.append('W')

        if cat_i % 2 == 0:
            if cat_i > 0 and cat_j < size and tiles[cat_i - 1][cat_j] == EMPTY_TILE:
                moves.append('NE')
            if cat_i > 0 and cat_j > 0 and tiles[cat_i - 1][cat_j - 1] == EMPTY_TILE:
                moves.append('NW')
            if cat_i < size - 1 and cat_j < size and tiles[cat_i + 1][cat_j] == EMPTY_TILE:
                moves.append('SE')
            if cat_i < size - 1 and cat_j > 0 and tiles[cat_i + 1][cat_j - 1] == EMPTY_TILE:
                moves.append('SW')
        else:
            if cat_i > 0 and cat_j < size - 1 and tiles[cat_i - 1][cat_j + 1] == EMPTY_TILE:
                moves.append('NE')
            if cat_i > 0 and cat_j >= 0 and tiles[cat_i - 1][cat_j] == EMPTY_TILE:
                moves.append('NW')
            if cat_i < size - 1 and cat_j < size - 1 and tiles[cat_i + 1][cat_j + 1] == EMPTY_TILE:
                moves.append('SE')
            if cat_i < size - 1 and cat_j > 0 and tiles[cat_i + 1][cat_j] == EMPTY_TILE:
                moves.append('SW')
        return moves

    def get_target_position(self, i, j, direction):
        """
        Get the target position based on the current position and direction.
        """
        target = [i, j]
        if direction == 'E':
            target = [i, j + 1]
        elif direction == 'W':
            target = [i, j - 1]
        elif direction == 'NE':
            target = [i - 1, j] if i % 2 == 0 else [i - 1, j + 1]
        elif direction == 'NW':
            target = [i - 1, j - 1] if i % 2 == 0 else [i - 1, j]
        elif direction == 'SE':
            target = [i + 1, j] if i % 2 == 0 else [i + 1, j + 1]
        elif direction == 'SW':
            target = [i + 1, j - 1] if i % 2 == 0 else [i + 1, j]
        return target

    def utility(self, moves, maximizing_player=True):
        """
        Calculate the utility of the current game state.
        """
        # Terminal cases
        if (
            self.cat_i == 0 or self.cat_i == self.size - 1 or
            self.cat_j == 0 or self.cat_j == self.size - 1
        ):
            return float(100)

        if len(moves) == 0:
            return float(-100)

        # Use the evaluation function
        # Evaluation function options: 'moves', 'custom', 'proximity'
        evaluation_function = 'proximity'

        if evaluation_function == 'moves':
            return self.eval_function.score_moves(self, maximizing_player)
        elif evaluation_function == 'proximity':
            return self.eval_function.score_proximity(self, maximizing_player)
        elif evaluation_function == 'custom':
            return self.eval_function.score_custom(self, maximizing_player)
        return 0

    def apply_move(self, move, maximizing_player):
        """
        Apply a move to the game state.
        """
        if self.tiles[move[0], move[1]] != EMPTY_TILE:
            raise InvalidMove('Invalid Move!')

        if maximizing_player:
            self.tiles[move[0], move[1]] = BLOCK_TILE
        else:
            self.tiles[move[0], move[1]] = CAT_TILE  # Place the cat
            self.tiles[self.cat_i, self.cat_j] = EMPTY_TILE  # Remove the old cat
            self.cat_i, self.cat_j = move

    def max_value(self, upper_game, move, maximizing_player, depth, max_depth):
        """
        Calculate the maximum value for the current game state in the minimax algorithm.
        """
        if self.time_left() < LAST_CALL_MS:
            self.terminated = True
            return [-1, -1], 0

        game = copy.deepcopy(upper_game)
        if move != [-1, -1]:
            maximizing_player = not maximizing_player
            game.apply_move(move, maximizing_player)
        
        legal_moves = game.get_valid_moves()  # Available directions: 'E', 'W', 'SE', 'SW', 'NE', 'NW'
        if not legal_moves or depth == max_depth:
            if depth == max_depth:
                self.reached_max_depth = True  
            return [self.cat_i, self.cat_j], (game.size**2 - depth) * game.utility(legal_moves, maximizing_player)
        
        best_value = float('-inf')
        best_move = game.get_target_position(self.cat_i, self.cat_j, legal_moves[0])
        for direction in legal_moves:
            target_pos = game.get_target_position(self.cat_i, self.cat_j, direction)
            _, value = self.min_value(game, target_pos, maximizing_player, depth + 1, max_depth)

            if self.terminated:
                return [-1, -1], 0
            if value > best_value:
                best_value = value
                best_move = target_pos
  
        return best_move, best_value

    def min_value(self, upper_game, move, maximizing_player, depth, max_depth):
        """
        Calculate the minimum value for the current game state in the minimax algorithm.

        Unlike max_value, min_value does not iterate over specific directions ('E', 'W', etc.).
        Instead, it examines every possible free tile on the board. This simplifies implementation
        for moves like blocking tiles, where legal positions are any unoccupied tiles, not directional.
        """
        if self.time_left() < LAST_CALL_MS:
            self.terminated = True
            return 0

        game = copy.deepcopy(upper_game)
        maximizing_player = not maximizing_player
        game.apply_move(move, maximizing_player)

        # Check if terminal state or depth limit is reached
        if depth == max_depth or (
            game.cat_i == 0 or game.cat_i == self.size - 1 or
            game.cat_j == 0 or game.cat_j == self.size - 1
        ):
            if depth == max_depth:
                self.reached_max_depth = True
            return (game.size**2 - depth) * game.utility([2, 3, 4], maximizing_player)
        
        best_value = float('inf')

        # Iterate through all possible moves (workaround for block placement)
        for i in range(game.size):
            for j in range(game.size):
                if game.tiles[i, j] != EMPTY_TILE:
                    continue
                
                move = [i, j]
                _, value = self.max_value(game, move, maximizing_player, depth + 1, max_depth)
                best_value = min(best_value, value)

                if self.terminated:
                    return 0
        
        return best_value

    def minimax(self, max_depth=float('inf'), maximizing_player=True):
        """
        Perform the Minimax algorithm to determine the best move.
        """
        best_move, best_value = self.max_value(self, [-1, -1], maximizing_player, 0, max_depth)
        return best_move, best_value

    def time_left(self):
        """
        Calculate the time remaining before the deadline.
        """
        return (self.deadline - time.time()) * 1000

    def print_tiles(self):
        """
        Print the current state of the game board.
        """
        for i in range(0, self.size, 2):
            print(self.tiles[i])
            if i + 1 < self.size:
                print('', self.tiles[i + 1])
        return

    def alpha_beta_max_value(self, upper_game, move, alpha, beta, maximizing_player, depth, max_depth):
        """
        Calculate the maximum value for the current game state using Alpha-Beta pruning.
        """
        if self.time_left() < LAST_CALL_MS:
            self.terminated = True
            return [-1, -1], 0

        game = copy.deepcopy(upper_game)
        if move != [-1, -1]:
            maximizing_player = not maximizing_player
            game.apply_move(move, maximizing_player)

        legal_moves = game.get_valid_moves()  # Available directions: 'E', 'W', 'SE', 'SW', 'NE', 'NW'
        if not legal_moves or depth == max_depth:
            if depth == max_depth:
                self.reached_max_depth = True
            return [self.cat_i, self.cat_j], (game.size**2 - depth) * game.utility(legal_moves, maximizing_player)

        best_value = float('-inf')
        best_move = game.get_target_position(self.cat_i, self.cat_j, legal_moves[0])
        for direction in legal_moves:
            target_pos = game.get_target_position(self.cat_i, self.cat_j, direction)
            _, value = self.alpha_beta_min_value(game, target_pos, alpha, beta, maximizing_player, depth + 1, max_depth)

            if self.terminated:
                return [-1, -1], 0
            if value > best_value:
                best_value = value
                best_move = target_pos

            if best_value >= beta:
                return best_move, best_value
            alpha = max(alpha, best_value)

        return best_move, best_value

    def alpha_beta_min_value(self, upper_game, move, alpha, beta, maximizing_player, depth, max_depth):
        """
        Calculate the minimum value for the current game state using Alpha-Beta pruning.

        Unlike max_value, min_value does not iterate over specific directions ('E', 'W', etc.).
        Instead, it examines every possible free tile on the board. This simplifies implementation
        for moves like blocking tiles, where legal positions are any unoccupied tiles, not directional.
        """
        if self.time_left() < LAST_CALL_MS:
            self.terminated = True
            return 0

        game = copy.deepcopy(upper_game)
        maximizing_player = not maximizing_player
        game.apply_move(move, maximizing_player)

        # Check if terminal state or depth limit is reached
        if depth == max_depth or (
            game.cat_i == 0 or game.cat_i == self.size - 1 or
            game.cat_j == 0 or game.cat_j == self.size - 1
        ):
            if depth == max_depth:
                self.reached_max_depth = True
            return (game.size**2 - depth) * game.utility([2, 3, 4], maximizing_player)

        best_value = float('inf')

        # Iterate through all possible moves (workaround for block placement)
        for i in range(game.size):
            for j in range(game.size):
                if game.tiles[i, j] != EMPTY_TILE:
                    continue

                move = [i, j]
                _, value = self.alpha_beta_max_value(game, move, alpha, beta, maximizing_player, depth + 1, max_depth)
                best_value = min(best_value, value)

                if self.terminated:
                    return 0
                if best_value <= alpha:
                    return best_value
                beta = min(beta, best_value)

        return best_value

    def alpha_beta(self, max_depth=float('inf'), alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
        """
        Perform the Alpha-Beta pruning algorithm to determine the best move.
        """
        best_move, best_value = self.alpha_beta_max_value(self, [-1, -1], alpha, beta, maximizing_player, 0, max_depth)
        return best_move, best_value

    def iterative_deepening(self, use_alpha_beta):
        """
        Perform iterative deepening search with an option to use Alpha-Beta pruning.
        """
        self.terminated = False
        best_depth = 0
        output_move, utility = [self.cat_i, self.cat_j], 0
        for depth in range(1, self.size**2):
            self.reached_max_depth = False
            if use_alpha_beta:
                best_move, utility = self.alpha_beta(max_depth=depth)
            else:
                best_move, utility = self.minimax(max_depth=depth)

            if self.terminated:
                break
            else:
                output_move = best_move
                best_depth = depth
                elapsed_time = (time.time() - self.start_time) * 1000
                print(f'Done with a tree of depth {depth} in {elapsed_time:.3f}ms')

                if not self.reached_max_depth:
                    break

        print('Depth reached:', best_depth)
        return output_move, utility

class CatEvaluationFunction:
    """
    Evaluation function class containing different scoring methods.
    """

    def score_moves(self, game, maximizing_player_turn=True):
        """
        Evaluate based on the number of valid moves available for the cat.
        """
        cat_moves = game.get_valid_moves()
        return len(cat_moves) if maximizing_player_turn else len(cat_moves) - 1

    def score_proximity(self, game, maximizing_player_turn=True):
        """
        Evaluate based on the proximity of the cat to the board edges.
        """
        distances = [100, 100]  # High initial distances
        cat_moves = game.get_valid_moves()
        for move in cat_moves:
            distance = 0
            i, j = game.cat_i, game.cat_j
            while True:
                distance += 1
                i, j = game.get_target_position(i, j, move)
                if i < 0 or i >= game.size or j < 0 or j >= game.size:
                    break
                if game.tiles[i, j] != EMPTY_TILE:
                    distance *= 5
                    break
            distances.append(distance)

        distances.sort()
        return game.size * 2 - (distances[0] if maximizing_player_turn else distances[1])

    def score_custom(self, game, maximizing_player_turn=True):
        """
        Placeholder for a custom evaluation function.
        """
        # Write your custom logic here
        return 1 if maximizing_player_turn else -1
