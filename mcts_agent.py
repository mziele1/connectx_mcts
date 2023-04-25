import numpy as np
import random
from timeit import default_timer
import multiprocessing as mp
import psutil


def drop_piece_on_grid(col, piece, grid):
    """Get the board after dropping mark in col on the supplied grid"""
    # get the last nonzero idx
    row = (grid[:, col]==0).nonzero()[0][-1]
    grid[row, col] = piece
    return grid


def get_move_winner(board, inarow, player, col):
    """Call get_winner on board with player's piece dropped in col"""
    next_board = drop_piece_on_grid(col, player, board.copy())
    return get_winner(next_board, inarow)


def get_winner(board, inarow):
    """
        Returns the winner {None, 0, 1, 2}
            None: game isn't finished
            0: tie
            1: player 1
            2: player 2
    """
    # horizontal
    rows, cols = board.shape
    for row in range(rows):
        for col in range(cols-(inarow-1)):
            window = board[row, col:col+inarow]
            if (window == 1).all():
                return 1
            elif (window == 2).all():
                return 2
    # vertical
    for row in range(rows-(inarow-1)):
        for col in range(cols):
            window = board[row:row+inarow, col]
            if (window == 1).all():
                return 1
            elif (window == 2).all():
                return 2
    # positive diagonal
    for row in range(rows-(inarow-1)):
        for col in range(cols-(inarow-1)):
            window = board[range(row, row+inarow), range(col, col+inarow)]
            if (window == 1).all():
                return 1
            elif (window == 2).all():
                return 2
    # negative diagonal
    for row in range(inarow-1, rows):
        for col in range(cols-(inarow-1)):
            window = board[range(row, row-inarow, -1), range(col, col+inarow)]
            if (window == 1).all():
                return 1
            elif (window == 2).all():
                return 2
    if 0 in board[0]:
        # empty space on board
        return None
    else:
        # board full
        return 0

class Board:
    """A class implementing the connectx game"""
    def __init__(self, board, cfg, player):
        np.random.seed()
        self.cols = cfg["cols"]
        self.rows = cfg["rows"]
        self.inarow = cfg["inarow"]
        self.player = player
        # board in matrix form
        self.board = board.reshape(self.rows, self.cols)
        if cfg["playout_type"] == "light":
            self.next_move_fn = self.get_light_move
        elif cfg["playout_type"] == "heavy":
            self.next_move_fn = self.get_heavy_move
        else:
            raise ValueError("Invalid playout type")
            
    def get_valid_moves(self):
        """Get valid moves on self.board"""
        valid_moves = (self.board[0]==0).nonzero()[0]
        return valid_moves
            
    def get_light_move(self):
        """Get a random valid move"""
        valid_moves = self.get_valid_moves()
        return random.choice(valid_moves)
    
    def get_heavy_move(self):
        """Prioritize winning, then blocking a opponent's win, finally choose a random move"""
        valid_moves = self.get_valid_moves()
        winners = np.zeros(self.cols, dtype=np.int8)
        for col in valid_moves:
            winner = get_move_winner(self.board, self.inarow, self.player, col)
            if winner == self.player:
                return col
            winners[col] = 0 if winner == None else winner
        opponent_winning_moves = (winners==(self.player%2+1)).nonzero()[0]
        if opponent_winning_moves.shape[0] > 0:
            return opponent_winning_moves[0]
        else:
            return random.choice(valid_moves)
        
    def drop_piece(self, col, piece):
        """Get the board after dropping mark in col"""
        next_board = self.board.copy()
        # get the last nonzero idx
        row = (next_board[:, col]==0).nonzero()[0][-1]
        next_board[row, col] = piece
        return next_board
        
    def check_game_over(self):
        """Returns true if there is a winning configuration on the board of if there are no valid moves"""
        board = self.board
        if self.get_valid_moves().shape[0] == 0:
            return True
        else:
            # horizontal
            for row in range(self.rows):
                for col in range(self.cols-(self.inarow-1)):
                    window = board[row, col:col+self.inarow]
                    if (window == self.player).all():
                        return True
            # vertical
            for row in range(self.rows-(self.inarow-1)):
                for col in range(self.cols):
                    window = board[row:row+self.inarow, col]
                    if (window == self.player).all():
                        return True
            # positive diagonal
            for row in range(self.rows-(self.inarow-1)):
                for col in range(self.cols-(self.inarow-1)):
                    window = board[range(row, row+self.inarow), range(col, col+self.inarow)]
                    if (window == self.player).all():
                        return True
            # negative diagonal
            for row in range(self.inarow-1, self.rows):
                for col in range(self.cols-(self.inarow-1)):
                    window = board[range(row, row-self.inarow, -1), range(col, col+self.inarow)]
                    if (window == self.player).all():
                        return True
            return False
        
class Node:
    """A class representing nodes in MCTS"""
    def __init__(self, parent, board, cfg, player):
        self.board = Board(board, cfg, player)
        self.visits = 0
        self.wins = np.zeros(3, dtype=np.int32)
        self.player = player
        self.c = 2
        self.parent = parent
        self.valid_actions = self.board.get_valid_moves()
        self.children = {}
        
    def is_leaf(self):
        """True if not all potential children have been visited"""
        for action in self.valid_actions:
            if action not in self.children:
                return True
        return False
    
    def is_terminal(self):
        return self.board.check_game_over()
    
    def expand(self):
        for col in self.board.get_valid_moves():
            self.children[valid_move] = Node(self, self.board.drop_piece(col), self.cfg, self.player%2+1)
            
    def get_UCT(self, player):
        """Get the UCT for the node"""
        uct = self.wins[player] / self.visits + self.c * np.sqrt(np.log(self.parent.visits/self.visits))
        return uct


class MCTS:
    """The MCTS search manager"""
    def __init__(self, obs, cfg, playout_type, decision_time=0.1):
        np.random.seed()
        self.time_budget = cfg["timeout"]
        self.decision_time = decision_time
        self.simulation_time = self.time_budget - self.decision_time
        self.cfg = {
            "cols": cfg["columns"],
            "rows": cfg["rows"],
            "inarow": cfg["inarow"],
            "playout_type": playout_type
        }
        self.player = obs["mark"]
        self.first_step = obs["step"]
        self.root = Node(None, np.array(obs["board"]), self.cfg, self.player)        
            
    def run_until_timeout(self, start):
        """Call run_round until time runs out"""
        while default_timer() - start < self.simulation_time:
            self.run_round()
            
    def run_round(self):
        """
            Run the 4 main MCTS steps:
                1: Selection
                2: Expansion
                3: Simulation
                4: Backpropagation
        """
        selected_node, actions = self.selection()
        expanded_node, actions = self.expansion(selected_node, actions)
        won = self.playout(expanded_node, actions)
        self.backpropagate(won, actions)
        
    def selection(self):
        """
            Select a leaf node for expansion
            if the current node has a valid child that has not been visited
                pick current node
            else
                calc UCTs
                pick child with highest UCT
                repeat
        """
        actions = []
        node = self.root
        while (node.is_leaf() == False) and (node.is_terminal()==False):
            ucts = np.array([node.children[action].get_UCT(node.player) for action in node.valid_actions])
            action = node.valid_actions[ucts.argmax()]
            actions.append(action)
            # update node to the child that maximizes UCT
            node = node.children[action]
        return node, actions
    
    def expansion(self, node, actions):
        """Expand non-terminal nodes by creating a random child"""
        curr_player = self.player if len(actions) % 2 == 0 else self.player%2+1
        if not node.is_terminal():
            unexplored_actions = [action for action in node.valid_actions if action not in node.children]
            selected_action = random.choice(unexplored_actions)
            actions.append(selected_action)
            next_player = curr_player%2+1
            node.children[selected_action] = Node(node, node.board.drop_piece(selected_action, curr_player).flatten(), self.cfg, next_player)
            curr_player = curr_player%2+1
            return node.children[selected_action], actions
        return node, actions
    
    def playout(self, node, actions):
        """Return the winner when selecting random actions from the supplied node"""
        curr_player = self.player if len(actions) % 2 == 0 else self.player%2+1
        while get_winner(node.board.board, node.board.inarow) == None:
            selected_action = node.board.next_move_fn()
            next_player = curr_player%2+1
            node = Node(node, node.board.drop_piece(selected_action, curr_player).flatten(), self.cfg, next_player)
            curr_player = next_player            
        winner = get_winner(node.board.board, node.board.inarow)
        return winner        
    
    def backpropagate(self, winner, actions):
        """Update all nodes along the action path"""
        node = self.root
        node.visits += 1
        node.wins[winner] += 1
        for action in actions:
            node = node.children[action]
            node.visits += 1
            node.wins[winner] += 1
            
    def get_best_action(self):
        """Select the best action as the one that has been simulated the most"""
        try:
            actions, children = map(list, zip(*self.root.children.items()))
        except ValueError:
            # no children have been expanded
            actions = self.root.valid_actions
            children = None
            
        # check for win, block opponent win
        winners = np.zeros(self.cfg["cols"], dtype=np.int8)
        for col in self.root.valid_actions:
            winner = get_move_winner(self.root.board.board, self.cfg["inarow"], self.player, col)
            if winner == self.player:
                return col
            winners[col] = 0 if winner == None else winner
        opponent_winning_moves = (winners==(self.player%2+1)).nonzero()[0]
        if opponent_winning_moves.shape[0] > 0:
            return opponent_winning_moves[0]

        if children != None:
            # if no win/block win move, choose by visits
            n_visits = [child.visits for child in children]
            n_wins = [child.wins for child in children]
            opt_idx = np.argmax(n_visits)
            return actions[opt_idx]
        else:
            return random.choice(actions)

        
def mcts_agent(obs, cfg):
    start = default_timer()
    global my_mcts
    # delete mcts for repeated simulation
    if obs["step"] == 0:
        if "my_mcts" in globals():
            del(my_mcts)
            
    if (obs["step"]==0) or (obs["step"]==1):
        # init
        my_mcts = MCTS(obs, cfg, "heavy")
        start += 55
    else:
        # need to update board based on opponent's tree
        board_diff = my_mcts.root.board.board - np.array(obs["board"]).reshape(cfg["rows"], cfg["columns"])
        opponent_action = board_diff.nonzero()[1][0]
        # if agent is slow, this branch may not have been explored and needs to be manually created
        if opponent_action not in my_mcts.root.children:
            my_mcts.root.children[opponent_action] = Node(my_mcts.root, my_mcts.root.board.drop_piece(opponent_action, my_mcts.player%2+1).flatten(), my_mcts.cfg, my_mcts.player)
        my_mcts.root = my_mcts.root.children[opponent_action]
    my_mcts.run_until_timeout(start)
    action = my_mcts.get_best_action()
    my_mcts.root = my_mcts.root.children[action]
    return int(action)
