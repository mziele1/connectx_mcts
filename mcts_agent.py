import numpy as np
import random
import numba
from numba import njit
from timeit import default_timer
import psutil
import multiprocessing as mp
import time


numba_fastmath = True
# caching will raise a pickling error when executed in the competition env
numba_cache = False

@njit(fastmath=numba_fastmath, cache=numba_cache)
def drop_piece_on_grid(col, piece, grid):
    """Get the board after dropping mark in col on the supplied grid"""
    # get the last nonzero idx
    row = (grid[:, col]==0).nonzero()[0][-1]
    grid[row, col] = piece
    return grid


@njit(fastmath=numba_fastmath, cache=numba_cache)
def get_move_winner(board, inarow, player, col):
    """Call get_winner on board with player's piece dropped in col"""
    next_board = drop_piece_on_grid(col, player, board.copy())
    return get_winner(next_board, inarow)


@njit(fastmath=numba_fastmath, cache=numba_cache)
def get_winner(board, inarow):
    """
        Returns the winner {None, 0, 1, 2} on the matrix style grid
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
    window = np.zeros(inarow, dtype=np.int8)
    # positive diagonal
    for row in np.arange(rows-(inarow-1)):
        for col in np.arange(cols-(inarow-1)):
            for i in range(inarow):
                window[i] = board[row+i, col+i]
            if (window == 1).all():
                return 1
            elif (window == 2).all():
                return 2
    # negative diagonal
    for row in np.arange(inarow-1, rows):
        for col in np.arange(cols-(inarow-1)):
            for i in range(inarow):
                window[i] = board[row-i, col+i]
            if (window == 1).all():
                return 1
            elif (window == 2).all():
                return 2
    if 0 in board[0]:
        # empty space on board
        return -1
    else:
        # board full
        return 0
    
    
@njit(fastmath=numba_fastmath, cache=numba_cache)
def get_valid_moves(board):
    """Get valid moves on self.board"""
    return (board[0]==0).nonzero()[0]


@njit(fastmath=numba_fastmath, cache=numba_cache)
def get_light_move(board):
    """Get a random valid move"""
    return np.random.choice(get_valid_moves(board))


@njit(fastmath=numba_fastmath, cache=numba_cache)
def get_heavy_move(board, cols, inarow, player):
    """Prioritize winning, then blocking a opponent's win, finally choose a random move"""
    valid_moves = get_valid_moves(board)
    winners = np.zeros(cols, dtype=np.int8)
    for col in valid_moves:
        winner = get_move_winner(board, inarow, player, col)
        if winner == player:
            return col
        winners[col] = winner
    opponent_winning_moves = (winners==(player%2+1)).nonzero()[0]
    if opponent_winning_moves.shape[0] > 0:
        return opponent_winning_moves[0]
    else:
        return np.random.choice(valid_moves)
    
    
@njit(fastmath=numba_fastmath, cache=numba_cache)
def drop_piece(board, col, piece):
    """Get the board after dropping mark in col"""
    next_board = board.copy()
    # get the last nonzero idx
    row = (next_board[:, col]==0).nonzero()[0][-1]
    next_board[row, col] = piece
    return next_board


@njit(fastmath=numba_fastmath, cache=numba_cache)
def check_game_over(board, rows, cols, inarow, player):
    """Returns true if there is a winning configuration on the board of if there are no valid moves"""
    if get_valid_moves(board).shape[0] == 0:
        return True
    else:
        # horizontal
        for row in range(rows):
            for col in range(cols-(inarow-1)):
                window = board[row, col:col+inarow]
                if (window == player).all():
                    return True
        # vertical
        for row in range(rows-(inarow-1)):
            for col in range(cols):
                window = board[row:row+inarow, col]
                if (window == player).all():
                    return True
        window = np.zeros(inarow, dtype=np.int8)
        # positive diagonal
        for row in np.arange(rows-(inarow-1)):
            for col in np.arange(cols-(inarow-1)):
                for i in range(inarow):
                    window[i] = board[row+i, col+i]
                if (window == player).all():
                    return True
        # negative diagonal
        for row in np.arange(inarow-1, rows):
            for col in np.arange(cols-(inarow-1)):
                for i in range(inarow):
                    window[i] = board[row-i, col+i]
                if (window == player).all():
                    return True
        return False

    
@njit(fastmath=numba_fastmath, cache=numba_cache)
def get_UCT(wins, visits, c, parent_visits):
    """Get the UCT for the node"""
    return wins / visits + c * np.sqrt(np.log(parent_visits/visits))


@njit(fastmath=numba_fastmath, cache=numba_cache)
def light_playout(curr_player, board, inarow, cols):
    """Return the winner when selecting random actions from the supplied node"""
    while get_winner(board, inarow) == -1:
        selected_action = get_light_move(board)
        next_player = curr_player%2+1
        board = drop_piece(board, selected_action, curr_player)
        curr_player = next_player
    return get_winner(board, inarow)        


@njit(fastmath=numba_fastmath, cache=numba_cache)
def heavy_playout(curr_player, board, inarow, cols):
    """Return the winner when selecting random actions from the supplied node"""
    while get_winner(board, inarow) == -1:
        selected_action = get_heavy_move(board, cols, inarow, curr_player)
        next_player = curr_player%2+1
        board = drop_piece(board, selected_action, curr_player)
        curr_player = next_player
    return get_winner(board, inarow)        


class Board:
    """A class implementing the connectx game"""
    def __init__(self, board, cfg, player):
        self.cols = cfg["cols"]
        self.rows = cfg["rows"]
        self.inarow = cfg["inarow"]
        self.player = player
        # board in matrix form
        self.board = board.reshape(self.rows, self.cols)
            
    def get_valid_moves(self):
        """Get valid moves on self.board"""
        return get_valid_moves(self.board)
        
    def drop_piece(self, col, piece):
        """Get the board after dropping mark in col"""
        return drop_piece(self.board, col, piece)
        
    def check_game_over(self):
        """Returns true if there is a winning configuration on the board of if there are no valid moves"""
        return check_game_over(self.board, self.rows, self.cols, self.inarow, self.player)
    
    
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
                
    def get_UCT(self, player):
        """Get the UCT for the node"""
        return get_UCT(self.wins[player], self.visits, self.c, self.parent.visits)


class MCTS:
    """The MCTS search manager"""
    def __init__(self, obs, cfg, playout_type, decision_time=0.1, board=None):
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
        if board is None:
            self.root = Node(None, np.array(obs["board"], dtype=np.int8), self.cfg, self.player)
        else:
            self.root = Node(None, np.array(board, dtype=np.int8), self.cfg, self.player)
            
    def run_until_timeout(self, start=None, q=None):
        """Call run_round until time runs out"""
        if start == None:
            start = default_timer()
        while default_timer() - start < self.simulation_time:
            self.run_round()
        if q == None:
            return self
        else:
            q.put(self.get_best_action())
            
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
            # update node to the child that maximizes the UCT
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
        if self.cfg["playout_type"] == "light":
            return light_playout(self.player if len(actions) % 2 == 0 else self.player%2+1, node.board.board, node.board.inarow, node.board.cols)
        elif self.cfg["playout_type"] == "heavy":
            return heavy_playout(self.player if len(actions) % 2 == 0 else self.player%2+1, node.board.board, node.board.inarow, node.board.cols)
        else:
            raise ValueError("Invalid playout type")
    
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
            winners[col] = 0 if winner == -1 else winner
        opponent_winning_moves = (winners==(self.player%2+1)).nonzero()[0]
        if opponent_winning_moves.shape[0] > 0:
            return opponent_winning_moves[0]

        if children != None:
            visits = np.zeros(self.cfg["cols"], dtype=np.int32)
            for action, child in self.root.children.items():
                visits[action] = child.visits
            return visits
        else:
            return random.choice(actions)


mp_or_sp = "mp"
playout_type = "light"


def sp_mcts_agent(obs, cfg):
    start = default_timer()
    global my_mcts
    if (obs["step"]==0) or (obs["step"]==1):
        # delete for repeated simulation
        if "my_mcts" in globals():
            np.random.seed()
            del(my_mcts)
        my_mcts = MCTS(obs, cfg, playout_type)
        start += 15
    else:
        # need to update board based on opponent's tree
        board_diff = my_mcts.root.board.board - np.array(obs["board"]).reshape(cfg["rows"], cfg["columns"])
        opponent_action = board_diff.nonzero()[1][0]
        # if agent is slow, this branch may not have been explored and needs to be manually created
        if opponent_action not in my_mcts.root.children:
            my_mcts.root.children[opponent_action] = Node(my_mcts.root, my_mcts.root.board.drop_piece(opponent_action, my_mcts.player%2+1).flatten(), my_mcts.cfg, my_mcts.player)
        my_mcts.root = my_mcts.root.children[opponent_action]
        my_mcts.root.parent = None
    my_mcts.run_until_timeout(start)
    action = my_mcts.get_best_action()
    if isinstance(action, np.ndarray):
        action = action.argmax()
    my_mcts.root = my_mcts.root.children[action]
    my_mcts.root.parent = None
    return int(action)


def mp_mcts_agent(obs, cfg):
    start = default_timer()
    global my_mctss
    global my_q
    if (obs["step"]==0) or (obs["step"]==1):
        # delete mcts for repeated simulation
        if "my_mctss" in globals():
            np.random.seed()
            del(my_mctss)
            my_q.close()
            del(my_q)
        my_mctss = [MCTS(obs, cfg, playout_type, decision_time=-2.0) for i in range(psutil.cpu_count(logical=False))]
        my_q = mp.Queue()
        start += 10
    else:
        # need to update board based on opponent's tree
        board_diff = my_mctss[0].root.board.board - np.array(obs["board"]).reshape(cfg["rows"], cfg["columns"])
        opponent_action = board_diff.nonzero()[1][0]
        # if agent is slow, this branch may not have been explored and needs to be manually created
        if opponent_action not in my_mctss[0].root.children:
            my_mctss[0].root.children[opponent_action] = Node(my_mctss[0].root, my_mctss[0].root.board.drop_piece(opponent_action, my_mctss[0].player%2+1).flatten(), my_mctss[0].cfg, my_mctss[0].player)
        my_mctss[0].root = my_mctss[0].root.children[opponent_action]
        my_mctss[0].root.parent = None
        # reinit to avoid long put/get
        for i in range(1, len(my_mctss)):
            my_mctss[i] = MCTS(obs, cfg, playout_type, decision_time=-2.0, board=my_mctss[0].root.board.board)
    my_procs = [mp.Process(target=my_mctss[i+1].run_until_timeout, args=[start, my_q]) for i in range(len(my_mctss) - 1)]
    for my_proc in my_procs:
        my_proc.start()
    # q WILL have len(my_mctss) - 1 objects, but may appear empty due to async puts
    # joining before getting results in a deadlock
    # first, wait mcts.simulation_time
    # then, attempt to get len(my_mctss) objs from the q, pausing if it is empty
    n_received = 0
    my_mctss[0].run_until_timeout(start)
    proc_actions = []
    while n_received < len(my_procs):
        if my_q.empty():
            time.sleep(0.01)
        else:
            proc_actions.append(my_q.get())
            n_received += 1
    for proc in my_procs:
        proc.join()
    main_best_actions = my_mctss[0].get_best_action()
    if not isinstance(main_best_actions, np.ndarray):
        action = main_best_actions
    else:
        int_actions = []
        array_actions = [main_best_actions]
        for proc_action in proc_actions:
            if isinstance(proc_action, int):
                int_actions.append(proc_action)
            else:
                array_actions.append(proc_action)

        if len(int_actions) > 0:
            action = int_actions[0]
        else:
            array_actions = np.sum(array_actions, axis=0)
            action = array_actions.argmax()
    action = int(action)
    my_mctss[0].root = my_mctss[0].root.children[action]
    my_mctss[0].root.parent = None
    return action


if mp_or_sp == "sp":
    mcts_agent = sp_mcts_agent
elif mp_or_sp == "mp":
    mcts_agent = mp_mcts_agent
    