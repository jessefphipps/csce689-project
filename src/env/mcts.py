from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from random import choice
import math
import chess
import chess.engine
import numpy as np


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            # print(f'{node} -- ', end="")
            if node.is_terminal():
                reward = node.reward()
                # print(reward)
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True

def _find_winner(chess_board, color):
    "Returns None if no winner, True if agent wins, False if opponent wins"
    if(chess_board.is_game_over()):
        board = chess_board
        result = None
        if(board.result()=='1-0'):  # white won
            #print("h1")
            # return True
            result = True
        elif(board.result()=='0-1'):    # black won
            #print("h2")
            # return False
            result = False
        if result is not None:
            return (not result) if color == 'black' else result
    return None

_chess = namedtuple("ChessBoard", ["board", "turn", "winner", "terminal", "color"])
class RandomEngineMCTSNode(_chess, Node):

    def is_terminal(self):
        return self.terminal

    def make_move(self, san):
        tmp_board = chess.Board(self.board)
        tmp_board.push_san(san)
        # print(san)
        turn = not self.turn
        winner = _find_winner(tmp_board, self.color)
        is_terminal = tmp_board.is_game_over()
        return RandomEngineMCTSNode(tmp_board.fen(), turn, winner, is_terminal, self.color)

    def find_children(self):
        board = chess.Board(self.board)
        if self.terminal:  # if the game is finished then no moves can be made
            return set()
        return {
            self.make_move(board.san(i)) for i in list(board.legal_moves)
        }
    
    def find_random_child(self):
        board = chess.Board(self.board)
        if self.terminal:
            return None
        legal_moves = [board.san(i) for i in list(board.legal_moves)]
        return self.make_move(choice(legal_moves))
    
    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self}")
        if self.winner is None:
            return 0.5
        if self.turn is (not self.winner):
            return 0
        if self.winner is self.turn:
            # it's your turn and you've already won. should be impossible
            raise RuntimeError(f"reward called on unreachable board {self}")

class RandomEngine:
    def __init__(self):
        pass
    def make_move(self, board):
        legal_moves = list(board.legal_moves)
        move = legal_moves[np.random.randint(0, len(legal_moves))]
        return move


if __name__ == "__main__":
    # create chess game
    # board = chess.Board()
    node = RandomEngineMCTSNode(board=chess.Board().fen(), turn=True, winner=None, terminal=False, color='white')

    # max num iterations
    max_iterations = 10

    # create agents
    mcts_white = MCTS()    # white
    # mcts_black = MCTS()    # black
    engine = RandomEngine()
    # engine = chess.engine.SimpleEngine.popen_uci("/home/bryant/csce689-project/engines/stockfish_15_win_x64_avx2/stockfish_15_x64_avx2.exe")
    # engine.configure({"Skill Level" : 2})

    steps = 0
    while True:
        # mcts for white
        for _ in range(max_iterations):
            mcts_white.do_rollout(node)
        node = mcts_white.choose(node)
        # print(f"Current FEN: {node.board}")
        print(chess.Board(node.board))
        print("==================")
        if node.is_terminal():
            break

        # # invert turn and winner for black
        # node = RandomEngineMCTSNode(board=node.board, turn=(not node.turn), winner=(not node.winner if node.winner is not None else None), terminal=node.terminal, color='black')

        # # mcts for black
        # for _ in range(max_iterations):
        #     mcts_black.do_rollout(node)
        # node = mcts_black.choose(node)
        # # print(f"Current FEN: {node.board}")
        # print(chess.Board(node.board))
        # print("==================")
        # if node.is_terminal():
        #     break
    
        # # invert turn and winner for white
        # node = RandomEngineMCTSNode(board=node.board, turn=(not node.turn), winner=(not node.winner if node.winner is not None else None), terminal=node.terminal, color='white')

        board = chess.Board(node.board)
        eng_move = engine.make_move(board)
        # eng_move = engine.play(board, chess.engine.Limit(time=0.001))
        board.push(eng_move)
        # board.push(eng_move.move)
        node = RandomEngineMCTSNode(board=board.fen(), turn=True, winner=None, terminal=board.is_game_over(), color='white')
        print(chess.Board(node.board))
        print("==================")
        if node.is_terminal():
            break


        steps += 1

    print(chess.Board(node.board).result())