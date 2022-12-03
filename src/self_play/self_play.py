#%%
import chess
import numpy as np
from tqdm import tqdm
import chess.engine
import chess.svg
import tensorflow as tf
import sys
sys.path.append('../model')
sys.path.append('../mcts_a2c')
from model import Engine689, actor_loss, critic_loss
from mcts_a2c import MCTS_Agent

# engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\Jesse Phipps\Documents\GitHub\csce689-project\engines\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe")

class RandomEngine:
    def __init__(self):
        pass
    def make_move(self, board):
        legal_moves = list(board.legal_moves)
        move = legal_moves[np.random.randint(0, len(legal_moves))]
        return move

# def step(board, engine):
    
#%%
board = chess.Board()
random_engine = RandomEngine()
load_saved_model = tf.keras.models.load_model('../../models/prelim_model', custom_objects={'loss': {'actor_output_loss' : actor_loss, 'critic_output_loss' : critic_loss}})
engine689 = Engine689(1)
engine689.model = load_saved_model
mcts_agent = MCTS_Agent(engine689, {'num_simulations': 50})

episode = []
steps = 0
while not board.is_game_over():  
    # trained engine move
    # move = engine689.make_move(board)
    move = mcts_agent.make_move(board, player=1)    # 1 white, -1 black
    board.push(move)

    # random engine move
    move = random_engine.make_move(board)
    episode.append((board.fen(), move, 0))
    board.push(move)
    print(board)
    print("===============")
print(board.result())