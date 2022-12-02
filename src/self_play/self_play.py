import chess
import numpy as np
from tqdm import tqdm
import chess.engine
import chess.svg


# engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\Jesse Phipps\Documents\GitHub\csce689-project\engines\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe")

class RandomEngine:
    def __init__(self):
        pass
    def make_move(self, board):
        legal_moves = list(board.legal_moves)
        move = legal_moves[np.random.randint(0, len(legal_moves))]
        return move

# def step(board, engine):
    

board = chess.Board()
engine = RandomEngine()
episode = []
while not board.is_game_over():  
    move = engine.make_move(board)
    episode.append((board.fen(), move, 0))
    board.push(move)
    
print(board)

print(episode[:3])