import chess
import numpy as np
from tqdm import tqdm
import chess.engine
import chess.svg

import tensorflow as tf
import tensorflow
from tensorflow import keras
from keras import layers
import keras.backend as K

# def policy_loss():

class EngineHelpers:

    def __init__(self):
        self.white_uci_to_output_mapping = self.generate_white_uci_to_output_mapping()
        self.white_output_to_uci_mapping = self.generate_white_output_to_uci_mapping()
        self.black_uci_to_output_mapping = self.generate_black_uci_to_output_mapping()
        self.black_output_to_uci_mapping = self.generate_black_output_to_uci_mapping()

    def generate_white_uci_to_output_mapping(self):
        move_ucis = [f"{f1}{r1}{f2}{r2}" for f1 in "abcdefgh" for r1 in range(1, 9) for f2 in "abcdefgh" for r2 in range(1, 9)]

        files = "abcdefgh"
        promotion_pieces = "bnrq"

        promotion_ucis = [f"{file}7{file}8{p}" for file in "abcdefgh" for p in promotion_pieces]
        promotion_ucis.extend([f"{files[i]}7{files[i+1]}8{p}" for i in range(7) for p in promotion_pieces])
        promotion_ucis.extend([f"{files[i-1]}7{files[i]}8{p}" for i in range(1, 8) for p in promotion_pieces])

        all_ucis = move_ucis + promotion_ucis

        return dict([(uci, index) for index, uci in enumerate(all_ucis)])

    def generate_white_output_to_uci_mapping(self):
        move_ucis = [f"{f1}{r1}{f2}{r2}" for f1 in "abcdefgh" for r1 in range(1, 9) for f2 in "abcdefgh" for r2 in range(1, 9)]
        
        files = "abcdefgh"
        promotion_pieces = "bnrq"

        promotion_ucis = [f"{file}7{file}8{p}" for file in "abcdefgh" for p in promotion_pieces]
        promotion_ucis.extend([f"{files[i]}7{files[i+1]}8{p}" for i in range(7) for p in promotion_pieces])
        promotion_ucis.extend([f"{files[i-1]}7{files[i]}8{p}" for i in range(1, 8) for p in promotion_pieces])  

        all_ucis = move_ucis + promotion_ucis

        return dict([(index, uci) for index, uci in enumerate(all_ucis)]) 

    def generate_black_uci_to_output_mapping(self):
        move_ucis = list(reversed([f"{f1}{r1}{f2}{r2}" for f1 in "abcdefgh" for r1 in range(1, 9) for f2 in "abcdefgh" for r2 in range(1, 9)]))

        files = list(reversed("abcdefgh"))
        promotion_pieces = "bnrq"

        promotion_ucis = [f"{file}2{file}1{p}" for file in files for p in promotion_pieces]
        promotion_ucis.extend([f"{files[i]}2{files[i+1]}1{p}" for i in range(7) for p in promotion_pieces])
        promotion_ucis.extend([f"{files[i-1]}2{files[i]}1{p}" for i in range(1, 8) for p in promotion_pieces])

        all_ucis = move_ucis + promotion_ucis

        return dict([(uci, index) for index, uci in enumerate(all_ucis)])

    def generate_black_output_to_uci_mapping(self):
        move_ucis = list(reversed([f"{f1}{r1}{f2}{r2}" for f1 in "abcdefgh" for r1 in range(1, 9) for f2 in "abcdefgh" for r2 in range(1, 9)]))

        files = list(reversed("abcdefgh"))
        promotion_pieces = "bnrq"

        promotion_ucis = [f"{file}2{file}1{p}" for file in files for p in promotion_pieces]
        promotion_ucis.extend([f"{files[i]}2{files[i+1]}1{p}" for i in range(7) for p in promotion_pieces])
        promotion_ucis.extend([f"{files[i-1]}2{files[i]}1{p}" for i in range(1, 8) for p in promotion_pieces])

        all_ucis = move_ucis + promotion_ucis

        return dict([(index, uci) for index, uci in enumerate(all_ucis)]) 

    def en_passant_to_coord(self, en_passant):
        rank_index = int(en_passant[1]) - 1
        file_index = "abcdefgh".index(en_passant[0])
        return (rank_index, file_index)

    def flip_case(self, character):
        if character.isalpha():
            if character.isupper():
                return character.lower()
            else:
                return character.upper()
        else:
            return character

    def flip_en_passant(self, en_passant):
        if en_passant != "-":
            file_index = "abcdefgh".index(en_passant[0])
            flipped_file_index = 7 - file_index
            flipped_file = "abcdefgh"[flipped_file_index]
            flipped_rank = 8 - int(en_passant[1])
            return "".join([flipped_file, flipped_rank])
        else:
            return en_passant

    def flip_row(self, row):
        return "".join([self.flip_case(character) for character in list(reversed(row))])

    def flip_castling(self, castling):
        return "".join(sorted("".join([self.flip_case(character) for character in list(reversed(castling))])))
        
    def flipped_board_fen(self, board):
        fen = board.fen()
        split_fen = fen.split(" ")
        pieces_fen = split_fen[0]
        flipped_pieces_fen = "/".join([self.flip_row(row) for row in list(reversed(pieces_fen.split("/")))])
        flipped_castling = self.flip_castling(split_fen[2])
        
        return " ".join([flipped_pieces_fen, "w", flipped_castling, self.flip_en_passant(split_fen[3]), split_fen[4], split_fen[5]])


    def generate_board_fen(self, board):
        if board.fen().split(" ")[1] == "w":
            return board.fen()
        else:
            return self.flipped_board_fen(board)


class Engine689:
    def __init__(self, n_res_filters):
        self.n_res_filters = n_res_filters
        self.helpers = EngineHelpers()
        self.build()
        self.model.compile()
        

    def build(self):
        input_dim = (8, 8, 19)

        input_layer = layers.Input(input_dim)

        x = layers.Conv2D(self.n_res_filters, (3, 3), padding="same")(input_layer)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation("relu")(x)
        
        for _ in range(0, 10):
            x = self.add_residual_layer(x)

        end_of_residuals = x

        # Policy head
        x = layers.Conv2D(self.n_res_filters, (3, 3), padding="same")(end_of_residuals)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation("relu")(x)
        x = layers.Flatten()(x)

        policy_head_out = layers.Dense(64*64, activation="softmax")(x)

        #Value head
        x = layers.Conv2D(self.n_res_filters, (3, 3), padding="same")(end_of_residuals)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation("relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        value_head_out = layers.Dense(1, activation="tanh")(x)

        self.model = keras.Model(input_layer, [policy_head_out, value_head_out])

    def add_residual_layer(self, x):

        layer_start = x
        x = layers.Conv2D(self.n_res_filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(self.n_res_filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Add()([layer_start, x])
        x = layers.Activation("relu")(x)

        return x

    def fen_to_input(self, fen):
        piece_dict = {
            "P": 1,
            "N": 2,
            "B": 3,
            "R": 4,
            "Q": 5,
            "K": 6,
            "p": 7,
            "n": 8,
            "b": 9,
            "r": 10,
            "q": 11,
            "k": 12,
        }

        split_fen = fen.split(" ")

        board_matrix = np.zeros((8, 8, 13))

        for row_index, row_epd in enumerate(split_fen[0].split("/")):
            col_index = 0
            for character in row_epd:
                if character.isdigit():
                    for _ in range(int(character)):
                        board_matrix[row_index, col_index, 0] = 1
                        col_index += 1
                else:
                    board_matrix[row_index, col_index, piece_dict[character]] = 1
                    col_index += 1

        en_passant_str = split_fen[3]
        en_passant = np.zeros((8, 8, 1), dtype=np.float32)

        if en_passant_str != "-":
            coords = self.helpers.en_passant_to_coord()
            en_passant[coords[0], coords[1]] = 1
        
        castling = split_fen[2]
        if "K" in castling:
            king_castle = np.ones((8, 8, 1), dtype=np.float32)
        else:
            king_castle = np.zeros((8, 8, 1), dtype=np.float32)
        
        if "Q" in castling:
            queen_castle = np.ones((8, 8, 1), dtype=np.float32)
        else:
            queen_castle = np.zeros((8, 8, 1), dtype=np.float32)

        if "k" in castling:
            opponent_king_castle = np.ones((8, 8, 1), dtype=np.float32)
        else:
            opponent_king_castle = np.zeros((8, 8, 1), dtype=np.float32)

        if "q" in castling:
            opponent_queen_castle = np.ones((8, 8, 1), dtype=np.float32)
        else:
            opponent_queen_castle = np.zeros((8, 8, 1), dtype=np.float32)

        move_count = np.full((8, 8, 1), int(split_fen[4]), dtype=np.float32)

        input_matrix = np.concatenate([board_matrix, en_passant, king_castle, queen_castle, opponent_king_castle, opponent_queen_castle, move_count], axis=2)
        
        assert input_matrix.shape == (8, 8, 19)

        return input_matrix

    def make_move(self, board, verbose=0):
        board_fen = self.helpers.generate_board_fen(board)
        input_matrix = self.fen_to_input(board_fen)
        model_outputs = self.model.predict(np.array([input_matrix]), verbose=0)
        action_outputs = model_outputs[0][0]

        active_color = board.fen().split(" ")[1]
        legal_move_ucis = [move.uci() for move in board.legal_moves]

        if verbose:
            print("Active color:", active_color)

        if active_color == "w":
            valid_model_output_indices = [self.helpers.white_uci_to_output_mapping[uci] for uci in legal_move_ucis]
            legal_move_outputs = action_outputs[valid_model_output_indices]
            greedy_action_index = np.argmax(legal_move_outputs)
            greedy_uci = legal_move_ucis[greedy_action_index]

            assert action_outputs[self.helpers.white_uci_to_output_mapping[greedy_uci]] == np.max(legal_move_outputs)

            return chess.Move.from_uci(greedy_uci)
        elif active_color == "b":
            valid_model_output_indices = [self.helpers.black_uci_to_output_mapping[uci] for uci in legal_move_ucis]
            legal_move_outputs = action_outputs[valid_model_output_indices]
            greedy_action_index = np.argmax(legal_move_outputs)
            greedy_uci = legal_move_ucis[greedy_action_index]

            assert action_outputs[self.helpers.black_uci_to_output_mapping[greedy_uci]] == np.max(legal_move_outputs)

            return chess.Move.from_uci(greedy_uci)