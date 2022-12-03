import chess
import chess.pgn
import chess.engine
import random
import time
import numpy as np
from math import log,sqrt,e,inf

class node():
    def __init__(self):
        self.state = chess.Board()
        self.action = ''
        self.children = set()
        self.parent = None
        self.N = 0
        self.n = 0
        self.v = 0

def ucb1(curr_node):
    ans = curr_node.v+2*(sqrt(log(curr_node.N+e+(10**-6))/(curr_node.n+(10**-10))))
    return ans

def rollout(curr_node):
    
    if(curr_node.state.is_game_over()):
        board = curr_node.state
        if(board.result()=='1-0'):
            #print("h1")
            return (1,curr_node)
        elif(board.result()=='0-1'):
            #print("h2")
            return (-1,curr_node)
        else:
            return (0.5,curr_node)
    
    all_moves = [curr_node.state.san(i) for i in list(curr_node.state.legal_moves)]
    
    for i in all_moves:
        tmp_state = chess.Board(curr_node.state.fen())
        tmp_state.push_san(i)
        child = node()
        child.state = tmp_state
        child.parent = curr_node
        curr_node.children.add(child)
    rnd_state = random.choice(list(curr_node.children))

    return rollout(rnd_state)

def expand(curr_node,white):
    if(len(curr_node.children)==0):
        return curr_node
    max_ucb = -inf
    if(white):
        idx = -1
        max_ucb = -inf
        sel_child = None
        for i in curr_node.children:
            tmp = ucb1(i)
            if(tmp>max_ucb):
                idx = i
                max_ucb = tmp
                sel_child = i

        return(expand(sel_child,0))

    else:
        idx = -1
        min_ucb = inf
        sel_child = None
        for i in curr_node.children:
            tmp = ucb1(i)
            if(tmp<min_ucb):
                idx = i
                min_ucb = tmp
                sel_child = i

        return expand(sel_child,1)

def rollback(curr_node,reward):
    curr_node.n+=1
    curr_node.v+=reward
    while(curr_node.parent!=None):
        curr_node.N+=1
        curr_node = curr_node.parent
    return curr_node

def mcts_pred(curr_node,over,white,iterations=10):
    if(over):
        return -1
    all_moves = [curr_node.state.san(i) for i in list(curr_node.state.legal_moves)]
    map_state_move = dict()
    
    for i in all_moves:
        tmp_state = chess.Board(curr_node.state.fen())
        tmp_state.push_san(i)
        child = node()
        child.state = tmp_state
        child.parent = curr_node
        curr_node.children.add(child)
        map_state_move[child] = i
        
    while(iterations>0):
        if(white):
            idx = -1
            max_ucb = -inf
            sel_child = None
            for i in curr_node.children:
                tmp = ucb1(i)
                if(tmp>max_ucb):
                    idx = i
                    max_ucb = tmp
                    sel_child = i
            ex_child = expand(sel_child,0)
            reward,state = rollout(ex_child)
            curr_node = rollback(state,reward)
            iterations-=1
        else:
            idx = -1
            min_ucb = inf
            sel_child = None
            for i in curr_node.children:
                tmp = ucb1(i)
                if(tmp<min_ucb):
                    idx = i
                    min_ucb = tmp
                    sel_child = i

            ex_child = expand(sel_child,1)

            reward,state = rollout(ex_child)

            curr_node = rollback(state,reward)
            iterations-=1
    if(white):
        
        mx = -inf
        idx = -1
        selected_move = ''
        for i in (curr_node.children):
            tmp = ucb1(i)
            if(tmp>mx):
                mx = tmp
                selected_move = map_state_move[i]
        return selected_move
    else:
        mn = inf
        idx = -1
        selected_move = ''
        for i in (curr_node.children):
            tmp = ucb1(i)
            if(tmp<mn):
                mn = tmp
                selected_move = map_state_move[i]

        return selected_move

class RandomEngine:
    def __init__(self):
        pass
    def make_move(self, board):
        legal_moves = list(board.legal_moves)
        move = legal_moves[np.random.randint(0, len(legal_moves))]
        return move

if __name__ == "__main__":
    board = chess.Board()
    # engine = chess.engine.SimpleEngine.popen_uci('/home/bryant/csce689-project/engines/stockfish_15_win_x64_avx2/stockfish_15_x64_avx2.exe')
    engine = RandomEngine()

    white = 1
    moves = 0
    pgn = []
    game = chess.pgn.Game()
    evaluations = []
    sm = 0
    cnt = 0
    # while((not board.is_game_over())):
    while(True):
        # print(f'at move {moves}')
        # all_moves = [board.san(i) for i in list(board.legal_moves)]
        # #start = time.time()
        # root = node()
        # root.state = board
        # result = mcts_pred(root,board.is_game_over(),white)
        # #sm+=(time.time()-start)
        # board.push_san(result)
        # #print(result)
        # pgn.append(result)
        # white ^= 1
        # #cnt+=1
        
        # moves+=1
        ## board_evaluation = evaluate(board.fen().split()[0])
        ## evaluations.append(board_evaluation)

        # print(f'at move {moves}')
        # white turn
        all_moves = [board.san(i) for i in list(board.legal_moves)]
        #start = time.time()
        root = node()
        root.state = board
        result = mcts_pred(root,board.is_game_over(),white)
        #sm+=(time.time()-start)
        board.push_san(result)
        #print(result)
        pgn.append(result)
        # white ^= 1
        #cnt+=1
        print(board)
        print("==================")
        moves += 1
        if board.is_game_over():
            break

        # black turn
        eng_move = engine.make_move(board)
        pgn.append(board.san(eng_move))
        board.push(eng_move)
        print(board)
        print("==================")
        moves+=1
        if board.is_game_over():
            break
    #print("Average Time per move = ",sm/cnt)
    print(board)
    print(" ".join(pgn))
    print()
    #print(evaluations)
    print(board.result())
    game.headers["Result"] = board.result()
    #print(game)
    # engine.quit()