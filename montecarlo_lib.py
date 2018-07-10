
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import copy,deepcopy


class checkers_state:
    def __init__(self,board_size = 8, board = None, player = 0, verbose = False, notes = None, allow_kings = True, allow_draws = True, max_turns = 100, tiebreaker_rule = False):        

        if not board is None:
            self.board_size = board.shape[0]
            self.board = board
        else:
            self.board_size = board_size
            self.board = self.create_board(n = board_size)
        self.player = player
        #self.action_space = self.get_action_space()
        self.done = False
        self.num_just_jumped = 0
        self.update_action_space()
        self.verbose = verbose
        self.notes = None
        self.max_turns = max_turns
        self.turn_num = 0 ##Total number of turns moved thus far
        self.allow_draws = allow_draws
        self.allow_kings = allow_kings
        self.tiebreaker_rule = tiebreaker_rule
        
    def create_board(self, n):
        board = np.zeros([n,n]).astype(int)
        for i in range(n):
            for j in range(n):
                if i < n/2 - 1:
                    x = 1
                elif (n-i) < n/2:
                    x = -1
                else:
                    continue
                if (i + j) % 2 == 1:
                    board[j,i] = x
        return board    
    
    def show_board(self):
        #plt.clf()

        #fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
        # (or if you have an existing figure)
        # fig = plt.gcf()
        # ax = fig.gca()

        #ax.add_artist(circle2)
        fig, ax = plt.subplots() 

        #ax.add_artist(plt.Circle( (i + .5, j + .5), 0.2, color='blue'))

        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i + j) % 2 == 0:
                    square_color = 'grey'
                else:
                    square_color = 'white'

                ax.add_artist(plt.Rectangle( (i, j), 1,1, color=square_color))
                if self.board[i,j] > 0:
                    color = 'red'
                elif self.board[i,j] < 0:
                    color = 'black'
                else:
                    continue
                #print i,j
                ax.add_artist(plt.Circle( (i + .5, j + .5), 0.2, color=color))
                if abs(self.board[i,j]) > 1:
                    ###This piece is a king
                    ax.add_artist(plt.Circle( (i + .5, j + .5), 0.3, color=color, fill = False))

        plt.xlim(0,self.board_size)
        plt.ylim(0,self.board_size)

        plt.show()
         
    def player_pieces(self, player = None):
        if player is None:
            player = self.player
        pieces = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if player == 0 and self.board[i,j] > 0:
                    pieces.append((i,j))
                elif player == 1 and self.board[i,j] < 0:
                    pieces.append((i,j))
        return pieces
      
    def get_action_space(self):
        return self.action_space

    def update_action_space(self):
        possible_jumps = []
        possible_moves = []
        
        if self.num_just_jumped > 0:
            pieces_to_move = [self.last_jumper]
        else:
            pieces_to_move = self.player_pieces()
        
        for u in pieces_to_move:
            for move_vector in self.legal_vectors(u):
                v = (u[0] + move_vector[0], u[1] + move_vector[1])
                action = (u,v)
                if self.is_legal(action):
                    possible_moves.append(action)
                    if np.abs(move_vector[0]) > 1:
                        possible_jumps.append(action)
        
        if len(possible_jumps) > 0:
            #if you can jump, you must jump
            self.must_jump = True
            if self.verbose:
                print "MUST JUMP!!, possible jumps:", possible_jumps
            self.action_space = possible_jumps
        else:
            self.must_jump = False
            self.action_space =  possible_moves

    def get_action_from_int(self, i):
        if i > self.num_actions():
            raise ValueError("Action index %d not in range of action space of length %d" % (i, self.num_actions()))
        else:
            return self.action_space[i]

    def is_legal(self, action, explain = False):
        legal = True
        (u,v) = action
        #print 'Checking action', action

        if not all([0 <= u[i] < self.board_size for i in range(2)]):
            if explain:
                print "Player %d trying to move a piece not in range (0, %d)" % (self.player, self.board_size) 
            return False

        elif not all([0 <= v[i] < self.board_size for i in range(2)]):
            if explain:
                print "Player %d trying to move a piece to outside of (0, %d)" % (self.player, self.board_size) 
            return False
        
        if self.board[u] == 0:
            if explain:
                print "Player %d is trying to move from empty position (%d,%d)" % (self.player, u[0], u[1])
            return False
        
        elif self.board[v] != 0:
            if explain:
                print "Player %d is trying to move to nonempty position (%d,%d)" % (self.player, v[0], v[1])
            return False
        
        elif (self.board[u] > 0 and self.player == 1) or (self.board[u] > 0 and self.player == 1):
            if explain:
                print "Player %d trying to move a piece belonging to player %d" % (self.player, 1 - self.player)
            return False
            
        move_vector =  (v[0] - u[0], v[1] - u[1])   
        
        if not move_vector in self.legal_vectors(u):
            if explain:
                print "Player %d trying vector %s, not in legal vectors %s" % (self.player, move_vector, self.legal_vectors(u))
            return False
        
        if np.abs(move_vector[0]) > 1:
            
            w = (int((u[0] + v[0]) / 2), int((u[1] + v[1]) / 2))
            
            if self.player == 0 and self.board[w] < 0:
                pass
                #print "Jumped piece at position (%d,%d)" % w
            elif self.player == 1 and self.board[w] > 0:
                pass
                #print "Jumped piece at position (%d,%d)" % w
            else:
                if explain:
                    print "Player %d trying to jump own piece or empty position at %s" (self.player, w)
                return False
        return True

    def legal_vectors(self, piece):
        if np.abs(self.board[piece]) > 1:
            #this piece is a king
            legal_vectors = [(1,1), (-1,1), (2,2), (-2, 2), (1,-1), (-1,-1), (2,-2), (-2, -2)]
        elif self.player == 0:
            legal_vectors =  [(1,1), (-1,1), (2,2), (-2, 2)]
        elif self.player == 1:
            legal_vectors = [(1,-1), (-1,-1), (2,-2), (-2, -2)]
            
        return legal_vectors
    
    def take_action(self, action, inplace = True):
        
        # an action is a tuple of two (u,v) where u,v are 2x2 vectors;
        # u coordinates of a checker, v coordinates of a place to move to
        
        (u,v) = action
        
        #player = 0 -> positive pieces
        #player = 1 -> negative pieces
        
        if not all([0 <= u[i] < self.board_size for i in range(2)]):
            raise ValueError("Player %d trying to move a piece not in range (0, %d)" % (self.player, self.board_size) )
        if not all([0 <= v[i] < self.board_size for i in range(2)]):
            raise ValueError("Player %d trying to move a piece to outside of (0, %d)" % (self.player, self.board_size) )
        if self.board[u] == 0:
            raise ValueError("Player %d is trying to move from empty position (%d,%d)" % (self.player, u[0], u[1]) )
        elif self.board[v] != 0:
            raise ValueError("Player %d is trying to move to nonempty position (%d,%d)" % (self.player, v[0], v[1]))
        elif (self.board[u] > 0 and self.player == 1) or (self.board[u] > 0 and self.player == 1):
            raise ValueError("Player %d trying to move a piece belonging to player %d" % (self.player, 1 - self.player))
        
        if inplace:
            next_state = self
        else:
            next_state = deepcopy(self)
        
        if abs(u[0] - v[0]) == 2:
            w = (int((u[0] + v[0]) / 2), int((u[1] + v[1]) / 2))
            if (self.player == 0 and self.board[w] < 0) or (self.player == 1 and self.board[w] > 0):
                next_state.board[w] = 0
                if self.verbose:
                    print "Jumped piece at position (%d,%d)" % w
                next_state.num_just_jumped += 1
            else:
                raise ValueError("Trying to jump own piece or empty position at %s", w)
        
        
        next_state.board[v] = next_state.board[u]
        next_state.board[u] = 0
        if self.allow_kings and np.abs(next_state.board[v]) == 1:
            if (next_state.player == 0 and v[1] == next_state.board_size - 1) or (next_state.player == 1 and v[1] == 0):
                if self.verbose:
                    print "King me!"
                next_state.board[v] = next_state.board[v] * 2
            

        
        
        if next_state.num_just_jumped > 0:
            next_state.last_jumper = v
            next_state.update_action_space()
            if next_state.must_jump:
                if self.verbose:
                    adj_dict = {2:'DOUBLE', 3:'TRIPLE', 4:'QUADRUPLE', 5:'QUINTUPLE'}

                    print "%s JUMP!!" % (adj_dict[next_state.num_just_jumped + 1])
                    pass
            else:
                next_state.switch_players()
        else:
            next_state.switch_players()
        next_state.update_action_space()
        
        next_state.turn_num += 1
        
        if inplace:
            return None
        else:
            return next_state
        

            #self.action_space = self.get_action_from_int()
            
    def switch_players(self):
        self.player = (1 - self.player)
        self.num_just_jumped = 0
        self.must_jump = False
        self.update_action_space()
        self.last_jumper = None

            
    def is_done(self):
        return (len(self.action_space) == 0) or self.turn_num >= self.max_turns
    
    def reward(self):
        reward = 0
        done = self.is_done()
        if done:
            if self.winner() == self.player:
                reward = 1
        return reward

    def winner(self):

        if not self.is_done():
            raise ValueError("Game is not finished.")

        num_player_0_pieces = len(self.player_pieces(player = 0))
        num_player_1_pieces = len(self.player_pieces(player = 1))


        if num_player_0_pieces == 0 and num_player_1_pieces > 0:
            return 1
        elif num_player_1_pieces == 0 and num_player_0_pieces > 0:
            return 0
        else:

            if num_player_0_pieces == num_player_1_pieces:
                if self.allow_draws:
                    return 'draw'
                else:
                    return 0 #player 0 wins, just 'cuz.
            if self.tiebreaker_rule or not self.allow_draws:
                
                #Tiebreaker rule means -> if time up, the player with more pieces wins.
                
                if num_player_0_pieces > num_player_1_pieces:
                    return 0
                elif num_player_1_pieces > num_player_0_pieces:
                    return 1
            else:
                return 'draw'   

 
            
    def step(self, i, inplace = True):
        #self.take_action(self.action_space[i])
#         if not inplace:
#             orig_board = deepcopy(self.board)
        
        if inplace:
            self.take_action(self.get_action_from_int(i), inplace = True)
            next_state = self
        else:
            next_state = self.take_action(self.get_action_from_int(i), inplace = False)
            
        observation = next_state.board
        #done = len(next_state.action_space) == 0
        done = next_state.is_done()
        next_state.done = done


        #self.reward = reward
        reward = self.reward()
        info = None
#         if not inplace:
#             if (self.board == orig_board).all():
#                 pass
#             else:
#                 print "WTF?"
#                 raise ValueError("not inplace but changing state")
        
        
        if inplace:
            return observation, reward, done, info
        else:
            return observation, reward, done, info, next_state

    
        
    def step_random(self, inplace = True):
        return self.step(self.random_action(), inplace = inplace)
    
    def random_action(self):
        k = self.num_actions()
        #print 'choosing from %d actions.' % k
        return np.random.randint(k)
    
    def num_actions(self):
        return len(self.action_space)
    
def argmax(a_dict, random = True):
    if len(a_dict) == 0:
        raise ValueError("Trying to find argmax of empty dict")
    max_value = max(a_dict.values())
    max_keys = [key for key in a_dict.keys() if a_dict[key] == max_value]
    
    if len(max_keys) == 0:
        raise ValueError("Cannot choose max argmax for dict", a_dict)
    elif len(max_keys) >= 2 and random:
        return sample(max_keys)
    else:
        return sorted(max_keys)[0]

def sample(A):
    return A[np.random.randint(len(A))]

def refactor(df, columns, preserved  = []):
    groups = [group.rename_axis({col:col + '_%s' % (str(x)) for col in group.columns if not col in preserved + columns }, axis = 'columns').reset_index() for x,group in df.groupby(columns)]
    for i in range(len(groups)):
        groups[i]['index'] = 0
        groups[i] = groups[i][[col for col in groups[i].columns if not col in columns]]
    df_refactored =  reduce( lambda df1, df2:  df1.merge(df2, how = 'outer'), groups)
    del df_refactored['index']
    return df_refactored

class MonteCarloTree:
    def __init__(self, game_state, 
                 budget = 100, 
                 c = 1, 
                 twoplayer = True, 
                 log_tree_building = False, 
                 num_simulations = 1,
                 endgame_predictor = None,
                 max_steps_to_simulate = 100,
                 save_simulations = False):
        
        self.root = Node(game_state, index = 0, depth = 0)
        self.budget = budget
        self.tree = {0:self.root}  #tree is a dictionary of Nodes
        self.max_steps_to_simulate = max_steps_to_simulate
        self.c = 1
        self.twoplayer = twoplayer
        self.log_tree_building = log_tree_building
        self.root_player = game_state.player
        self.num_simulations = num_simulations
        self.endgame_predictor = endgame_predictor
        self.save_simulations = save_simulations
        
    def result(self):
        '''
        If final result of this game is known, assuming perfect play, return 'win' or 'loss' or 'draw'.
        If final result is not known (tree not complete), return None.
        '''
        
        if self.root.is_complete:
            return self.root.result

    def UCTSearch(self): 
        
        if self.budget == 0:
            #If budget is 0, return random move.
            return self.root.state.random_action()
        num_actions = self.root.state.num_actions()
        
        if num_actions == 0:
            raise ValueError("Game over; no tree search necessary.")
        elif num_actions == 1:
            return 0
            
        
        self.root.N = 1
        nsteps = 0
        while nsteps < self.budget:
            nsteps += 1
            next_node = self.TreePolicy()  #creates a new node and adds to tree
            
            if self.root.is_complete:
                #if self.log_tree_building:
                note = "Game is determined: %s for Player %d\n"  % (self.root.result, self.root_player)

                
                self.root.state.notes += note             
                if self.log_tree_building:
                    print note                             
                break
            else:
                total_Q = 0
                results = []
                if self.save_simulations:
                    games = []
                else:
                    games = None
                for i in range(self.num_simulations):
                    if self.save_simulations:
                        Q, game_states = self.Simulate(next_node, return_game_states = True)
                        games.append(game_states)
                    else:
                        Q = self.Simulate(next_node, return_game_states = False)
                    total_Q += Q
                    results.append(Q)
                    
                    #next_node.all_simulation_results.append(Q)
                self.BackPropogate(next_node, results,  self.num_simulations, games)

                
        if self.result() == 'loss':
            return None
        else:
            return self.BestChild(self.root, explore = False).action

    def TreePolicy(self):
        node = self.root
        while len(node.children) > 0:  #explore until a leaf is found
            if len(node.children) < node.state.num_actions(): #if not fully expanded
                return self.Expand(node)
            else:
                node = self.BestChild(node, self.c)
        return self.Expand(node)

    
    def Simulate(self,node, return_game_states = False):
        #node.N += 1  #this increment is now performed in BackPropogate()
        step_num = 0
        current_state = deepcopy(node.state)
        nsteps = 0
        #Should we be updating the tree here?
        game_states = [current_state]
        while nsteps < self.max_steps_to_simulate and not current_state.is_done():
            nsteps += 1
            action = current_state.random_action()
            observation, reward, done, info = current_state.step(action)
            if return_game_states:
                game_states.append(deepcopy(current_state))
            #ignore given reward.
                
        if current_state.is_done():
            #Reward is 1 if root player is winner, otherwise 0.
            winner = current_state.winner()

            if winner == self.root_player:
                reward = 1.0
            elif winner == (1 - self.root_player):
                reward = 0.0
            elif winner == 'draw':
                reward = 0.5
            else:
                raise ValueError("Winner of game is %s; not understood." % winner   )
            if return_game_states:
                return reward, game_states
            else:
                return reward
        else:
            if self.endgame_predictor is None:
                raise ValueError("Max number of steps exceeded")
            else:
                predicted_winner = self.endgame_predictor.predict(current_state)
                return predicted_winner

    def Expand(self,node):
        if len(node.children) >= node.state.num_actions():
            raise ValueError("Node fully expanded already")
        else:
            action = len(node.children)
            new_index = max(self.tree.keys()) + 1
            if self.log_tree_building:
                print "Adding node %d from parent %d with action %d" % (new_index, node.index, action)
            observation, reward, done, info, new_state = node.state.step(action, inplace = False)
            new_child = Node(new_state, new_index, action = action, parent = node.index, depth = node.depth + 1)
            node.children.append(new_index)
            self.tree[new_index] = new_child
            self.BackPropogate_EndGame(new_index)
            return new_child
            
            
    def BackPropogate_EndGame(self, index):
        '''
        Determines if the given node's game result is fully determined based on the current tree.
        
        Note, assumes game is completely deterministic!
        
        Logic:
        
            - If any child is the current player's turn and a WIN, current node is a win.
            - If any child is the opponent's turn and a LOSS, current node is a loss.
            - Otherwise, if all children have results completely determined by the tree:
               - If any is a DRAW, the current node is a draw.
               - Otherwise, the current node is a loss.
            - If the above do not apply, current game's result is unknown.
            
        
        '''
        
        
        node = self.tree[index]
        
        if self.twoplayer:
            if node.state.is_done():
                node.is_complete = True
                winner = node.state.winner()
                if winner == 'draw':
                    node.result = 'draw'
                elif winner == node.state.player:
                    node.result = 'win'
                elif winner == 1 - node.state.player:
                    node.result = 'loss'
                else:
                    raise ValueError("Winner reported as %s; not understood" % winner)

            
            elif any([self.tree[child_index].state.player != node.state.player and self.tree[child_index].is_complete and self.tree[child_index].result == 'loss' for child_index in node.children]):
                #if any child is opponent and is a loss to that player, the current node is a WIN for the current node's player.
                self.tree[index].is_complete = True
                self.tree[index].result = 'win'

            elif any([self.tree[child_index].state.player == node.state.player and self.tree[child_index].is_complete and self.tree[child_index].result == 'win' for child_index in node.children]):
                #if any child is current player and a win, the current node is a WIN for the current player.
                self.tree[index].is_complete = True
                self.tree[index].result = 'win'
                
            elif len(node.children) >= node.state.num_actions(): #fully expanded
                ## No win possible for the current node's player.  Draw if possible.
                
                
                if all([self.tree[child_index].is_complete for child_index in node.children]):
                    if any([self.tree[child_index].result == 'draw' for child_index in node.children]):
                    
                        '''
                        If any child is a draw, result is a draw for the current node's player.
                        '''
                        self.tree[index].is_complete = True
                        self.tree[index].result = 'draw'
                    else:
                        '''
                        If draw is not possible, result is a loss for the current node's player.
                        '''
                        self.tree[index].is_complete = True
                        self.tree[index].result = 'loss'

        else:
            #single-player game
            if node.state.done:
                #draws not implemented for single-player games.
                node.is_complete = True
                winner = node.state.winner()
                if node.state.winner == node.state.player:
                    node.result = 'win'
                else:
                    node.result = 'loss'
            elif any([self.tree[child_index].is_complete and self.tree[child_index].result == 'win' for child_index in node.children]):
                #if any child is a win, the current node is a win for the current player.
                self.tree[index].is_complete = True
                self.tree[index].result = 'win'
            elif len(node.children) >= node.state.num_actions(): #fully expanded
                if all([self.tree[child_index].is_complete and self.tree[child_index].result == 'loss' for child_index in node.children]):
                    #if ALL children are losses the the current node is a loss.                    
                    self.tree[index].is_complete = True
                    self.tree[index].result = 'loss'
                    
        
        if node.is_complete:
            if self.log_tree_building:
                print "Node %d is complete and is a %s!!" % (node.index, node.result)
            
            if not node.parent is None:
                self.BackPropogate_EndGame(node.parent)

    def BackPropogate(self, node, results, num_trials, games = None):
        orig_index = node.index
        node = deepcopy(node)
        node_index = node.index
        while not node_index is None:
            #print "Backpropogating results from node %d to node %d:" % (orig_index, node.index)
            #print results
            node = self.tree[node_index]
            #node.Q += reward
            node.N += num_trials
            node.all_simulation_results += results
            node_index = node.parent
            if not games is None:
                node.games += games
    
    def BestChild(self, node, explore = True, display_scores = False):
        #Perform Sophie's Choice
        if len(node.children) == 0:
            raise ValueError("Node has no children")
        if explore and node.is_complete:
            raise ValueError("Node has already been fully explored")
        
        S = {}
        if not self.twoplayer:
            #only two-player games currently implemented.
            raise ValueError("not implemented")
            
            
        if node.is_complete:
            if node.result == 'loss':
                raise ValueError("Node is a loss; no best child.")
                ##todo: if loss, can still choose node with highest probability of winning in random play
                
            else:
                for i in node.children:
                    child = self.tree[i]

                    ##First, check for any winning moves and take one if possible.

                    if node.result == 'win':
                        if child.state.player == node.state.player:
                            if child.is_complete and child.result == 'win':
                                #this child is a WIN to the current node's player.  Take this move and don't explore further.
                                return child
                        elif child.state.player != node.state.player:
                            if child.is_complete and child.result == 'loss':
                                #this child is a LOSS for the opponent.  Take this move and don't explore further.                            
                                return child
                    elif node.result == 'draw':
                        if child.is_complete and child.result == 'draw':
                            return child
                    
                else:
                    raise ValueError("Game is reported as a %s, but no %s move found." % (node.result, 'winning' if node.result == 'win' else 'drawing'))
        
        for i in node.children:
            child = self.tree[i]
            if explore:
                if child.is_complete:
                    #This child completely explored; do not bother.
                    continue
                
                if child.N == 0:
                    S[i] = np.inf
                else:
                    Q = child.Q
                    if node.state.player != self.root_player:
                        Q = -Q
                    S[i] = Q / float(child.N) + self.c * np.sqrt(  2 * np.log( node.N ) / float(child.N)  )

            else:
                try:
                    Q = child.Q
                    if node.state.player != self.root_player:
                        Q = -Q
                    S[i] = Q / float(child.N)  #Should we bias towards high N for conservative play?
                except:
                    raise ValueError("Node %d has N = 0" % i)
                    
        if display_scores:
            print S

        child = self.tree[argmax(S, random = False)]  #should we return an INDEX?
        return child

class Node:
    def __init__(self, state, index, action = None, parent = None, depth = None):
        self.state = state
        self.action = action  #The action that went from the parent of this state to this state.
        self.children = []
        self.index = index
        self.parent = parent
        #self.Q = 0  #total reward
        self.N = 0  #total visit count
        self.is_complete = False # Set to True if subtree with this root is FULLY expanded to endgame 
        self.result = None # Set to 'win' if game is completely determined as a win for the CURRENT player, 
                           #'loss' if game is completely determined as a loss for the CURRENT player, 
                           #'draw' if game is completely determined as a draw for the CURRENT player, 
                           # if unknown, is set to None
        self.all_simulation_results = []
        self.depth = depth
        self.games = []
        
    @property
    def Q(self):
        return sum(self.all_simulation_results)
        
        
class simple_checkers_predictor:
    def __init__(self, king_score = 2):
        self.scoring_dict = {0:0, 1:1, 2:king_score}

    def predict(self, state):
        ###Predicts the winner of a checkers game using a simple scoring system.
        if state.is_done():
            return state.winner()
        else:
            total_scores = [0,0]

            for i in range(state.board_size):
                for j in range(state.board_size):
                    score = self.scoring_dict[np.abs(state.board[i,j])]
                    player = 0 if state.board[i,j] > 0 else 1
                    total_scores[player] += score

            if total_scores[0] > total_scores[1]:
                return 0
            elif total_scores[1] > total_scores[0]:
                return 1
            else:
                ##Draw
                if state.allow_draws:
                    return 'draw'
                else:
                    return 0 #Player 0 wins draws, just 'cuz.
                
                
def get_random_subset(n, p):
    
    random_vector = np.random.rand(n)
    
    return [i for i in range(n) if random_vector[i] <= p]
    
    

def transform_board(state, encode_type = 'rectangle'):
    '''
    Given a checkers board with pieces as used in checkers_state, output a single large vector that a dnn can understand.
    Different encodings may have advantages or disadvantages.
    '''
    board = state.board
    n = state.board_size
    if encode_type == 'diamond':
     #assume square, and n even.
        M = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                if (i+j) %2 == 1:
                    x = (i+j - 1) / 2
                    y = (i - j + n) / 2
                    #print i,j, ',',board[i,j],',', x,y
                    #if M[x,y] != 0:
                    #    print 'reused??'
                    M[x,y] = board[i,j]

        return M
    elif encode_type == 'rectangle':
        M = np.zeros([n,(n+1)/2])
        
        for i in range(n):
            for j in range(n):
                if (i+j) %2 == 1:
                    y = j / 2 #integer division
                    x = i
                    M[x,y] = board[i,j]
    else:
        M = state.board
    return M 


def get_single_board_vector(state, encode_type = 'rectangle'):
    
    #TODO: encode information of forced jumps.
    board = copy(state.board)
    
    m,n = board.shape
    
    encoded_board = transform_board(state, encode_type = encode_type)
    
    m,n = encoded_board.shape
    N = m*n
    
    player_0_checkers_vector = ((encoded_board == 1)).reshape(1,N)[0].astype(int)
    player_1_checkers_vector = ((encoded_board == -1)).reshape(1,N)[0].astype(int)
    
    current_player_vector = [1,0] if state.player == 0 else [0,1]  #Q: Better with 2-dim encoding or 1-dim??
    

    
    
    if state.allow_kings:
        player_0_kings_vector = (encoded_board * (encoded_board == 2)).reshape(1,N)[0].astype(int)
        player_1_kings_vector = (encoded_board * (encoded_board == 2)).reshape(1,N)[0].astype(int)
        return sum( [list(A) for A in [player_0_checkers_vector, 
                                      player_0_kings_vector, 
                                      player_1_checkers_vector, 
                                      player_1_kings_vector, 
                                      current_player_vector] ], [])
    
    else:
        return sum( [list(A) for A in [player_0_checkers_vector, 
                                      player_1_checkers_vector, 
                                      current_player_vector] ], [])

    
    
    
    