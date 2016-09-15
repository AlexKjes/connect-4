import numpy as np


class Game(object):
    def __init__(self, player1=None, player2=None, size=(7, 6)):
        self.board = np.zeros(size, dtype=np.int)
        self.players = [player1, player2]
        self.players_turn = 1
        self.moves = []
        self.last_move_pos = []
        self.game_over = False
        self.board_size = size[0]*size[1]
        self.output = False

    def start_game(self):
        self.game_loop()

    def set_player1(self, player1):
        self.players[0] = player1

    def set_player2(self, player2):
        self.players[1] = player2

    # checks if the move is valid
    # invalid if row is full
    def validate_move(self, move):
        return self.board[move][0] == 0 and 0 <= move < self.board.shape[0]

    # processes the drop of the chip
    def process_move(self):
        last_move = self.moves[-1]
        for i in range(self.board.shape[1]-1, -1, -1):
            if self.board[last_move[1]][i] == 0:
                self.board[last_move[1]][i] = last_move[0]
                self.last_move_pos = (last_move[1], i)
                return

    # checks if a move is victorious
    def check_if_move_is_victorious(self):
        directions = (self.board[self.last_move_pos[0]],
                      self.board[:, self.last_move_pos[1]],
                      self.board.diagonal(self.last_move_pos[1]-self.last_move_pos[0]),
                      np.fliplr(self.board).diagonal(np.abs(self.last_move_pos[1]-(self.board.shape[1]-1))-self.last_move_pos[0]))
        for v in directions:
            if self._check_for_four_in_row(v, self.moves[-1][0]):
                self.game_over = True
                if self.output:
                    print("PLAYER" + str(self.players_turn) + " IS VICTORIOUS!!!")
                break

        if len(self.moves) == self.board_size:
            self.game_over = True
            if self.output:
                print("No moves left. Game draw")

    # Checks a direction vector for four in a row
    @staticmethod
    def _check_for_four_in_row(vector, player_id):
        num_in_row = 0
        for i in vector:
            if i == player_id:
                num_in_row += 1
            else:
                num_in_row = 0
            if num_in_row == 4:
                return True
        return False

    # The main game loop
    def game_loop(self):
        while not self.game_over:
            if self.output:
                print('--------------------------------------------\n')
                print(self.board.T)
                print('player' + str(self.players_turn) + '\'s turn: ')
            move = self.players[self.players_turn-1].take_turn()
            if self.validate_move(move):
                self.moves.append((self.players_turn, move))
                self.process_move()
                self.players[self.players_turn-1].post_process()
                if self.check_if_move_is_victorious():
                    break

                if self.players_turn == 1:
                    self.players_turn = 2
                else:
                    self.players_turn = 1

            else:
                if self.output:
                    print("Invalid move.. Try again")


class BasePlayer(object):

    def take_turn(self):
        pass

    def post_process(self):
        pass


class AIBasePlayer(BasePlayer):
    def __init__(self, game):
        self.game = game
        self.state_before_move = game.board
        self.same_state_in_row = 0

    def take_turn(self):
        self.state_before_move = np.copy(self.game.board)

    def calculate_reward(self):
        before = self.calculate_reward_at_state(self.state_before_move)
        after = self.calculate_reward_at_state(self.game.board)
        print(before)
        print(after)
        return np.subtract(before, after)

    def calculate_reward_at_state(self, state):
        total_in_row = [[0, 0, 0, 0], [0, 0, 0, 0]]
        vectors = (state[self.game.last_move_pos[0]],
                   state[:, self.game.last_move_pos[1]],
                   state.diagonal(self.game.last_move_pos[1]-self.game.last_move_pos[0]),
                   np.fliplr(state).diagonal(np.abs(self.game.last_move_pos[1]-(self.game.board.shape[1]-1))-self.game.last_move_pos[0]))
        for v in vectors:
            v_rev = self.calculate_reward_vector(v)
            total_in_row = np.add(total_in_row, v_rev)

        return total_in_row

    # sums up number of connected tiles players own in a vector
    def calculate_reward_vector(self, board_vector):
        total_in_row = [[0, 0, 0, 0], [0, 0, 0, 0]]
        in_row = [0, 0]
        for i in range(0, len(board_vector)):
            if board_vector[i] != 0:
                if board_vector[i] == in_row[0]:
                    in_row[1] += 1
                else:
                    if in_row[0] != 0:
                        total_in_row[in_row[0]-1][in_row[1]-1] += 1
                    in_row = [board_vector[i], 1]
            elif in_row[0] != 0:
                total_in_row[in_row[0]-1][in_row[1]-1] += 1
                in_row = [0, 0]
        if in_row[0] != 0:
            total_in_row[in_row[0]-1][in_row[1]-1] += 1

        return total_in_row


class HumanPlayer(AIBasePlayer):

    def __init__(self, game):
        super().__init__(game)

    def take_turn(self):
        super(HumanPlayer, self).take_turn()
        move = input()
        return int(move)

    def post_process(self):
        self.calculate_reward()


class EvenDistributionRandomPlayer(BasePlayer):
    def __init__(self, num_actions):
        self.numActions = num_actions

    def take_turn(self):
        return np.random.random_integers(0, self.numActions-1)
