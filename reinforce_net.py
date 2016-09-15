import numpy as np
import c4_game


class ReinforceNet(object):
    def __init__(self, network_size):
        self.generation_number = 0
        self.weights = self._init_weights(network_size)
        self.n_layers = len(network_size)
        self.size = network_size

    def feed_forward(self, inputs):
        y = inputs
        for i in range(0, self.n_layers-1):
            si = np.dot(self.weights[i].T, y)
            pi = self._squash(si)
            y = self.random_generator(pi)
        return y

    @staticmethod
    def random_generator(pi):
        return np.random.binomial(6, pi)

    @staticmethod
    def _squash(si):
        return 1/(1+np.exp(-si))

    @staticmethod
    def _init_weights(network_size):
        weights = []
        for i in range(0, len(network_size)-1):
            weights.append(np.zeros((network_size[i], network_size[i+1])))

        return weights


class ReinforceNetRev2(object):
    def __init__(self, network_size):
        self.generation_number = 0
        self.weights = self._init_weights(network_size)
        self.n_layers = len(network_size)
        self.size = network_size

    def feed_forward(self, inputs):
        y = inputs
        for i in range(0, self.n_layers-2):
            si = np.dot(self.weights[i].T, y)
            pi = self._squash(si)
            y = self.random_generator(pi)
        y = self._squash(np.dot(self.weights[-1].T, y))
        return y

    @staticmethod
    def random_generator(pi):
        return np.random.binomial(1, pi)

    @staticmethod
    def _squash(si):
        return 1/(1+np.exp(-si))

    @staticmethod
    def _init_weights(network_size):
        weights = []
        for i in range(0, len(network_size)-1):
            weights.append(np.random.rand(network_size[i], network_size[i+1]))

        return weights


class GeneticTrainer:
    def __init__(self, start_population):
        self.generation_num = 0
        self.population = self._set_up_population(start_population)
        self.mutation_rate = 0.01
        self.n_tests_per_population = 1000
        self.mutation_boarder_values = (-100, 100)

    def generate_next_generation(self):
        self.population[:, 1].sort()
        next_gen = []
        for i in range(0, len(self.population)):
            parent1 = int(np.random.triangular(0, len(self.population), len(self.population)))
            parent2 = int(np.random.triangular(0, len(self.population), len(self.population)))
            next_gen.append(self._pair(self.population[parent1][0], self.population[parent2][0]))
        return next_gen

    def evaluate_population(self):
        for i in range(0, self.n_tests_per_population):
            np.random.shuffle(self.population)
            for j in range(0, self.population.shape[0]//2, 2):
                g = c4_game.Game()
                g.set_player1(Connect4RFNetAdapter(g, self.population[j][0]))
                g.set_player2(Connect4RFNetAdapter(g, self.population[j + 1][0]))
                g.start_game()

                if g.players_turn == 1:
                    self.population[j][1] += 1
                else:
                    self.population[j+1][1] += 1

    @staticmethod
    def _set_up_population(start_population):
        ret_set = []
        for i in range(0, len(start_population)):
            ret_set.append([start_population[i], 0])

        return np.array(ret_set)

    def _pair(self, parent1, parent2):
        child = ReinforceNet(parent1.size)
        for i in range(0, len(parent1.weights)):
            for j in range(0, len(parent1.weights[i])):
                for k in range(0, len(parent1.weights[i][j])):
                    if np.random.rand()*100 > self.mutation_rate:
                        if np.random.randint(0, 1) > 0:
                            child.weights[i][j][k] = parent1.weights[i][j][k]
                        else:
                            child.weights[i][j][k] = parent2.weights[i][j][k]
                    else:
                        child.weights[i][j][k] += np.random.randint(-np.abs(parent1.weights[i][j][k]), np.abs(parent1.weights[i][j][k]))
        return child

    def save_generation(self):
        np.save("/home/alex/data/connect-4_rev1/gen"+str(self.generation_num), [self.generation_num, self.population])

    def set_population(self, population):
        self.population = self._set_up_population(population)

    @staticmethod
    def load_generation(generation_file):
        data = np.load(generation_file)
        trainer = GeneticTrainer(data[1][:, 0])
        trainer.generation_num = data[0]

        return trainer

    @staticmethod
    def generate_random_pop(network_size, population_size):
        population = []
        for i in range(0, population_size):
            population.append(ReinforceNetRev2(network_size))

        return population


class Connect4RFNetAdapter(c4_game.AIBasePlayer):
    def __init__(self, game, net):
        super().__init__(game)
        self.net = net

    def take_turn(self):
        super().take_turn()
        if self.same_state_in_row > 3:
            self.game.game_over = True

        if np.array_equal(self.state_before_move, self.game.board):
            self.same_state_in_row += 1
        else:
            self.same_state_in_row = 0


        return np.argmax(self.net.feed_forward(self.game.board.flatten()))

    def post_process(self):
        totalReward = 0











