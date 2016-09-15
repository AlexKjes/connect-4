import c4_game
import reinforce_net as net
import numpy as np
import matplotlib.pyplot as plt




# net_dim = [board_tiles, 50, 25, 1]
# analysis_ai = net.ReinforceNet(net_dim)




def play_generation(n_generation):
    rn = np.load("/home/alex/data/connect-4_rev1/gen" + str(n_generation) + ".npy")[1][0][0]
    #print(rn[0])

    g = c4_game.Game()
    g.set_player1(c4_game.HumanPlayer())
    g.set_player2(net.Connect4RFNetAdapter(g, rn))
    g.output = True
    g.start_game()


def trainer():
    network_size = (42, 25, 6)
    #god = net.GeneticTrainer.load_generation("/home/alex/data/connect-4_rev1/gen48.npy")
    #print("Generation population..")
    god = net.GeneticTrainer(net.GeneticTrainer.generate_random_pop(network_size, 100))
    god.mutation_rate = .01
    while True:
        print("Evaluating population at generation " + str(god.generation_num) + "..")
        god.evaluate_population()
        print("Saving generation")
        god.save_generation()
        print("Generating next generation..")
        god.set_population(god.generate_next_generation())
        god.generation_num += 1



#print(np.load("/home/alex/data/connect-4_rev1/gen48.npy")[1][0][0].weights)
#play_generation(57)
#trainer()

#god = net.GeneticTrainer(population)
#god.evaluate_population()



g = c4_game.Game()
g.set_player1(c4_game.HumanPlayer(g))
g.set_player2(c4_game.HumanPlayer(g))
g.output = True
g.start_game()



"""
arr = np.zeros((7, 6))
arr[5][4] = 1

print(np.fliplr(arr))
print(np.fliplr(arr).diagonal(np.abs(4-5)-5))
"""