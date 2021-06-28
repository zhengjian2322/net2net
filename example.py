from hill_climb import HillClimb
from network_config import se_init_config
from network_morphisms import NetworkMorphisms


def initial_network():
    model = NetworkMorphisms(se_init_config)
    model.initial_network(epochs=20, model_folder='initial/')


def hill_climb():
    evolution = HillClimb(number_of_organism=8, epochs=8, load_model_path='initial/model.pkl')
    evolution.start()
    evolution.eval()


if __name__ == '__main__':
    # initial_network()
    hill_climb()
