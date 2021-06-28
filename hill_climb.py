import glob
import shutil
import random
import datetime
import os
import numpy as np

from network_morphisms import NetworkMorphisms


class Organism(object):
    def __init__(self, number, epoch=''):
        self.number = number
        self.folder = epoch + 'model' + str(self.number) + '/'
        if os.path.isdir(self.folder[:-1]):
            shutil.rmtree(self.folder)
            os.mkdir(self.folder)
        else:
            os.mkdir(self.folder)
        self.model = NetworkMorphisms()

    def random_modification(self):
        # Select random modification
        available_modifications = self.model.return_available_nodes()
        while True:
            random_modification = random.choice(list(available_modifications.keys()))
            if len(available_modifications[random_modification]) > 0:
                break
        random_index = random.choice(list(available_modifications[random_modification]))
        print(random_modification, random_index)
        function = getattr(self.model, random_modification)
        function(random_index)
        self.model.plot_model(self.folder)
        return random_modification

    def train(self, epochs=17, lr=0.05, save_folder='./'):
        return self.model.train(epochs, lr, save_folder=save_folder)


class HillClimb(object):
    def __init__(self, number_of_organism, epochs, load_model_path):
        self.number_of_organism = number_of_organism
        self.epochs = epochs
        self.load_model_path = load_model_path

    def start(self):
        model_dirs = glob.glob('model*/')
        for model_dir in model_dirs:
            shutil.rmtree(model_dir)
        if os.path.isdir('best'):
            shutil.rmtree('best')
            os.mkdir('best')
        else:
            os.mkdir('best')
        shutil.copyfile(self.load_model_path, 'best/model.pkl')

        previous_best = -1
        for epoch in range(self.epochs):
            print('Epoch %d' % epoch)
            list_of_organisms = []
            list_of_result = []
            for i in range(self.number_of_organism):
                list_of_organisms.append(Organism(i))
            for i in range(self.number_of_organism):
                while True:
                    print('Model loading %d' % i)
                    list_of_organisms[i].model.load_teacher(model_path='best/model.pkl')
                    if i == 0:
                        list_of_organisms[i].model.plot_model(list_of_organisms[i].folder)
                        break
                    modifications = []
                    number_of_modifications = 3
                    # Select random modifications
                    for _ in range(number_of_modifications):
                        modification = list_of_organisms[i].random_modification()
                        modifications.append(modification)
                    print('Organism %d: modifications: %s' % (i, modifications))
                    if list_of_organisms[i].model.number_of_parameter() < 50000000:
                        print('Number of parameters: %d' % list_of_organisms[i].model.number_of_parameter())
                        break
                    else:
                        print('Repeat drawing of network morphism function: %d' % list_of_organisms[
                            i].model.number_of_parameter())

                history = list_of_organisms[i].train(epochs=17, lr=0.05, save_folder=list_of_organisms[i].folder)
                # TODO: With what to evaluate
                organism_result = np.mean(history[-3:])
                list_of_result.append(organism_result)
                print('Organism %d result: %f' % (i, list_of_result[i]))
            best = list_of_result.index(max(list_of_result))
            print('\nBest: %d, result: %f, previous: %f\n' % (best, list_of_result[best], previous_best))
            if max(list_of_result) > previous_best:
                shutil.copyfile(list_of_organisms[best].folder + 'model.pkl', 'best/model.pkl')
                if os.path.exists(list_of_organisms[best].folder + 'model.onnx'):
                    shutil.copyfile(list_of_organisms[best].folder + 'model.onnx', 'best/model.onnx')
                previous_best = max(list_of_result)
            else:
                shutil.copyfile(list_of_organisms[0].folder + 'model.pkl', 'best/model.pkl')
                if os.path.exists(list_of_organisms[0].folder + 'model.onnx'):
                    shutil.copyfile(list_of_organisms[0].folder + 'model.onnx', 'best/model.onnx')

            with open('best/results.txt', 'a') as result_file:
                result_file.write(str(datetime.datetime.now()))
                for i in range(self.number_of_organism):
                    result_file.write('Epoch: %d, organism %d accuracy: %f\n' % (epoch, i, list_of_result[i]))
                result_file.write('Epoch: %d, best accuracy: %f\n\n\n' % (epoch, list_of_result[best]))

    @staticmethod
    def eval(epochs=200, lr=0.005):
        model = NetworkMorphisms()
        model.load_teacher(model_path='best/model.pkl')
        train_history = model.train(epochs=epochs, lr=lr, save_folder='test')
        print(max(train_history))
        with open('best/results.txt', 'a') as result_file:
            result_file.write('final_model acc(epoch:%d): %.4f' % (epochs, max(train_history)))
