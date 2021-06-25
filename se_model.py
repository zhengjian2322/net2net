import glob
import os
import shutil
from copy import deepcopy
import random
import datetime

import torch


import numpy as np
from torch import optim
from tqdm import tqdm

from data_loader import data_loader
from network_config import se_init_config
from utils import GenerateNet


class SEModel():
    def __init__(self, model_config=None, ):
        self.teacher_config = model_config
        self.student_config = model_config
        self.teacher_weights = None
        self.teacher = None
        self.student = None

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.train_loader, self.test_loader = data_loader(train_batch_size=128, test_batch_size=100)

    def load_teacher(self, model_path):
        assert os.path.isfile(model_path), 'The model path does not exist'
        check_point = torch.load(model_path)
        self.teacher = GenerateNet(check_point['model_config'])
        self.teacher_config = check_point['model_config']
        self.teacher.load_state_dict(check_point['model_state_dict'])

    def initial_network(self, epochs=20, lr=0.05, model_folder=''):
        assert self.teacher_config, 'model_config is None'
        self.teacher = GenerateNet(self.teacher_config)
        self.teacher = self.teacher.to(self.device)

        optimizer = optim.SGD(params=self.teacher.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        loss_func = torch.nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

        best_acc = 0
        for epoch in range(epochs):
            self.teacher.train()
            train_loss, correct, total = 0, 0, 0
            with tqdm(total=len(self.train_loader), desc='train epoch %d' % epoch, colour='black') as t:
                for step, (train_x, train_y) in enumerate(self.train_loader):
                    train_x, train_y = train_x.to(self.device), train_y.to(self.device)
                    optimizer.zero_grad()
                    output = self.teacher(train_x)
                    loss = loss_func(output, train_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    total += train_y.size(0)
                    _, predict = output.max(1)
                    correct += predict.eq(train_y).sum().item()
                    t.set_postfix({
                        'step': step,
                        'length of test': len(self.train_loader),
                        'Loss': '%.3f' % (train_loss / (step + 1)),
                        'Acc': '%.3f%% (%d/%d)' % (100. * correct / total, correct, total)
                    })
                    t.update(1)
            # test
            self.teacher.eval()
            test_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                with tqdm(total=len(self.test_loader), desc='eval epoch %d' % epoch, colour='black') as t:
                    for step, (test_x, test_y) in enumerate(self.test_loader):
                        test_x, test_y = test_x.to(self.device), test_y.to(self.device)
                        output = self.teacher(test_x)
                        loss = loss_func(output, test_y)
                        test_loss += loss
                        _, predict = output.max(1)
                        total += test_y.size(0)
                        correct += predict.eq(test_y).sum().item()
                        t.set_postfix({
                            'step': step,
                            'length of test': len(self.test_loader),
                            'Loss': '%.3f' % (test_loss / (step + 1)),
                            'Acc': '%.3f%% (%d/%d)' % (100. * correct / total, correct, total)
                        })
                        t.update(1)
            acc = correct / total
            if acc > best_acc:
                self.save_model(best_acc, self.teacher.state_dict(), self.teacher_config, model_folder)
                best_acc = acc
            scheduler.step()

    def train(self, epochs=17, lr=0.05, folder='./'):
        optimizer = optim.SGD(params=self.teacher.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        loss_func = torch.nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
        run_history = []
        self.teacher = self.teacher.to(self.device)
        for epoch in range(epochs):
            self.teacher.train()
            train_loss, correct, total = 0, 0, 0
            with tqdm(total=len(self.train_loader), desc='train epoch %d' % epoch, colour='black') as t:
                for step, (train_x, train_y) in enumerate(self.train_loader):
                    train_x, train_y = train_x.to(self.device), train_y.to(self.device)
                    optimizer.zero_grad()
                    output = self.teacher(train_x)
                    loss = loss_func(output, train_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    total += train_y.size(0)
                    _, predict = output.max(1)
                    correct += predict.eq(train_y).sum().item()
                    t.set_postfix({
                        'step': step,
                        'length of test': len(self.train_loader),
                        'Loss': '%.3f' % (train_loss / (step + 1)),
                        'Acc': '%.3f%% (%d/%d)' % (100. * correct / total, correct, total)
                    })
                    t.update(1)

            # test
            self.teacher.eval()
            test_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                with tqdm(total=len(self.test_loader), desc='eval epoch %d' % epoch, colour='black') as t:
                    for step, (test_x, test_y) in enumerate(self.test_loader):
                        test_x, test_y = test_x.to(self.device), test_y.to(self.device)
                        output = self.teacher(test_x)
                        loss = loss_func(output, test_y)
                        test_loss += loss
                        _, predict = output.max(1)
                        total += test_y.size(0)
                        correct += predict.eq(test_y).sum().item()
                        progress_bar(step, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                     % (test_loss / (step + 1), 100. * correct / total, correct, total))
                        t.set_postfix({
                            'step': step,
                            'length of test': len(self.test_loader),
                            'Loss': '%.3f' % (test_loss / (step + 1)),
                            'Acc': '%.3f%% (%d/%d)' % (100. * correct / total, correct, total)
                        })
                        t.update(1)

            acc = correct / total
            run_history.append(acc)
            scheduler.step()
        self.save_model(np.mean(run_history[-3:]), self.teacher.state_dict(), self.teacher_config, folder)
        return run_history

    @staticmethod
    def save_model(acc, model_state_dict, model_config, folder):
        check_point = {
            'best_acc': acc,
            'model_state_dict': model_state_dict,
            'model_config': model_config
        }
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save(check_point, os.path.join(folder, 'model.pkl'))

    def change_teacher(self, student_weight):
        self.teacher = GenerateNet(self.student_config)
        self.teacher_config = deepcopy(self.student_config)
        self.teacher.load_state_dict(student_weight)

    def generate_node_name(self, name):
        same_node = 0
        for node_name in self.student_config:
            if name in node_name:
                same_node += 1
        return name + str(same_node + 1)

    def add(self, layer_index, change_teacher):

        self.student_config = deepcopy(self.teacher_config)
        student_weight = self.teacher.state_dict()
        nodes_list = self.get_nodes_list()
        node_name = nodes_list[layer_index][0]

        assert 'conv' in node_name, 'Wrong layer index'

        bn_index, bn_name, relu_index, relu_name = self.get_next_bn_and_relu(nodes_list, node_name)
        assert all([bn_index, bn_name, relu_index,
                    relu_name]), 'bn_index or  bn_name or relu_index or relu_name must not be None'

        lambda1 = self.generate_node_name('lambda')
        self.student_config[lambda1] = {
            'config': '', 'inbound_nodes': [relu_name]
        }

        conv1 = self.generate_node_name('conv')
        bn1 = self.generate_node_name('bn')
        relu1 = self.generate_node_name('relu')
        self.student_config[conv1] = deepcopy(self.student_config[node_name])
        self.student_config[bn1] = deepcopy(self.student_config[bn_name])
        self.student_config[bn1]['inbound_nodes'] = [conv1]
        self.student_config[relu1] = deepcopy(self.student_config[relu_name])
        self.student_config[relu1]['inbound_nodes'] = [bn1]

        lambda2 = self.generate_node_name('lambda')
        self.student_config[lambda2] = deepcopy(self.student_config[lambda1])
        self.student_config[lambda2]['inbound_nodes'] = [relu1]

        add1 = self.generate_node_name('add')
        self.student_config[add1] = {'config': '', 'inbound_nodes': [lambda1, lambda2]}

        layer_list = self.get_nodes_list_with_details()
        next_conv_index = self.return_next_layer(relu_index)

        for index in next_conv_index:
            for idx, element in enumerate(self.student_config[layer_list[index][0]]['inbound_nodes']):
                if element == relu_name:
                    self.student_config[layer_list[index][0]]['inbound_nodes'][idx] = add1

        self.student = GenerateNet(self.student_config)
        student_weight[conv1 + '.weight'] = student_weight[node_name + '.weight']
        student_weight[conv1 + '.bias'] = student_weight[node_name + '.bias']
        student_weight[bn1 + '.weight'] = student_weight[bn_name + '.weight']
        student_weight[bn1 + '.bias'] = student_weight[bn_name + '.bias']
        student_weight[bn1 + '.running_mean'] = student_weight[bn_name + '.running_mean']
        student_weight[bn1 + '.running_var'] = student_weight[bn_name + '.running_var']
        print(type(student_weight), '------------------------------------------------')
        self.student.load_state_dict(student_weight)
        if change_teacher:
            self.change_teacher(student_weight)

    def concat(self, layer_index, change_teacher=False):
        '''Create 'concatenation motif' as in https://arxiv.org/pdf/1806.02639.pdf'''

        self.student_config = deepcopy(self.teacher_config)
        student_weight = self.teacher.state_dict()
        nodes_list = self.get_nodes_list()
        node_name = nodes_list[layer_index][0]

        assert 'conv' in node_name, 'Wrong layer index'

        bn_index, bn_name, relu_index, relu_name = self.get_next_bn_and_relu(nodes_list, node_name)
        assert all([bn_index, bn_name, relu_index,
                    relu_name]), 'bn_index or  bn_name or relu_index or relu_name must not be None'

        next_conv_index = self.return_next_layer(relu_index)

        filters = self.student_config[node_name]['config']['out_channels']
        self.student_config[node_name]['config']['out_channels'] = int(filters / 2)

        conv1 = self.generate_node_name('conv')
        bn1 = self.generate_node_name('bn')
        relu1 = self.generate_node_name('relu')
        self.student_config[conv1] = deepcopy(self.student_config[node_name])

        self.student_config[bn_name]['config']['input_size'] = int(filters / 2)
        self.student_config[bn1] = deepcopy(self.student_config[bn_name])
        self.student_config[bn1]['inbound_nodes'] = [conv1]

        self.student_config[relu1] = deepcopy(self.student_config[relu_name])
        self.student_config[relu1]['inbound_nodes'] = [bn1]

        concat1 = self.generate_node_name('concat')
        self.student_config[concat1] = {'config': '', 'inbound_nodes': [relu1, relu_name]}

        layer_list = self.get_nodes_list_with_details()

        # TODO: Is here bug
        for index in next_conv_index:
            for idx, element in enumerate(self.student_config[layer_list[index][0]]['inbound_nodes']):
                if element == relu_name:
                    self.student_config[layer_list[index][0]]['inbound_nodes'][idx] = concat1

        self.student = GenerateNet(self.student_config)
        student_weight[conv1 + '.weight'] = student_weight[node_name + '.weight'][:int(filters / 2), :, :, :]
        student_weight[conv1 + '.bias'] = student_weight[node_name + '.bias'][:int(filters / 2)]

        student_weight[node_name + '.weight'] = student_weight[node_name + '.weight'][int(filters / 2):, :, :, :]
        student_weight[node_name + '.bias'] = student_weight[node_name + '.bias'][int(filters / 2):]

        student_weight[bn1 + '.weight'] = student_weight[bn_name + '.weight'][:int(filters / 2)]
        student_weight[bn1 + '.bias'] = student_weight[bn_name + '.bias'][:int(filters / 2)]
        student_weight[bn1 + '.running_mean'] = student_weight[bn_name + '.running_mean'][:int(filters / 2)]
        student_weight[bn1 + '.running_var'] = student_weight[bn_name + '.running_var'][:int(filters / 2)]

        student_weight[bn_name + '.weight'] = student_weight[bn_name + '.weight'][int(filters / 2):]
        student_weight[bn_name + '.bias'] = student_weight[bn_name + '.bias'][int(filters / 2):]
        student_weight[bn_name + '.running_mean'] = student_weight[bn_name + '.running_mean'][int(filters / 2):]
        student_weight[bn_name + '.running_var'] = student_weight[bn_name + '.running_var'][int(filters / 2):]
        self.student.load_state_dict(student_weight)

        if change_teacher:
            self.change_teacher(student_weight)

    def wider2net_conv2d(self, layer_index, change_teacher=False, new_width=None):
        '''Function that add filters to convolutional filter. If new_width is not provided it double numbers of filters'''
        self.student_config = deepcopy(self.teacher_config)
        student_weight = self.teacher.state_dict()
        nodes_list = self.get_nodes_list()
        node_name = nodes_list[layer_index][0]

        assert 'conv' in node_name, 'Layer is not convolutional'

        bn_index, bn_name, relu_index, relu_name = self.get_next_bn_and_relu(nodes_list, node_name)
        assert all([bn_index, bn_name, relu_index,
                    relu_name]), 'bn_index or  bn_name or relu_index or relu_name must not be None'

        next_node_index = self.return_next_layer(relu_index)
        assert len(next_node_index) == 1, 'Wrong place for add'
        next_node_index = next_node_index[0]
        next_node_name = nodes_list[next_node_index][0]

        assert 'lambda' not in next_node_name, 'Wider inside add or concatenate block'

        if 'max' in next_node_name:
            for idx, node in enumerate(nodes_list):
                if node[1] == next_node_name:
                    next_node_index, next_node_name = idx, node[0]
                    break
        assert 'fc' not in next_node_name, 'Last convolutional layer'

        teacher_w1, teacher_b1 = student_weight[node_name + '.weight'], student_weight[node_name + '.bias']
        alpha, beta, mean, std = student_weight[bn_name + '.weight'], student_weight[bn_name + '.bias'], student_weight[
            bn_name + '.running_mean'], student_weight[bn_name + '.running_var']
        teacher_w2, teacher_b2 = student_weight[next_node_name + '.weight'], student_weight[next_node_name + '.bias']

        original_filters = teacher_w1.shape[0]
        if new_width is None:
            new_width = self.student_config[node_name]['config']['out_channels'] * 2
        n = new_width - original_filters
        assert n > 0, "New width smaller than teacher width"
        index = np.random.randint(original_filters, size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[index, :, :, :]
        new_b1 = teacher_b1[index]
        new_w2 = (teacher_w2[:, index, :, :] / torch.from_numpy(factors.reshape((1, -1, 1, 1))).to(teacher_w2.device))
        # teacher_w2.to(self.device)
        # print(teacher_w2.device)
        new_alpha = alpha[index]
        new_beta = beta[index]
        new_mean = mean[index]
        new_std = std[index]
        # new_w1 = new_w1 + np.random.normal(scale=new_w1.std() * 0.05, size=new_w1.shape)
        student_w1 = torch.cat((teacher_w1, new_w1), 0)
        student_b1 = torch.cat((teacher_b1, new_b1), 0)

        alpha = torch.cat((alpha, new_alpha))
        beta = torch.cat((beta, new_beta))
        mean = torch.cat((mean, new_mean))
        std = torch.cat((std, new_std))
        # new_w2 = new_w2 + np.random.normal(scale=new_w2.std() * 0.05, size=new_w2.shape)

        student_w2 = torch.cat((teacher_w2, new_w2), 1)
        student_w2[:, index, :, :] = new_w2

        self.student_config[node_name]['config']['out_channels'] = new_width
        self.student_config[bn_name]['config']['input_size'] = new_width
        self.student_config[next_node_name]['config']['in_channels'] = new_width
        student_weight[node_name + '.weight'], student_weight[node_name + '.bias'] = student_w1, student_b1
        student_weight[next_node_name + '.weight'], student_weight[next_node_name + '.bias'] = student_w2, teacher_b2
        student_weight[bn_name + '.weight'], student_weight[bn_name + '.bias'], student_weight[
            bn_name + '.running_mean'], student_weight[bn_name + '.running_var'] = alpha, beta, mean, std

        self.student = GenerateNet(self.student_config)
        self.student.load_state_dict(student_weight)
        if change_teacher:
            self.change_teacher(student_weight)

    def wider2net_conv2d_fc(self, layer_index, change_teacher=False, new_width=None):

        '''Add filters to the convolutional layer that is placed before fully connected layer'''
        self.student_config = deepcopy(self.teacher_config)
        student_weight = self.teacher.state_dict()
        nodes_list = self.get_nodes_list()
        node_name = nodes_list[layer_index][0]

        assert 'conv' in node_name, 'Wrong layer index'
        bn_index, bn_name, relu_index, relu_name = self.get_next_bn_and_relu(nodes_list, node_name)
        assert all([bn_index, bn_name, relu_index,
                    relu_name]), 'bn_index or  bn_name or relu_index or relu_name must not be None'

        next_node_index = self.return_next_layer(relu_index)
        assert len(next_node_index) == 1, 'Wrong place for add'
        next_node_index = next_node_index[0]
        next_node_name = nodes_list[next_node_index][0]

        if 'max' in next_node_name:
            next_node_index = self.return_next_layer(next_node_index)[0]
            next_node_name = nodes_list[next_node_index][0]
        assert 'fc' in next_node_name, 'there is not a fully connected layer'

        teacher_w1, teacher_b1 = student_weight[node_name + '.weight'], student_weight[node_name + '.bias']
        alpha, beta, mean, std = student_weight[bn_name + '.weight'], student_weight[bn_name + '.bias'], student_weight[
            bn_name + '.running_mean'], student_weight[bn_name + '.running_var']
        teacher_w2, teacher_b2 = student_weight[next_node_name + '.weight'], student_weight[next_node_name + '.bias']

        original_filters = teacher_w1.shape[0]
        if new_width is None:
            new_width = self.student_config[node_name]['config']['out_channels'] * 2
        n = new_width - original_filters
        assert n > 0, "New width smaller than teacher width"
        index = np.random.randint(original_filters, size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[index, :, :, :]
        new_b1 = teacher_b1[index]

        new_w2 = teacher_w2.T
        new_w2 = new_w2[index, :] / factors.reshape((-1, 1))

        new_alpha = alpha[index]
        new_beta = beta[index]
        new_mean = mean[index]
        new_std = std[index]

        alpha = torch.cat((alpha, new_alpha))
        beta = torch.cat((beta, new_beta))
        mean = torch.cat((mean, new_mean))
        std = torch.cat((std, new_std))

        # new_w1 = new_w1 + np.random.normal(scale=new_w1.std() * 0.05, size=new_w1.shape)
        student_w1 = torch.cat((teacher_w1, new_w1))
        student_b1 = torch.cat((teacher_b1, new_b1))
        # new_w2 = new_w2 + np.random.normal(scale=new_w2.std() * 0.05, size=new_w2.shape)
        student_w2 = torch.cat((teacher_w2.T, new_w2))
        student_w2[index, :] = new_w2
        student_w2 = student_w2.T

        self.student_config = deepcopy(self.student_config)

        self.student_config[node_name]['config']['out_channels'] = new_width
        self.student_config[bn_name]['config']['input_size'] = new_width
        self.student_config[next_node_name]['config']['input_size'] = new_width
        student_weight[node_name + '.weight'], student_weight[node_name + '.bias'] = student_w1, student_b1
        student_weight[next_node_name + '.weight'], student_weight[next_node_name + '.bias'] = student_w2, teacher_b2
        student_weight[bn_name + '.weight'], student_weight[bn_name + '.bias'], student_weight[
            bn_name + '.running_mean'], student_weight[bn_name + '.running_var'] = alpha, beta, mean, std

        self.student = GenerateNet(self.student_config)
        self.student.load_state_dict(student_weight)

        if change_teacher:
            self.change_teacher(student_weight)

    def deeper2net_conv2d(self, layer_index, change_teacher=False):

        '''Add convolutional layer after convolutional layer'''
        self.student_config = deepcopy(self.teacher_config)
        student_weight = self.teacher.state_dict()
        nodes_list = self.get_nodes_list()
        node_name = nodes_list[layer_index][0]

        assert 'conv' in node_name, 'Layer is not convolutional'

        bn_index, bn_name, relu_index, relu_name = self.get_next_bn_and_relu(nodes_list, node_name)
        assert all([bn_index, bn_name, relu_index,
                    relu_name]), 'bn_index or  bn_name or relu_index or relu_name must not be None'

        conv_or_maxpool_index = self.return_next_layer(relu_index)

        filters = self.student_config[node_name]['config']['out_channels']
        kh = kw = self.student_config[node_name]['config']['kernel_size']

        conv1 = self.generate_node_name('conv')
        bn1 = self.generate_node_name('bn')
        relu1 = self.generate_node_name('relu')
        self.student_config[conv1] = {
            'config': {'in_channels': filters, 'out_channels': filters, 'kernel_size': 3, 'padding': 1, 'stride': 1},
            'inbound_nodes': [relu_name]}

        self.student_config[bn1] = deepcopy(self.student_config[bn_name])
        self.student_config[bn1]['inbound_nodes'] = [conv1]

        self.student_config[relu1] = deepcopy(self.student_config[relu_name])
        self.student_config[relu1]['inbound_nodes'] = [bn1]

        layer_list = self.get_nodes_list_with_details(teacher=True)

        for index in conv_or_maxpool_index:
            for idx, element in enumerate(self.student_config[layer_list[index][0]]['inbound_nodes']):
                if element == relu_name:
                    self.student_config[layer_list[index][0]]['inbound_nodes'][idx] = relu1

        student_w = torch.zeros((filters, filters, kh, kw))
        for i in range(filters):
            student_w[i, i, (kh - 1) // 2, (kw - 1) // 2] = 1.
        # student_w = student_w + np.random.normal(scale=student_w.std() * 0.01, size=student_w.shape)
        student_weight[conv1 + '.weight'] = student_w
        student_weight[conv1 + '.bias'] = torch.zeros(student_weight[node_name + '.bias'].shape)

        student_weight[bn1 + '.weight'] = student_weight[bn_name + '.weight']
        student_weight[bn1 + '.bias'] = student_weight[bn_name + '.bias']
        student_weight[bn1 + '.running_mean'] = student_weight[bn_name + '.running_mean']
        student_weight[bn1 + '.running_var'] = student_weight[bn_name + '.running_var']
        self.student = GenerateNet(self.student_config)

        self.student.load_state_dict(student_weight)
        if change_teacher:
            self.change_teacher(student_weight)

    def skip(self, layer_index, change_teacher=False):
        '''Add skip conection. This is combination of 'add' and 'deeper2net_conv2d' functions'''

        nodes_before_deeper = self.get_nodes_list(teacher=True)
        nodes_before_deeper = [item[0] for item in nodes_before_deeper]
        self.deeper2net_conv2d(layer_index, change_teacher=True)
        nodes_after_deeper = self.get_nodes_list(teacher=True)
        nodes_after_deeper = [item[0] for item in nodes_after_deeper]
        difference = list(set(nodes_after_deeper) - set(nodes_before_deeper))
        new_relu_name = [x for x in difference if 'relu' in x][0]
        new_conv_name = [x for x in difference if 'conv' in x][0]

        self.student_config = deepcopy(self.teacher_config)
        student_weight = self.teacher.state_dict()

        lambda1 = self.generate_node_name('lambda')
        self.student_config[lambda1] = {'config': '', 'inbound_nodes': [new_relu_name]}

        lambda2 = self.generate_node_name('lambda')
        self.student_config[lambda2] = {'config': '', 'inbound_nodes': [new_conv_name]}

        add1 = self.generate_node_name('add')
        self.student_config[add1] = {'config': '', 'inbound_nodes': [lambda1, lambda2]}
        nodes_list = self.get_nodes_list_with_details(teacher=True)

        new_relu_index = None
        for index, node in enumerate(nodes_list):
            if node[0] == new_relu_name:
                new_relu_index = index

        next_layer_index = self.return_next_layer(new_relu_index)

        for index in next_layer_index:
            for idx, element in enumerate(self.student_config[nodes_list[index][0]]['inbound_nodes']):
                if element == new_relu_name:
                    self.student_config[nodes_list[index][0]]['inbound_nodes'][idx] = add1

        self.student = GenerateNet(self.student_config)
        self.student.load_state_dict(student_weight)

        if change_teacher:
            self.change_teacher(student_weight)

    def get_nodes_list(self, teacher=False):
        nodes_list = []

        _nodes_config = self.teacher_config if teacher else self.student_config
        for node_name in _nodes_config:
            nodes_list.append([node_name] + [_nodes_config[node_name]['inbound_nodes'][0]])

        return nodes_list

    def get_nodes_list_with_details(self, teacher=False):
        nodes_list = []

        _nodes_config = self.teacher_config if teacher else self.student_config
        for node_name in _nodes_config:
            nodes_list.append([node_name] + _nodes_config[node_name]['inbound_nodes'])

        return nodes_list

    def return_next_layer(self, layer_index):
        layer_list = self.get_nodes_list_with_details(teacher=True)
        next_layers = []
        for i in range(1, len(layer_list)):
            if layer_list[layer_index][0] in layer_list[i][1:]:
                next_layers.append(i)
        return list(next_layers)

    @staticmethod
    def get_next_bn_and_relu(nodes_list, node_name):
        bn_index, bn_name, relu_index, relu_name = None, None, None, None
        for idx, node in enumerate(nodes_list):
            if node[1] == node_name and 'bn' in node[0]:
                bn_index, bn_name = idx, node[0]
        for idx, node in enumerate(nodes_list):
            if node[1] == bn_name and 'relu' in node[0]:
                relu_index, relu_name = idx, node[0]
        return bn_index, bn_name, relu_index, relu_name

    def return_available_nodes(self):
        wider2net_conv2d = []
        deeper2net_conv2d = []
        wider2net_conv2d_fc = []
        add = []
        concat = []
        skip = []
        # layer_list 返回的是二位列表,[layer_name, 连接的input层]
        layer_list = self.get_nodes_list(teacher=True)
        # 返回的是[layer_name,[]], 第二个元素是input的集合
        proper_layer_list = self.get_nodes_list_with_details(teacher=True)
        # 来判断哪些是可以横向扩展的层，但是判断的条件？
        for i, element in enumerate(layer_list):
            if 'conv' not in element[0]:
                continue
            second = self.return_next_layer(i)
            if len(second) > 1:
                continue

            third = self.return_next_layer(second[0])
            fourth = self.return_next_layer(third[0])

            if len(fourth) > 1:
                continue
            if len(proper_layer_list[fourth[0]][1:]) > 1:
                continue
            if 'fc' in layer_list[fourth[0]][0]:
                continue
            if 'lambda' in layer_list[fourth[0]][0]:
                continue
            if 'conv' or 'max' in layer_list[fourth[0]][0]:
                fifth = self.return_next_layer(fourth[0])
                if len(fifth) > 1:
                    continue
                if len(fifth) == 1 and 'fc' in layer_list[fifth[0]][0]:
                    continue
            wider2net_conv2d.append(i)

        for i, element in enumerate(layer_list):
            if 'conv' not in element[0]:
                continue
            second = self.return_next_layer(i)
            third = self.return_next_layer(second[0])
            fourth = self.return_next_layer(third[0])
            if 'max' in layer_list[fourth[0]][0]:
                fifth = self.return_next_layer(fourth[0])
                if len(fifth) == 1 and 'fc' in layer_list[fifth[0]][0]:
                    wider2net_conv2d_fc.append(i)
            if 'fc' in layer_list[fourth[0]]:
                wider2net_conv2d_fc.append(i)

        for i, element in enumerate(layer_list):
            if 'conv' in element[0]:
                deeper2net_conv2d.append(i)

        for i, element in enumerate(layer_list):
            if 'conv' not in element[0]:
                continue
            next_layer = self.return_next_layer(i)
            if len(next_layer) > 1:
                continue
            skip.append(i)

        for i, element in enumerate(layer_list):
            if 'conv' not in element[0]:
                continue
            next_layer = self.return_next_layer(i)
            if len(next_layer) > 1:
                continue
            add.append(i)
            concat.append(i)

        available = {'wider2net_conv2d': wider2net_conv2d, 'wider2net_conv2d_fc': wider2net_conv2d_fc,
                     'deeper2net_conv2d': deeper2net_conv2d, 'add': add, 'concat': concat, 'skip': skip}

        return available

    def number_of_student_parameter(self):
        return sum(p.numel() for p in self.teacher.parameters())

    def plot_model(self, folder):
        torch.onnx.export(self.teacher, torch.rand(12, 3, 32, 32), folder + 'model.onnx')


class Organism(object):
    def __init__(self, number, epoch=''):
        self.numer = number
        self.folder = epoch + 'model' + str(number) + '/'
        if os.path.isdir(self.folder[:-1]):
            shutil.rmtree(self.folder)
            os.mkdir(self.folder)
        else:
            os.mkdir(self.folder)
        self.model = SEModel()

    def random_modification(self):
        # Select random modification
        available_modifications = self.model.return_available_nodes()  # 选择哪些可以扩展的层
        # print(available_modifications)
        while True:
            random_modification = random.choice(list(available_modifications.keys()))
            if random_modification != 'add':
                continue
            if len(available_modifications[random_modification]) > 0:
                break
        random_index = random.choice(list(available_modifications[random_modification]))
        print(random_modification, random_index)
        function = getattr(self.model, random_modification)
        function(random_index, change_teacher=True)

        self.model.plot_model(self.folder)
        return random_modification

    def train(self, epochs=17, lr=0.05, folder='./'):
        return self.model.train(epochs, lr, folder)


class HillClimb(object):
    def __init__(self, number_of_organism, epochs):
        self.number_of_organism = number_of_organism
        self.epochs = epochs

    def start(self):
        model_dirs = glob.glob('model*/')
        for model_dir in model_dirs:
            shutil.rmtree(model_dir)
        if os.path.isdir('best'):
            shutil.rmtree('best')
            os.mkdir('best')
        else:
            os.mkdir('best')
        shutil.copyfile('init/model.pkl', 'best/model.pkl')

        previous_best = -1
        for epoch in range(self.epochs):
            print('\nEpoch %d' % epoch)
            list_of_organisms = []
            list_of_result = []
            for i in range(self.number_of_organism):
                list_of_organisms.append(Organism(i))
            for i in range(self.number_of_organism):

                while True:
                    print('\nModel loading %d' % i)
                    list_of_organisms[i].model.load_teacher(model_path='best/model.pkl')
                    # if i == 0:
                    #     list_of_organisms[i].model.plot_model(list_of_organisms[i].folder)
                    #     break
                    modifications = []
                    number_of_modifications = 3
                    '''Select random modifications'''
                    for _ in range(number_of_modifications):
                        modification = list_of_organisms[i].random_modification()
                        modifications.append(modification)
                    print('Organism %d: modifications: %s' % (i, modifications))
                    if list_of_organisms[i].model.number_of_student_parameter() < 50000000:
                        print('Number of parameters: %d' % list_of_organisms[i].model.number_of_student_parameter())
                        break
                    else:
                        print('Repeat drawing of network morphism function: %d' % list_of_organisms[
                            i].model.number_of_student_parameter())

                history = list_of_organisms[i].train(epochs=1, lr=0.05, folder=list_of_organisms[i].folder)
                # TODO: 用什么来评价好坏？
                organism_result = np.mean(history[-3:])
                list_of_result.append(organism_result)
                print('Organism %d result: %f' % (i, list_of_result[i]))
            best = list_of_result.index(max(list_of_result))
            print('\n=============================\nBest: %d, result: %f, previous: %f\n===========================' %
                  (best, list_of_result[best], previous_best))

            if max(list_of_result) > previous_best:
                shutil.copyfile(list_of_organisms[best].folder + 'model.pkl', 'best/model.pkl')

                if os.path.exists(list_of_organisms[best].folder + 'model.onnx'):
                    shutil.copyfile(list_of_organisms[best].folder + 'model.onnx', 'best/model.onnx')
                # print('Algorithm found new best organism')
                previous_best = max(list_of_result)
            else:
                shutil.copyfile(list_of_organisms[0].folder + 'model.pkl', 'best/model.pkl')

                if os.path.exists(list_of_organisms[0].folder + 'model.onnx'):
                    shutil.copyfile(list_of_organisms[0].folder + 'model.onnx', 'best/model.onnx')

            with open('best/results.txt', 'a') as myfile:
                myfile.write(str(datetime.datetime.now()))
                for i in range(self.number_of_organism):
                    myfile.write('Epoch: %d, organism %d accuracy: %f\n' % (epoch, i, list_of_result[i]))
                myfile.write('Epoch: %d, best accuracy: %f\n\n\n' % (epoch, list_of_result[best]))


# if __name__ == '__main__':
    # model = SEModel()
    # model.create_initial_network(epochs=1)
    # evolution = HillClimb(number_of_organism=8, epochs=8)
    # evolution.start()
    # model = SEModel()
    # model.load_teacher(model_folder='best')

    # model.teacher.summary()
    # train_history = model.train(epochs=120, lr=0.005, folder='./')
    # with open('final_result.txt', 'w') as f:
    #     f.write('%.4f' % max(train_history))
    # print(max(train_history))

if __name__ == '__main__':
    model = SEModel(se_init_config)
    model.initial_network(epochs=20,lr=0.05,model_folder='init')
