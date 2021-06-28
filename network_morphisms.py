import os
from copy import deepcopy
import torch

import numpy as np
from torch import optim
from tqdm import tqdm

from data_loader import data_loader
from network_config import se_init_config
from utils import GenerateNet


class NetworkMorphisms(object):
    def __init__(self, in_channels=3, picture_size=(32, 32)):
        self.in_channels = in_channels
        self.picture_size = picture_size
        self.teacher_config = None
        self.student_config = None
        self.teacher = None
        self.student = None

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.train_loader, self.test_loader = data_loader(train_batch_size=128, test_batch_size=100)

    def load_teacher(self, model_path):
        """
        load teacher network from check point file
        """
        assert os.path.isfile(model_path), 'The model path does not exist'
        check_point = torch.load(model_path)
        self.teacher = GenerateNet(check_point['model_config'])
        self.teacher_config = check_point['model_config']
        self.teacher.load_state_dict(check_point['model_state_dict'])

    def initial_network(self, epochs=20, lr=0.05, model_folder='', model_config=None):
        """
        Initialize the network as the basic network
        """
        if model_config is None:
            model_config = deepcopy(se_init_config)
        self.teacher_config = model_config
        self.teacher = GenerateNet(model_config)
        self.teacher = self.teacher.to(self.device)

        optimizer = optim.SGD(params=self.teacher.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        loss_func = torch.nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

        best_acc = 0
        for epoch in range(epochs):
            self._train(epoch, optimizer, loss_func)
            correct, total = self._eval(epoch, loss_func)
            acc = correct / total
            if acc > best_acc:
                self.save_model(best_acc, self.teacher.state_dict(), self.teacher_config, model_folder)
                best_acc = acc
            scheduler.step()

    def train(self, epochs=17, lr=0.05, save_folder='./'):
        optimizer = optim.SGD(params=self.teacher.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        loss_func = torch.nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
        run_history = []
        self.teacher = self.teacher.to(self.device)
        for epoch in range(epochs):
            self._train(epoch, optimizer, loss_func)
            correct, total = self._eval(epoch, loss_func)
            acc = correct / total
            run_history.append(acc)
            scheduler.step()
        self.save_model(np.mean(run_history[-3:]), self.teacher.state_dict(), self.teacher_config, save_folder)
        return run_history

    def change_teacher(self, student_weight):
        self.teacher = GenerateNet(self.student_config)
        self.teacher_config = deepcopy(self.student_config)
        self.teacher.load_state_dict(student_weight)

    def generate_node_name(self, name):
        """
        Generate a new node name
        """
        same_node = 0
        for node_name in self.student_config:
            if name in node_name:
                same_node += 1
        return name + str(same_node + 1)

    def add(self, node_index: int):
        """
        Create add modification
        """
        self.student_config = deepcopy(self.teacher_config)
        student_weight = self.teacher.state_dict()
        nodes_list = self.get_nodes_list()
        node_name, bn_index, bn_name, relu_index, relu_name = self.get_conv_bn_relu(nodes_list, node_index)

        lambda1 = self.generate_node_name('lambda')
        self.student_config[lambda1] = {'config': '', 'inbound_nodes': [relu_name]}

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

        next_nodes_index = self.get_next_nodes(relu_index)
        self.replace_student_node_inbound(nodes_list, next_nodes_index, relu_name, add1)

        self.student = GenerateNet(self.student_config)
        node_weight = student_weight[node_name + '.weight']
        student_weight[conv1 + '.weight'] = node_weight + np.random.normal(scale=node_weight.std() * 0.01,
                                                                           size=node_weight.shape)
        student_weight[conv1 + '.bias'] = student_weight[node_name + '.bias']
        student_weight[bn1 + '.weight'] = student_weight[bn_name + '.weight']
        student_weight[bn1 + '.bias'] = student_weight[bn_name + '.bias']
        student_weight[bn1 + '.running_mean'] = student_weight[bn_name + '.running_mean']
        student_weight[bn1 + '.running_var'] = student_weight[bn_name + '.running_var']
        self.student.load_state_dict(student_weight)

        self.change_teacher(student_weight)

    def concat(self, node_index: int):
        """
        Create 'concatenation motif' as in https://arxiv.org/pdf/1806.02639.pdf
        """
        self.student_config = deepcopy(self.teacher_config)
        student_weight = self.teacher.state_dict()
        nodes_list = self.get_nodes_list()
        node_name, bn_index, bn_name, relu_index, relu_name = self.get_conv_bn_relu(nodes_list, node_index)

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

        next_conv_index = self.get_next_nodes(relu_index)
        self.replace_student_node_inbound(nodes_list, next_conv_index, relu_name, concat1)

        self.student = GenerateNet(self.student_config)
        node_weight = student_weight[node_name + '.weight'][:int(filters / 2), :, :, :]
        student_weight[conv1 + '.weight'] = node_weight + np.random.normal(scale=node_weight.std() * 0.01,
                                                                           size=node_weight.shape)
        # student_weight[conv1 + '.weight'] = student_weight[node_name + ".weight"][:int(filters / 2), :, :, :]
        student_weight[conv1 + '.bias'] = student_weight[node_name + ".bias"][:int(filters / 2)]

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

        self.change_teacher(student_weight)

    def wider2net_conv2d(self, node_index: int, new_width=None):
        """
        Function that add filters to convolutional filter. If new_width is not provided it double numbers of filters
        """
        self.student_config = deepcopy(self.teacher_config)
        student_weight = self.teacher.state_dict()
        nodes_list = self.get_nodes_list()
        node_name, bn_index, bn_name, relu_index, relu_name = self.get_conv_bn_relu(nodes_list, node_index)

        next_node_index = self.get_next_nodes(relu_index)
        assert len(next_node_index) == 1, 'Wrong place for widder'
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

        new_alpha = alpha[index]
        new_beta = beta[index]
        new_mean = mean[index]
        new_std = std[index]

        new_w1 = new_w1 + np.random.normal(scale=new_w1.std() * 0.05, size=new_w1.shape)
        student_w1 = torch.cat((teacher_w1, new_w1), 0)
        student_b1 = torch.cat((teacher_b1, new_b1), 0)

        alpha = torch.cat((alpha, new_alpha))
        beta = torch.cat((beta, new_beta))
        mean = torch.cat((mean, new_mean))
        std = torch.cat((std, new_std))
        new_w2 = new_w2 + np.random.normal(scale=new_w2.std() * 0.05, size=new_w2.shape)

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

        self.change_teacher(student_weight)

    def wider2net_conv2d_fc(self, node_index: int, new_width=None):

        """
        Add filters to the convolutional layer that is placed before fully connected layer
        """

        self.student_config = deepcopy(self.teacher_config)
        student_weight = self.teacher.state_dict()
        nodes_list = self.get_nodes_list()
        node_name, bn_index, bn_name, relu_index, relu_name = self.get_conv_bn_relu(nodes_list, node_index)

        next_node_index = self.get_next_nodes(relu_index)
        assert len(next_node_index) == 1, 'Wrong place for widder'
        next_node_index = next_node_index[0]
        next_node_name = nodes_list[next_node_index][0]

        if 'max' in next_node_name:
            next_node_index = self.get_next_nodes(next_node_index)[0]
            next_node_name = nodes_list[next_node_index][0]
        assert 'fc' in next_node_name, 'there is not a fully connected layer'

        teacher_w1, teacher_b1 = student_weight[node_name + ".weight"], student_weight[node_name + '.bias']
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

        new_w1 = new_w1 + np.random.normal(scale=new_w1.std() * 0.05, size=new_w1.shape)
        student_w1 = torch.cat((teacher_w1, new_w1))
        student_b1 = torch.cat((teacher_b1, new_b1))
        new_w2 = new_w2 + np.random.normal(scale=new_w2.std() * 0.05, size=new_w2.shape)
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

        self.change_teacher(student_weight)

    def deeper2net_conv2d(self, node_index: int):

        """
        Add convolutional layer after convolutional layer
        """
        self.student_config = deepcopy(self.teacher_config)
        student_weight = self.teacher.state_dict()
        nodes_list = self.get_nodes_list()
        node_name, bn_index, bn_name, relu_index, relu_name = self.get_conv_bn_relu(nodes_list, node_index)

        conv1 = self.generate_node_name('conv')
        bn1 = self.generate_node_name('bn')
        relu1 = self.generate_node_name('relu')
        filters = self.student_config[node_name]['config']['out_channels']
        kh = kw = self.student_config[node_name]['config']['kernel_size']

        self.student_config[conv1] = {
            'config': {'in_channels': filters, 'out_channels': filters, 'kernel_size': 3, 'padding': 1, 'stride': 1},
            'inbound_nodes': [relu_name]}

        self.student_config[bn1] = deepcopy(self.student_config[bn_name])
        self.student_config[bn1]['inbound_nodes'] = [conv1]

        self.student_config[relu1] = deepcopy(self.student_config[relu_name])
        self.student_config[relu1]['inbound_nodes'] = [bn1]

        next_nodes_index = self.get_next_nodes(relu_index)
        self.replace_student_node_inbound(nodes_list, next_nodes_index, relu_name, relu1)

        student_w = torch.zeros((filters, filters, kh, kw))
        for i in range(filters):
            student_w[i, i, (kh - 1) // 2, (kw - 1) // 2] = 1.
        student_w = student_w + np.random.normal(scale=student_w.std() * 0.01, size=student_w.shape)
        student_weight[conv1 + '.weight'] = student_w
        student_weight[conv1 + '.bias'] = torch.zeros(student_weight[node_name + '.bias'].shape)
        student_weight[bn1 + '.weight'] = student_weight[bn_name + '.weight']
        student_weight[bn1 + '.bias'] = student_weight[bn_name + '.bias']
        student_weight[bn1 + '.running_mean'] = student_weight[bn_name + '.running_mean']
        student_weight[bn1 + '.running_var'] = student_weight[bn_name + '.running_var']
        self.student = GenerateNet(self.student_config)

        self.student.load_state_dict(student_weight)
        self.change_teacher(student_weight)

    def skip(self, node_index: int, change_teacher=False):
        """
        Add skip connection. This is combination of 'add' and 'deeper2net_conv2d' functions
        """

        nodes_before_deeper = self.get_nodes_list(teacher=True)
        nodes_before_deeper = [item[0] for item in nodes_before_deeper]
        self.deeper2net_conv2d(node_index)
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
        nodes_list = self.get_nodes_list()

        new_relu_index = None
        for index, node in enumerate(nodes_list):
            if node[0] == new_relu_name:
                new_relu_index = index

        next_node_index = self.get_next_nodes(new_relu_index)
        self.replace_student_node_inbound(nodes_list, next_node_index, new_relu_name, add1)

        self.student = GenerateNet(self.student_config)
        self.student.load_state_dict(student_weight)

        if change_teacher:
            self.change_teacher(student_weight)

    def _train(self, epoch, optimizer, loss_func):
        self.teacher.train()
        train_loss, correct, total = 0, 0, 0
        with tqdm(total=len(self.train_loader), desc='train epoch %d' % epoch, colour='black') as t_train:
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
                t_train.set_postfix({'step': step, 'length of train': len(self.train_loader),
                                     'Loss': '%.3f' % (train_loss / (step + 1)),
                                     'Acc': '%.3f%% (%d/%d)' % (100. * correct / total, correct, total)})
                t_train.update(1)

    def _eval(self, epoch, loss_func):
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
                    t.set_postfix({'step': step, 'length of eval': len(self.test_loader),
                                   'Loss': '%.3f' % (test_loss / (step + 1)),
                                   'Acc': '%.3f%% (%d/%d)' % (100. * correct / total, correct, total)})
                    t.update(1)
        return correct, total

    def replace_student_node_inbound(self, node_list, nodes_index, original_inbound_node_name, new_inbound_node_name):
        """
        Replace the old inbound node of the nodes with the new inbound node name
        """
        for index in nodes_index:
            for idx, element in enumerate(self.student_config[node_list[index][0]]['inbound_nodes']):
                if element == original_inbound_node_name:
                    self.student_config[node_list[index][0]]['inbound_nodes'][idx] = new_inbound_node_name

    def get_nodes_list(self, teacher=False):
        nodes_list = []
        _nodes_config = self.teacher_config if teacher else self.student_config
        for node_name in _nodes_config:
            nodes_list.append([node_name] + _nodes_config[node_name]['inbound_nodes'])

        return nodes_list

    def get_next_nodes(self, node_index, teacher=True):
        nodes_list = self.get_nodes_list(teacher=teacher)
        next_node = []
        for i in range(1, len(nodes_list)):
            if nodes_list[node_index][0] in nodes_list[i][1:]:
                next_node.append(i)
        return list(next_node)

    def return_available_nodes(self):
        """
        Before the network morphism, we will check the correspondence between points and operations
        """
        wider2net_conv2d = []
        deeper2net_conv2d = []
        wider2net_conv2d_fc = []
        add = []
        concat = []
        skip = []

        nodes_list = self.get_nodes_list(teacher=True)
        for i, element in enumerate(nodes_list):
            if 'conv' not in element[0]:
                continue
            second = self.get_next_nodes(i)
            if len(second) > 1:
                continue

            third = self.get_next_nodes(second[0])
            fourth = self.get_next_nodes(third[0])

            if len(fourth) > 1:
                continue
            if len(nodes_list[fourth[0]][1:]) > 1:
                continue
            if 'fc' in nodes_list[fourth[0]][0]:
                continue
            if 'lambda' in nodes_list[fourth[0]][0]:
                continue
            if 'conv' or 'max' in nodes_list[fourth[0]][0]:
                fifth = self.get_next_nodes(fourth[0])
                if len(fifth) > 1:
                    continue
                if len(fifth) == 1 and 'fc' in nodes_list[fifth[0]][0]:
                    continue
            wider2net_conv2d.append(i)

        for i, element in enumerate(nodes_list):
            if 'conv' not in element[0]:
                continue
            second = self.get_next_nodes(i)
            third = self.get_next_nodes(second[0])
            fourth = self.get_next_nodes(third[0])
            if 'max' in nodes_list[fourth[0]][0]:
                fifth = self.get_next_nodes(fourth[0])
                if len(fifth) == 1 and 'fc' in nodes_list[fifth[0]][0]:
                    wider2net_conv2d_fc.append(i)
            if 'fc' in nodes_list[fourth[0]]:
                wider2net_conv2d_fc.append(i)

        for i, element in enumerate(nodes_list):
            if 'conv' in element[0]:
                deeper2net_conv2d.append(i)

        for i, element in enumerate(nodes_list):
            if 'conv' not in element[0]:
                continue
            next_layer = self.get_next_nodes(i)
            if len(next_layer) > 1:
                continue
            skip.append(i)

        for i, element in enumerate(nodes_list):
            if 'conv' not in element[0]:
                continue
            next_layer = self.get_next_nodes(i)
            if len(next_layer) > 1:
                continue
            add.append(i)
            concat.append(i)

        available = {'wider2net_conv2d': wider2net_conv2d, 'wider2net_conv2d_fc': wider2net_conv2d_fc,
                     'deeper2net_conv2d': deeper2net_conv2d, 'add': add, 'concat': concat, 'skip': skip}

        return available

    @staticmethod
    def get_conv_bn_relu(nodes_list, node_index):
        node_name = nodes_list[node_index][0]

        assert 'conv' in node_name, 'Wrong layer index'
        bn_index, bn_name, relu_index, relu_name = None, None, None, None
        for idx, node in enumerate(nodes_list):
            if node[1] == node_name and 'bn' in node[0]:
                bn_index, bn_name = idx, node[0]
        for idx, node in enumerate(nodes_list):
            if node[1] == bn_name and 'relu' in node[0]:
                relu_index, relu_name = idx, node[0]
        assert all([bn_index, bn_name, relu_index,
                    relu_name]), 'bn_index or  bn_name or relu_index or relu_name must not be None'
        return node_name, bn_index, bn_name, relu_index, relu_name

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

    def number_of_parameter(self):
        return sum(p.numel() for p in self.teacher.parameters())

    def plot_model(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.onnx.export(self.teacher, torch.rand(1, self.in_channels, self.picture_size[0], self.picture_size[1]),
                          folder + 'model.onnx')

# if __name__ == '__main__':
#     model = SEModel(se_init_config)
#     model.initial_network(epochs=20,lr=0.05,model_folder='init')
