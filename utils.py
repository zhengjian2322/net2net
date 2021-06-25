from copy import deepcopy

import torch
import torch.nn as nn


class GenerateNet(nn.Module):
    """
    Generate network with network configuration
    """

    def __init__(self, net_config):
        super(GenerateNet, self).__init__()
        self.net_config = net_config
        self.node_list = []
        self.get_conv_from_dict = lambda x: nn.Conv2d(in_channels=x['in_channels'], out_channels=x['out_channels'],
                                                      kernel_size=x['kernel_size'],
                                                      padding=x['padding'], stride=x['stride'])
        self.get_bn_from_dict = lambda x: nn.BatchNorm2d(x['input_size'])
        self.get_linear_from_dict = lambda x: nn.Linear(x['input_size'], x['output_size'])
        self.get_maxpooling_from_dict = lambda x: nn.MaxPool2d(kernel_size=x['kernel_size'], stride=x['stride'])
        self._add_model_from_dict()

        for node_name in self.net_config:
            self.node_list.append([node_name] + self.net_config[node_name]['inbound_nodes'])

    def _add_model_from_dict(self):
        for node_name in self.net_config:
            node_config = self.net_config[node_name]['config']
            if 'conv' in node_name:
                self.add_module(node_name, self.get_conv_from_dict(node_config))
            elif 'bn' in node_name:
                self.add_module(node_name, self.get_bn_from_dict(node_config))
            elif 'relu' in node_name:
                self.add_module(node_name, nn.ReLU())
            elif 'fc' in node_name:
                self.add_module(node_name, self.get_linear_from_dict(node_config))
            elif 'max' in node_name:
                self.add_module(node_name, self.get_maxpooling_from_dict(node_config))

    def forward(self, x):
        layers = dict(self.named_children())
        _node_list = deepcopy(self.node_list)
        final_node = None
        layer_out = {'input': x}
        while len(_node_list) > 0:
            _node_list_len = len(_node_list)
            for node in _node_list:
                node_name = node[0]
                inbound_nodes = node[1:]
                if set(inbound_nodes) <= set(layer_out.keys()):
                    if 'add' in node_name:
                        assert len(inbound_nodes) == 2 or len(inbound_nodes == 0), ValueError('Inbound_nodes error')
                        layer_out[node_name] = layer_out[inbound_nodes[0]] + layer_out[inbound_nodes[1]]
                    elif 'concat' in node_name:
                        assert len(inbound_nodes) == 2 or len(inbound_nodes == 0), ValueError('Inbound_nodes error')
                        layer_out[node_name] = torch.cat(
                            (layer_out[inbound_nodes[0]][:, :, :, :], layer_out[inbound_nodes[1]][:, :, :, :]), 1)
                    elif 'fc' in node_name:
                        out = layer_out[inbound_nodes[0]]
                        out = out.view(out.size()[0], -1)
                        layer_out[node_name] = layers[node_name](out)
                    elif 'lambda' in node_name:
                        out = layer_out[inbound_nodes[0]]
                        layer_out[node_name] = 0.5 * out
                    else:
                        layer_out[node_name] = layers[node_name](layer_out[inbound_nodes[0]])
                    final_node = node_name
                    _node_list.remove(node)
            assert len(_node_list) < _node_list_len, 'Net configuration error!'

        return layer_out[final_node]
