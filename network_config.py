vgg11_config = {

    'conv1': {'config': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1},
              'inbound_nodes': ['input']},
    'bn1': {'config': {'input_size': 64}, 'inbound_nodes': ['conv1']},
    'relu1': {'config': '', 'inbound_nodes': ['bn1']},

    'max_pooling1': {'config': {'kernel_size': 2, 'stride': 2}, 'inbound_nodes': ['relu1']},

    'conv2': {'config': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 1},
              'inbound_nodes': ['max_pooling1']},
    'bn2': {'config': {'input_size': 128}, 'inbound_nodes': ['conv2']},
    'relu2': {'config': '', 'inbound_nodes': ['bn2']},

    'max_pooling2': {'config': {'kernel_size': 2, 'stride': 2}, 'inbound_nodes': ['relu2']},

    'conv3': {'config': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 1},
              'inbound_nodes': ['max_pooling2']},
    'bn3': {'config': {'input_size': 256}, 'inbound_nodes': ['conv3']},
    'relu3': {'config': '', 'inbound_nodes': ['bn3']},
    'conv4': {'config': {'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 1},
              'inbound_nodes': ['relu3']},
    'bn4': {'config': {'input_size': 256}, 'inbound_nodes': ['conv4']},
    'relu4': {'config': '', 'inbound_nodes': ['bn4']},

    'max_pooling3': {'config': {'kernel_size': 2, 'stride': 2}, 'inbound_nodes': ['relu4']},

    'conv5': {'config': {'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 1},
              'inbound_nodes': ['max_pooling3']},
    'bn5': {'config': {'input_size': 512}, 'inbound_nodes': ['conv5']},
    'relu5': {'config': '', 'inbound_nodes': ['bn5']},
    'conv6': {'config': {'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 1},
              'inbound_nodes': ['relu5']},
    'bn6': {'config': {'input_size': 512}, 'inbound_nodes': ['conv6']},
    'relu6': {'config': '', 'inbound_nodes': ['bn6']},

    'max_pooling4': {'config': {'kernel_size': 2, 'stride': 2}, 'inbound_nodes': ['relu6']},

    'conv7': {'config': {'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 1},
              'inbound_nodes': ['max_pooling4']},
    'bn7': {'config': {'input_size': 512}, 'inbound_nodes': ['conv7']},
    'relu7': {'config': '', 'inbound_nodes': ['bn7']},
    'conv8': {'config': {'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 1},
              'inbound_nodes': ['relu7']},
    'bn8': {'config': {'input_size': 512}, 'inbound_nodes': ['conv8']},
    'relu8': {'config': '', 'inbound_nodes': ['bn8']},

    'max_pooling5': {'config': {'kernel_size': 2, 'stride': 2}, 'inbound_nodes': ['relu8']},

    'fc1': {'config': {'input_size': 512, 'output_size': 10}, 'inbound_nodes': ['max_pooling5']},
}

se_init_config = {

    'conv1': {'config': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1},
              'inbound_nodes': ['input']},
    'bn1': {'config': {'input_size': 64}, 'inbound_nodes': ['conv1']},
    'relu1': {'config': '', 'inbound_nodes': ['bn1']},

    'max_pooling1': {'config': {'kernel_size': 2, 'stride': 2}, 'inbound_nodes': ['relu1']},

    'conv2': {'config': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 1},
              'inbound_nodes': ['max_pooling1']},
    'bn2': {'config': {'input_size': 128}, 'inbound_nodes': ['conv2']},
    'relu2': {'config': '', 'inbound_nodes': ['bn2']},

    'max_pooling2': {'config': {'kernel_size': 2, 'stride': 2}, 'inbound_nodes': ['relu2']},

    'conv3': {'config': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 1},
              'inbound_nodes': ['max_pooling2']},
    'bn3': {'config': {'input_size': 256}, 'inbound_nodes': ['conv3']},
    'relu3': {'config': '', 'inbound_nodes': ['bn3']},

    'fc1': {'config': {'input_size': 256*8*8, 'output_size': 10}, 'inbound_nodes': ['relu3']},
}
