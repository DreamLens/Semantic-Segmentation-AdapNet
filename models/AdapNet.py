''' AdapNet:  Adaptive  Semantic  Segmentation
              in  Adverse  Environmental  Conditions

 Copyright (C) 2018  Abhinav Valada, Johan Vertens , Ankit Dhall and Wolfram Burgard

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

import tensorflow as tf
import network_base

class AdapNet(network_base.Network):
    def __init__(self, num_classes=12, learning_rate=0.001, float_type=tf.float32, weight_decay=0.0005,
                 decay_steps=30000, power=0.9, training=True, ignore_label=True, global_step=0,
                 has_aux_loss=False):
        super(AdapNet, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.initializer = 'xavier'
        self.has_aux_loss = has_aux_loss
        self.float_type = float_type
        self.power = power
        self.decay_steps = decay_steps
        self.training = training
        self.bn_decay_ = 0.99
        self.residual_units = [3, 4, 6, 3]
        self.filters = [256, 512, 1024, 2048]
        self.strides = [1, 2, 2, 1]
        self.global_step = global_step
        if self.training:
            self.keep_prob = 0.3
        else:
            self.keep_prob = 1.0
        if ignore_label:
            self.weights = tf.ones(self.num_classes-1)
            self.weights = tf.concat((tf.zeros(1), self.weights), 0)
        else:
            self.weights = tf.ones(self.num_classes)

    def _setup(self, data):
        self.input_shape = data.get_shape()
        self.conv_3x3_out = self.conv_batchN_relu(data, 3, 1, 3, name='conv0')
        self.conv_7x7_out = self.conv_batchN_relu(self.conv_3x3_out, 7, 2, 64, name='conv1')
        self.max_pool_out = self.pool(self.conv_7x7_out, 3, 2)

        ##block1
        self.m_b1_out = self.unit_v1(self.max_pool_out, self.filters[0], 1, 1, 1, shortcut=True)
        for unit_index in range(1, self.residual_units[0]):
            self.m_b1_out = self.unit_v1(self.m_b1_out, self.filters[0], 1, 1, unit_index+1)

        ##block2
        self.m_b2_out = self.