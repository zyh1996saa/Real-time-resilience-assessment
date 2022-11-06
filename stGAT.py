#from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import gat
import random




class Attention(tf.keras.layers.Layer):
    def __init__(self, units, activation=tf.identity, l2=0.0):
        super(Attention, self).__init__()
        #print('units:',units)
        self.l2 = l2
        self.activation = activation
        self.units = units

    def build(self, input_shape):
        #print(input_shape)
        H_shape, A_shape = input_shape
        
        #self.W 维度为[4,7]
        self.W = self.add_weight(
          shape=(H_shape[2], self.units),
          initializer='glorot_uniform',
          dtype=tf.float32,
          #regularizer=tf.keras.regularizers.l2(self.l2)
        )
        #print('self.W.shape:',self.W.shape)
        #1.self.W.shape:(4, 4)
        #2.self.W.shape:(32, 4)
        
        self.a_1 = self.add_weight(
          shape=(self.units, 1),
          initializer='glorot_uniform',
          dtype=tf.float32,
          #regularizer=tf.keras.regularizers.l2(self.l2)
        )
        #1.self.a1.shape:(4, 1)
        #self.a1 shape [7,1]
        self.a_2 = self.add_weight(
          shape=(self.units, 1),
          initializer='glorot_uniform',
          dtype=tf.float32,
          #regularizer=tf.keras.regularizers.l2(self.l2)
        )
        #1.self.a2.shape:(4, 1)
        
    def call(self, inputs):
        # H :[None,1433] A:[None,2708]
        H, A = inputs
        #print(H.shape,A.shape)
        #(None, 118, 118) (None, 118, 5)
        # X:[None,7]
        X = tf.matmul(H, tf.cast(self.W,dtype=tf.float32))
        #print(X.shape)
        #X.shape:(None, 39, 4)
        
        attn_self = tf.matmul(X, tf.cast(self.a_1,dtype=tf.float32))
        #print(attn_self.shape)
        #attn_self.shape:(None, 39, 1)
        
        attn_neighbours = tf.matmul(X, tf.cast(self.a_2,dtype=tf.float32))
        #print(attn_neighbours.shape)
        #attn_neighbours.shape:(None, 39, 1)
        
        
        attention = attn_self + tf.transpose(attn_neighbours,perm=[0, 2, 1])
        #print(tf.transpose(attn_neighbours,perm=[0, 2, 1]).shape)
        #print(attention.shape)
        
        
        E = tf.nn.leaky_relu(tf.math.real(attention))

        #E = tf.complex(E1,E2)
        #print('---------')
        #print(E.shape)
        mask = mask = -10e9 * (1.0 - A)
        #print(mask.shape)
        masked_E = E + mask

        # A = tf.cast(tf.math.greater(A, 0.0), dtype=tf.float32)
        alpha = tf.nn.softmax(tf.math.real(masked_E))
        #alpha2 = tf.nn.softmax(tf.math.imag(masked_E))
        #alpha = tf.complex(alpha1,alpha2)
        H_cap = alpha @ X
        out = self.activation(tf.math.real(H_cap))
        #out2 = self.activation(tf.math.imag(H_cap))
        #out = tf.complex(out1,out2)
        return out



class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, output_layer=False, activation=tf.identity, l2=0.0):
        super(GraphAttentionLayer, self).__init__()

        self.activation = activation
        self.num_heads = num_heads
        self.output_layer = output_layer

        self.attn_layers = [Attention(units, l2=l2) for x in range(num_heads)]

    def call(self, inputs):

        H, A = inputs
        #print('H.shape:',H.shape)
        #print('A.shape:',A.shape)
        #H.shape: (None, 39, 4)
        #A.shape: (None, 39, 39)
        H_out = [self.attn_layers[i]([H, A]) for i in range(self.num_heads)]

        if self.output_layer:
            multi_head_attn = tf.reduce_mean(tf.stack(H_out), axis=0)
            out = self.activation(tf.math.real(multi_head_attn))
            #out2 = self.activation(tf.math.imag(multi_head_attn))
            #out = tf.complex(out1,out2)
        else:
            multi_head_attn = tf.concat(H_out, axis=-1)
            tf.math.real(multi_head_attn)
            out = self.activation(tf.math.real(multi_head_attn))
            #out2 = self.activation(tf.math.imag(multi_head_attn))
            #out = tf.complex(out1,out2)
        return out
    


input1 = tf.keras.Input(shape=(118,118))
input2 = tf.keras.Input(shape=(118,118))
input3 = tf.keras.Input(shape=(118,118))
input4 = tf.keras.Input(shape=(118,118))
input5 = tf.keras.Input(shape=(118,118))
input6 = tf.keras.Input(shape=(118,118))
input7 = tf.keras.Input(shape=(118,118))
input8 = tf.keras.Input(shape=(118,118))
input9 = tf.keras.Input(shape=(118,118))
input10 = tf.keras.Input(shape=(118,118))

input11 = tf.keras.Input(shape=(118,5))
input12 = tf.keras.Input(shape=(118,5))
input13 = tf.keras.Input(shape=(118,5))
input14 = tf.keras.Input(shape=(118,5))
input15 = tf.keras.Input(shape=(118,5))
input16 = tf.keras.Input(shape=(118,5))
input17 = tf.keras.Input(shape=(118,5))
input18 = tf.keras.Input(shape=(118,5))
input19 = tf.keras.Input(shape=(118,5))
input20 = tf.keras.Input(shape=(118,5))

head_num = 8
unit1 = 6 
unit2 = 16
x1 = GraphAttentionLayer(unit1,8,activation=tf.nn.elu)([input11,input1])
x1 = GraphAttentionLayer(unit2,8,activation=tf.nn.elu)([x1,input1])
x2 = GraphAttentionLayer(unit1,8,activation=tf.nn.elu)([input12,input2])
x2 = GraphAttentionLayer(unit2,8,activation=tf.nn.elu)([x2,input2])
x3 = GraphAttentionLayer(unit1,8,activation=tf.nn.elu)([input13,input3])
x3 = GraphAttentionLayer(unit2,8,activation=tf.nn.elu)([x3,input3])
x4 = GraphAttentionLayer(unit1,8,activation=tf.nn.elu)([input14,input4])
x4 = GraphAttentionLayer(unit2,8,activation=tf.nn.elu)([x4,input4])
x5 = GraphAttentionLayer(unit1,8,activation=tf.nn.elu)([input15,input5])
x5 = GraphAttentionLayer(unit2,8,activation=tf.nn.elu)([x5,input5])
x6 = GraphAttentionLayer(unit1,8,activation=tf.nn.elu)([input16,input6])
x6 = GraphAttentionLayer(unit2,8,activation=tf.nn.elu)([x6,input6])
x7 = GraphAttentionLayer(unit1,8,activation=tf.nn.elu)([input17,input7])
x7 = GraphAttentionLayer(unit2,8,activation=tf.nn.elu)([x7,input7])
x8 = GraphAttentionLayer(unit1,8,activation=tf.nn.elu)([input18,input8])
x8 = GraphAttentionLayer(unit2,8,activation=tf.nn.elu)([x8,input8])
x9 = GraphAttentionLayer(unit1,8,activation=tf.nn.elu)([input19,input9])
x9 = GraphAttentionLayer(unit2,8,activation=tf.nn.elu)([x9,input8])
x10 = GraphAttentionLayer(unit1,8,activation=tf.nn.elu)([input20,input10])
x10 = GraphAttentionLayer(unit2,8,activation=tf.nn.elu)([x10,input10])

for i in range(1,11):
    exec('x%s = tf.keras.layers.Reshape((1,118,unit2*head_num))(x%s)'%(i,i))
x=tf.keras.layers.Concatenate(axis=1)([x1, x2,x3,x4,x5,x6,x7,x8,x9,x10])
x=tf.keras.layers.Reshape((10,118*unit2*head_num))(x)

unit3 = 256
x=tf.keras.layers.GRU(unit3)(x)
x = tf.keras.layers.Dense(64,activation='gelu',)(x)
x = tf.keras.layers.Dense(64,activation='gelu',)(x)
output = tf.keras.layers.Dense(1,)(x)
#tf.keras.layers.Concatenate(axis=1)([x, y])
#x1 = tf.keras.layers.Reshape((3, 4), input_shape=(12,))
#x = tf.keras.layers.Dense(64,name='dense1',activation='gelu',kernel_regularizer=tf.keras.regularizers.l1(0.01))(input0)
#x = tf.keras.layers.Dense(16,name='dense2',activation='gelu',kernel_regularizer=tf.keras.regularizers.l1(0.01))(x)
#output = tf.keras.layers.Dense(1,name='dense3')(x)
model = tf.keras.Model([eval('input%s'%i) for i in range(1,21)], output)

'''
xx = np.random.random((1,118,118))
xxx = np.random.random((1,118,5))
model.predict([xx,xx,xx,xx,xx,xx,xx,xx,xx,xx,xxx,xxx,xxx,xxx,xxx,xxx,xxx,xxx,xxx,xxx])'''