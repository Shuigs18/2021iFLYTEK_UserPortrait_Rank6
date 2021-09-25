from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
from tensorflow import keras
import tensorflow as tf
from tensorflow.compat.v1.keras.layers import CuDNNGRU,CuDNNLSTM

import warnings

from sklearn.model_selection import StratifiedKFold

from gensim.models.fasttext import FastText
from gensim.models import Word2Vec

import time
import gc
import sys

warnings.filterwarnings('ignore')

def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss
    
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed

def TextCNN_DeepFM_model(SPARSE_FEATURES, DENSE_FEATURES, embedding_matrix, 
                         nb_words, SPARSE_DICT, activation ='relu', focal=False):
    # dense features的input
    dense_inputs = []
    for f in DENSE_FEATURES:
        _input = Input([1], name=f)
        dense_inputs.append(_input)
    concat_dense_inputs = Concatenate(axis=1)(dense_inputs)
    # dense 特征的FM一阶部分
    fm_concat_dense_inputs = Concatenate(axis=1)(dense_inputs)
    fst_order_dense_layer = Dense(1)(fm_concat_dense_inputs)

    # tagid feature的 input
    tagid_embedding_input = Input(shape=(128,), dtype='int32')
    tagid_embedder = Embedding(nb_words,
                         400,
                         input_length=128,
                         weights=[embedding_matrix],
                         trainable=False
                       )
    tagid_embed = tagid_embedder(tagid_embedding_input)
    
#    TextCNN
    x = SpatialDropout1D(0.2)(tagid_embed)
    convs = []
    for kernel_size in [1, 2, 3, 4]:
        c = Conv1D(200, kernel_size, kernel_initializer='he_uniform')(x)
        c = BatchNormalization()(c)
        c = Activation(activation="relu")(c)
        c = Conv1D(200, kernel_size, kernel_initializer='he_uniform')(c)
        c = BatchNormalization()(c)
        c = Activation(activation="relu")(c)
        c_max = GlobalMaxPooling1D()(c)
        convs.extend([c_max])
        
    x = Concatenate()(convs)
    x = Dense(256)(x)
    x = Activation(activation="relu")(x)
    tagid_layer = Dropout(0.2)(x)

    sparse_inputs = []
    for f in SPARSE_FEATURES:
        _input = Input([1], name=f)
        sparse_inputs.append(_input)
    
    # sparse特征的FM一阶部分
    sparse_1d_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = SPARSE_FEATURES[i]
        voc_size = SPARSE_DICT[f].nunique()
        _embed = Flatten()(Embedding(voc_size, 1)(_input))
        sparse_1d_embed.append(_embed)
    fst_order_sparse_layer = Add()(sparse_1d_embed)
    
    # FM线性部分相加
    linear_part = Add()([fst_order_dense_layer, fst_order_sparse_layer])
    
    # 建立FM二阶部分
    k = 32
    
    ## sparse部分的二阶交叉
    sparse_kd_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = SPARSE_FEATURES[i]
        voc_size = SPARSE_DICT[f].nunique()
        _embed = Embedding(voc_size, k)(_input)
        sparse_kd_embed.append(_embed)
        
    ## 1. 将所有的sparse的embedding拼接起来，得到(?, n, k)矩阵，n为特征数，k为embeddings_size
    concat_kd_embed = Concatenate(axis=1)(sparse_kd_embed)
    
    ## 2. axis=1列向求和再平方
    sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(concat_kd_embed)
    square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])  # ?, k
    
    ## 3. 先平方在求和
    square_kd_embed = Multiply()([concat_kd_embed, concat_kd_embed])
    sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(square_kd_embed) 
    
    ## 4. 相减除以2
    sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])
    sub = Lambda(lambda x: x*0.5)(sub)
    snd_order_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub)    
    
    ## DNN 部分
    flatten_embed = Flatten()(concat_kd_embed)  
    DNN_input = Concatenate(axis=1)([flatten_embed, concat_dense_inputs, tagid_layer])
    
    fc_layer = Dense(128)(DNN_input)
    fc_layer = ReLU()(fc_layer)
    fc_layer = Dense(64)(fc_layer)
    fc_layer = ReLU()(fc_layer)
    fc_layer = Dense(8)(fc_layer)
    fc_layer = ReLU()(fc_layer)
    fc_output_layer = Dense(1)(fc_layer)
    
    output_layer = Add()([linear_part, snd_order_layer,fc_output_layer])
    output_layer = Activation('sigmoid')(output_layer)
    
    model = Model(dense_inputs+sparse_inputs+[tagid_embedding_input], 
                  output_layer)
    
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    if focal == True:
        model.compile(optimizer=optimizer,
                  loss=[binary_focal_loss(alpha=.25, gamma=2)],
                  metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')]
                  )
    else:
        model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')]
                  )
    return model


def TextbiRNN_DeepFM_model(SPARSE_FEATURES, DENSE_FEATURES, embedding_matrix, 
                            nb_words, SPARSE_DICT, activation='relu', focal=False):
    # dense features的input
    dense_inputs = []
    for f in DENSE_FEATURES:
        _input = Input([1], name=f)
        dense_inputs.append(_input)
    concat_dense_inputs = Concatenate(axis=1)(dense_inputs)
    # dense 特征的FM一阶部分
    fm_concat_dense_inputs = Concatenate(axis=1)(dense_inputs)
    fst_order_dense_layer = Dense(1)(fm_concat_dense_inputs)

    # tagid feature的 input
    embedding_input = Input(shape=(128,), dtype='int32')
    embedder = Embedding(nb_words,
                         400,
                         input_length=128,
                         weights=[embedding_matrix],
                         trainable=False
                        )
    tagid_embed = embedder(embedding_input)
    
    # TextbiRNN
    x = SpatialDropout1D(0.2)(tagid_embed)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    avg_pool_1 = GlobalAveragePooling1D()(x)
    max_pool_1 = GlobalMaxPooling1D()(x)
    conc = concatenate([max_pool_1, avg_pool_1])
    x = Dense(256)(conc)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    tagid_layer = Dropout(0.2)(x)
    
    sparse_inputs = []
    for f in SPARSE_FEATURES:
        _input = Input([1], name=f)
        sparse_inputs.append(_input)
    
    # sparse特征的FM一阶部分
    sparse_1d_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = SPARSE_FEATURES[i]
        voc_size = SPARSE_DICT[f].nunique()
        _embed = Flatten()(Embedding(voc_size, 1)(_input))
        sparse_1d_embed.append(_embed)
    fst_order_sparse_layer = Add()(sparse_1d_embed)
    
    # FM线性部分相加
    linear_part = Add()([fst_order_dense_layer, fst_order_sparse_layer])
    
    # 建立FM二阶部分
    k = 32
    
    ## sparse部分的二阶交叉
    sparse_kd_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = SPARSE_FEATURES[i]
        voc_size = SPARSE_DICT[f].nunique()
        _embed = Embedding(voc_size, k)(_input)
        sparse_kd_embed.append(_embed)
        
    ## 1. 将所有的sparse的embedding拼接起来，得到(?, n, k)矩阵，n为特征数，k为embeddings_size
    concat_kd_embed = Concatenate(axis=1)(sparse_kd_embed)
    
    ## 2. axis=1列向求和再平方
    sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(concat_kd_embed)
    square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])  # ?, k
    
    ## 3. 先平方在求和
    square_kd_embed = Multiply()([concat_kd_embed, concat_kd_embed])
    sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(square_kd_embed) 
    
    ## 4. 相减除以2
    sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])
    sub = Lambda(lambda x: x*0.5)(sub)
    snd_order_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub)    
    
    ## DNN 部分
    flatten_embed = Flatten()(concat_kd_embed) 
    DNN_input = Concatenate(axis=1)([flatten_embed, concat_dense_inputs, tagid_layer])
    
    fc_layer = Dense(128)(DNN_input)
    fc_layer = Activation(activation=activation)(fc_layer)
    fc_layer = Dense(64)(fc_layer)
    fc_layer = Activation(activation=activation)(fc_layer)
    fc_layer = Dense(8)(fc_layer)
    fc_layer = Activation(activation=activation)(fc_layer)    
    
    fc_output_layer = Dense(1)(fc_layer)
    
    output_layer = Add()([linear_part, snd_order_layer,fc_output_layer])
    output_layer = Activation('sigmoid')(output_layer)
    
    model = Model(dense_inputs+sparse_inputs+[embedding_input], output_layer)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    if focal == True:
        model.compile(optimizer=optimizer,
                  loss=[binary_focal_loss(alpha=.25, gamma=2)],
                  metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')]
                  )
    else:
        model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')]
                  )
    return model


Num_capsule=1
Dim_capsule=8
Routings=3

def squash(x, axis=-1):
    s_squared_norm = tf.keras.backend.sum(tf.keras.backend.square(x), axis, keepdims=True)
    scale = tf.keras.backend.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return x / scale

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), 
                 share_weights=True,activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = tf.keras.backend.conv1d(u_vecs, kernel=self.W)
        else:
            u_hat_vecs = tf.keras.backend.local_conv1d(u_vecs, kernel=self.W, kernel_size=[1], strides=[1])

        batch_size = tf.shape(u_vecs)[0]
        input_num_capsule = tf.shape(u_vecs)[1]
        u_hat_vecs = tf.reshape(u_hat_vecs, [batch_size, input_num_capsule,self.num_capsule, self.dim_capsule])
        u_hat_vecs = tf.transpose(u_hat_vecs,perm=[0, 2, 1, 3])# final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        b = tf.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = tf.transpose(b, perm=[0, 2, 1])  # shape = [None, input_num_capsule, num_capsule] 
            c = tf.nn.softmax(b) # shape = [None, input_num_capsule, num_capsule] 
            c = tf.transpose(c, perm=[0, 2, 1])  # shape = [None, num_capsule, input_num_capsule] 
            s_j = tf.reduce_sum(tf.multiply(tf.expand_dims(c,axis=3) , u_hat_vecs) , axis=2)        
            outputs = self.activation(s_j) #[None,num_capsule,dim_capsule]
            if i < self.routings - 1:
                b = tf.reduce_sum(tf.multiply(tf.expand_dims(outputs,axis=2) , u_hat_vecs) , axis=3)
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)



def Capsule_DeepFM_model(SPARSE_FEATURES, DENSE_FEATURES, embedding_matrix, 
                         nb_words, SPARSE_DICT, activation = 'relu', focal=False):
    # dense features的input
    dense_inputs = []
    for f in DENSE_FEATURES:
        _input = Input([1], name=f)
        dense_inputs.append(_input)
    concat_dense_inputs = Concatenate(axis=1)(dense_inputs)
    # dense 特征的FM一阶部分
    fm_concat_dense_inputs = Concatenate(axis=1)(dense_inputs)
    fst_order_dense_layer = Dense(1)(fm_concat_dense_inputs)
    
    # tagid feature的 input
    embedding_input = Input(shape=(128,), dtype='int32')
    embedder = Embedding(nb_words,
                         400,
                         input_length=128,
                         weights=[embedding_matrix],
                         trainable=False
                        )
    tagid_embed = embedder(embedding_input)
    
    embed = SpatialDropout1D(0.2)(tagid_embed)
    x = Bidirectional(GRU(200, return_sequences=True))(embed)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    capsule = Flatten()(capsule)
    x = Dense(256)(capsule)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    tagid_layer = Dropout(0.2)(x)
    
    # sparse特征的输入层
    sparse_inputs = []
    for f in SPARSE_FEATURES:
        _input = Input([1], name=f)
        sparse_inputs.append(_input)
    
    # sparse特征的FM一阶部分
    sparse_1d_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = SPARSE_FEATURES[i]
        voc_size = SPARSE_DICT[f].nunique()
        _embed = Flatten()(Embedding(voc_size, 1)(_input))
        sparse_1d_embed.append(_embed)
    fst_order_sparse_layer = Add()(sparse_1d_embed)
    
    # FM线性部分相加
    linear_part = Add()([fst_order_dense_layer, fst_order_sparse_layer])
    
    # 建立FM二阶部分
    k = 32
    
    ## sparse部分的二阶交叉
    sparse_kd_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = SPARSE_FEATURES[i]
        voc_size = SPARSE_DICT[f].nunique()
        _embed = Embedding(voc_size, k)(_input)
        sparse_kd_embed.append(_embed)
    
    
    ## 1. 将所有的sparse的embedding拼接起来，得到(?, n, k)矩阵，n为特征数，k为embeddings_size
    concat_kd_embed = Concatenate(axis=1)(sparse_kd_embed)
    
    ## 2. axis=1列向求和再平方
    sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(concat_kd_embed)
    square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])  # ?, k
    
    ## 3. 先平方在求和
    square_kd_embed = Multiply()([concat_kd_embed, concat_kd_embed])
    sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(square_kd_embed) 
    
    ## 4. 相减除以2
    sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])
    sub = Lambda(lambda x: x*0.5)(sub)
    snd_order_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub)    
    
    ## DNN 部分
    flatten_embed = Flatten()(concat_kd_embed) 
    DNN_input = Concatenate(axis=1)([flatten_embed, concat_dense_inputs, tagid_layer])
    

    fc_layer = Dense(128)(DNN_input)
    fc_layer = BatchNormalization()(fc_layer)
    fc_layer = ReLU()(fc_layer)
    fc_layer = Dense(64)(fc_layer)
    fc_layer = ReLU()(fc_layer)
    fc_layer = Dense(8)(fc_layer)
    fc_layer = ReLU()(fc_layer)
    
    fc_output_layer = Dense(1)(fc_layer)
    
    output_layer = Add()([linear_part, snd_order_layer,fc_output_layer])
    output_layer = Activation('sigmoid')(output_layer)
    
    model = Model(dense_inputs+sparse_inputs+[embedding_input], output_layer)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    if focal == True:
        model.compile(optimizer=optimizer,
                  loss=[binary_focal_loss(alpha=.25, gamma=2)],
                  metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')]
                  )
    else:
        model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')]
                  )
    return model