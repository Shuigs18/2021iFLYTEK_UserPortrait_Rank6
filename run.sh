#!/bin/bash
#1.创建目录
mkdir feats
mkdir result
mkdir model_ckpt
mkdir 预训练模型

#1.提取特征
python extract_features.py


#2.NN model
# 参数 --training_model CNN --fake_label True --fake_rato 0.8 
# --tagid_trick False --sim_feats False --focal False --activation ule 
python NN.py --training_model CNN
python NN.py --training_model RNN
python NN.py --training_model Capsule

python NN.py --training_model RNN --fake_label True --fake_rato 0.8
python NN.py --training_model RNN --fake_label True --fake_rato 0.85
python NN.py --training_model CNN --fake_label True --fake_rato 0.85
python NN.py --training_model Capsule --fake_label True --fake_rato 0.85
python NN.py --training_model RNN --fake_label True --fake_rato 0.85 --tagid_trick True --focal True --activation elu
python NN.py --training_model RNN --fake_label True --fake_rato 0.85 --tagid_trick True --focal True --activation elu --sim_feats True
python NN.py --training_model RNN --fake_label True --fake_rato 0.85 --tagid_trick True --sim_feats False --focal False --activation reluos

#3.模型融合
python blending.py
