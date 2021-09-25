import pandas as pd
import numpy as np

from sklearn import preprocessing
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
from tensorflow import keras
import tensorflow as tf
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from pandarallel import pandarallel

from gensim.models.fasttext import FastText
from gensim.models import Word2Vec

import model

import gc
import os

warnings.filterwarnings('ignore')

pandarallel.initialize()

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("fake_label", False, "fake label")
flags.DEFINE_float("fake_rato", None, "the point of using fake label")
flags.DEFINE_boolean("tagid_trick", False, "using tagid trick")
flags.DEFINE_boolean("sim_feats", False, "using sim_feat, model_in, model_make_in")
flags.DEFINE_string("training_model", "CNN", "model")
flags.DEFINE_boolean("focal", False, "focal")
flags.DEFINE_string("activation", 'relu', "activation func")

embed_size = 200
MAX_SEQUENCE_LENGTH = 128
LEARNING_RATE = 0.001


def load_base_feats(fake_label=False,fake_rato=None):
    base_feats = pd.read_feather("./feats/复赛_base_feats.feather")
    time_feats = pd.read_feather('./feats/复赛_time_feats.feather')
    user_tagid_svd100 = pd.read_feather('./feats/复赛_tagid_svd100.feather')
    sim_feats = pd.read_feather('./feats/复赛_相似度特征.feather')

    data = time_feats
    data = data.merge(base_feats, on='userid', how='left')
    data = data.merge(user_tagid_svd100, on='userid', how='left')
    data = data.merge(sim_feats, on='userid', how='left')
    if fake_label == True:
        model1_result = pd.read_csv('./result/prob_CNN_Nonefake_NoneFocal_NoneTrick_NoneSim_relu.csv')
        model2_result = pd.read_csv('./result/prob_RNN_Nonefake_NoneFocal_NoneTrick_NoneSim_relu.csv')
        model3_result = pd.read_csv('./result/prob_Capsule_Nonefake_NoneFocal_NoneTrick_NoneSim_relu.csv')
        blending_result = model1_result[['pid']]
        blending_result['tmp'] = (model1_result['tmp'] + model2_result['tmp'] + model3_result['tmp'])/3
        fake_label = blending_result[(blending_result.tmp > fake_rato) | (blending_result.tmp < (1 - fake_rato))]
        fake_label['label'] = 1
        fake_label.loc[fake_label.tmp < (1-fake_rato), 'label'] = 0
        fake_label['userid'] = fake_label['pid'].astype(str).values + '_test'
        train_temp = data.loc[data.pid < 1400001, :]
        test_temp = data.loc[data.pid >= 1400001, :]
        test_temp.drop(columns=['label'], inplace=True)
        test_temp = test_temp.merge(fake_label, on=['pid', 'userid'], how="left")
        data = pd.concat([train_temp, test_temp])
    return data

def load_tagid_feats(data):
    tagid_feats = []

    train_temp = data.loc[data.pid <= 1400000]
    test_temp = data.loc[data.pid >= 1400001]

    for f in [
        "age", "province",  "city", "city_level",
        "city_cluster", "model_cluster","make_cluster"
        ]:
        file_path = f'./feats/tagid_feats/复赛_500_train_{f}_tagid.feather'
        feats_col = [
            'userid',
            f+'_tagid_label_mean',
            f+'_tagid_buy_count_mean',
        ]
        train_f_tagid = pd.read_feather(file_path, columns=feats_col)
        train_temp = train_temp.merge(train_f_tagid, on=['userid'], how="left")
        tagid_feats.extend([f+'_tagid_label_mean', 
                            f+'_tagid_buy_count_mean',
                            ])

    for f in ["age", "province",  "city", "city_level",
            "city_cluster", "model_cluster","make_cluster"]:
        file_path = f'./feats/tagid_feats/复赛_500_test_{f}_tagid.feather'
        feats_col = [
            'userid',
            f+'_tagid_label_mean', 
            f+'_tagid_buy_count_mean', 
        ]
        test_f_tagid = pd.read_feather(file_path, columns=feats_col)
        test_temp = test_temp.merge(test_f_tagid, on=['userid'], how="left")
        
    for f1 in ["age", "province",  "city", "city_level",
            "city_cluster", "model_cluster","make_cluster"]:
        for f2 in ['label', 'buy_count']:
            f_mean_m = train_temp[f"{f1}_tagid_{f2}_mean"].mean()
            train_temp[f"{f1}_tagid_{f2}_mean"] = train_temp[f"{f1}_tagid_{f2}_mean"].fillna(f_mean_m)

    train_tagid_feat = pd.read_feather('./feats/tagid_feats/复赛_500_train_tagid.feather',
                                    columns=['userid','tagid_label_mean',
                                            'tagid_buy_count_mean'])
    test_tagid_feat = pd.read_feather('./feats/tagid_feats/复赛_500_test_tagid.feather',
                                    columns=['userid','tagid_label_mean', 
                                            'tagid_buy_count_mean'])

    train_temp = train_temp.merge(train_tagid_feat, on=['userid'], how="left")
    test_temp = test_temp.merge(test_tagid_feat, on=['userid'], how="left")
    for agg_ in ["mean", ]:
        tagid_feats.extend([f'tagid_label_{agg_}', f'tagid_buy_count_{agg_}'])
        agg_label_m = train_temp[f'tagid_label_{agg_}'].mean()
        agg_buy_count_m = train_temp[f'tagid_buy_count_{agg_}'].mean()
        test_temp[f'tagid_label_{agg_}'] = test_temp[f'tagid_label_{agg_}'].fillna(agg_label_m)
        test_temp[f'tagid_buy_count_{agg_}'] = test_temp[f'tagid_buy_count_{agg_}'].fillna(agg_buy_count_m)
        train_temp[f'tagid_label_{agg_}'] = train_temp[f'tagid_label_{agg_}'].fillna(agg_label_m)
        train_temp[f'tagid_buy_count_{agg_}'] = train_temp[f'tagid_buy_count_{agg_}'].fillna(agg_buy_count_m)

    data = pd.concat([train_temp, test_temp])

    return data

def load_time_feats(data):
    time_feats = []

    train_temp = data[data['pid'] < 1400001]
    test_temp = data[data['pid'] >= 1400001]


    for f in ['Y-M-D', 'Hour']:
        train_tagid_feat = pd.read_feather(f'./feats/time_feats/复赛_500_train_time_{f}.feather',
                                        columns=['userid',f'{f}_label_mean',
                                                f'{f}_buy_count_mean'])
        test_tagid_feat = pd.read_feather(f'./feats/time_feats/复赛_500_test_time_{f}.feather',
                                        columns=['userid',f'{f}_label_mean',
                                                f'{f}_buy_count_mean'])

        train_temp = train_temp.merge(train_tagid_feat, on=['userid'], how="left")
        test_temp = test_temp.merge(test_tagid_feat, on=['userid'], how="left")
        for agg_ in ["mean", ]:
            time_feats.extend([f'{f}_label_{agg_}', f'{f}_buy_count_{agg_}'])
            agg_label_m = train_temp[f'{f}_label_{agg_}'].mean()
            agg_buy_count_m = train_temp[f'{f}_buy_count_{agg_}'].mean()
            test_temp[f'{f}_label_{agg_}'] = test_temp[f'{f}_label_{agg_}'].fillna(agg_label_m)
            test_temp[f'{f}_buy_count_{agg_}'] = test_temp[f'{f}_buy_count_{agg_}'].fillna(agg_buy_count_m)
            train_temp[f'{f}_label_{agg_}'] = train_temp[f'{f}_label_{agg_}'].fillna(agg_label_m)
            train_temp[f'{f}_buy_count_{agg_}'] = train_temp[f'{f}_buy_count_{agg_}'].fillna(agg_buy_count_m)

    data = pd.concat([train_temp, test_temp])
    del train_temp
    del test_temp
    return data

def preprocess_feats(data):
    SPARSE_FEATURES = ['age',
        'province',
        'city_level',
        'model_in',
        'model_make_in',
        'city_cluster',
        'model_cluster',
        'make_cluster',
        'time_count_seg',
        'time_long_seg'
        ]

    for f in SPARSE_FEATURES:
        data[f] = data[f].astype(str)
        label_encoder = LabelEncoder()
        label_encoder.fit(data[f])
        data[f] = label_encoder.transform(data[f])

    DENSE_FEATURES = ['age_sale_count', 'age_sale_rato', 
                  'province_sale_count', 'province_sale_rato',
                  'city_sale_count','city_sale_rato',
                  'city_level_sale_count','city_level_sale_rato',
                  'city_cluster_sale_count','city_cluster_sale_rato',
                  'model_cluster_sale_count','model_cluster_sale_rato',
                  'make_cluster_sale_count', 'make_cluster_sale_rato',
                  'weekend_mean', 'friday_mean', 'month_start_mean', 'month_end_mean',
                  'month_5_mean', 'month_6_mean',
                  'hour_seg_0_mean', 'hour_seg_1_mean', 'hour_seg_2_mean', 'hour_seg_3_mean',
                  'Quarter_0_mean', 'Quarter_1_mean', 'Quarter_2_mean', 'Quarter_3_mean',
                  'year_2021_mean','year_2020_mean','year_2019_mean',
                  'new_timestamp_mean','new_timestamp_max','new_timestamp_min','new_timestamp_median',
                  'month_mean', 'month_max', 'month_min', 'month_median',
                  'pid_tagid_change_time',
                  'shopping_festival_mean',
                  'time_diff_mean', 'time_diff_max', 'time_diff_min','time_diff_var',
                  'day_to_weekend_mean','is_year_start_mean',
                  'time_skew', 'time_kurtosis',
                  'w2v_cos_sim_in_tagid_model_make'
                 ]
    data['w2v_cos_sim_in_tagid_model_make'] = data['w2v_cos_sim_in_tagid_model_make'].fillna(0)

    tfidf_svd_feature = []
    n_components = 100
    for i in (range(n_components)):
        tfidf_svd_feature.append('tfidf_svd_{}'.format(i))
    
    time_feats = [
    'Y-M-D_label_mean',                   
    'Y-M-D_buy_count_mean',               
    'Hour_label_mean',                    
    'Hour_buy_count_mean'                 
    ]

    tagid_feats = []
    tagid_feats.extend(['tagid_label_mean', 'tagid_buy_count_mean'])
    for f in ["age", 
            "province",
            "city",
            "city_level",
            "city_cluster",
            "model_cluster",
            "make_cluster"
            ]:
        tagid_feats.extend([
                            f+'_tagid_label_mean',
                            f+'_tagid_buy_count_mean', 
                            ])
    
    temp = []
    temp.extend(DENSE_FEATURES.copy() + tagid_feats + time_feats + tfidf_svd_feature)
    temp.remove('time_skew')
    temp.remove('time_kurtosis')
    # 标准化
    #Z-Score标准化
    #建立StandardScaler对象
    zscore = preprocessing.StandardScaler()
    # 标准化处理
    data[temp] = zscore.fit_transform(data[temp])

    return data

def tagid_trick(data):
    # 前64个 后64个
    def tagid_trick1(x):
        res = []
        if len(x) > 128:
            res.extend(x[:64] + x[-64:])
        else:
            res = x
        return res
    data['tagid'] = data['tagid'].parallel_apply(lambda x: list(x))
    data['tagid'] = data['tagid'].parallel_apply(tagid_trick1)

    return data

def load_word_vector(data):
    data['tagid'] = data['tagid'].parallel_apply(lambda x: eval(str(x)))
    w2v_model = Word2Vec.load(f'./预训练模型/复赛_tagid_200维_window1_word2vec_model.model')
    FastText_model = FastText.load(f'./预训练模型/复赛_tagid_200维_window1_FastText_model.model')
    tagid_train = data.loc[data.label.notnull().T, 'tagid']
    tagid_test = data.loc[data.label.isnull().T, 'tagid']

    # 创建词典
    tokenizer = text.Tokenizer(num_words=None)
    tokenizer.fit_on_texts(list(tagid_train)+list(tagid_test))
    tagid_train = tokenizer.texts_to_sequences(tagid_train)
    tagid_test = tokenizer.texts_to_sequences(tagid_test)

    tagid_train = sequence.pad_sequences(tagid_train, maxlen=MAX_SEQUENCE_LENGTH)
    tagid_test = sequence.pad_sequences(tagid_test, maxlen=MAX_SEQUENCE_LENGTH)
    word_index = tokenizer.word_index

    nb_words = len(word_index) + 1

    embedding_matrix = np.zeros((nb_words, embed_size*2))

    for word, i in word_index.items():
        if (
            (word in w2v_model.wv.key_to_index) 
            and (word in FastText_model.wv.key_to_index) 
            ):
            embedding_vector = np.concatenate((w2v_model.wv[word],
                                            FastText_model.wv[word],
                                            ))
        else:
            embedding_vector = np.random.random(embed_size*2) * 0.5
            embedding_vector = embedding_vector - embedding_vector.mean()
        embedding_matrix[i] = embedding_vector

    del w2v_model
    del FastText_model
    del word_index
    del tokenizer
    gc.collect()

    return tagid_train, tagid_test, embedding_matrix, nb_words

def dataset(data, tagid_train, tagid_test, sim_feats=False):
    SPARSE_FEATURES = [
    'age',
    'province',
    #'city_level',
    #'model_in',
    #'model_make_in',
    'city_cluster',
    'model_cluster',
    'make_cluster',
    'time_count_seg',
    'time_long_seg'
    ]

    DENSE_FEATURES = ['age_sale_count', 'age_sale_rato', 
                    'province_sale_count', 'province_sale_rato',
                    'city_sale_count','city_sale_rato',
                    #'province_area_sale_count','province_area_sale_rato',
                    #'city_level_sale_count','city_level_sale_rato',
                    'city_cluster_sale_count','city_cluster_sale_rato',
                    'model_cluster_sale_count','model_cluster_sale_rato',
                    'make_cluster_sale_count', 'make_cluster_sale_rato',
                    'weekend_mean', 'friday_mean', 'month_start_mean', 'month_end_mean',
                    'month_5_mean', 'month_6_mean',
                    'hour_seg_0_mean', 'hour_seg_1_mean', 'hour_seg_2_mean', 'hour_seg_3_mean',
                    'Quarter_0_mean', 'Quarter_1_mean', 'Quarter_2_mean', 'Quarter_3_mean',
                    'year_2021_mean','year_2020_mean','year_2019_mean',
                    'new_timestamp_mean','new_timestamp_max','new_timestamp_min','new_timestamp_median',
                    'month_mean', 'month_max', 'month_min', 'month_median',
                    'pid_tagid_change_time',
                    'shopping_festival_mean',
                    'day_to_weekend_mean','is_year_start_mean',
                    'time_diff_mean', 'time_diff_max', 'time_diff_min','time_diff_var',
                    'time_skew', 'time_kurtosis',
                    #'w2v_cos_sim_in_tagid_model_make',
                    ]

    tfidf_svd_feature = []
    n_components = 100
    for i in (range(n_components)):
        tfidf_svd_feature.append('tfidf_svd_{}'.format(i))
        
    time_feats = [
        'Y-M-D_label_mean',
        'Y-M-D_buy_count_mean',
        'Hour_label_mean',
        'Hour_buy_count_mean'
    ]

    DENSE_FEATURES.extend(['tagid_label_mean', 'tagid_buy_count_mean'])
    for f in ["age", 
            "province",
            "city",
            "city_cluster",
            "model_cluster",
            "make_cluster"
            ]:
        DENSE_FEATURES.extend([
                            f+'_tagid_label_mean', 
                            f+'_tagid_buy_count_mean',
                            ])
    DENSE_FEATURES += tfidf_svd_feature + time_feats
    if sim_feats == True:
        SPARSE_FEATURES.extend(['model_in', 'model_make_in'])
        DENSE_FEATURES.append('w2v_cos_sim_in_tagid_model_make')
   
    train = data.loc[data.label.notnull().T]
    test = data.loc[data.label.isnull().T]

    train_dense_x = [train[f].values for f in DENSE_FEATURES]
    train_sparse_x = [train[f].values for f in SPARSE_FEATURES]
    test_dense_x = [test[f].values for f in DENSE_FEATURES]
    test_sparse_x = [test[f].values for f in SPARSE_FEATURES]
    y_categorical = train['label'].values

    X_train = train_dense_x + train_sparse_x + [tagid_train] 
    X_test = test_dense_x + test_sparse_x + [tagid_test]

    SPARSE_DICT = {
    'age': data['age'],
    'province': data['province'],
    #'city_level': data['city_level'],
    'model_in':data['model_in'],
    'model_make_in': data['model_make_in'],
    'city_cluster': data['city_cluster'],
    'model_cluster': data['model_cluster'],
    'make_cluster': data['make_cluster'],
    'time_count_seg': data['time_count_seg'],
    'time_long_seg': data['time_long_seg'],
    }
    
    return X_test, X_train, y_categorical, SPARSE_FEATURES, DENSE_FEATURES, SPARSE_DICT

def modelTrain(model_name, train, test, X_train, X_test, y_categorical,
               SPARSE_FEATURES, DENSE_FEATURES, embedding_matrix, nb_words, SPARSE_DICT):
    model_root_path = "./model_ckpt"
    if not os.path.exists(model_root_path):
        os.mkdir(model_root_path)
    # 五折交叉验证
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    oof = np.zeros([len(train), 1])
    predictions = np.zeros([len(test), 1])
    EPOCHS = 10
    PATIENCE = 4
    BATCH_SIZE = 64
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        print("fold n{}".format(fold_ + 1))
        if FLAGS.training_model == "CNN":
            MyModel = model.TextCNN_DeepFM_model(SPARSE_FEATURES, DENSE_FEATURES, embedding_matrix, 
                                nb_words, SPARSE_DICT, focal=FLAGS.focal)
        elif FLAGS.training_model == "RNN":
            MyModel = model.TextbiRNN_DeepFM_model(SPARSE_FEATURES, DENSE_FEATURES, embedding_matrix, 
                                nb_words, SPARSE_DICT, activation=FLAGS.activation, focal=FLAGS.focal)
        else:
            MyModel = model.Capsule_DeepFM_model(SPARSE_FEATURES, DENSE_FEATURES, embedding_matrix, 
                                nb_words, SPARSE_DICT, focal=FLAGS.focal) 
        if fold_ == 0:
            MyModel.summary()
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)
        bst_model_path = f"./model_ckpt/{model_name}_fold{fold_}_epochs{EPOCHS}.h5"
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        
        #减小学习率  'val_binary_crossentropy',
        Reduce_lr=ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.1,
                                    patience=1,
                                    verbose=1,
                                    mode='auto',
                                    min_delta=0.0001,
                                    cooldown=0,
                                    min_lr=0)
        
        X_trn = []
        X_val = []
        
        for f in X_train:
            X_trn.append(f[trn_idx])
            X_val.append(f[val_idx])
        y_trn, y_val = y_categorical[trn_idx], y_categorical[val_idx]
        
        MyModel.fit(X_trn, y_trn,
                validation_data = (X_val, y_val),
                epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
                callbacks=[early_stopping, model_checkpoint, Reduce_lr]
                )
        MyModel.load_weights(bst_model_path)
            
        oof[val_idx] = MyModel.predict(X_val)
        
        predictions += MyModel.predict(X_test)
        del MyModel
        gc.collect()
        
    predictions /= folds.n_splits
    print("训练结束")
    return oof, predictions

def save_prob(oof, predictions, train, test, data, model_name):
    result_dir = "./result"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    test_data = pd.read_csv('./data/复赛/test.txt',header=None)
    test_data.columns = ['pid', 'gender', 'age', 'tagid', 'timestamp', 'province', 'city', 'model', 'make']
    
    train['predict'] = oof
    train['rank'] = train['predict'].rank()
    train['p'] = 1
    train.loc[train['rank'] <= train.shape[0] * (1 - len(train[train['label']==1]) / len(train)), 'p'] = 0
    bst_f1_tmp = f1_score(train['label'].values, train['p'].values)
    trn_f1 = str(bst_f1_tmp).split('.')[1]
    
    if FLAGS.fake_label:
        model1_result = pd.read_csv('./result/prob_CNN_Nonefake_NoneFocal_NoneTrick_NoneSim_relu.csv')
        model2_result = pd.read_csv('./result/prob_RNN_Nonefake_NoneFocal_NoneTrick_NoneSim_relu.csv')
        model3_result = pd.read_csv('./result/prob_Capsule_Nonefake_NoneFocal_NoneTrick_NoneSim_relu.csv')
        test_result = test[['pid']]
        test_result['tmp'] = predictions

        fake_label = data.loc[data.pid >= 1400001][['pid', 'label']]
        fake_label = fake_label.dropna(subset=['label'])
        fake_label = fake_label.drop(columns=['label'])
        blending_result = model1_result[['pid']]
        blending_result['tmp'] = (model1_result['tmp'] + model2_result['tmp'] + model3_result['tmp'])/3
        fake_label = fake_label.merge(blending_result, on='pid', how='left')
        submit_temp = pd.concat([fake_label, test_result])
        submit = test_data[["pid"]]
        submit = submit.merge(submit_temp, on=["pid"], how="left")
        submit = submit.fillna(int(1))

        prob_path = f'./result/prob_{model_name}_{trn_f1}.csv'
        submit[['pid', 'tmp']].to_csv(prob_path, index=False)
    else:
        submit_temp = test[['pid']]
        submit_temp['tmp'] = predictions
        submit = test_data[["pid"]]
        submit = submit.merge(submit_temp, on=["pid"], how="left")
        submit = submit.fillna(int(1))
        prob_path = f'./result/prob_{model_name}.csv'
        submit[['pid', 'tmp']].to_csv(prob_path,index=False)


def main(argv):
    data = load_base_feats(FLAGS.fake_label, FLAGS.fake_rato)
    data = load_tagid_feats(data)
    data = load_time_feats(data)
    data = preprocess_feats(data)
    if FLAGS.tagid_trick:
        data = tagid_trick(data)
    tagid_train, tagid_test, embedding_matrix, nb_words = load_word_vector(data)
    X_test, X_train, y_categorical, SPARSE_FEATURES, DENSE_FEATURES, SPARSE_DICT = dataset(data, tagid_train,
                                                                                           tagid_test, FLAGS.sim_feats)
    train = data.loc[data.label.notnull().T]
    test = data.loc[data.label.isnull().T]

    fake_name = 'fake'+str(int(FLAGS.fake_rato * 100)) if FLAGS.fake_label else 'Nonefake'
    focal_name = 'focal' if FLAGS.focal else 'NoneFocal'
    trick_name = 'tagid_trick' if FLAGS.tagid_trick else 'NoneTrick'
    sim_name = 'sim' if FLAGS.sim_feats else 'NoneSim'
    activation_name = 'relu' if FLAGS.activation=='relu' else 'elu'
    model_name = f"{FLAGS.training_model}_{fake_name}_{focal_name}_{trick_name}_{sim_name}_{activation_name}"
    
    oof, predictions = modelTrain(model_name, train, test, X_train, X_test, y_categorical,
                                  SPARSE_FEATURES, DENSE_FEATURES, embedding_matrix, nb_words, SPARSE_DICT)
    save_prob(oof, predictions, train, test, data, model_name)

if __name__ == "__main__":
    tf.app.run(main)