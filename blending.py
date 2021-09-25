import pandas as pd
import os

result_dir = "./result"
file_list = os.listdir(result_dir)
remove_columns = [
    'prob_CNN_Nonefake_NoneFocal_NoneTrick_NoneSim_relu.csv',
    'prob_RNN_Nonefake_NoneFocal_NoneTrick_NoneSim_relu.csv',
    'prob_Capsule_Nonefake_NoneFocal_NoneTrick_NoneSim_relu.csv',
]

for f in remove_columns:
    file_list.remove(f)

test_data = pd.read_csv('./data/复赛/test.txt',header=None)
test_data.columns = ['pid', 'gender', 'age', 'tagid', 'timestamp', 'province', 'city', 'model', 'make']
blending = test_data[['pid']]
normal_ = 0
for index, res_file in enumerate(file_list):
    path_temp = os.path.join(result_dir, res_file)
    result_temp = pd.read_csv(path_temp)
    if 'CNN' in res_file or 'Capsule' in res_file:
        result_temp['tmp'] *= 2
        normal_ += 2
    else:
        result_temp['tmp'] *= 3
        normal_ += 3
    result_temp = result_temp.rename(columns={'tmp': f'tmp_{index}'})
    blending = blending.merge(result_temp, on='pid', how='left')

blending['tmp'] = 0
for i in range(len(file_list)):
    blending['tmp'] += blending[f'tmp_{i}']
blending['tmp'] /= normal_

submit_path = './submit'
if not os.path.exists(submit_path):
    os.mkdir(submit_path)

blending['rank'] = blending['tmp'].rank()
blending['pred_label'] = 1
blending.loc[blending['rank'] <= 87000, 'pred_label'] = 0

blending[['pid', 'pred_label']].to_csv('result.csv', index=None)
