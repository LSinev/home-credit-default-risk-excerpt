# inspired by Pavel's script at https://www.kaggle.com/pavetr/stacking-lb-0-285
# this script just requires typing the names of files to stack 
# at the initial section and it is ready to run

import pandas as pd
import numpy as np

RANDOM_STATE = 2042
np.random.seed(RANDOM_STATE)

# key in the files to stack here
# START
files = {}
files['iv_079581CV_0795PL'] = '../stack_it/iv_079581CV_0795PL.csv'
files['11, 10 fold Full AUC score 0.795349'] = '../submit/11, 10 fold Full AUC score 0.795349.zip'
files['blend_Home_blendv9+7'] = '../submit/blend_Home_blendv9+7.zip'
files['blend+6'] = '../submit/blend+6.zip'
files['exp_sub'] = '../submit/exp_sub.zip'
files['Home_blend(version 9'] = '../submit/Home_blend(version 9.csv'
files['HC_sub_7_079581CV'] = '../submit/Kaggle_HC_sub_7_079581CV.zip'
files['HC_sub_8_CV079465'] = '../submit/Kaggle_HC_sub_8_CV079465.zip'
files['kcostya_10CV_0794958_LB_0796'] = '../submit/kcostya_10CV_0794958_LB_0796.csv'
files['prediction_9'] = '../submit/prediction_9.zip'
files['prediction_mte'] = '../submit/prediction_mte.zip'
# END


dfs = []
for key, _file in files.items():
    df = pd.read_csv(_file)
    df.rename(columns = {'target': key, 'TARGET': key}, inplace=True)
    dfs.append(df)



_submission = pd.concat(dfs, axis=1)


for key in files.keys():
    _submission[key + '_rank'] = _submission[key].rank()


_submission['rank_sum'] = np.sum(
        _submission[col] for col in _submission.columns if '_rank' in col)
_submission['TARGET'] = _submission['rank_sum']/(len(files) *
        _submission.shape[0])

# take the first (id) and last column (target)
submission = _submission.iloc[:, [0, -1]]


#filename = f"leonid03blend_rank_average-{','.join(files.keys())}"
filename = "leonid03blend_rank_average"
submit_file = f'../submit/{filename}.csv'
print(f'creating {submit_file}')

submission.to_csv(submit_file, index=False)
print('Done')
