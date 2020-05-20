# -*-conding:utf-8-*-
import os
import pandas as pd
import sys

cur_path=os.path.abspath(os.path.dirname(__file__))
parent_path=os.path.abspath(os.path.join(cur_path,'..'))

def save_mean_std(mean_std_dict,fc,fj,model_kind):
    mean_std_df = pd.DataFrame(mean_std_dict)
    path_saved = parent_path + '/data/' + 'mean_std/' + fc + '_' + fj + '_' + model_kind + '/'
    if not os.path.exists(path_saved):
        os.makedirs(path_saved)
    data_file_name=fc+'_'+fj+'_'+model_kind+'.csv'
    mean_std_df.to_csv(path_saved+data_file_name,encoding='utf-8')

def get_mean_std(fc,fj,model_kind):
    path_saved = parent_path + '/data/' + 'mean_std/' + fc + '_' + fj + '_' + model_kind + '/'
    print('mean_std data file path:',path_saved)
    data_file_name = fc + '_' + fj + '_' + model_kind + '.csv'
    if not os.path.exists(path_saved+data_file_name):
        print('no mean_std data file exist.')
        sys.exit(1)
    df=pd.read_csv(path_saved+data_file_name,index_col=0)
    dic=df.to_dict(orient='records')
    return dic
