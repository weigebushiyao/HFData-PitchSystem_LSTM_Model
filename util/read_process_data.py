import pandas as pd
from util.save_get_mean_std import save_mean_std

class ReadProcessData:
    def __init__(self,):
        pass

    def read_data(self,file_dir=None):
        df = pd.read_csv(file_dir, encoding='utf-8', index_col=0)
        dscr = df.describe()
        data_set = df.iloc[:, :].values
        x_dataset = data_set[:, :-1]
        y_dataset = data_set[:, -1]
        return x_dataset,y_dataset,dscr

    def get_std_mean(self, dscr,params):
        ret_list = []
        for i in range(len(params) - 1):
            ret_list.append({'mean': dscr.loc['mean', params[i]], 'std': dscr.loc['std', params[i]]})
            print(params[i] + ':',
                  {'mean': dscr.loc['mean', params[i]], 'std': dscr.loc['std', params[i]]})
        return ret_list