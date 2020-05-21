from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from util.show_save_result import ShowAndSave
from util.save_get_mean_std import save_mean_std, get_mean_std
from model.params_dict import ParamsDict
from model.get_data_path import get_train_data_path, get_test_data_path
import os
from util.user_logger import logger_process
from util.read_process_data import ReadProcessData
from util.data_normalized import data_normalized

cur_path = os.path.abspath(os.path.dirname(__file__))


class RFModel(ShowAndSave):
    def __init__(self, jobname='rf_model', fc='wfzc', fj='A2', model_kind='cap_temp_1', max_depth=20,
                 n_estimators=100, mf=3, params_kind=None):
        super().__init__()
        self.job_name = jobname + '_' + fc + '_' + fj + '_' + str(max_depth) + '_' + str(n_estimators) + '_' + str(
            mf)
        self.logger = logger_process(cur_path, self.job_name)
        self.logger.info(self.job_name)
        self.model_folder_name = fj + '_' + model_kind + '_' + params_kind
        self.model_name = fj
        self.model_kind = model_kind
        self.params_kind = params_kind
        self.fc = fc
        self.fj = fj
        self.model_kind = model_kind
        self.cur_path = cur_path
        self.init_param()
        self.params = ParamsDict.model_params[params_kind][model_kind]
        self.fj_model_kind = fj + '_' + model_kind
        self.model_file_name = self.fj_model_kind + '_rf.model'
        self.feature_importance_path = self.single_model_path + 'feature_importance/'
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.mf = mf

    def train_rf(self, data_kind='train_data'):
        self.data_kind=data_kind
        file_dir = get_train_data_path(self.fc, self.fj, self.model_kind, self.params_kind)
        rpd = ReadProcessData()
        x_dataset, y_dataset, dscr = rpd.read_data(file_dir)
        mean_std_dic = rpd.get_std_mean(dscr, self.params)
        save_mean_std(mean_std_dic, self.fc, self.fj, self.model_kind)
        normalized_data = data_normalized(x_dataset, mean_std_dic)
        x_train, x_test, y_train, y_test = train_test_split(normalized_data, y_dataset, train_size=0.8)
        self.logger.info('when training,train data number:{}'.format(str(len(y_train))))
        self.logger.info('when training,test data number:{}'.format(len(y_test)))
        self.logger.info('training model:{}'.format(self.model_kind))
        rfr = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,oob_score=True)
        rfr.fit(x_train, y_train)
        #self.logger(rfr.feature_importances_)
        self.logger.info(self.model_path)
        joblib.dump(rfr, self.model_path+self.model_file_name)
        #self.logger(rfr.score(x_train, y_train))
        pred = rfr.predict(x_test)
        plt.show()
        plt.close()
        self.save_result_dataframe(y_test, pred, )
        self.set_var(true_v=y_test, pred_v=pred)
        self.show_save_figure(detal_idx=4)
        t_mean = self.cal_mean(self.true)
        p_mean = self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean, pred_mean=p_mean, train_n=len(x_train), test_n=len(x_test))

    def test_rf(self,data_kind='test',delta_idx=4):
        self.data_kind=data_kind
        file_dir=get_test_data_path(self.fc, self.fj, self.model_kind, self.params_kind)
        df = pd.read_csv(file_dir, encoding='utf-8', index_col=0)
        data = df.iloc[50000:60000, :].values
        x = data[:, :-1]
        y = data[:, -1]
        mean_std_dict = get_mean_std(self.fc, self.fj, self.model_kind)
        x = data_normalized(x, mean_std_dict)
        rfr=joblib.load(self.model_path+self.model_file_name)
        pred=rfr.predict(x)
        self.save_result_dataframe(y, pred)
        self.set_var(true_v=y, pred_v=pred)
        # self.show_rolling_fig()
        self.show_save_figure(detal_idx=delta_idx)
        t_mean = self.cal_mean(self.true)
        p_mean = self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean, pred_mean=p_mean)

rfm=RFModel(jobname='rf_model_motor', fc='wfzc', fj='A2', model_kind='motor_temp_3',
                    params_kind='model_params_v1', max_depth=32, n_estimators=300, mf=1)
rfm.train_rf()