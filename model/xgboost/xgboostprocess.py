# -*-coding:utf-8-*-
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from model.get_data_path import get_test_data_path, get_train_data_path
from sklearn.model_selection import train_test_split
from util.show_save_result import ShowAndSave
from util.read_process_data import ReadProcessData
from util.data_normalized import data_normalized
from util.save_get_mean_std import save_mean_std,get_mean_std
from model.params_dict import ParamsDict
import os
from util.user_logger import logger_process

cur_path=os.path.abspath(os.path.dirname(__file__))

class XgboostModel(ShowAndSave):
    def __init__(self, jobname='xgb_model', fc='wfzc', fj='A2', model_kind='cap_temp_1', max_depth=20,
                 n_estimator=512, min_child_weight=3, params_kind=None):
        super().__init__()
        self.job_name = jobname + '_' + fc + '_' + fj + '_' + str(max_depth) + '_' + str(n_estimator) + '_' + str(
            min_child_weight)
        self.logger=logger_process(cur_path,self.job_name)
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
        self.model_file_name = self.fj_model_kind + '_xgb.model'
        self.feature_importance_path = self.single_model_path + 'feature_importance/'
        self.max_depth = max_depth
        self.n_estimator = n_estimator
        self.min_child_weight = min_child_weight

    def xgboostmodel(self, data_kind='train_data'):
        self.data_kind=data_kind
        train_data_path = get_train_data_path(self.fc, self.fj, self.model_kind, self.params_kind)
        rpd=ReadProcessData()
        x_dataset,y_dataset,dscr=rpd.read_data(train_data_path)
        mean_std_dic = rpd.get_std_mean(dscr,self.params)
        save_mean_std(mean_std_dic, self.fc,self.fj,self.model_kind)
        normalized_data = data_normalized(x_dataset, mean_std_dic)
        x_train, x_test, y_train, y_test = train_test_split(normalized_data, y_dataset, train_size=0.8)
        self.logger.info('when training,train data number:{}'.format(str(len(y_train))))
        self.logger.info('when training,test data number:{}'.format(len(y_test)))
        self.logger.info('training model:{}'.format(self.model_kind))
        # params={'booster':'gbtree','objective':'reg:squarederror','eval_metric':'rmse','seed':0,'n_jobs':10,'max_depth':self.max_depth,'n_estimators':self.n_estimator,'min_child_weight':self.min_child_weight,
        #         'verbosity':1,'learning_rate':0.05}
        raw_model = xgb.XGBRegressor(max_depth=self.max_depth,
                                     n_estimators=self.n_estimator, learning_rate=0.02, silent=False,
                                     min_child_weight=self.min_child_weight)
        # raw_model = xgb.XGBRegressor(**params)
        raw_model.fit(x_train, y_train)
        self.logger.info(self.model_path)
        raw_model.save_model(self.model_path + self.model_file_name)
        #self.logger(raw_model.score(x_train,y_train))
        pred = raw_model.predict(x_test)
        plot_importance(raw_model)
        plt.show()
        plt.close()
        self.save_result_dataframe(y_test, pred, )
        self.set_var(true_v=y_test, pred_v=pred)
        self.show_save_figure(detal_idx=4)
        t_mean = self.cal_mean(self.true)
        p_mean = self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean, pred_mean=p_mean, train_n=len(x_train), test_n=len(x_test))

    def test_model(self, data_kind='fault_data', delta_idx=2):
        self.data_kind = data_kind
        fault_test_file_path = get_train_data_path(self.fc, self.fj, self.model_kind, self.params_kind)
        df = pd.read_csv(fault_test_file_path, encoding='utf-8', index_col=0)
        data = df.iloc[50000:60000, :].values
        x = data[:, :-1]
        y = data[:, -1]
        mean_std_dict=get_mean_std(self.fc,self.fj,self.model_kind)
        x=data_normalized(x,mean_std_dict)
        xgbr = xgb.XGBRegressor()
        self.logger.info(self.model_path + self.model_file_name)
        xgbr.load_model(self.model_path + self.model_file_name)
        pred = xgbr.predict(x)
        self.save_result_dataframe(y, pred)
        self.set_var(true_v=y, pred_v=pred)
        # self.show_rolling_fig()
        self.show_save_figure(detal_idx=delta_idx)
        t_mean = self.cal_mean(self.true)
        p_mean = self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean, pred_mean=p_mean)

    def params_tuned(self):
        xgbr = xgb.XGBRegressor(objective='reg:squarederror')
        datafile = get_train_data_path(self.fc, self.fj, self.model_kind, self.params_kind)
        params = {'max_depth': [16, 32, 48], 'n_estimators': [128, 256, 512], 'min_child_weight': [3]}
        grid = RandomizedSearchCV(xgbr, params, cv=3, scoring='neg_mean_squared_error', n_iter=6)
        df = pd.read_csv(datafile, encoding='utf-8', index_col=0)
        traindata = df.iloc[100000:250000, :].values
        x = traindata[:, :-1]
        y = traindata[:, -1]
        grid.fit(x, y)
        self.logger.info(grid.best_score_)
        self.logger.info(grid.best_params_)
        self.params = grid.best_params_
        df = pd.DataFrame(list(self.params.items()))
        df.to_csv(self.params_file_path + 'params.csv', encoding='utf-8')

    def train_test_model(self):
        self.xgboostmodel()
        self.test_model()

    def predict(self, x):
        xgbr = xgb.XGBRegressor()
        # self.logger.info(self.model_path + self.model_file_name)
        xgbr.load_model(self.model_path + self.model_file_name)
        pred = xgbr.predict(x)
        return pred


xgbm = XgboostModel(jobname='xgb_model_motor', fc='wfzc', fj='A2', model_kind='motor_temp_3',
                    params_kind='model_params_v1', max_depth=32, n_estimator=300, min_child_weight=1)
# xgbm.params_tuned()
xgbm.xgboostmodel()
xgbm.test_model()
