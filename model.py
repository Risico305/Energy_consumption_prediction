import time
import pandas as pd
from keras import Sequential
from keras.layers import GRU, Dense, Dropout
from keras import backend as K
from sklearn import neighbors, metrics
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from attributes import Attributes
from numpy.random import seed


seed(1337)


class RegressionModel:
    def __init__(self, data_path, ablation):
        self.data_path = data_path
        self.attr = Attributes()
        self.ablation = ablation

    def __rmse(self,y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def __mse(self,y_true, y_pred):

        return K.mean(K.square(y_pred - y_true), axis=-1)

    def __r_square(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    def __scaler(self, data):
        scale = StandardScaler()
        return scale.fit_transform(data)

    def prepare_data(self, features):
        dataset = pd.read_csv(self.data_path)
        dataset = dataset.drop(labels='date', axis=1)
        data = dataset[features]
        train = data.sample(frac=0.7, random_state=333)
        test = data.drop(train.index)
        df_train = pd.DataFrame(columns=train.columns, index=train.index)
        df_train[df_train.columns] = self.__scaler(train)
        df_test = pd.DataFrame(columns=test.columns, index=test.index)
        df_test[df_test.columns] = self.__scaler(test)
        train_y = df_train['Appliances']
        test_y = df_test['Appliances']
        train_x = df_train.drop(labels='Appliances', axis=1)
        test_x = df_test.drop(labels='Appliances', axis=1)
        return train_x, test_x, train_y, test_y

    def __train(self, train_x, test_x, train_y, test_y):
        results = list()
        gbr = GradientBoostingRegressor()
        svr = SVR()
        knn = neighbors.KNeighborsRegressor()
        tree = ExtraTreesRegressor()
        models = {'GBR': gbr, 'SVR': svr, 'kNN': knn, 'ETR': tree}
        for name, model in models.items():
            print('model %s training' % name)
            model.fit(train_x, train_y)
            tr_r = metrics.r2_score(train_y, model.predict(train_x))
            te_r = metrics.r2_score(test_y, model.predict(test_x))
            mse = (mean_squared_error(test_y, model.predict(test_x)))
            results.append({name: {'R2_train': tr_r
                , 'R2_test': te_r
                , 'mean-squared_error': mse}})
        return results

    def run(self, _feat=None):
        ablation_stats = []
        feat_sets = {'T_only': self.attr.T_ONLY
            , 'Rh_only': self.attr.RH_ONLY
            , 'conditions_only': attr.COND_ONLY
            , 'rv_only': attr.RV_ONLY
            , 'all_features': attr.ALL_FEATURES}
        start_time = time.time()
        if self.ablation == True:
            for feat_name, feat in feat_sets.items():
                print('Training model with %s ' % feat_name)
                train_x, test_x, train_y, test_y = reg.prepare_data(feat)
                result = self.__train(train_x, test_x, train_y, test_y)
                ablation_stats.append({feat_name: {'result': result}})
            print('Printing feature Ablation test results')
            print(ablation_stats)
        else:
            print('Normal results against the feature choice')
            train_x, test_x, train_y, test_y = reg.prepare_data(_feat)
            print(self.__train(train_x, test_x, train_y, test_y))
        print("\nThis took %s minutes to run !!" % ((time.time() - start_time) / 60))

    def build_correlaions(self):
        dataset = pd.read_csv(self.data_path)
        dataset = dataset.drop(labels='date', axis=1)
        attr = Attributes()
        data = dataset[attr.ALL_FEATURES]
        relation = data.corr()
        masking = np.zeros_like(relation, dtype=np.bool)
        masking[np.triu_indices_from(masking)] = True
        sns.heatmap(relation, annot=True, fmt=".2f", mask=masking)
        plt.xticks(range(len(relation.columns)), relation.columns)
        plt.yticks(range(len(relation.columns)), relation.columns)
        plt.show()
        print(self.__abs_correlations(data, 40))

    def __get_redundant_pairs(self, df):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i + 1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def __abs_correlations(self, df, n=None):
        au_corr = df.corr().abs().unstack()
        labels_to_drop = self.__get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]

    def neural(self, X_Train, X_Test, Y_Train, Y_Test):
        model = Sequential()
        model.add((GRU(  units=100
                       , activation='relu'
                       , recurrent_dropout=0.2
                       , input_shape=(X_Train.shape[1]
                       , X_Train.shape[2]))))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='rmsprop', metrics=[self.__r_square, self.__rmse])
        model.fit(X_Train, Y_Train, epochs=100, batch_size=10, validation_data=(X_Test, Y_Test), verbose=2,
                  shuffle=True)

        test_pred = model.predict(X_Test)
        train_pred = model.predict(X_Train)

        print("The R2 score on the Train set is:\t{:0.3f}".format(self.__r_square(Y_Train, train_pred)))
        print("The R2 score on the Test set is:\t{:0.3f}".format(self.__r_square(Y_Test, test_pred)))


if __name__ == '__main__':
    attr = Attributes()
    reg = RegressionModel(data_path=r'data/data.csv', ablation=False)
    reg.run(attr.BEST_FIT)
    reg.build_correlaions()
