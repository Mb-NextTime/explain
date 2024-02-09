import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class DatasetExplorer:
    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self._data = data
        self._target = target

    def _corr(self):
        return self._data.corr(), self._data.corrwith(self._target.squeeze())

    def display(self):
        features_corr, target_corr = self._corr()
        sns.heatmap(features_corr)
        plt.title('Features collinearity')
        plt.show()

        target_corr.plot(kind='bar', title='Correlation with target')
        plt.show()


class VirtualDataset:
    def __init__(self, model, eval_metric, n_iter=5,):
        self._Model = type(model)
        self._params = model.get_params() | {'silent': True}
        self._metric = eval_metric
        self._max_iter = n_iter
        self._sizes = np.linspace(0.2, 1, self._max_iter)

    def fit(self, X, y, density=20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        sizes = []
        scores = []
        for size in self._sizes:
            for _ in range(density):
                if size != 1:
                    X, _, y, _ = train_test_split(
                        X_train,
                        y_train,
                        test_size=1-size,
                        stratify=y_train
                    )
                else:
                    X, y = X_train, y_train
                model = self._Model(**self._params)
                model.fit(X, y)
                score = self._metric(model.predict(X_test), y_test)
                sizes.append(len(X))
                scores.append(score)

        return {'DatasetSize': sizes, 'Score': scores}

    def display(self, X, y, density=20):
        vds_res = self.fit(X.to_numpy(), y.to_numpy(), density=density)
        sns.violinplot(vds_res, x='DatasetSize', y='Score')
