import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ModelEvaluator:
    def __init__(self, model):
        self._model = model

    def confusion_matrix(self, X_test, y_test):
        confusion = confusion_matrix(y_test, self._model.predict(X_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion)
        disp.plot()
        plt.show()

    def calc_metric_folded(self, X_test, y_test, metric, n_folds=5):
        scores = []
        for X_test_fold, y_test_fold in zip(np.array_split(X_test, n_folds), np.array_split(y_test, n_folds)):
            scores.append(metric(y_test_fold, self._model.predict(X_test_fold)))
        return np.array(scores)


class ModelExplainer:
    def __init__(self, model, data):
        self._model = model
        self._data = data

    def plot_shap(self, max_display=10):
        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer(self._data)

        shap.plots.beeswarm(shap_values, max_display=max_display)

    def plot_ICE(self, feature_indexes=None):
        if feature_indexes is None:
            feature_indexes = range(self._data.shape[1])
        for i in feature_indexes:
            PartialDependenceDisplay.from_estimator(self._model, self._data, [i], kind='both')
            plt.show()

    def lime(self, *args):
        explainer = LimeTabularExplainer(
            self._data.to_numpy(),
            feature_names=self._data.columns.values,
            discretize_continuous=True
        )
        ...
        # TODO: finish LIME & add other methods

