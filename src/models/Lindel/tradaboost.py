"""
    description:
        Scikit-learn compatible implementation of the 
        TrAdaBoost algorithm from the ICML'07 paper
        "Boosting for Transfer Learning"
    author: Suraj Iyer
"""
import numpy as np
import multiprocessing as mp
import os, json
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Flatten
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
from keras import backend as K
from tensorflow.keras.models import clone_model
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.spatial import distance
from scipy.stats import pearsonr
import tensorflow_probability as tfp

class TrAdaBoostClassifier():

    def __init__(self, n_iters = 10, verbose=True, input_shape=None, output_size=None):
        assert isinstance(input_shape, int)
        assert isinstance(output_size, int)

        self.input_shape = input_shape
        self.output_size = output_size
        self.n_iters = n_iters
        self.verbose = verbose
        self.estimators_ = []
        self.num_folds = 5
        self.random_state = 42

    def _get_model(self, reg):
        model = Sequential()
        model.add(Dense(self.output_size,  activation='softmax', input_shape=(self.input_shape,), kernel_regularizer=reg))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
        return model

    def _normalise_weights(self, weights):
        return weights/np.sum(weights)

    # calculate the MSE per sample
    def _mse(self, x, y, w=None):
        if w is None:
            w = np.ones(y.shape[0])
        return np.dot(w, ((x-y)**2)).sum()/w.sum()

    def _mean_absolute_error(self, x, y):
        return np.abs(x-y).mean(axis=1)

    # calculate jenson shannon distance
    def _jenson_shannon(self, x, y):
        nrows = x.shape[0]
        d = []
        for i in np.arange(nrows):
            d.append(distance.jensenshannon(x[i,:], y[i,:], 2))
        return np.array(d)

    def _corr(self, x, y):
        nrows = x.shape[0]
        d = []
        for i in np.arange(nrows):
            d.append(pearsonr(x[i,:], y[i,:])[0])
        return np.array(d)

    # get the MSE per sample and take the weighted average
    def _calculate_error(self, x, y, weights=None):
        if weights is None: 
            weights = 1.
        else:
            check_consistent_length(x, y, weights)
        err = 0.5 * self._jenson_shannon(x, y)
        return np.dot(weights, err)/np.sum(weights)

    def _train_model(self, x_train, y_train, w_train, reg, x_valid=None, y_valid=None, w_valid=None):
        np.random.seed(0)
        model = self._get_model(reg)
        if x_valid is not None and y_valid is not None:
            model.fit(x_train, y_train, sample_weight=w_train, epochs=200, validation_data=(x_valid, y_valid, w_valid), 
                    callbacks=[EarlyStopping(patience=1)], verbose=0)
            y_hat = model.predict(x_valid)
            return model, self._mse(y_hat, y_valid, w_valid)
        else:
            model.fit(x_train, y_train, epochs=200, verbose=0)
            return model, None

    def _benchmark_models(self, X, Y, weights, lambdas, split):
        errors_l1 = []
        errors_l2 = []

        x_train = X[split[0], :]
        y_train = Y[split[0], :]
        w_train = weights[split[0]]
        x_valid = X[split[1], :]
        y_valid = Y[split[1], :]
        w_valid = weights[split[1]]

        for l in lambdas:
            _, err = self._train_model(x_train, y_train, w_train, l1(l), x_valid=x_valid, y_valid=y_valid, w_valid=w_valid)
            errors_l1.append(err)
            _, err = self._train_model(x_train, y_train, w_train, l2(l), x_valid=x_valid, y_valid=y_valid, w_valid=w_valid)
            errors_l2.append(err)

        return errors_l1, errors_l2

    def _average_fold_errors(self, errors):
        means = np.mean(np.array(errors), axis=0)
        min_mean = np.min(means)
        min_index = np.argmin(means)
        return min_mean, min_index

    def _fit(self, X, y, weights):
        lambdas = 10 ** np.arange(-10, -1, 0.2) # for activation function test
        kf = KFold(n_splits=self.num_folds, random_state=self.random_state, shuffle=True)
        folds = list(kf.split(X, y))

        errors_l1 = []
        errors_l2 = []
        def log_results(results):
            e1, e2 = results
            errors_l1.append(e1)
            errors_l2.append(e2)
        pool = mp.Pool(5)
        for fold in folds:
            # log_results(self._benchmark_models(X, y, weights, lambdas, fold))
            pool.apply_async(self._benchmark_models, args=(X, y, weights, lambdas, fold), callback = log_results)
        pool.close()
        pool.join()

        # find best params and train final model
        min_mean_l1, min_idx_l1 = self._average_fold_errors(errors_l1)
        min_mean_l2, min_idx_l2 = self._average_fold_errors(errors_l2)

        best_lambda, reg = None, None
        if min_mean_l1 < min_mean_l2:
            best_lambda = lambdas[min_idx_l1]
            reg = l1(best_lambda)
        else:
            best_lambda = lambdas[min_idx_l2]
            reg = l2(best_lambda)
        m, _ = self._train_model(X, y, weights, reg)
        return m

    # in TrAdaBoost notation, source = diff_distribution and target = same_distribution
    def fit(self, X_source, y_source, X_target, y_target):
        X = np.append(X_source, X_target, axis=0)
        y = np.append(y_source, y_target, axis=0)
        n = y_source.shape[0]
        m = y_target.shape[0]
        n_samples = n + m
        weights = np.ones(n_samples)
        if self.verbose:
            print("Training on {} Source samples, {} Target Samples".format(n, m))

        # 0 = source, 1 = target
        mask = np.append(np.zeros(n), np.ones(m)) == 1

        P = np.empty((self.n_iters, n_samples))

        # initialise error vector
        error = np.empty(self.n_iters)
        beta0 = 1 / (1 + np.sqrt(2 * np.log(n / self.n_iters)))
        beta = np.empty(self.n_iters)

        # initialise estimator list for each iteration
        estimators = []

        for t in np.arange(self.n_iters):
            # call Learner over combined dataset of source and target data
            P[t] = weights/np.median(weights)

            # include cross validation steps
            est = self._fit(X, y, P[t])

            if self.verbose:
                print("Proportional weight of source samples: {}".format(np.sum(P[t][~mask])/np.sum(P[t])))
                print("Proportional weight of target samples: {}".format(np.sum(P[t][mask])/np.sum(P[t])))

            # calculate error over target data
            y_same_pred = est.predict(X[mask,:])
            error[t] = self._calculate_error(y[mask, :], y_same_pred, weights=weights[mask])

            if error[t] > 0.5 or error[t] == 0:
                # if the error is 0 or > 0.5, stop updating weights
                self.n_iters = t
                beta = beta[:t]

                if self.verbose:
                    if error[t] > 0.5:
                        print("Early stopping because error: {} > 0.5".format(error[t]))
                    else:
                        print("Early stopping because error is zero.")
                break

            if self.verbose:
                print('Weighted error_{}: {}'.format(t, error[t]))
                unweighted_js = np.nanmean(self._jenson_shannon(y[mask, :], y_same_pred))
                print('Jenson Shannon_{}: {}'.format(t, unweighted_js))
                y_diff_pred = est.predict(X[~mask, :])
                d_unweighted_corr = np.nanmean(self._corr(y[~mask, :], y_diff_pred))
                print('Source Domain Pearson Corr_{}: {}'.format(t, d_unweighted_corr))
                unweighted_corr = np.nanmean(self._corr(y[mask, :], y_same_pred))
                print('Target Domain Pearson Corr_{}: {}'.format(t, unweighted_corr))

            # set beta[t]
            beta[t] = error[t] / (1 - error[t])
            if self.verbose:
                print('beta_{}: {}'.format(t, beta[t]))

            # Update the new weight vector
            if t < self.n_iters - 1:
                y_diff_pred = est.predict(X[~mask, :])
                weights[~mask] = weights[~mask] * (beta0 ** (self._calculate_error(y[~mask, :], y_diff_pred)))
                weights[mask] = weights[mask] * (beta[t] ** -(self._calculate_error(y[mask, :], y_same_pred)))

            # add estimator
            estimators.append(est)

        if self.verbose:
            print("Number of iterations run: {}".format(self.n_iters))

        self.fitted_ = True
        self.diff_sample_weights_ = weights
        self.beta_ = beta
        self.estimators_ = estimators
        self.classes_ = getattr(estimators[0], 'classes_', None)

        return self

    def _predict_one(self, x):
        """
        Output the hypothesis for a single instance
        :param x: array-like
            target label of a single instance from each iteration in order
        :return: 0 or 1
        """
        x, N = check_array(x, ensure_2d=False), self.n_iters
        # replace 0 by 1 to avoid zero division and remove it from the product
        beta = np.array([self.beta_[t] if self.beta_[t] != 0 else 1 for t in range(len(self.estimators_))])
        beta = np.log(1/beta)
        s = np.dot(beta, x)
        y = s/sum(s)
        return y
    
    def predict(self, X, domain_column=None):
        check_is_fitted(self, 'fitted_')
        y_pred_list = np.array([est.predict(X)[0] for est in self.estimators_])
        y_pred = np.array(self._predict_one(y_pred_list))
        y_pred = y_pred.reshape((1,len(y_pred)))
        return y_pred
        # return self.estimators_[0].predict(X)

    def save(self, d, f):
        models_dir = d + f + "/models/"
        os.makedirs(models_dir, exist_ok=True)
        for i, e in enumerate(self.estimators_):
            e.save(models_dir + str(i) + ".h5")
        params = {
            "input_shape": self.input_shape,
            "output_size": self.output_size,
            "n_iters": int(self.n_iters),
            "verbose": self.verbose,
            "beta_": self.beta_.tolist(),
            "fitted_": self.fitted_,
            "classes_": self.classes_
        }
        with open(d + f + '/params.json', 'w') as fp:
            json.dump(params, fp)
        
    @staticmethod
    def load(d, f):
        with open(d + f + '/params.json') as fp:
            data = json.load(fp)
            input_shape = data["input_shape"]
            output_size = data["output_size"]
            n_iters = data["n_iters"]
            verbose = data["verbose"]

            clf = TrAdaBoostClassifier(n_iters=n_iters, verbose=verbose, input_shape=input_shape, output_size=output_size)
            clf.beta_ = data["beta_"]
            clf.fitted_ = data["fitted_"]
            clf.classes_ = data["classes_"]

            models_dir = d + f + "/models/"
            for i in range(clf.n_iters):
                clf.estimators_.append(load_model(models_dir + str(i) + ".h5"))
        return clf
