import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from hyperopt import hp, fmin, tpe
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import random

# Set the random seed
random.seed(42)
np.random.seed(42)


class LightGBMClassifier:
    """A class for training and evaluating a LightGBM classifier.

        Parameters
        ----------
        df_train : pandas.DataFrame
            A DataFrame containing the training data. The last column should contain
            the target labels.
        df_test : pandas.DataFrame
            A DataFrame containing the test data. The last column should contain the
            target labels.

        Attributes
        ----------
        df_train : pandas.DataFrame
            A DataFrame containing the training data.
        df_test : pandas.DataFrame
            A DataFrame containing the test data.
        clf : lightgbm.LGBMClassifier
            The LightGBM classifier object.
        params : dict
            A dictionary of hyperparameters for the classifier.
        space : dict
            A dictionary of hyperparameter search space for Bayesian optimization.
        clf_baesian : lightgbm.LGBMClassifier
            The LightGBM classifier object trained using Bayesian optimization.

            Methods
    -------
    split_data()
        Splits the training and test data into feature and target arrays.
    fit(X_train, y_train)
        Fits the classifier to the training data.
    predict(X_test)
        Makes predictions on the test data.
    evaluate(y_test, y_pred)
        Calculates and returns the average F1 score on the test data.
    bayesian_tuning(X_train, y_train, X_test, y_test)
        Performs Bayesian optimization of the classifier's hyperparameters on the
        training data and returns the F1 score on the test data using the tuned
        model.
    compare(X_test, y_test, f1_default, f1_bayesian)
        Compares the models and returns the best model,the confidence level that
         the one is significantly better than the other model and the best models name

    """


    def __init__(self, df_train, df_test):
        """Initializes the LightGBMClassifier object.

        Parameters
        ----------
        df_train : pandas.DataFrame
            A DataFrame containing the training data. The last column should contain
            the target labels.
        df_test : pandas.DataFrame
            A DataFrame containing the test data. The last column should contain the
            target labels.

        Attributes
        ----------
        df_train : pandas.DataFrame
            A DataFrame containing the training data.
        df_test : pandas.DataFrame
            A DataFrame containing the test data.
        clf : lightgbm.LGBMClassifier
            The LightGBM classifier object.
        params : dict
            A dictionary of hyperparameters for the classifier.
        space : dict
            A dictionary of hyperparameter search space for Bayesian optimization.

        """
        self.df_train = df_train
        self.df_test = df_test
        self.clf = lgb.LGBMClassifier()
        self.params = {}
        self.space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0)
        }

    def split_data(self):
        """Splits the training and test data into feature and target.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - X_train : pandas.DataFrame
                A DataFrame containing the feature arrays for the training data.
            - y_train : pandas.Series
                A Series containing the target labels for the training data.
            - X_test : pandas.DataFrame
                A DataFrame containing the feature arrays for the test data.
            - y_test : pandas.Series
                A Series containing the target labels for the test data.

        Raises
        ------
        ValueError
            If df_train or df_test is not a non-empty dataframe, or if the shapes
            of df_train and df_test number of columns is not equal.

        """

        try:
            # Check if df_train and df_test are dataframes and are not empty
            if not isinstance(self.df_train, pd.DataFrame) or self.df_train.empty:
                raise ValueError("df_train is not a non-empty dataframe")
            if not isinstance(self.df_test, pd.DataFrame) or self.df_test.empty:
                raise ValueError("df_test is not a non-empty dataframe")

            # Check if the shapes of df_train and df_test are equal
            if self.df_train.shape[1] != self.df_test.shape[1]:
                raise ValueError("Input data shape is incorrect")

            # Split df_train into X_train and y_train
            X_train = self.df_train.iloc[:, :-1]
            y_train = self.df_train.iloc[:, -1]

            # Split df_test into X_test and y_test
            X_test = self.df_test.iloc[:, :-1]
            y_test = self.df_test.iloc[:, -1]

        except Exception as e:
            raise ValueError("Input data shape is incorrect")

        return X_train, y_train, X_test, y_test

    def fit(self, X_train, y_train):
        """
                Fit the classifier to the training data.

                Parameters
                ----------
                X_train : pandas.DataFrame
                    The training input samples as a DataFrame.
                y_train : pandas.Series
                    The target values as a Series.

                Returns
                -------
                self : object
                    Returns self.
                """
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions on the test data.

        Parameters
        ----------
        X_test : pandas.DataFrame, shape (n_samples, n_features)
            The test data to make predictions on.

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            The predicted target values.
        """
        y_pred = self.clf.predict(X_test)
        return y_pred

    def evaluate(self, y_test, y_pred):
        """
        Calculate the average F1 score for the test data.

        Parameters
        ----------
        y_test : array-like, shape (n_samples,)
            The true target values for the test data.
        y_pred : array-like, shape (n_samples,)
            The predicted target values for the test data.

        Returns
        -------
        f1 : float
            The average F1 score.
        """
        f1 = f1_score(y_test, y_pred, average='macro')
        return f1

    def bayesian_tuning(self, X_train, y_train, X_test, y_test):
        """
                Tune the hyperparameters of the classifier using Bayesian optimization.

                Parameters
                ----------
                X_train : pandas.DataFrame
                    The training input samples as a DataFrame.
                y_train : pandas.Series
                    The target values as a Series.
                X_test : pandas.DataFrame
                    The test input samples as a DataFrame.
                y_test : pandas.Series
                    The true target values as a Series.

                Returns
                -------
                f1 : float
                    The F1 score of the tuned classifier on the test data.
                clf_baesian : lightgbm.LGBMClassifier
                    The tuned classifier.
                """
        def objective(params):
            params['num_leaves'] = int(params['num_leaves'])
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return -f1_score(y_test, y_pred, average='macro')

        best_params = fmin(objective, self.space, algo=tpe.suggest, max_evals=10)
        best_params['num_leaves'] = int(best_params['num_leaves'])
        self.params.update(best_params)
        self.clf_baesian = lgb.LGBMClassifier(**self.params)
        self.clf_baesian.fit(X_train, y_train)

        # Make predictions on the test data using the tuned model
        y_pred = self.clf_baesian.predict(X_test)

        # Calculate and return the F1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        return f1, self.clf_baesian

    def compare(self, X_test, y_test, f1_default, f1_bayesian):
        """
        Compare the default and Bayesian-tuned classifiers using a Mann-Whitney U test.

        Parameters
        ----------
        X_test : pandas.DataFrame
            The test input samples as a DataFrame.
        y_test : pandas.Series
            The true target values as a Series.
        f1_default : float
            The F1 score of the default classifier on the test data.
        f1_bayesian : float
            The F1 score of the Bayesian-tuned classifier on the test data.

        Returns
        -------
        best_model : DecisionTreeClassifier or lightgbm.LGBMClassifier
            The classifier that performed better on the test data.
        confidence_level : float
            The confidence level of the Mann-Whitney U test.
        best_model_name : str
            The name of the classifier that performed better on the test data.
        """

        def generate_f1(model):
            combined_f1_scores = []
            df = X_test.join(y_test)
            subarrays = np.array_split(df, 3)
            for subarray in subarrays:
                _, _, f1_scores, _ = precision_recall_fscore_support(subarray.iloc[:,-1], model.predict(subarray.iloc[:,:-1]), average=None)
                combined_f1_scores.extend(f1_scores)
            return combined_f1_scores

        combined_f1_scores_default = generate_f1(self)
        combined_f1_scores_baesian = generate_f1(self.clf_baesian)
        t_statistic, p_value = mannwhitneyu(combined_f1_scores_default, combined_f1_scores_baesian)

        # Determine which model is better based on the F1 scores
        if f1_default > f1_bayesian:
            best_model = self
            confidence_level = 1 - p_value
            best_model_name = 'Default'
        else:
            best_model = self.clf_baesian
            confidence_level = 1 - p_value
            best_model_name = 'Bayesian'

        return best_model, confidence_level, best_model_name




def main():
    """
        Compare the default and Bayesian-tuned classifiers using a Mann-Whitney U test. The classifiers are trained and evaluated on intersect and combined feature sets.
        The best models are saved as pickle files and their hyperparameters as csv files.
        A DataFrame is created ,saved and showed to the screen with a comparison of all the results obtained.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
    # Load the training and test data into pandas DataFrames
    # Load df_train_scaled_intersect
    with open('df_train_scaled_intersect.pkl', 'rb') as f:
        df_train_scaled_intersect = pickle.load(f)

    # Load df_test_scaled_intersect
    with open('df_test_scaled_intersect.pkl', 'rb') as f:
        df_test_scaled_intersect = pickle.load(f)

    # Load df_train_scaled_combined
    with open('df_train_scaled_combined.pkl', 'rb') as f:
        df_train_scaled_combined = pickle.load(f)

    # Load df_test_scaled_combined
    with open('df_test_scaled_combined.pkl', 'rb') as f:
        df_test_scaled_combined = pickle.load(f)
    # Define data types
    features_types = ['intersect','combined']
    df_comparison = pd.DataFrame(columns = ['Features num', 'F1 Default score', 'F1 Bayesian score', 'Confidence level', 'Best model'])
    for type_ in features_types:
        if type_ == 'intersect':
            # Create an instance of the LightGBMClassifier class
            lgbm_classifier = LightGBMClassifier(df_train_scaled_intersect, df_test_scaled_intersect)
            # Split the data into training and test sets
            X_train, y_train, X_test, y_test = lgbm_classifier.split_data()
        else:
            # Create an instance of the LightGBMClassifier class
            lgbm_classifier = LightGBMClassifier(df_train_scaled_combined, df_test_scaled_combined)
            # Split the data into training and test sets
            X_train, y_train, X_test, y_test = lgbm_classifier.split_data()

        # Fit the classifier to the training data
        lgbm_classifier.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = lgbm_classifier.predict(X_test)

        # Evaluate the predictions and get the F1 score
        f1_default = lgbm_classifier.evaluate(y_test, y_pred)

        # Tune the hyperparameters using Bayesian optimization
        f1_bayesian, baesian_model = lgbm_classifier.bayesian_tuning(X_train, y_train, X_test, y_test)

        # Get the best model, cobfidence level and the best model name
        best_model, confidence_level, best_model_name = lgbm_classifier.compare(X_test, y_test, f1_default, f1_bayesian)

        rows = []
        rows.append({'feature': 'Type', 'value': str(type(best_model))})
        for k, v in best_model.__dict__.items():
            rows.append({'feature': k, 'value': v})

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(rows)

        # Save the model with the features type as a name
        with open(f'Model {type_}.pickle', 'wb') as f:
            pickle.dump(best_model, f)

        # Save the hyperparameters with the features type as a name
        df.to_csv(f'Hyperparameters {type_}.csv')


        df_comparison.loc[type_,'Features num'] = X_train.shape[1]
        df_comparison.loc[type_, 'F1 Default score'] = f'{f1_default:.6f}'
        df_comparison.loc[type_, 'F1 Bayesian score'] = f'{f1_bayesian:.6f}'
        df_comparison.loc[type_, 'Confidence level'] = f'{confidence_level:.3f}'
        df_comparison.loc[type_, 'Best model'] = best_model_name

    print(df_comparison)

    # Save the DataFrame to a file
    df_comparison.to_csv('df_comparison.csv')

    fig, ax = plt.subplots()

    # Use the 'table' function to draw the DataFrame
    table = ax.table(cellText=df_comparison.values, rowLabels=df_comparison.index, colLabels=df_comparison.columns, loc='center')

    # Hide the x and y axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Adjust the layout and font size of the table
    table.set_fontsize(14)
    table.scale(1, 2)

    # Save the figure to a file
    plt.savefig('df_comparison.png')

    # Show the figure
    plt.show()




if __name__ == '__main__':
    main()


