import unittest
import Model_training_and_evaluation as lgb
import pandas as pd
from math import isclose

df_train = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [8, 10, 12], 'target': [1, 2, 3]})
df_test = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [8, 10, 12], 'target': [1, 2, 3]})

class Test_LGBM(unittest.TestCase):

    def test_split_data(self):
        # Create an instance of the LightGBMClassifier class
        lgbm_classifier = lgb.LightGBMClassifier(df_train, df_test)

        # Split the data into training and test sets
        X_train, y_train, X_test, y_test = lgbm_classifier.split_data()

        # Check that the data types are correct
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)

        # Check that the shapes of the feature arrays and target labels are correct
        self.assertEqual(X_train.shape[1], df_train.shape[1] - 1)
        self.assertEqual(y_train.shape[0], df_train.shape[0])
        self.assertEqual(X_test.shape[1], df_test.shape[1] - 1)
        self.assertEqual(y_test.shape[0], df_test.shape[0])


    def test_fit(self):
        # Create an instance of the LightGBMClassifier class
        lgbm_classifier = lgb.LightGBMClassifier(df_train, df_test)

        # Split the data into training and test sets
        X_train, y_train, _, _  = lgbm_classifier.split_data()

        # Fit the classifier to the training data
        lgbm_classifier.fit(X_train, y_train)

        # Check that the classifier object has been updated with the training data
        self.assertIsNotNone(lgbm_classifier.clf)
        self.assertTrue(hasattr(lgbm_classifier, 'clf'))

    def test_predict(self):
        # Create an instance of the LightGBMClassifier class
        lgbm_classifier = lgb.LightGBMClassifier(df_train, df_test)

        # Split the data into training and test sets
        X_train, y_train, X_test, y_test = lgbm_classifier.split_data()

        # Fit the classifier to the training data
        lgbm_classifier.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = lgbm_classifier.predict(X_test)

        # Check that the shape of the predictions is correct
        self.assertEqual(y_pred.shape[0], y_test.shape[0])

    def test_evaluate(self):
        # Create an instance of the LightGBMClassifier class
        lgbm_classifier = lgb.LightGBMClassifier(df_train, df_test)

        # Generate some test data
        y_test = [1, 2, 3, 1, 2]
        y_pred = [1, 2, 3, 1, 2]

        # Calculate the average F1 score
        f1 = lgbm_classifier.evaluate(y_test, y_pred)

        # Check that the returned value is correct
        self.assertAlmostEqual(f1, 0.99, places=1)


    def test_bayesian_tuning(self):
        # Create an instance of the LightGBMClassifier class
        lgbm_classifier = lgb.LightGBMClassifier(df_train, df_test)

        # Split the data into training and test sets
        X_train, y_train, X_test, y_test = lgbm_classifier.split_data()

        # Call bayesian_tuning method
        f1_besian, clf_baesian = lgbm_classifier.bayesian_tuning(X_train, y_train, X_test, y_test)

        # Verify output
        self.assertTrue(f1_besian > 0 and f1_besian < 1)
        self.assertIsNotNone(clf_baesian)

    def compare(self):

        # Calculate F1 scores
        f1_default = 0.8
        f1_bayesian = 0.7

        # Create an instance of the LightGBMClassifier class
        lgbm_classifier = lgb.LightGBMClassifier(df_train, df_test)

        # Split the data into training and test sets
        _, _, X_test, y_test = lgbm_classifier.split_data()


        # Call compare method
        best_model, confidence_level, best_model_name = lgbm_classifier.compare(X_test, y_test, f1_default, f1_bayesian)

        self.assertTrue(confidence_level >= 0 and confidence_level < 1)
        self.assertEqual(best_model_name, 'Default')
        self.assertIsNotNone(best_model)




if __name__ == '__main__':
    unittest.main()

