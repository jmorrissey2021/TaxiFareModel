# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        self.pipeline = self.run()

    def run(self):
        """set and train the pipeline"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, [
                                    "pickup_latitude",
                                    "pickup_longitude",
                                    'dropoff_latitude',
                                    'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])],
            remainder="drop")

        pipe = Pipeline([('preproc', preproc_pipe),
                        ('linear_model', LinearRegression())])

        pipe.fit(self.X, self.y)

        return pipe

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.rmse = rmse

        return rmse


if __name__ == "__main__":
    # get data
    data = get_data()

    # clean data
    data = clean_data(data)

    # set X and y
    y = data.pop("fare_amount")
    X = data

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train
    model = Trainer(X_train, y_train)
    model.set_pipeline()

    # evaluate
    model.evaluate(X_test, y_test)

    print(f'RMSE is {model.rmse}')
