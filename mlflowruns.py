import sqlite3
import pandas as pd
import dill

# Connect to SQLite database
conn = sqlite3.connect('videogamessales.db')

# SQL query to fetch data using JOINs
sql_query = '''
    SELECT g.Name, g.Platform, g.Year_of_Release, g.Genre, g.Publisher, g.Rating,
       s.NA_Sales, s.EU_Sales, s.JP_Sales, s.Other_Sales, s.Global_Sales,
       u.User_Score, u.User_Count
FROM Games g
LEFT JOIN Sales s ON g.Name = s.Name
LEFT JOIN User_Reviews u ON g.Name = u.Name
'''
# Execute the SQL query and fetch data into a Pandas DataFrame
df = pd.read_sql_query(sql_query, conn)
# type cast year of release to int if it is not null and null if empty
df['Year_of_Release'] = df['Year_of_Release'].apply(lambda x: int(x) if x else None)
# Close the connection
conn.close()

# Train test split the data for target variable Global_Sales

from sklearn.model_selection import train_test_split

df.reset_index(inplace=True,drop=True)

# Drop the Name and index columns

df = df.drop(columns=['Name'])

# Split dataset into train and test dataframes without extracting the target variable

train, test = train_test_split(df, test_size=0.25, random_state=42)

train.head()

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Filter warnings
import warnings
warnings.filterwarnings("ignore")


class Preprocessor(BaseEstimator, TransformerMixin):

    numerical_columns = ['Year_of_Release', 'NA_Sales',
       'EU_Sales', 'JP_Sales', 'Other_Sales', 'User_Score',
       'User_Count']

    # Columns to be label encoded
    categorical_columns = [
        'Platform', 
        'Publisher'
    ]

    # Columns to be one hot encoded
    one_hot_columns = [ "Genre", "Rating" ]
    

    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.onehot = OneHotEncoder(handle_unknown='ignore')
        self.label = LabelEncoder()

    def fit(self, X, y=None):

        self.imputer.fit(X[self.numerical_columns])
        self.scaler.fit(X[self.numerical_columns])
        self.onehot.fit(X[self.one_hot_columns])
        return self

    def compute_days_since_release(self,year):
        return 2024 - year

    def transform(self, X, y=None):
        
        imputed_cols = self.imputer.transform(X[self.numerical_columns])
        onehot_cols = self.onehot.transform(X[self.one_hot_columns])
        
        transformed_df = X.copy()
         
        # Apply transformed columns
        transformed_df[self.numerical_columns] = imputed_cols
        transformed_df[self.numerical_columns] = self.scaler.transform(transformed_df[self.numerical_columns])
        

        transformed_df['Platform'] = self.label.fit_transform(X['Platform'])
        transformed_df['Rating'] = self.label.fit_transform(X['Rating'])
        transformed_df['Publisher'] = self.label.fit_transform(X['Publisher'])        
        
       
        # Impute Rating with RP
        transformed_df['Rating'] = transformed_df['Rating'].fillna('RP')

        # Drop existing categorical columns and replace with one hot equivalent
        transformed_df = transformed_df.drop(self.one_hot_columns, axis=1) 
        transformed_df[self.onehot.get_feature_names_out()] = onehot_cols.toarray().astype(int)

        # FEATURE ENGINEERING
        
        # Feature Engnieer Days Since release
        transformed_df['Days_Since_Release'] = transformed_df['Year_of_Release'].apply(self.compute_days_since_release)

        return transformed_df

# Implement the preprocessor on the training data and display the first five rows of the transformed data.
preprocessor = Preprocessor()
preprocessor.fit(train)
train_fixed = preprocessor.transform(train)
train_fixed.head()

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

rfr = make_pipeline(Preprocessor(), RandomForestRegressor(n_estimators=50))
gbr = make_pipeline(Preprocessor(), GradientBoostingRegressor(n_estimators=50))
abr = make_pipeline(Preprocessor(), AdaBoostRegressor(n_estimators=50))

X_train = train.drop(columns=['Global_Sales'])
y_train = train['Global_Sales']


rfr.fit(X_train, y_train)
gbr.fit(X_train, y_train)
abr.fit(X_train, y_train)

y_pred_rfr = rfr.predict(X_train)
y_pred_gbr = gbr.predict(X_train)
y_pred_abr = abr.predict(X_train)

rmse_rfr = root_mean_squared_error(y_train, y_pred_rfr)
rmse_gbr = root_mean_squared_error(y_train, y_pred_gbr)
rmse_abr = root_mean_squared_error(y_train, y_pred_abr)

mae_rfr = mean_absolute_error(y_train, y_pred_rfr)
mae_gbr = mean_absolute_error(y_train, y_pred_gbr)
mae_abr = mean_absolute_error(y_train, y_pred_abr)

accuracy_rfr = rfr.score(X_train, y_train)
accuracy_gbr = gbr.score(X_train, y_train)
accuracy_abr = abr.score(X_train, y_train)

print(f"Random Forest Regressor RMSE: {rmse_rfr}, MAE: {mae_rfr}, Accuracy: {accuracy_rfr}")
print(f"Gradient Boosting Regressor RMSE: {rmse_gbr}, MAE: {mae_gbr}, Accuracy: {accuracy_gbr}")
print(f"Ada Boost Regressor RMSE: {rmse_abr}, MAE: {mae_abr}, Accuracy: {accuracy_abr}")


import mlflow
import os
from mlflow.models import infer_signature

mlflow_exp_name = 'Video_Games_Sales_Model'
# os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:8080'

# Create a new MLflow Experiment
mlflow.set_experiment(mlflow_exp_name)

# os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/Zorrat/EAS-503-Group7-ML-Project.mlflow'
# os.environ["MLFLOW_TRACKING_USERNAME"] = "Zorrat"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "df70bc1aa2b8de0b1f0ce7fc8c6644604a677f92"
# MLFLOW_TRACKING_URI=https://dagshub.com/Zorrat/EAS-503-Group7-ML-Project.mlflow \
# MLFLOW_TRACKING_USERNAME=Zorrat \
# MLFLOW_TRACKING_PASSWORD=df70bc1aa2b8de0b1f0ce7fc8c6644604a677f92 \
# python script.py

def train_and_log(model, name):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        rmse = root_mean_squared_error(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)
        accuracy = model.score(X_train, y_train)
        signature = infer_signature(X_train, model.predict(X_train))

        
        # Log parameters, metrics, and model
        mlflow.log_params(model.named_steps)
        mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "Accuracy": accuracy})
        mlflow.sklearn.log_model(model, name)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="Video_Games_Sales_Model",
            signature=signature,
            input_example=preprocessor.transform(X_train),
            registered_model_name=name,
        )


rfr = make_pipeline(Preprocessor(), RandomForestRegressor(n_estimators=50))
gbr = make_pipeline(Preprocessor(), GradientBoostingRegressor(n_estimators=50))
abr = make_pipeline(Preprocessor(), AdaBoostRegressor(n_estimators=50))

# Train models with different parameters and log with MLflow

rfr_learning_rate_model = RandomForestRegressor(n_estimators=75, learning_rate=0.5)
gbr_learning_rate_model = GradientBoostingRegressor(n_estimators=75, learning_rate=0.5)
abr_learning_rate_model = AdaBoostRegressor(n_estimators=75, learning_rate=0.5)




train_and_log(rfr, "Random Forest Regressor 50 Estimators")
train_and_log(gbr, "Gradient Boosting Regressor 50 Estimators")
train_and_log(abr, "Ada Boost Regressor 50 Estimators")

# Print out metrics if needed (optional)
# These will also be available in the MLflow UI
print("Models trained and logged successfully.")

# Grid Search with Cross Validation

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn

# Define the parameter grids for each model
rfr_param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20]
}

gbr_param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.5],
    'model__loss': ['ls', 'lad', 'huber']
}

abr_param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.5],
    'model__loss': ['linear', 'square', 'exponential']
}

# Create pipelines for each model
rfr_pipeline = Pipeline([
    ('preprocessor', Preprocessor()),
    ('model', RandomForestRegressor())
])

gbr_pipeline = Pipeline([
    ('preprocessor', Preprocessor()),
    ('model', GradientBoostingRegressor())
])

abr_pipeline = Pipeline([
    ('preprocessor', Preprocessor()),
    ('model', AdaBoostRegressor())
])

# Perform GridSearchCV
with mlflow.start_run():
    rfr_grid_search = GridSearchCV(rfr_pipeline, rfr_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
    rfr_grid_search.fit(X_train, y_train)
    mlflow.sklearn.log_model(rfr_grid_search.best_estimator_,"Random Forest Regressor GridSearchCV" )
    signature = infer_signature(X_train, rfr_grid_search.predict(X_train))
    model_info = mlflow.sklearn.log_model(
            sk_model=rfr_grid_search.best_estimator_,
            artifact_path="Video_Games_Sales_Model",
            signature=signature,
            input_example=preprocessor.transform(X_train),
            registered_model_name="Random Forest Regressor GridSearchCV",
        )
    
with mlflow.start_run():
    gbr_grid_search = GridSearchCV(gbr_pipeline, gbr_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
    gbr_grid_search.fit(X_train, y_train)
    mlflow.sklearn.log_model(gbr_grid_search.best_estimator_, "Gradient Boosting Regressor GridSearchCV")
    signature = infer_signature(X_train, gbr_grid_search.predict(X_train))
    model_info = mlflow.sklearn.log_model(
            sk_model=gbr_grid_search.best_estimator_,
            artifact_path="Video_Games_Sales_Model",
            signature=signature,
            input_example=preprocessor.transform(X_train),
            registered_model_name="Gradient Boosting Regressor GridSearchCV",
        )

with mlflow.start_run():
    abr_grid_search = GridSearchCV(abr_pipeline, abr_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
    abr_grid_search.fit(X_train, y_train)
    mlflow.sklearn.log_model(abr_grid_search.best_estimator_, "Ada Boost Regressor GridSearchCV")
    signature = infer_signature(X_train, abr_grid_search.predict(X_train))
    model_info = mlflow.sklearn.log_model(
            sk_model=abr_grid_search.best_estimator_,
            artifact_path="Video_Games_Sales_Model",
            signature=signature,
            input_example=preprocessor.transform(X_train),
            registered_model_name="Ada Boost Regressor GridSearchCV",
        )

# Print the best parameters and best scores
print("Random Forest Regressor Best Params:", rfr_grid_search.best_params_)
print("Random Forest Regressor Best Score:", -rfr_grid_search.best_score_)
print()
print("Gradient Boosting Regressor Best Params:", gbr_grid_search.best_params_)
print("Gradient Boosting Regressor Best Score:", -gbr_grid_search.best_score_)
print()
print("Ada Boost Regressor Best Params:", abr_grid_search.best_params_)
print("Ada Boost Regressor Best Score:", -abr_grid_search.best_score_)

print("GridSearchCV completed and models logged with MLflow.")
 

rfr_grid_search = GridSearchCV(rfr_pipeline, rfr_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
gbr_grid_search = GridSearchCV(gbr_pipeline, gbr_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
abr_grid_search = GridSearchCV(abr_pipeline, abr_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)

# Fit the grid search models
rfr_grid_search.fit(X_train, y_train)
gbr_grid_search.fit(X_train, y_train)
abr_grid_search.fit(X_train, y_train)

# Save the best models along with pipelines
# Save the best models along with pipelines
with open('rfr_model_best.pkl', 'wb') as rfr_file:
    dill.dump((rfr_grid_search.best_estimator_, rfr_pipeline), rfr_file)

with open('gbr_model_best.pkl', 'wb') as gbr_file:
    dill.dump((gbr_grid_search.best_estimator_, gbr_pipeline), gbr_file)

with open('abr_model_best.pkl', 'wb') as abr_file:
    dill.dump((abr_grid_search.best_estimator_, abr_pipeline), abr_file)
print("Models saved successfully.")