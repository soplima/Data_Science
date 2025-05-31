#%%
import pandas as pd
from sklearn import random
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from feature_engine import imputation
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
# %%
df = pd.read_csv('../data/rs_data.csv')
df
# %%
features = df.columns.tolist()[:-1]
target = 'price'
#%%
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42)
# %%
print('Taxa Resposta Treino:', y_train.mean())
print('Taxa Resposta Teste:',  y_test.mean())

#%%

num_features = ['bed', 'bath', 'acre_lot', 'house_size']
num_transformer = SimpleImputer(strategy='constant', fill_value=-1)


target_encode_features = ['brokered_by', 'street', 'city', 'state', 'zip_code', 'month_sell']
target_encoder_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('encoder', TargetEncoder(smoothing=10, min_samples_leaf=100))
])
# %%
onehot_features = ['status']
onehot_encoder_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
# %%
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('target', target_encoder_pipeline, target_encode_features),
    ('onehot', onehot_encoder_pipeline, onehot_features)
])
# %%
model = RandomForestRegressor(verbose=2, n_jobs=-1)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])
pipeline
# %%
pipeline.fit(X_train, y_train)

# %%
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('r2 metrics:', r2)
print('mean absolute error:', mae)
print('mean squared Error:', mse)
print('root of mean squared error:', rmse)
# %%
#?First Model
#! r2 metrics: 0.035872927793664355
#! mean absolute error: 360562.18163763377
#! mean squared Error: 2074558367328.424
#! root of mean squared error: 1440332.7279932315