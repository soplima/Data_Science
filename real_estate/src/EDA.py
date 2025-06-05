#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
# %%
df = pd.read_csv('../data/rs_data.csv')
df.head(10)
#%
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(df[['status']])
encoded_cols = encoder.get_feature_names_out(['status'])
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)
result_df = pd.concat([df, encoded_df], axis=1)
result_df
# %%
result_df = result_df.drop(columns='status')
#%%
result_df = result_df.drop(columns=[col for col in result_df.columns if col.endswith('_encoded')], errors='ignore')

encoder = ce.TargetEncoder(cols=['city','state'])
result_df_encoded = encoder.fit_transform(result_df[['city','state']], result_df['price'])
result_df = pd.concat([result_df, result_df_encoded.add_suffix('_encoded')], axis=1)
result_df
# %%
result_df = result_df.drop(columns=['state', 'city'])
# %%
result_df = result_df.drop(columns='price')
# %%
log_price = np.log(df['price'])
log_price.name = 'price_log'
result_df = pd.concat([result_df, log_price], axis=1)
result_df
# %%
result_df
# %%
plt.figure(figsize=(20,12))
corr_matrix = result_df.corr() > 0.003
corr_matrix['price_log'].sort_values(ascending=False)
sb.heatmap(corr_matrix, annot=True)
plt.show()
# %%
