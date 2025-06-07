#%%
import pandas as pd
from sklearn import random
from sklearn import model_selection
from feature_engine import imputation
from sklearn.model_selection import RandomizedSearchCV
from sklearn import pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
import numpy as np
from tqdm import tqdm

#%%
df = pd.read_csv('../data/test.csv')
df_price = pd.read_csv('../data/rs_data.csv')
# %%
df['price'] = df_price['price'].values
#%%
df = df[df['price'] > 0].copy()
df['price_log'] = np.log(df['price'])
#%%
df = df.drop(columns='Unnamed: 0')
# %%
features = ['bed', 'bath', 'acre_lot', 'street', 'zip_code',
       'status_for_sale', 'status_ready_to_build',
       'status_sold', 'city_encoded', 'state_encoded']
target = 'price_log'

# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42) 
# %%
print('taxa resposta y_teste', y_test.mean())
print('taxa resposta y_train', y_train.mean())

# %%
X_train.isna().sum()
# %%
imputation_max = imputation.ArbitraryNumberImputer(arbitrary_number=-1,
                                                   variables=['bed', 'bath', 'acre_lot',
                                                              'street', 'zip_code'])  
  
pipe = pipeline.Pipeline([
    ('imput', imputation_max),
    ('model', RandomForestRegressor(random_state=42, max_samples=0.5))
])
  
              
params = {
    'model__n_estimators': [50, 100],
    'model__min_samples_leaf': [10, 100],
    'model__max_depth': [10, 20],
    'model__max_features': ['sqrt', 0.3]
}

class TqdmSearchCV(RandomizedSearchCV):
    def fit(self, X, y=None, **fit_params):
        total = self.n_iter * self.cv
        with tqdm(total=total, desc="RandomizedSearchCV Progress") as pbar:
            self._pbar = pbar
            return super().fit(X, y, **fit_params)

    def _run_search(self, evaluate_candidates):

            def evaluate_candidates_with_progress(candidate_params):
                results = evaluate_candidates(candidate_params)
                self._pbar.update(len(candidate_params))
                return results

            return super()._run_search(evaluate_candidates_with_progress)


# Initialize search
search = TqdmSearchCV(
    estimator=pipe,
    param_distributions=params,
    n_iter=6,
    scoring='neg_mean_squared_error',
    n_jobs=1,
    cv=3,
    random_state=42,
    verbose=0
)

search.fit(X_train, y_train)

print("Melhores parâmetros:", search.best_params_)
print("Melhor score (neg MSE):", search.best_score_)
#%%
y_test_pred = search.predict(X_test)

r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = np.exp(mae)    

print('r2 metrics:', r2)
print('mean absolute error em $:', mae)
print('mean squared Error:', mse)
print('root of mean squared error:', rmse)
# %%
# bath                     0.406812
# city_encoded             0.406795
# state_encoded            0.334616
# bed                      0.275859
# zip_code                 0.110305
# status_sold              0.087587
# status_ready_to_build    0.047870
# acre_lot                 0.009904
# status_for_sale         -0.097359
# street                  -0.143355


#! Melhores parâmetros: {'model__n_estimators': 100, 
#! 'model__min_samples_leaf': 10, 
#! 'model__max_features': 0.3, 'model__max_depth': 20}
#! Melhor score (neg MSE): -0.3441161774225743

#! r2 metrics: 0.7484303361720022
#! mean absolute error em $: 1.448312148228779
#! mean squared Error: 0.3395497808835535
#! root of mean squared error: 0.5827090018899257

#%%
# Previsão (log) e real (log)
y_test_pred_log = search.predict(X_test)
y_test_real = np.exp(y_test)
y_pred_real = np.exp(y_test_pred_log)

# DataFrame com erros
erro_df = X_test.copy()
erro_df['real_price'] = y_test_real
erro_df['predicted_price'] = y_pred_real
erro_df['abs_error'] = np.abs(erro_df['real_price'] - erro_df['predicted_price'])
erro_df['pct_error'] = erro_df['abs_error'] / erro_df['real_price']
#%%
# Top 10 imóveis com maior erro absoluto
erro_df.sort_values(by='abs_error', ascending=False).head(10)
# Top 10 com maior erro percentual
erro_df.sort_values(by='pct_error', ascending=False).head(10)
#%%
erro_df['faixa_preco'] = pd.cut(erro_df['real_price'],
                                bins=[0, 100000, 200000, 500000, 1000000, 5000000],
                                labels=['até 100k', '100k-200k', '200k-500k', '500k-1M', '1M+'])

erro_por_faixa = erro_df.groupby('faixa_preco')['abs_error'].mean()
print(erro_por_faixa)

# %%
erro_por_faixa_pct = erro_df.groupby('faixa_preco')['pct_error'].mean()
print(erro_por_faixa_pct)

# %%
import matplotlib.pyplot as plt
erro_por_faixa_pct = erro_por_faixa_pct.sort_index()

# Plot
plt.figure(figsize=(8, 5))
plt.bar(erro_por_faixa_pct.index, erro_por_faixa_pct.values, color='steelblue')
plt.yscale('log')  # Escala logarítmica
plt.title('Erro Percentual Médio por Faixa de Preço (Escala Log)')
plt.ylabel('Erro Percentual Médio (%) [log]')
plt.xlabel('Faixa de Preço')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
df.head()
# %%
bins = [0, 100000, 200000, 500000, 1000000, float('inf')]
labels = ['até 100k', '100k-200k', '200k-500k', '500k-1M', '1M+']
df['faixa_preco'] = pd.cut(df['price'], bins=bins, labels=labels)
contagem_por_faixa = df['faixa_preco'].value_counts().sort_index()

contagem_por_faixa.plot(kind='bar', color='coral', figsize=(8, 5))
plt.title('Quantidade de Imóveis por Faixa de Preço')
plt.ylabel('Quantidade')
plt.xlabel('Faixa de Preço')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(contagem_por_faixa)
# %%
import seaborn as sns
df['log_price'] = np.log1p(df['price'])  

plt.figure(figsize=(10, 6))
sns.histplot(df['log_price'], bins=50, kde=True, stat='density', color='coral')

plt.title('Distribuição Logarítmica dos Preços dos Imóveis')
plt.xlabel('log(Preço + 1)')
plt.ylabel('Densidade')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# %%
import scipy.stats as stats
stats.probplot(df['log_price'], dist="norm", plot=plt)
plt.title("Q-Q Plot for log(Price)")
plt.show()
# %%
from scipy.stats import shapiro

stat, p = shapiro(df['log_price'].sample(5000))  # Use sample if too large
print('Shapiro-Wilk Test: stat=%.3f, p=%.3f' % (stat, p))

if p > 0.05:
    print("Data looks Gaussian (normal distribution)")
else:
    print("Data does NOT look Gaussian")
# %%
