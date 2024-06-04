### Acá van las utilidades a usar

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve



class MeanImputer(BaseEstimator, TransformerMixin):
    """ This is the imputer for numerical values without outliers

    Args:
        BaseEstimator ( type ): Class object that tell us about how to imput values
        TransformerMixin (type_) : _description_
    """

    def __init__(self, variables=None):
        self.variables = variables
        
    def fit(self, X, y=None):
        self.mean_dict = {}
        for column in self.variables:
            self.mean_dict[column] = X[column].mean()
        return self
        
    def transform(self, X):
        X = X.copy()
        for column in self.variables:
            X[column].fillna(self.mean_dict[column], inplace = True)
        return X
    
class MedianImputer(BaseEstimator, TransformerMixin):
    """ This is the imputer for numerical values without outliers

    Args:
        BaseEstimator ( type ): Class object that tell us about how to imput values
        TransformerMixin (type_) : _description_
    """

    def __init__(self, variables=None):
        self.variables = variables
        
    def fit(self, X, y=None):
        self.median_dict = {}
        for column in self.variables:
            self.median_dict[column] = X[column].median()
        return self
        
    def transform(self, X):
        X = X.copy()
        for column in self.variables:
            X[column].fillna(self.median_dict[column], inplace=True)
        return X
    
class ModeImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.mode_dict = {}
        for column in self.variables:
            self.mode_dict[column] = X[column].mode()[0]
        return self
    
    def transform(self, X):
        X = X.copy()
        for column in self.variables:
            X[column].fillna(self.mode_dict[column], inplace = True)
        return X
    
class Carga:

    def load_from_csv(self, path):
        return pd.read_csv(path, decimal=",")
    
    def convert_to_category(self, data, columns):
        for column in columns:
            data[column] = data[column].astype('category')
        return data
    

class Categoricalfeat:
    def process_categorical(self, data, number):
        categorical_feat = list(data.select_dtypes(include=['category', 'object']).columns)
        data[categorical_feat] = data[categorical_feat].applymap(str)
        selected_variables = [var for var in categorical_feat if data[var].nunique() > number]
        return selected_variables
    
    def process_categorical_variables(self, data, selected_variables, numeric_thresholds, categorical_labels):
        # Generar variables temporales
        temp_variables = {}
        for var in selected_variables:
            temp_var_name = 'temp_' + var
            temp_variables[var] = data[var].value_counts()

        # Calcular los conteos de cada variable seleccionada
        for var in selected_variables:
            count_name = var + '_count'
            data[count_name] = data[var].apply(lambda x: temp_variables[var][x])

        # Aplicar el umbral cuantil para cada variable seleccionada
        for var, label, threshold in zip(selected_variables, categorical_labels, numeric_thresholds):
            count_name = var + '_count'
            threshold_value = data[count_name].quantile(threshold)
            data[var] = data.apply(lambda row: label if row[count_name] < threshold_value else row[var], axis=1)
            # Eliminar la columna de conteo temporal
            del data[count_name]

        return data
    


class Chi2:
    def __init__(self):
        pass

    def chi2_feature_selection(self, X, y):
        # Define an empty dictionary to store chi-squared test results
        chi2_check = {}

        # Loop over each column in the training set to calculate chi-statistic with the target variable
        for column in X:
            chi, p, dof, ex = chi2_contingency(pd.crosstab(y, X[column]))
            chi2_check.setdefault('Feature', []).append(column)
            chi2_check.setdefault('p-value', []).append(round(p, 10))

        # Convert the dictionary to a DataFrame
        chi2_result = pd.DataFrame(data=chi2_check)

        # Merge with describe to get unique count
        data_merge = chi2_result.merge(X.describe().T.reset_index(),
                                        left_on='Feature',
                                        right_on='index').sort_values(by=['p-value', 'unique'])
        return data_merge

    def drop_high_chi2_features(self, data_merge, x=0.05):
        # Create a list called chi_alto with elements in the 'Feature' column of data_merge
        chi_alto = data_merge[data_merge['p-value'] > x]['Feature'].tolist()
        return chi_alto

    def remove_columns(self, X, to_drop):
        # Remove the columns in the list to_drop from X
        X = X.drop(columns=to_drop)
        return X
    



class Anova:
    def __init__(self):
        pass

    def anova_feature_selection(self, X, y):
        # Calcular estadísticas ANOVA
        f_statistics, p_values = f_classif(X.fillna(X.median()), y)
        
        # Crear DataFrame con los resultados
        anova_f_table = pd.DataFrame(data={'Feature': X.columns.values,
                                           'F-Score': f_statistics,
                                           'p-value': p_values.round(decimals=10)})
        
        # Combinar con describe para obtener el conteo
        anova_merge = anova_f_table.merge(X.describe().T.reset_index(),
                                           left_on='Feature',
                                           right_on='index').sort_values(['F-Score', 'count'], ascending=False).head(50)
        
        return anova_merge

    def anova_features(self, anova_merge, x=0.05):
        # Generar lista de características a eliminar
        p_anova = anova_merge[(anova_merge['p-value']<0.05)].sort_values(by='p-value')
        list_anova = list(p_anova['Feature'])
        return list_anova

    def anova_dataset(self, X, list_anova):
        # Eliminar columnas de X
        A = X[list_anova]
        return A

class Corr:
    def __init__(self):
        pass

    def drop_high_corr_features(self, X, x=0.7):

        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > x)]
        
        return to_drop

    def remove_columns(self, X, to_drop):
        # Eliminar las columnas especificadas de X
        X = X.drop(columns=to_drop)
        return X
    
class Dummy:
    def __init__(self):
        pass

    def dummy_creation(self, df, cols):
        # Crear variables dummy
        df_dummies = pd.get_dummies(df[cols], prefix_sep=':')
        
        # Convertir las variables dummy a enteros (0s y 1s)
        df_dummies = df_dummies.astype(int)
        
        # Concatenar las variables dummy al DataFrame original
        df = pd.concat([df, df_dummies], axis=1)
        
        return df

    


class Woe_iv:
    def __init__(self):
        pass

    def woe_discrete(self, df, cat_variable_name, y_df):
        # Agregar la variable categórica y la variable objetivo al DataFrame
        df = pd.concat([df[cat_variable_name], y_df], axis=1)
        
        # Agrupar por la variable categórica y calcular el recuento de observaciones y el número de observaciones positivas
        df = df.groupby(cat_variable_name, as_index=False).agg({df.columns[1]: ['count', 'sum']})
        df.columns = [cat_variable_name, 'n_obs', 'n_good']
        df['n_bad'] = df['n_obs'] - df['n_good']
        
        # Calcular la proporción de observaciones, observaciones positivas y observaciones negativas
        df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
        df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
        df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
        
        # Calcular el WOE
        df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
        df = df.sort_values(['WoE'])
        df = df.reset_index(drop=True)
        
        # Calcular la diferencia entre las filas consecutivas
        df['diff_prop_good'] = (df['n_good'] / df['n_obs']).diff().abs()
        df['diff_WoE'] = df['WoE'].diff().abs()
        
        # Calcular el IV
        df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
        df['IV'] = df['IV'].sum()
        
        return df

    def plot_by_woe(self, df_WoE, rotation_of_x_axis_labels=0):
        # Configurar el estilo de los gráficos
        sns.set()
        
        # Graficar el WoE
        plt.figure(figsize=(16, 4))
        x = np.array(df_WoE.iloc[:, 0].apply(str))
        y = df_WoE['WoE']
        sns.lineplot(x=x, y=y, marker='o', linestyle='dotted', color='red')
        plt.xlabel(df_WoE.columns[0])
        plt.ylabel('Weight of Evidence')
        plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
        plt.xticks(rotation=rotation_of_x_axis_labels)
        plt.show()
    
    def woe_ordered_continuous(self, df, cat_variable_name, y_df):
        # Agregar la variable continua y la variable objetivo al DataFrame
        df = pd.concat([df[cat_variable_name], y_df], axis=1)
        
        # Agrupar por la variable continua y calcular el recuento de observaciones y el número de observaciones positivas
        df = df.groupby(cat_variable_name, as_index=False).agg({df.columns[1]: ['count', 'sum']})
        df.columns = [cat_variable_name, 'n_obs', 'n_good']
        df['n_bad'] = df['n_obs'] - df['n_good']
        
        # Calcular la proporción de observaciones, observaciones positivas y observaciones negativas
        df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
        df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
        df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
        
        # Calcular el WOE
        df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
        
        # Calcular la diferencia entre las filas consecutivas
        df['diff_prop_good'] = (df['n_good'] / df['n_obs']).diff().abs()
        df['diff_WoE'] = df['WoE'].diff().abs()
        
        # Calcular el IV
        df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
        df['IV'] = df['IV'].sum()
        return df

class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stats.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values
        
    def crear_summary_p(self, X, y):
        """Crea un resumen de coeficientes y valores p."""
        feature_name = X.columns.values

        summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
        summary_table['Coefficients'] = np.transpose(self.model.coef_)
        summary_table.index = summary_table.index + 1
        summary_table.loc[0] = ['Intercept', self.model.intercept_[0]]
        summary_table = summary_table.sort_index()
        p_values = self.p_values
        p_values = np.append(np.nan, np.array(p_values))
        summary_table['p_values'] = p_values

        return summary_table
        
class Testeos:
    """Class for conducting various tests and evaluations."""

    def __init__(self, y_test, y_hat_test_proba):
        self.y_test = y_test
        self.y_hat_test_proba = y_hat_test_proba

    def calculate_AUROC(self):
        """Calculate AUROC (Area Under the Receiver Operating Characteristic Curve)."""
        results = []
        real = list(np.linspace(0.2, 0.9, 100))

        for r in real:
            AUROC_TEST = []
            x = {}

            # Set predicted labels based on threshold r
            y_pred_test = np.where(self.y_hat_test_proba > r, 1, 0)

            # Calculate AUROC
            AUROC = roc_auc_score(self.y_test, y_pred_test)
            AUROC_TEST.append(AUROC)
            AUROC_TEST = np.mean(AUROC_TEST)

            x['real'] = r
            x['auroc'] = AUROC_TEST
            results.append(x)

        return pd.DataFrame(results)
    def get_optimal_threshold(self):
        """Get the threshold corresponding to the maximum AUROC."""
        auroc_results = self.calculate_AUROC()
        max_auroc_row = auroc_results.loc[auroc_results['auroc'] == auroc_results['auroc'].max()]
        return max_auroc_row['real'].values[0]

    def calculate_AUROC_score(self, threshold):
        """Calculate AUROC for a given threshold."""
        y_pred_test = np.where(self.y_hat_test_proba > threshold, 1, 0)
        return roc_auc_score(self.y_test, y_pred_test)

    def calculate_Gini(self, threshold):
        """Calculate Gini coefficient for a given threshold."""
        AUROC = self.calculate_AUROC_score(threshold)
        return AUROC * 2 - 1
    
    
class ScoreCard:
    """Class for generating a scorecard based on coefficients and p-values."""
    
    def __init__(self):
        pass
    
    def create_scorecard(self, list, table):
        """Create a scorecard from reference categories and summary table."""
        df_ref_categories = pd.DataFrame(list, columns=['Feature name'])
        df_ref_categories['Coefficients'] = 0
        df_ref_categories['p_values'] = np.nan
        df_scorecard = pd.concat([table, df_ref_categories])
        df_scorecard = df_scorecard.reset_index()
        df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]
        return df_scorecard
    
    def score_max_min(self, min_score, max_score, df_scorecard):
        """Calculate scores based on minimum and maximum scores."""
        min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
        max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
        df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
        df_scorecard.loc[0, 'Score - Calculation'] = ((df_scorecard.loc[0, 'Coefficients'] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
        df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()
        df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']
        df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']
        df_scorecard['Score - Final'] = df_scorecard['Score - Final'].astype(int)
        return df_scorecard, min_sum_coef, max_sum_coef


class Computing_score:
    """Class for computing scores and related metrics."""

    @staticmethod
    def add_intercept(inputs_test_with_ref_cat):
        """Add an intercept column to the test data."""
        inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat.copy()
        inputs_test_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)
        return inputs_test_with_ref_cat_w_intercept

    @staticmethod
    def calculate_scores(inputs_test_with_ref_cat_w_intercept, df_scorecard, min_score, max_score, min_sum_coef, max_sum_coef):
        """Calculate scores based on the provided inputs."""
        scorecard_scores = df_scorecard['Score - Final'].values.reshape(-1, 1)
        y_scores = inputs_test_with_ref_cat_w_intercept.dot(scorecard_scores)
        
        sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)) * (max_sum_coef - min_sum_coef) + min_sum_coef
        y_hat_proba_from_score = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)
        
        return y_scores, y_hat_proba_from_score

    @staticmethod
    def n_approved(y_hat_proba, threshold):
        """Calculate the number of approved applications above a given probability."""
        return np.where(y_hat_proba >= threshold, 1, 0).sum()

    @staticmethod
    def calculate_cutoffs(y_test, y_hat_proba,min_score, max_score, min_sum_coef,max_sum_coef):
        """Calculate ROC curve cutoffs and related metrics."""
        fpr, tpr, thresholds = roc_curve(y_test, y_hat_proba)
        df_cutoffs = pd.DataFrame({'thresholds': thresholds, 'fpr': fpr, 'tpr': tpr})
        df_cutoffs['thresholds'][0] = 1 - 1 / np.power(10, 16)
        df_cutoffs['Score'] = ((np.log(df_cutoffs['thresholds'] / (1 - df_cutoffs['thresholds'])) - min_sum_coef) * ((max_score - min_score) / (max_sum_coef - min_sum_coef)) + min_score).round().astype(int)
        df_cutoffs['N Approved'] = df_cutoffs['thresholds'].apply(lambda x: Computing_score.n_approved(y_hat_proba, x))
        df_cutoffs['N Rejected'] = y_hat_proba.shape[0] - df_cutoffs['N Approved']
        df_cutoffs['Approval Rate'] = df_cutoffs['N Approved'] / y_hat_proba.shape[0]
        df_cutoffs['Rejection Rate'] = 1 - df_cutoffs['Approval Rate']
        df_cutoffs['score_factor'] = pd.cut(df_cutoffs['Score'], 15)
        df_cutoffs['score_factor_4'] = pd.cut(df_cutoffs['Score'], 4)
        df_cutoffs['approval_rejection'] = np.where(df_cutoffs['N Approved'] < df_cutoffs['N Rejected'], 1, 0)
        return df_cutoffs
