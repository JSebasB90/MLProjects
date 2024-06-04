### Modelo logistico
from utils import LogisticRegression_with_p_values, Testeos, ScoreCard
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import joblib

class Model:
    """Class for creating and analyzing logistic regression models."""

    
    def create_model(X_train, X_test, y_train, y_test):
        """Create a logistic regression model and analyze its performance."""
       
        inputs_train_with_ref_cat = X_train.loc[: , [ 
                                             'mora_de_30_60:0', 
                                             'mora_de_30_60:1', 
                                             'mora_de_60_90:0', 
                                             'mora_de_60_90:1',
                                             'mora_mayor_90:0', 
                                             'mora_mayor_90:1', 
                                             'regional:DC', 
                                             'regional:DN', 
                                             'regional:DS', 
                                             'regional:OC', 
                                             'unidad_negocio:GRN',
                                             'unidad_negocio:OTRO',  
                                             'condicion_pago:08D', 
                                             'condicion_pago:15D', 
                                             'centro_operacion:CHIQUINQUIRA', 
                                             'centro_operacion:CUCUTA', 
                                             'centro_operacion:MANIZALES', 
                                             'centro_operacion:SIBERIA', 
                                             'centro_operacion:VILLANUEVA_YUMBO_BELLO', 
                                             'centro_operacion:CARTAGENA_BUCARAMANGA', 
                                             'unidad_negocio:ADM_CIL', 
                                             'condicion_pago:30D_CTD', 
                                             'condicion_pago:OTRO', 
                                             'porcentaje_h_60_90: entre 0 y 20', 
                                             'porcentaje_h_60_90: entre 20 y 80', 
                                             'porcentaje_h_60_90: entre 80 y 100', 
                                             'porcentaje_h_30_60: entre 0 y 30', 
                                             'porcentaje_h_30_60: entre 30 y 70', 
                                             'porcentaje_h_30_60: entre 70 y 100', 
                                             'antiguedad: menor a 6 meses', 
                                             'antiguedad: entre 6 meses y 1 año', 
                                             'antiguedad: entre 1 y 3 años', 
                                             'antiguedad: entre 3 y 5 años', 
                                             'antiguedad: entre 5 y 6 años', 
                                             'antiguedad: entre 6 y 8 años', 
                                             'antiguedad: entre 8 y 10 años', 
                                             'antiguedad: entre 10 y 12 años', 
                                             'antiguedad: mayor a 12 años', 
                                             'porcentaje_h_1_30: entre 0 y 25', 
                                             'porcentaje_h_1_30: entre 25 y 50',
                                             'porcentaje_h_1_30: entre 50 y 75', 
                                             'porcentaje_h_1_30: entre 75 y 100'
        ]]

        inputs_test_with_ref_cat = X_test.loc[: , [ 
                                             'mora_de_30_60:0', 
                                             'mora_de_30_60:1', 
                                             'mora_de_60_90:0', 
                                             'mora_de_60_90:1',
                                             'mora_mayor_90:0', 
                                             'mora_mayor_90:1', 
                                             'regional:DC', 
                                             'regional:DN', 
                                             'regional:DS', 
                                             'regional:OC', 
                                             'unidad_negocio:GRN',
                                             'unidad_negocio:OTRO',  
                                             'condicion_pago:08D', 
                                             'condicion_pago:15D', 
                                             'centro_operacion:CHIQUINQUIRA', 
                                             'centro_operacion:CUCUTA', 
                                             'centro_operacion:MANIZALES', 
                                             'centro_operacion:SIBERIA', 
                                             'centro_operacion:VILLANUEVA_YUMBO_BELLO', 
                                             'centro_operacion:CARTAGENA_BUCARAMANGA', 
                                             'unidad_negocio:ADM_CIL', 
                                             'condicion_pago:30D_CTD', 
                                             'condicion_pago:OTRO', 
                                             'porcentaje_h_60_90: entre 0 y 20', 
                                             'porcentaje_h_60_90: entre 20 y 80', 
                                             'porcentaje_h_60_90: entre 80 y 100', 
                                             'porcentaje_h_30_60: entre 0 y 30', 
                                             'porcentaje_h_30_60: entre 30 y 70', 
                                             'porcentaje_h_30_60: entre 70 y 100', 
                                             'antiguedad: menor a 6 meses', 
                                             'antiguedad: entre 6 meses y 1 año', 
                                             'antiguedad: entre 1 y 3 años', 
                                             'antiguedad: entre 3 y 5 años', 
                                             'antiguedad: entre 5 y 6 años', 
                                             'antiguedad: entre 6 y 8 años', 
                                             'antiguedad: entre 8 y 10 años', 
                                             'antiguedad: entre 10 y 12 años', 
                                             'antiguedad: mayor a 12 años', 
                                             'porcentaje_h_1_30: entre 0 y 25', 
                                             'porcentaje_h_1_30: entre 25 y 50',
                                             'porcentaje_h_1_30: entre 50 y 75', 
                                             'porcentaje_h_1_30: entre 75 y 100'
                ]]


        ref_categories = [                  
                  'mora_de_30_60:0', #segundo intento
                  'mora_de_30_60:1', 
                  'mora_de_60_90:0', #segundo intento
                  'mora_de_60_90:1',
                  'mora_mayor_90:0', #segundo intento
                  'mora_mayor_90:1', 
                  'regional:DC', 
                  'unidad_negocio:OTRO', 
                  'centro_operacion:SIBERIA', 
                  'condicion_pago:30D_CTD', 
                  'porcentaje_h_60_90: entre 80 y 100', 
                  'porcentaje_h_30_60: entre 70 y 100', 
                  'antiguedad: entre 3 y 5 años', 
                  'porcentaje_h_1_30: entre 0 y 25'
                  ]
                #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
                ### aqui arranca la prueba 

        inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis = 1)
        inputs_test = inputs_test_with_ref_cat.drop(ref_categories, axis = 1)

        reg = LogisticRegression(max_iter = 6, 
                                        solver = 'liblinear',
                                        C = 1,
                                        class_weight={0:1-0.445455, 1:0.445455},
                                        random_state = 1727,
                                        verbose=False)

        reg.fit(inputs_train, y_train)
        feature_name = inputs_train.columns.values

        y_prediction = reg.predict(inputs_test)

                # Generamos el informe de clasificación
        y_hat_test_proba = reg.predict_proba(inputs_test)
        y_hat_test_proba = y_hat_test_proba[: ][: , 1]
        y_test_temp = y_test.copy()
        y_test_temp.reset_index(drop = True, inplace = True)
        df_actual_predicted_probs = pd.concat([y_test_temp, pd.DataFrame(y_hat_test_proba)], axis = 1)
        df_actual_predicted_probs.columns = ['y_test', 'y_hat_test_proba']
        df_actual_predicted_probs.index = y_test.index

        pickle.dump(reg, open('in/pd_model.sav', 'wb'))
        
        #exportar modelo
        joblib.dump(reg, open('in/pd_model.pkl', 'wb'))
                # Con esto calculamos AUROC, tr y gini
        bestauroc = Testeos(y_test, y_hat_test_proba)
        tr = bestauroc.get_optimal_threshold()
        auroc = bestauroc.calculate_AUROC_score(tr)
        gini = bestauroc.calculate_Gini(tr)

        return inputs_train, inputs_test, ref_categories
    
    def create_scorecard(inputs_train, y_train, ref_categories, min_score, max_score):
        logreg = LogisticRegression_with_p_values(max_iter = 6, 
                         solver = 'liblinear',
                         C = 1,
                         class_weight={0:1-0.445455, 1:0.445455},
                         random_state = 1727,
                         verbose=False)
        logreg.fit(inputs_train, y_train)
        # Crear el resumen de coeficientes y valores p
        summary_table = logreg.crear_summary_p(inputs_train, y_train)

        # Crear el scorecard
        scorecard = ScoreCard()
        df_scorecard = scorecard.create_scorecard(ref_categories, summary_table)

        # Calcular los puntajes, 

        df_scorecard, min_sum_coef, max_sum_coef = scorecard.score_max_min(min_score, max_score, df_scorecard)

        return df_scorecard

