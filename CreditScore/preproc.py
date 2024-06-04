### Preprocesamiento del modelo
### Preprocesamiento del modelo
from utils import Carga,  MedianImputer, ModeImputer, Categoricalfeat, Chi2, Anova, Woe_iv, Corr
from utils import Dummy
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class Process:
    def __init__(self):
        pass

    def preprocess_data(self, enlace):
        carga = Carga()
        data = carga.load_from_csv(enlace)
        # Convertir ciertas columnas a tipo categoría
        columns_to_convert = ['mora_de_1_30', 'mora_de_30_60', 'mora_de_60_90', 'mora_mayor_90']
        data = carga.convert_to_category(data, columns_to_convert)

        # Convertimos ciertas columnas a tipo categoría
        columns_to_convert = ['mora_de_1_30', 'mora_de_30_60', 'mora_de_60_90', 'mora_mayor_90']
        data = carga.convert_to_category(data, columns_to_convert)

        # Creamos la columna "good-bad"
        data["good-bad"] = np.where(data["dias_mora_promedio"] > 30, '0', '1')
        data = carga.convert_to_category(data, ["good-bad"])

        # Obtenemos las columnas numéricas y no numéricas
        variables_numericas = data.select_dtypes(include=['number']).columns
        variables_no_numericas = data.select_dtypes(exclude=['number']).columns

        # Creamos instancias de MedianImputer y ModeImputer
        median_imputer = MedianImputer(variables=variables_numericas)
        mode_imputer = ModeImputer(variables=variables_no_numericas)

        # Imputamos los valores faltantes
        data[variables_numericas] = median_imputer.fit_transform(data[variables_numericas])
        data[variables_no_numericas] = mode_imputer.fit_transform(data[variables_no_numericas])

        #variables categoricas
        categorical = Categoricalfeat()
        categorical_features = categorical.process_categorical(data, 5)
        print(categorical_features)
        numeric_thresholds = [0.2, 0.1, 0.1]
        categorical_labels = ["OTRO", "OTRO", "OTRO"]
        data = categorical.process_categorical_variables(data, categorical_features, numeric_thresholds, categorical_labels)

        numerical_feat = list(data.select_dtypes(include=['int64','float64','Int64']).columns)
        categorical_feat = list(data.select_dtypes(include=['category','object']).columns)

        y = data['good-bad'].astype(int)
        X = data.drop(['good-bad'],axis=1)

        return X, y

    def separate(self, X, y):
        
        #Let’s split X and y using Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state= 1727, stratify=y)
        return X_train, X_test, y_train, y_test

    def statistics_process(self, X_train, X_test, y_train, y_test):
        #grouping X_train by values
        X_train_cat = X_train.select_dtypes(include=['object', 'category']).copy()
        X_train_num = X_train.select_dtypes(include = 'number').copy()

        #ahora va chi2
        # Creamos una instancia de la clase Chi2, Anova y Corr
        chi2_selector = Chi2()
        anova_selector = Anova()
        corr_selector = Corr()

        #hacemos las evaluaciones de chi2 y anova
        data_merge = chi2_selector.chi2_feature_selection(X_train_cat, y_train)
        to_drop = chi2_selector.drop_high_chi2_features(data_merge)
        list_chi = X_train_cat.columns.difference(to_drop)

        anova_merge = anova_selector.anova_feature_selection(X_train_num, y_train)
        list_anova = anova_selector.anova_features(anova_merge)
        X_train_num = anova_selector.anova_dataset(X_train_num, list_anova)
        list_anova.extend(list_chi)

        to_drop_corr = corr_selector.drop_high_corr_features(X_train_num)
        selected_cols = [x for x in list_anova if x not in to_drop_corr]

        #para X_test
        X_train = X_train[selected_cols]
        X_test = X_test[selected_cols]
        
        dummy_creator = Dummy()
        X_train = dummy_creator.dummy_creation(X_train, list_chi)
        X_test = dummy_creator.dummy_creation(X_test, list_chi)

        X_test = X_test.reindex(labels=X_train.columns, axis=1, fill_value=0)
        y_test = y_test.reindex(axis=1, fill_value=0)

        return X_train, X_test, y_train, y_test
    
     
    def categ_woe(self, X):
         
        inputs_prepr = X.copy()
        
        #categoricas
        mapeo_centros = {
            'VILLANUEVA': 'VILLANUEVA_YUMBO_BELLO',
            'OFICINA CENTRAL': 'VILLANUEVA_YUMBO_BELLO',
            'YUMBO': 'VILLANUEVA_YUMBO_BELLO',
            'BELLO': 'VILLANUEVA_YUMBO_BELLO',
            'PITALITO': 'CARTAGENA_BUCARAMANGA',
            'CARTAGENA': 'CARTAGENA_BUCARAMANGA',
            'OTRO': 'CARTAGENA_BUCARAMANGA',
            'BUCARAMANGA': 'CARTAGENA_BUCARAMANGA'
        }

        # Aplicar el mapeo a la columna 'centro_operacion'
        inputs_prepr['centro_operacion'] = inputs_prepr['centro_operacion'].replace(mapeo_centros)

        inputs_prepr['centro_operacion:VILLANUEVA_YUMBO_BELLO'] = sum([inputs_prepr['centro_operacion:VILLANUEVA'], inputs_prepr['centro_operacion:OFICINA CENTRAL'], inputs_prepr['centro_operacion:YUMBO'], inputs_prepr['centro_operacion:BELLO']])
        inputs_prepr['centro_operacion:CARTAGENA_BUCARAMANGA'] = sum([inputs_prepr['centro_operacion:CARTAGENA'], inputs_prepr['centro_operacion:PITALITO'], inputs_prepr['centro_operacion:OTRO'], inputs_prepr['centro_operacion:BUCARAMANGA']])

        mapeo_un = {
            'ADM': 'ADM_CIL',
            'CIL': 'ADM_CIL'
        }

        inputs_prepr['unidad_negocio'] = inputs_prepr['unidad_negocio'].replace(mapeo_un)
        inputs_prepr['unidad_negocio:ADM_CIL'] = sum([inputs_prepr['unidad_negocio:ADM'], inputs_prepr['unidad_negocio:CIL']])
        mapeo_cp = {
            '30D': '30D o CTD',
            'CTD': '30D o CTD',
            'OTRO': 'OTROS',
            '60D': 'OTROS'
        }
        inputs_prepr['condicion_pago'] = inputs_prepr['condicion_pago'].replace(mapeo_cp)
        inputs_prepr['condicion_pago:30D_CTD'] = sum([inputs_prepr['condicion_pago:30D'], inputs_prepr['condicion_pago:CTD']])
        inputs_prepr['condicion_pago:OTROS'] = sum([inputs_prepr['condicion_pago:OTRO'], inputs_prepr['condicion_pago:60D']])
        #Numericas
        inputs_prepr['min_mora_factor'] = np.where(inputs_prepr['min_mora']<0,'negativo',
                                              np.where(inputs_prepr['min_mora']>=0,'no negativo',0))
        inputs_prepr['max_mora_factor'] = np.where(inputs_prepr['max_mora']<100,50,
                                              np.where(inputs_prepr['max_mora']<=300,200,
                                                    np.where(inputs_prepr['max_mora']>300,300,0)))
        inputs_prepr['porcentaje_h_60_90_factor'] = pd.cut(inputs_prepr['porcentaje_h_60_90'], 5)
        inputs_prepr['porcentaje_h_60_90_factor'] = np.where(inputs_prepr['porcentaje_h_60_90']<=20,'0 a 20',
                                            np.where(inputs_prepr['porcentaje_h_60_90']<=80,'20 a 80',
                                              np.where(inputs_prepr['porcentaje_h_60_90']>80,'80 a 100',0)))
        inputs_prepr['porcentaje_h_60_90: entre 0 y 20'] = np.where(inputs_prepr['porcentaje_h_60_90']<=20,1,0)
        inputs_prepr['porcentaje_h_60_90: entre 20 y 80'] = np.where((inputs_prepr['porcentaje_h_60_90']>20)&(inputs_prepr['porcentaje_h_60_90']<=80),1,0)
        inputs_prepr['porcentaje_h_60_90: entre 80 y 100'] = np.where(inputs_prepr['porcentaje_h_60_90']>80,1,0)
        
        inputs_prepr['porcentaje_h_mayor_90_factor'] = pd.cut(inputs_prepr['porcentaje_h_mayor_90'], 10)
        inputs_prepr['porcentaje_h_30_60_factor'] = pd.cut(inputs_prepr['porcentaje_h_30_60'], 3)
        
        inputs_prepr['porcentaje_h_30_60: entre 0 y 30'] = np.where(inputs_prepr['porcentaje_h_30_60']<=30,1,0)
        inputs_prepr['porcentaje_h_30_60: entre 30 y 70'] = np.where((inputs_prepr['porcentaje_h_30_60']>30)&(inputs_prepr['porcentaje_h_30_60']<=70),1,0)
        inputs_prepr['porcentaje_h_30_60: entre 70 y 100'] = np.where(inputs_prepr['porcentaje_h_30_60']>70,1,0)
        inputs_prepr['Antiguedad_factor'] = np.where(inputs_prepr['antiguedad']<=180, 'menor a 6 meses',
                                     np.where(inputs_prepr['antiguedad']<=360, 'entre 6 meses y 1 año',
                                      np.where(inputs_prepr['antiguedad']<=1080, 'entre 1 y 3 años',
                                        np.where(inputs_prepr['antiguedad']<=1800, 'entre 3 y 5 años',
                                         np.where(inputs_prepr['antiguedad']<=2160, 'entre 5 y 6 años',
                                          np.where(inputs_prepr['antiguedad']<=2880, 'entre 6 y 8 años',
                                           np.where(inputs_prepr['antiguedad']<=3600, 'entre 8 y 10 años',
                                            np.where(inputs_prepr['antiguedad']<=4320, 'entre 10 y 12 años',
                                               np.where(inputs_prepr['antiguedad']>4320, 'mayor a 12 años',0)))))))))
        
        inputs_prepr['antiguedad: menor a 6 meses'] = np.where(inputs_prepr['antiguedad']<=180,1,0)
        inputs_prepr['antiguedad: entre 6 meses y 1 año'] = np.where((inputs_prepr['antiguedad']>180)&(inputs_prepr['antiguedad']<=360),1,0)
        inputs_prepr['antiguedad: entre 1 y 3 años'] = np.where((inputs_prepr['antiguedad']>360)&(inputs_prepr['antiguedad']<=1080),1,0)
        inputs_prepr['antiguedad: entre 3 y 5 años'] = np.where((inputs_prepr['antiguedad']>1080)&(inputs_prepr['antiguedad']<=1800),1,0)
        inputs_prepr['antiguedad: entre 5 y 6 años'] = np.where((inputs_prepr['antiguedad']>1800)&(inputs_prepr['antiguedad']<=2160),1,0)
        inputs_prepr['antiguedad: entre 6 y 8 años'] = np.where((inputs_prepr['antiguedad']>2160)&(inputs_prepr['antiguedad']<=2880),1,0)
        inputs_prepr['antiguedad: entre 8 y 10 años'] = np.where((inputs_prepr['antiguedad']>2880)&(inputs_prepr['antiguedad']<=3600),1,0)
        inputs_prepr['antiguedad: entre 10 y 12 años'] = np.where((inputs_prepr['antiguedad']>3600)&(inputs_prepr['antiguedad']<=4320),1,0)
        inputs_prepr['antiguedad: mayor a 12 años'] = np.where(inputs_prepr['antiguedad']>4320,1,0)
        
        inputs_prepr['porcentaje_h_pago_a_tiempo_factor'] = pd.cut(inputs_prepr['porcentaje_h_pago_a_tiempo'], 2)
        inputs_prepr['porcentaje_h_1_30_factor'] = np.where(inputs_prepr['porcentaje_h_1_30']<25,'Entre 0 y 25',
                                            np.where(inputs_prepr['porcentaje_h_1_30']<50,'Entre 25 y 50',
                                            np.where(inputs_prepr['porcentaje_h_1_30']<75,'Entre 50 y 75',
                                              np.where(inputs_prepr['porcentaje_h_1_30']>=75,'Entre 75 y 100',0))))
        inputs_prepr['porcentaje_h_1_30: entre 0 y 25'] = np.where(inputs_prepr['porcentaje_h_1_30']<=25,1,0)
        inputs_prepr['porcentaje_h_1_30: entre 25 y 50'] = np.where((inputs_prepr['porcentaje_h_1_30']>25)&(inputs_prepr['porcentaje_h_1_30']<=50),1,0)
        inputs_prepr['porcentaje_h_1_30: entre 50 y 75'] = np.where((inputs_prepr['porcentaje_h_1_30']>50)&(inputs_prepr['porcentaje_h_1_30']<=75),1,0)
        inputs_prepr['porcentaje_h_1_30: entre 75 y 100'] = np.where(inputs_prepr['porcentaje_h_1_30']>75,1,0)

        return inputs_prepr
    
    