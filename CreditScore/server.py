import joblib
import numpy as np
import pandas as pd
from flask import Flask
from flask import jsonify

app = Flask(__name__)

input_array = np.array([0.0,41.0,0.0,3.95,15.34,80.69,-18.0,3712,'CARTAGENA_BUCARAMANGA','DN','GRN','30D o CTD',0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,0,1,0,1,0,'negativo',50,'0 a 20',1,0,0,"(-0.1, 10.0]","(-0.1, 33.333]",1,0,0,'entre 10 y 12 años',0,0,0,0,0,0,0,1,0,"(-0.1, 50.0]",'Entre 75 y 100',0,0,0,1])

nombres_columnas = ['porcentaje_h_mayor_90','max_mora','porcentaje_h_60_90','porcentaje_h_pago_a_tiempo','porcentaje_h_30_60','porcentaje_h_1_30', 'min_mora','antiguedad','centro_operacion','regional','unidad_negocio','condicion_pago','mora_de_1_30','mora_de_30_60','mora_de_60_90','mora_mayor_90','centro_operacion:BELLO',
                    'centro_operacion:BUCARAMANGA','centro_operacion:CARTAGENA','centro_operacion:CHIQUINQUIRA','centro_operacion:CUCUTA','centro_operacion:MANIZALES','centro_operacion:OFICINA CENTRAL','centro_operacion:OTRO','centro_operacion:PITALITO','centro_operacion:SIBERIA','centro_operacion:VILLANUEVA','centro_operacion:YUMBO',
                    'regional:DC','regional:DN','regional:DS','regional:OC','unidad_negocio:ADM','unidad_negocio:CIL','unidad_negocio:GRN','unidad_negocio:OTRO','condicion_pago:08D','condicion_pago:15D','condicion_pago:30D','condicion_pago:60D','condicion_pago:CTD','condicion_pago:OTRO','mora_de_1_30:0','mora_de_1_30:1','mora_de_30_60:0',
                    'mora_de_30_60:1','mora_de_60_90:0','mora_de_60_90:1','mora_mayor_90:0','mora_mayor_90:1','centro_operacion:VILLANUEVA_YUMBO_BELLO','centro_operacion:CARTAGENA_BUCARAMANGA','unidad_negocio:ADM_CIL','condicion_pago:30D_CTD','condicion_pago:OTROS','min_mora_factor','max_mora_factor','porcentaje_h_60_90_factor',
                    'porcentaje_h_60_90: entre 0 y 20','porcentaje_h_60_90: entre 20 y 80','porcentaje_h_60_90: entre 80 y 100','porcentaje_h_mayor_90_factor','porcentaje_h_30_60_factor','porcentaje_h_30_60: entre 0 y 30','porcentaje_h_30_60: entre 30 y 70','porcentaje_h_30_60: entre 70 y 100','Antiguedad_factor','antiguedad: menor a 6 meses',
                    'antiguedad: entre 6 meses y 1 año','antiguedad: entre 1 y 3 años','antiguedad: entre 3 y 5 años','antiguedad: entre 5 y 6 años','antiguedad: entre 6 y 8 años','antiguedad: entre 8 y 10 años','antiguedad: entre 10 y 12 años','antiguedad: mayor a 12 años','porcentaje_h_pago_a_tiempo_factor','porcentaje_h_1_30_factor',
                    'porcentaje_h_1_30: entre 0 y 25','porcentaje_h_1_30: entre 25 y 50','porcentaje_h_1_30: entre 50 y 75','porcentaje_h_1_30: entre 75 y 100']

df_server = pd.DataFrame([input_array], columns=nombres_columnas)

inputs_train_with_ref_cat = df_server.loc[: , [ 
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

X = inputs_train_with_ref_cat.drop(ref_categories, axis = 1)


#POSTMAN PARA PRUEBAS
@app.route('/predict', methods=['GET'])
def predict():
    prediction = model.predict(X)
    prediction_list = prediction.tolist()
    return jsonify({'prediccion' : list(prediction_list)})
    
if __name__ == "__main__":
    model = joblib.load('./in/pd_model.pkl')
    app.run(port=8080)