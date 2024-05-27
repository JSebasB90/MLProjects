from preproc import Process
from model import Model 

if __name__== "__main__":
    
    ruta= 'in/data.csv'
    processor = Process()
    X, y = processor.preprocess_data(ruta)
    X_train, X_test, y_train, y_test = processor.statistics_process(X,y)
    X_train = processor.categ_woe(X_train)
    X_test = processor.categ_woe(X_test)
    #X_train.to_csv('in/X_train_prepr.csv', index=False)
    #y_train.to_csv('in/y_train_prepr.csv', index=False)
    #X_test.to_csv('in/X_test_prepr.csv', index=False)
    #y_test.to_csv('in/y_test_prepr.csv', index=False)
    
    #X_train= processor.remove_reference_categories(X_train)
    #X_test= processor.remove_reference_categories(X_test)
    #guardamos los csv
    
    # Ejecutar la creación del modelo
    inputs_train, inputs_test, ref_categories = Model.create_model(X_train, X_test, y_train, y_test)
    # Ejecutar la creación del scorecard
    min_score = 150
    max_score = 950
    df_scorecard = Model.create_scorecard(inputs_train, y_train, ref_categories, min_score, max_score)
    print(df_scorecard[['Feature name','Score - Final']])

    

    
