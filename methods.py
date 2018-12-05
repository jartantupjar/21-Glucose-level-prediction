from sklearn.metrics import  accuracy_score,r2_score,mean_squared_error,mean_absolute_error,explained_variance_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import Normalizer,StandardScaler,MinMaxScaler
def acc_score(y_pred,y_true,show_res=True):

    RMSE=rmse(y_true,y_pred)
    MAE=mean_absolute_error(y_true,y_pred)
    R2=r2_score(y_true,y_pred)
    MAPE=np.mean(np.abs((y_true-y_pred)/y_true))*100
    if show_res==True:
        print(' ERROR MEASURES ')
        print('Root Mean Squared Error: ', RMSE)
        print('Mean Absolute Error: ', MAE)
        print('Mean Absolute Percent Error: ', MAPE)
        print('R2 score: ', R2)
    
    return RMSE,MAE,R2,MAPE

def feature_plot(imp_features,X):
    indices = np.argsort(imp_features)[::-1]
    num_features=len(imp_features[imp_features>0])
    columns = X.columns.values[indices][:num_features]
    values=imp_features[indices][:num_features]

    plt.figure(figsize = (15,5))
    plt.title("Feature importances")
    plt.barh(range(num_features), values, align="center")
    plt.yticks(range(num_features), columns)
    plt.ylim([ num_features,-1])
    plt.show() 
    
def plot_test_data(y_test,y_pred):
    pred_data=y_test.copy()
    pred_data=pred_data.to_frame()
    pred_data['pred']=y_pred
    pred_data.plot(kind='line',use_index=False)
    pred_data.plot(kind='bar',use_index=True)
    plt.show();
    display(pred_data)
    
def get_model_prediction(X_train,y_train,X_test,y_test,clf_models,print_metrics=False,print_var_imp=False):
    imp_features=None
    for clf in clf_models:
        model_name=clf.__class__.__name__
        print('**Model name: ',model_name)
        clf.fit(X_train,y_train)
        pred=clf.predict(X_test)
        print(pred)
        
        if print_metrics:
            clf_metrics=acc_score(y_test,pred)
            plot_test_data(y_test,pred)
        if print_var_imp:
            try:
                imp_features=clf.feature_importances_
                feature_plot(imp_features,X_train)  
            except AttributeError:
                pass
    return imp_features


def gen_summary_data(X):   
    summary_data=pd.DataFrame()
    summary_data['max']=X.max(axis=1)
    summary_data['min']=X.min(axis=1)
    summary_data['avg']=X.mean(axis=1)
    summary_data['std']=X.std(axis=1)
    summary_data['mode']=X.mode(axis=1)[0]
    summary_data['median']=X.median(axis=1)
    summary_data['max-min']=X.max(axis=1)-X.min(axis=1)
    summary_data['Q1']=X.quantile(0.25,axis=1)
    summary_data['Q3']=X.quantile(0.75,axis=1)
   
    return summary_data



def scale_numerical_data(data,scaler=None):
    numerical_scaler=None
    if scaler is not None:
        if scaler=='minmax':
            numerical_scaler=MinMaxScaler()
        elif scaler=='standard':
            numerical_scaler=StandardScaler()
        else:
            numerical_scaler=Normalizer()
            
        columns_to_encode=list(data.select_dtypes(include=['float64','int64','object']))
        features_transform = pd.DataFrame(data = data)
        features_transform[columns_to_encode] = numerical_scaler.fit_transform(data[columns_to_encode])
        
        return features_transform
    
    else:
        return data


def get_reg_results(X_train,y_train,X_test,y_test,clf_models,print_var_imp):
    imp_features=None
    for clf in clf_models:
        model_name=clf.__class__.__name__
        print('classifier: ',model_name)
        clf.fit(X_train,y_train)
        pred=clf.predict(X_test)
        
        clf_metrics=acc_score(y_test,pred)
        plot_test_data(y_test,pred)
        if print_var_imp:
            try:
                imp_features=clf.feature_importances_
                feature_plot(imp_features,X_train)  
            except AttributeError:
                pass
    return imp_features
