import pandas as pd, numpy as np, xgboost as xgb, pickle, matplotlib, matplotlib.pyplot as pl, shap
from sklearn.datasets import load_boston
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
matplotlib.use('Agg')


def main():
    st.title("Feature Interpreation using SHAP")
    st.subheader("Sayantan Ghosh")
    
    @st.cache
    #Loading teh Boston Data---------------------------------------
    def load_data():
        boston = load_boston()
        return boston
    
    #Laoding the Dataset
    data_load_state = st.text("Loading Data")   
    boston = load_data()
    data_load_state = st.text("Data Loaded")
    
    
    #-----------------------------------------------------------------
    @st.cache
    def load_dataframe():
        Boston = pd.DataFrame(boston.data, columns=boston.feature_names)
        Boston['MEDV'] = boston.target
        return Boston
    
    Boston = load_dataframe()
    #Showing the snapshot of the data
    st.write(Boston.head(5))
    
    
    #Defining X and Y
    x = Boston.loc[:, Boston.columns != 'MEDV'].values
    y = Boston.loc[:, Boston.columns == 'MEDV'].values
    x_train, x_test, y_train, y_test = train_test_split (Boston[boston.feature_names],y, test_size = 0.25, random_state=34)
    
    # Building the dashboard on XGBOOST model:
    st.title('Model the Boston Housing Dataset using XGBOOST')
    
    # creating DMatrices for XGBOOST application
    #dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=boston.feature_names)
    #dtest  = xgb.DMatrix(x_test, label=y_test, feature_names=boston.feature_names)
   
    # Loading the cross-validated tuned XGBOOST model
    #loaded_model = pickle.load(open("xgboost_cv_best_pickle.dat", "rb"))
    #loaded_predictions = loaded_model.predict(dtest)
    
    
    
    loaded_model = xgb.XGBRegressor(
            n_estimators=150,
          reg_lambda=1,
            gamma=0,
            max_depth=8
        )    
    
    loaded_model.fit(x_train,y_train)
    loaded_predictions = loaded_model.predict(x_test)
    st.write('RMSE of the XGBoost model on test set:', round(np.sqrt(metrics.mean_squared_error(y_test, loaded_predictions)),2))
    
    
    #Tree Visualization------------------------------------------------------------------------------------------
    try:
        st.write('below, all seperate decision trees that have been build by training the model can be reviewed')
        ntree=st.number_input('Select the desired record for detailed explanation on the training set'
                                           ,min_value=1,max_value=5)
                                           
        tree=xgb.to_graphviz(loaded_model,num_trees=ntree)
        st.graphviz_chart(tree)
    except:
        pass
    
    
    #feature importance-------------------------------------------------------------------------------------------
    try:
        
        st.write('Using the standard XGBOOST importance plot feature, exposes the fact that the most important feature is not stable, select'
                 ' different importance types using the selectbox below')
        importance_type = st.selectbox('Select the desired importance type', ('weight','gain','cover'),index=0)
        importance_plot = xgb.plot_importance(loaded_model,importance_type=importance_type)
        pl.title ('xgboost.plot_importance(best XGBoost model) importance type = '+ str(importance_type))
        st.pyplot(bbox_inches='tight')
        pl.clf()
    except:
        pass
    
    
    #Feature Importance------------------------------------------------------------------------------------------
    st.write('To handle this inconsitency, SHAP values give robust details, among which is feature importance')
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(x_train)
    pl.title('Assessing feature importance based on Shap values')
    shap.summary_plot(shap_values,x_train,plot_type="bar",show=False)
    st.pyplot(bbox_inches='tight')
    pl.clf()
    
    
    #--------------------------------------------------------------------------------------------------------------
    st.write('SHAP values can also be used to represent the distribution of the training set of the respectable'
             'SHAP value in relation with the Target value, in this case the Median House Value (MEDV)')
    pl.title('Total distribution of observations based on Shap values, colored by Target value')
    shap.summary_plot(shap_values,x_train,show=False)
    st.pyplot(bbox_inches='tight')
    pl.clf()
    
    
    #----------------------------------------------
    st.write('Another example of SHAP values is for GDPR regulation, one should be able to give detailed information as to'
              ' why a specific prediction was made.')
    expectation = explainer.expected_value
    
    individual = st.number_input('Select the desired record from the training set for detailed explanation.'
                                           ,min_value=1
                                           ,max_value=1000)
    predicted_values = loaded_model.predict(x_train)
    real_value = y_train[individual]
    st.write('The real median house value for this individual record is: '+str(real_value))
    st.write('The predicted median house value for this individual record is: '+str(predicted_values[individual]))
    st.write('This prediction is calculated as follows: '
              'The average median house value: ('+str(expectation)+')'+
               ' + the sum of the SHAP values. ')
    st.write(  'For this individual record the sum of the SHAP values is: '+str(sum(shap_values[individual,:])))
    st.write(  'This yields to a predicted value of median house value of:'+str(expectation)+' + '+str(sum(shap_values[individual,:]))+
               '= '+str(expectation+(sum(shap_values[individual,:]))))
    st.write('Which features caused this specific prediction? features in red increased the prediction, in blue decreased them')
    shap.force_plot(explainer.expected_value, shap_values[individual,:],x_train.iloc[individual,:],matplotlib=True,show=False
                    ,figsize=(16,5))
    st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
    pl.clf()
    
   
    
    
    
    
    
    
if __name__ == '__main__':
    main()
   