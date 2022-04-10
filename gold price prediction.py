import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from flask import *  
app = Flask(__name__)  
gold_data = pd.read_csv('gld_price_data.csv')
gold_data.head(10)
gold_data['PLT_Price']=gold_data.PLT_Price/10
gold_data['PLD_Price']=gold_data.PLD_Price/10
gold_data.tail()
gold_data.shape
gold_data.info()
gold_data.isnull().sum()
gold_data.describe()
gold_data.describe().T
correlation = gold_data.corr()
plt.figure(figsize= (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Reds')
X=gold_data.drop(['GOLD'],axis=1)
X=X.drop(['SPX'],axis=1)
X.corrwith(gold_data['GOLD']).plot.bar(
        figsize = (10, 5), title = "Correlation with GOLD", fontsize = 20,
        rot = 40, grid = True)
for col in gold_data.columns:
    sns.displot(gold_data[col],kde=True)
sns.heatmap(gold_data.corr(), annot=True) 
sns.displot(gold_data['GOLD'], color='green')
X = gold_data.drop(['Date', 'GOLD'], axis=1)
Y = gold_data['GOLD']
print("X")
print(X)
print("Y")
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
regressor = RandomForestRegressor(n_estimators=100)
regressor
regressor.fit(X_train,Y_train)
test_data_prediction = regressor.predict(X_test)
score_1 = metrics.r2_score(Y_test, test_data_prediction)
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)
score_3 = metrics.mean_squared_error(Y_test, test_data_prediction)
score_4 = np.sqrt(score_3)
print("R squared error : ", score_1*100)
print('Mean Absolute Error : ', score_2)
print("Mean squared error : ", score_3)
print('Root Mean squared error : ', score_4)
print(test_data_prediction)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)
Y_test = list(Y_test)
sns.pairplot(data=gold_data)
def fun(i1,i2,i3,i4,i5,i6):
     l1=[i1]
     l2=[i2]
     l3=[i3]
     l4=[i4]
     l5=[i5]
     l6=[i6]
     d={"SPX":l1,"USO":l2,"SILVER":l3,"PLT_Price":l4,"PLD_Price":l5,"EUR/USD":i6}
     d=pd.DataFrame(d)
     return regressor.predict(d)
@app.route('/')  
def ho():  
    return render_template("Welcomepage.html");
@app.route('/fis',methods = ['POST'])  
def hoo():  
    return render_template("fis.html");
@app.route('/input',methods = ['POST'])  
def home():  
      i1=request.form['i1']  
      i2=request.form['i2']
      i3=request.form['i3']
      i4=request.form['i4']
      i5=request.form['i5']
      i6=request.form['i6']
      f=fun(i1,i2,i3,i4,i5,i6)
      return render_template("output.html",name=str(f[0]));
app.run()


      