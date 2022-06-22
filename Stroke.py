#Import Libraries and Packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Upload dataset
stroke_data = pd.read_csv(r"\Users\pbroo\OneDrive\Documents\Grad School\data mining\stroke_data.csv")
#print(stroke_data.describe().T)


#Pre-processing
#Replace missing variables in BMI with its mean and remove all other missing variables
#print(stroke_data.isna().sum())
stroke_data['bmi']=stroke_data['bmi'].fillna(stroke_data['bmi'].mean())
stroke_data= stroke_data.dropna()
#print(stroke_data.isna().sum())
#Remove "ID" column
stroke_data.drop(columns='id',axis=1,inplace=True)

#print(stroke_data.head().iloc[:5])

#Data Exploration (with index words)
#Change numerical data into categorical
data_copy = stroke_data.copy()
data_copy["hypertension"]= stroke_data["hypertension"].map({1: "Yes",0: "No"})
data_copy["stroke"]= stroke_data["stroke"].map({1: "Yes",0:"No"})
data_copy["heart_disease"]= stroke_data["heart_disease"].map({1:"Yes",0: "No"})
#print(data_copy.head())
#print(data_copy.describe().T)
gender_ct = stroke_data.gender.value_counts()
hyper_ct = stroke_data.hypertension.value_counts()
heart_ct = stroke_data.heart_disease.value_counts()
married_ct = stroke_data.ever_married.value_counts()
work_ct = stroke_data.work_type.value_counts()
res_ct = stroke_data.Residence_type.value_counts()
smoke_ct = stroke_data.smoking_status.value_counts()
stroke_ct = stroke_data.stroke.value_counts()
#print(gender_ct)
#Categorical data bar plots
sns.countplot(data=data_copy, x = 'stroke', palette='PuBuGn')
#plt.show()


#Change categorical variables into numerical
#stroke_data = pd.get_dummies(data=stroke_data)
#print(stroke_data["gender"].unique())
label_encoder = preprocessing.LabelEncoder()
stroke_data["gender"]= label_encoder.fit_transform(stroke_data["gender"])
stroke_data["ever_married"]= label_encoder.fit_transform(stroke_data["ever_married"])
stroke_data["work_type"]= label_encoder.fit_transform(stroke_data["work_type"])
stroke_data["Residence_type"]= label_encoder.fit_transform(stroke_data["Residence_type"])
stroke_data["smoking_status"]= label_encoder.fit_transform(stroke_data["smoking_status"])

#histogram
fig2= stroke_data.hist(figsize=(10,8))
plt.tight_layout()
#plt.show()

#Interfeature correlation
f,ax = plt.subplots(figsize=(12, 12))
ax.set_title('Correlation map for variables')
sns.heatmap(stroke_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,cmap="icefire")
#plt.show()

#Correlation analysis between numerical data
plt.figure(1, figsize=(15,7))
n = 0
for x in ['age','avg_glucose_level','bmi']:
    for y in ['age','avg_glucose_level','bmi']:
        n += 1
        plt.subplot(3,3,n)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        sns.regplot(x = x, y = y, data = stroke_data)
        plt.ylabel(y.split()[0] + ' ' + y.split()[1] if len(y.split()) > 1 else y)
#plt.show()

#Sorted correlation matrix: 1= strong correlation -= negative correlation
#print(stroke_data.corr()['stroke'].sort_values(ascending = False))


#Classify Data
from imblearn import over_sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

X = stroke_data.drop(['stroke'], axis=1)
y = stroke_data['stroke']
#X , y = oversample.fit_resample(X,y)

#Distribution before oversampling
print('Distribution_of_Stroke:',np.array(X==0).sum())
print('Distribution_of_Not_Stroke',np.array(y==1).sum())

#Oversampling
oversample = SMOTE(random_state=100, sampling_strategy='minority')
X_sm, y_sm = oversample.fit_resample(X, y)
print('Distribution_of_Stroke:',np.array(X_sm==0).sum())
print('Distribution_of_Not_Stroke',np.array(y_sm==1).sum())

#Split Data
from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X_sm,y_sm,test_size=0.2,random_state=100)
print("x train:", X_train.shape, "y train:",y_train.shape, "x test:", X_test.shape,"y test:", y_test.shape)


# Normalization / Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(X_train,y_train)
print("LR Test Accuracy: {}%".format(round(lr.score(X_test,y_test)*100,2)))

pred_lr = lr.predict(X_test)
y_true_lr = y_test
cm = confusion_matrix(y_true_lr, pred_lr)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,fmt = ".0f",ax=ax)
plt.xlabel("LR Predicted Values")
plt.ylabel("LR True Values")
plt.show()
print(classification_report(lr.predict(X_test), y_test))

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)

#best_Kvalue = 0
#best_score = 0
#for i in range(1,10):
   # knn = KNeighborsClassifier(n_neighbors=i)
   # knn.fit(X_train,y_train)
    #if knn.score(X_test,y_test) > best_score:
     #   best_score = knn.score(X_train,y_train)
     #   best_Kvalue = i
knn_pred= knn.predict(X_test)
knn_acc= knn.score(X_test, y_test)
print("KNN Test Accuracy:", knn_acc)
print(classification_report(knn.predict(X_test),y_test))
knn_cm= confusion_matrix(y_test, knn_pred)
sns.heatmap(knn_cm, annot=True, fmt = "d")
plt.xlabel("KNN Predicted Values")
plt.ylabel("KNN True Values")
plt.show()


#svc
from sklearn.svm import SVC
svc = SVC(random_state=100)
svc.fit(X_train,y_train)
svc_pred= svc.predict(X_test)
svc.acc = svc.score(X_test, y_test)
print("SVC Test Accuracy: {}%".format(round(svc.score(X_test,y_test)*100,2)))
print(classification_report(svc.predict(X_test),y_test))
svc_cm = confusion_matrix(y_test, svc_pred)
sns.heatmap(svc_cm, annot = True)
plt.xlabel("SVC Predicted Values")
plt.ylabel("SVC True Values")
plt.show()

#naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
print("Test Accuracy: {}%".format(round(nb.score(X_test,y_test)*100,2)))
print(classification_report(nb.predict(X_test),y_test))
nb_pred= nb.predict(X_test)
nb_cm = confusion_matrix(y_test, nb_pred)
sns.heatmap(nb_cm, annot= True)
plt.xlabel("NB Predicted Values")
plt.ylabel("NB True Values")
plt.show()

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
import sklearn.datasets as datasets
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
dt = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)
tree =dt.fit(X_train,y_train)
print("DT Test Accuracy: {}%".format(round(dt.score(X_test,y_test)*100,2)))
print(classification_report(dt.predict(X_test), y_test))


fig, ax = plt.subplots(figsize=(10,10))
plot_tree(dt, fontsize= 10)
plt.show()


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf = rf.fit(X_train,y_train)
y_pred_test = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_test)
rf_acc2= accuracy_score(y_train,rf.predict(X_train))
print("Random Forest Test Accuracy:", (rf_acc)*100,"%")
print("Random Forest Train Accuracy:", (rf_acc2)*100,"%")
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot = True, fmt = "d")
plt.xlabel("RF Predicted Values")
plt.ylabel("RF True Values")
plt.show()
print(classification_report(rf.predict(X_test), y_test))
print(rf.feature)


#Voting Classifier
from sklearn.ensemble import VotingClassifier
clf1 = RandomForestClassifier()
clf2= DecisionTreeClassifier()

eclf1 = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2)], voting='soft')
eclf1.fit(X_train, y_train)
predictions = eclf1.predict(X_test)
print("Voting Classifier Accuracy Score is: ")
print(accuracy_score(y_test, predictions))
eclf1_cm = confusion_matrix(y_test, predictions)
sns.heatmap(eclf1_cm, annot = True, fmt="d")
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
print(classification_report(eclf1.predict(X_test), y_test))


