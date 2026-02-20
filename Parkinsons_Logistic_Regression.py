import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
     
# URL ของ dataset บน UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

# โหลดข้อมูลเข้า Pandas DataFrame
df1 = pd.read_csv(url)
df1
df1.describe()

#copy data1 มาใส่ data2
df2 = df1.copy()


''' ทำการ info เพื่อดูสรุปเกี่ยวกับ DataFrame พบว่า status ซึ่งเอาไว้แสดงสถานะ การเป็นโรค
    พากินสัน เป็น int จึงทำการเปลี่่ยนเป็น catagory จากนั้นขยับ column ไปไว้ column ไปไว้
    ที่ท้ายสุดของ Dataframe และทำการลบ column name ออก จากนั้น info ดู dataframe อีกรอบว่าถูกต้อง'''

df2.info()
df2['status'] = df2['status'].astype('category')
y = df2['status']
df2 = df2.drop(['status'], axis = 1)
df2 = df2.drop(['name'], axis =1)
df2['status'] = y
df2.info()

# ทำการเช็ค ดูข้อมูลใน Dataframe ว่าแต่ละ column มีข้อมูลขาดหายหรือไม่  พบว่า Dataset นี้ไม่มีข้อมูลขาดหาย
df2.isnull().sum()

#ทำการ df2.describe เพื่อดูข้อมูล พบว่ามีข้อมูลที่ Min-Max ห่างกันเกินไป ดังนั้นจะทำการกำจัด outlier โดยการใช้ วิธี IQR ในการกำจัด
df2.describe()

#boxplot เพื่อดูแนวโน้มการกระจายของค่า สังเกตุว่ามี outlier จำนวนมาก จากทั้ง feature ทั้ง 23 column
plt.figure(figsize=(15,5))
sns.boxplot(df2,width=0.2)

# ทำการ copy df2 ลงใน df3 เพื่อที่จะ เอา df3 ไปใช้ คัด outlier ออก
df3 = df2.copy()

# สร้าง Function สำหรับการ กรอง Outlier ในข้อมูลออก ให้เหลืออยู่ใน ระดับ % ที่ต้องการ
def remove_outliers(df3,percent_accept):

    x_ol = df3.drop(['status'],axis = 1)
    y_ol = df3['status']

    while True:

        q1 = np.quantile(x_ol, 0.25, axis=0)
        q2 = np.quantile(x_ol, 0.50, axis=0)
        q3 = np.quantile(x_ol, 0.75, axis=0)
        iqr = q3 - q1
        lower_limit = q1 - 1.5*iqr
        upper_limit = q3 + 1.5*iqr

        count_not_outlier = 0
        count_outlier = 0

        for i in range(len(x_ol)):
            if  np.all((x_ol.iloc[i,:] >= lower_limit) & (x_ol.iloc[i,:] <= upper_limit)):
                count_not_outlier += 1
            else :
                count_outlier += 1

        percent_outlier = count_outlier/(count_not_outlier+count_outlier)*100

        if  percent_outlier <= percent_accept:
            break

        else :
            mask = np.all((x_ol >= lower_limit) & (x_ol <= upper_limit), axis=1)
            x_ol = x_ol[mask]
            y_ol = y_ol[mask]

    df3 =  pd.concat([x_ol,y_ol],axis=1)
    return df3

#กำจัด outlier ด้วยการเรียกใช้ function ที่ได้สร้างไว้
df3 = remove_outliers(df3,10)
df3

# หลังจาก คัด outlier ออกแล้ว ต่อมาทำการ describe ข้อมูลดูข้อมูล พบว่าข้อมูลมีการกระจายที่น้อยลงมาก
df3.describe()

# Boxplot เพื่อดูการกระจาย ของข้อมูลก่อน คัด outlier ออก และหลังคัด outlier ออก ได้ดังรูป จะเห็นว่า outlier หายไปเยอะมาก
plt.figure(figsize=(15,5))
sns.boxplot(df2,width=0.2)

plt.figure(figsize=(15,5))
sns.boxplot(df3,width=0.2)

#ทำการปรับข้อมูลให้อยู่ในรูปมาตราฐานเดียวกัน โดยใช้ การแปลงเป็น z-score เพื่อให้ นำไปใช้คำนวณ model ได้ค่าที่แม่นยำ

mean = df3.drop(['status'],axis=1).mean()
std = df3.drop(['status'],axis=1).std()

z = (df3.drop(['status'],axis=1)-mean)/std
z['status']  = df2['status']
df3  = z.copy()
df3


# จาก การทำdata preparation ที่ผ่านมาเราได้ทำการแปลง dataset column'status' เป็น catagory และทำการย้ายมาอยู่ column ท้ายสุด และลบ column 'name' ออก
# จากนั้นทำการจัดการ outlier ด้วยวิธี Iqr และสุดท้ายทำการแปลงข้ามูลทั้งหมดใน data set ให้เป็น z-score
#ทำให้ได้ Dataset ที่พร้อมนำไปทำการ Train medel ดังที่แสดงให้ดูด้านล่าง
df3


# copy df3 มาใส่ df4 เพื่อที่จะเอา df4 ไปทำการ สร้าง model Binomial Logistic Regression
df4 = df3.copy()
# แยก feature กับ target โดยใช้ feature ทั้งหมด
x_t = df4.drop(['status'],axis = 1)
y_t = df4['status']
y_t.unique()

# ทำการ Train
x_train,x_test,y_train,y_test = train_test_split(x_t,y_t,test_size=0.2,random_state=25)
model = LogisticRegression()
model = model.fit(x_train,y_train)
print(model.intercept_)
print()
print(model.coef_)



# Simple linear regression ineach 2 class
b0 = model.intercept_[0]
b1 = model.coef_[0,0]
b2 = model.coef_[0,1]
b3 = model.coef_[0,2]
b4 = model.coef_[0,3]
b5 = model.coef_[0,4]
b6 = model.coef_[0,5]
b7 = model.coef_[0,6]
b8 = model.coef_[0,7]
b9 = model.coef_[0,8]
b10 = model.coef_[0,9]
b11 = model.coef_[0,10]
b12 = model.coef_[0,11]
b13 = model.coef_[0,12]
b14 = model.coef_[0,13]
b15 = model.coef_[0,14]
b16 = model.coef_[0,15]
b17 = model.coef_[0,16]
b18 = model.coef_[0,17]
b19 = model.coef_[0,18]
b20 = model.coef_[0,19]
b21 = model.coef_[0,20]
b22 = model.coef_[0,21]

print('Simple linear ---->  %.4f + %.4fx1 + %.4fx2 + %.4fx3 + %.4fx4 + %.4fx5 + %.4fx6 + %.4fx7 + %.4fx8 + %.4fx9 + %.4fx10 + %.4fx11 + %.4fx12 + %.4fx13 + %.4fx14 + %.4fx15 + %.4fx16 + %.4fx17 + %.4fx18 + %.4fx19 + %.4fx20 + %.4fx21 + %.4fx22'%(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22))

     
# Predict test set to output 0's and 1's.
y_pred = model.predict(x_test)
print(y_pred)

# Use predict_proba to output a probability
# 0 ไม่เป็น Parkinsons , 1 เป็น Parkinsons
y_pred_prob = model.predict_proba(x_test)
print(y_pred_prob.round(4))


# Updated dataframe for x_test
df4 = x_test.copy()
df4['status_test'] = y_test
df4['status_pred'] = y_pred
df4['prob-0'] = y_pred_prob[:,0]
df4['prob-1'] = y_pred_prob[:,1]
df4.round(4)
     

     # Classification report
print(classification_report(y_test, y_pred, target_names=[str(c) for c in model.classes_]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, cmap='Oranges', fmt='d',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')
plt.show()

#***
# Encode the categorical labels into numeric values
encoder = LabelEncoder()
y_test_encoded = encoder.fit_transform(y_test)
y_pred_encoded = encoder.transform(y_pred)

print('Accuracy: %.4f' % accuracy_score(y_test_encoded, y_pred_encoded))

# Compute Precision, Recall, and F1 Score with multiclass support
print('Precision: %.4f' % precision_score(y_test_encoded, y_pred_encoded, average='macro'))
print('Recall: %.4f' % recall_score(y_test_encoded, y_pred_encoded, average='macro'))
print('F1 Score: %.4f' % f1_score(y_test_encoded, y_pred_encoded, average='macro'))


'''
จากผลการ ทดลองเพื่อสร้าง model Binomial Logistic Regression เพื่อที่จะใช้ทำนายว่าบุคคลใดมีโอกาสเป็น โรค Parkinsons
โดยขั้นตอนหลักๆ มีอยู่ 4 ส่วน
    1.ทำการ นำเข้า Dataset มาจาก uci.edu
    2.ทำการ Datapreparation , Visualization เพื่อทำให้ข้อมูลมีความเหมาะสมก่อนนำไปทำการ Train model ประกอบด้วย
      การจัดการกับข้อมูลที่สูญหาย,การเปลี่ยนชนิดข้อมูลให้เหมาะสม (Numeric data-> Categorical data),การกำจัดข้อมูล Outlier โดยการใช้ IQR,
      และ การทำให้อยู่ในรูปมาตรฐานโดยการแปลงเป็น Z-score
    3.การสร้าง model โดยใช้ sklearn ในการสร้าง model Binomial Logistic Regression
    4.การ ประเมินประสิทภาพ model Result and Evaluation ได้ค่า
      Accuracy: 0.8000
      Precision: 0.7250
      Recall: 0.6974
      F1 Score: 0.7086
'''