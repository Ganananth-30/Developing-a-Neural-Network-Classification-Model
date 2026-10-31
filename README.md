# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Write your own steps

### STEP 2: 



### STEP 3: 



### STEP 4: 



### STEP 5: 



### STEP 6: 





## PROGRAM

### Name:

### Register Number:

```

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("customers.csv")
df

df=df.drop(columns=["ID"])
df

df.columns

df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)
df

cat_columns=['Gender','Ever_Married', 'Graduated', 'Profession',
    'Spending_Score', 'Var_1']
for col in cat_columns:
    df[col]=LabelEncoder().fit_transform(df[col])

lbe=LabelEncoder()
df["Segmentation"]=lbe.fit_transform(df["Segmentation"])
df

x=df.drop(columns="Segmentation")
y=df["Segmentation"].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
xt=scaler.fit_transform(xt)
xst=scaler.transform(xst)

xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)

tr=TensorDataset(xt,yt)
tst=TensorDataset(xst,yst)
trl=DataLoader(tr,batch_size=16,shuffle=True)
tstl=DataLoader(tst,batch_size=16)

class classifier1(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1=nn.Linear(input_size,32)
        self.l2=nn.Linear(32,16)
        self.l3=nn.Linear(16,8)
        self.l4=nn.Linear(8,4)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=self.l4(x)
        return x


model=classifier1(input_size=xt.shape[1])
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)

epochs=100
for i in range(epochs):
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b.long())
        loss.backward()
        op.step()
    if i%10==0:
        print(f"Loss:{i}/{epochs}",loss.item())



pre=[]
act=[]
with torch.no_grad():
    output=model(xst)
    _,predicted=torch.max(output,1)
    pre.extend(predicted.numpy())
    act.extend(yst.numpy())
    

accuracy=accuracy_score(act,pre)
conf_matrix=confusion_matrix(act,pre)
cl_report=classification_report(act,pre,target_names=['A','B','C','D'])
print("Accuracy:",accuracy)
print("confusion_matrix:\n",conf_matrix)
print("classification_report:\n",cl_report)

import seaborn as sns
import matplotlib.pyplot as plt
xl=['A','B','C','D']
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=xl, yticklabels=xl,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Prediction for a sample input
sample_input = xst[12].clone().unsqueeze(0).to(torch.float32)

with torch.no_grad():
    output = model(sample_input)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = lbe.inverse_transform([predicted_class_index])[0]
print(f"Predicted class for sample input: {predicted_class_label}")
print(f"Actual class for sample input: {lbe.inverse_transform([int(yst[12].item())])[0]}")


```

### Dataset Information
Include screenshot of the dataset.


<img width="1367" height="487" alt="image" src="https://github.com/user-attachments/assets/02506f85-c722-4e65-b552-998f3da0ae18" />


### OUTPUT


<img width="1401" height="571" alt="image" src="https://github.com/user-attachments/assets/7784ff1c-8536-43ed-8c9f-5de87ebb7c99" />

## Confusion Matrix

Include confusion matrix here


<img width="1295" height="623" alt="image" src="https://github.com/user-attachments/assets/46e2ab2a-13d8-44d9-b0e8-270a8a1b62fc" />


<img width="1240" height="608" alt="image" src="https://github.com/user-attachments/assets/de86216b-3c7f-4b9b-9ed2-80839906478f" />


<img width="625" height="340" alt="image" src="https://github.com/user-attachments/assets/7988bb06-68d2-4c4d-92ef-dd124b462b17" />


<img width="797" height="512" alt="image" src="https://github.com/user-attachments/assets/da9efd9a-c120-4d19-af55-f1b2a90bc66d" />


## Classification Report
Include classification report here

<img width="842" height="600" alt="image" src="https://github.com/user-attachments/assets/ffa0c940-234b-4a18-a47c-119ee1b6e524" />


### New Sample Data Prediction
Include your sample input and output here


<img width="1090" height="381" alt="image" src="https://github.com/user-attachments/assets/63b9eb94-7b20-4ff0-901a-d37e5da26959" />

## RESULT
Include your result here
