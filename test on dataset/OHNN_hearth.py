import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fede_code.torch_functions as tof
import fede_code.objects as obj
import fede_code.model_trainer as mt
from torch.optim import AdamW, RMSprop
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score



#%%


headers = ['age', 'sex', 'chest_pain', 'resting_blood_pressure',
           'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
           'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak', "slope of the peak",
           'num_of_major_vessels', 'thal', 'heart_disease']

df_heart = pd.read_csv("data/heart.csv")

print(df_heart.head())

#%%

x = df_heart.drop(columns=['target'])
y = df_heart['target'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#%%
#standardize the dataset
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

#%%

train_set = mt.TrainDataset(torch.tensor(x_train),
                     torch.tensor(y_train))

test_set = mt.TrainDataset(torch.tensor(x_test),
                           torch.tensor(y_test))

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

#%%

svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(x_train,y_train)
predicted = svm_clf.predict(x_test)
f1_svm = f1_score(y_test,predicted)
acc_svm = accuracy_score(y_test, predicted)

print('f1 score SVM:', f1_svm)
print('accuracy SVM:', acc_svm)

#%%

# build the dataset of predictions to perform explanation
pred_set = mt.TrainDataset(torch.tensor(x_test),
                           torch.tensor(predicted))

pred_loader = DataLoader(pred_set, batch_size=16, shuffle=True)


#%%

ohnn_model = obj.oneHotRuleNN(input_size=13,
                              num_neurons=4,
                              num_rules=8,
                              final_activation=nn.Sigmoid,
                              rule_weight_constraint=tof.getMax,
                              out_weight_constraint=tof.forcePositive,
                              dtype=torch.float32)

loss_fn = nn.BCELoss()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('GPU')
else:
    device = torch.device('cpu')
    print('CPU')


trainer = mt.modelTrainer(model=ohnn_model,
                          loss_fn=loss_fn,
                          optimizer=AdamW)

trainer.train_model(train_data_loader=pred_loader,
                    num_epochs=10000,
                    device=device,
                    learning_rate=1e-5)


#%%

import seaborn as sns
import matplotlib.pyplot as plt

fig1 = sns.lineplot(trainer.history['train_loss'])
plt.show()

#%%

fig2 = sns.lineplot(trainer.history['train_accuracy'])
plt.show()

#%%

array = torch.cat(trainer.observer_history['rule_weight']).detach().numpy()

# Create a Seaborn lineplot
sns.set(style="whitegrid")  # Set the style if you prefer

plt.figure(figsize=(10,6))  # Set the figure size

# Loop through each column in the array and plot it as a line
for i in range(array.shape[1]):
    sns.lineplot(x=range(array.shape[0]), y=array[:, i], label=f"rule {i+1}")

plt.xlabel("Num epoch")
plt.ylabel("Value")
plt.title("Rule weight history")

plt.show()

#%%
rule_weight_num = 0
sns.lineplot(x=range(array.shape[0]), y=array[:,rule_weight_num])
plt.show()


#%%
import numpy as np

num_neuron = 3
num_rule = 'rule_2'

array = np.array([t.detach().numpy() for t in trainer.observer_history[num_rule]])

array = array[:, num_neuron, :]

# Create a Seaborn lineplot
sns.set(style="whitegrid")  # Set the style if you prefer

plt.figure(figsize=(10, 6))  # Set the figure size

# Loop through each column in the array and plot it as a line
for i in range(array.shape[1]):
    sns.lineplot(x=range(array.shape[0]), y=array[:, i], label=f'weight {i+1}')

plt.xlabel("Num epoch")
plt.ylabel("Value")
plt.title("Neuron " + str(num_neuron) + ' ' + num_rule + ' ' + ' history')

plt.show()

#%%


##############
# GRADIENTS
##############

array = torch.cat(trainer.grad_history['rule_weight']).detach().numpy()

# Create a Seaborn lineplot
sns.set(style="whitegrid")  # Set the style if you prefer

plt.figure(figsize=(10,6))  # Set the figure size

# Loop through each column in the array and plot it as a line
for i in range(array.shape[1]):
    sns.lineplot(x=range(array.shape[0]), y=array[:, i])

plt.xlabel("Row Index")
plt.ylabel("Value")
plt.title("Tensor Columns as Lines")

plt.show()





