import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fede_code.torch_functions as tof
import fede_code.objects as obj
import fede_code.model_trainer as mt
import fede_code.graphics_functions as grafun
from torch.optim import AdamW, RMSprop
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


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

train_set = mt.TrainDataset(torch.tensor(x_train, dtype=torch.float32),
                     torch.tensor(y_train, dtype=torch.float32))

test_set = mt.TrainDataset(torch.tensor(x_test, dtype=torch.float32),
                           torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

#%%

"""
-----------------------------------------------------------------------
FIT A SVM MODEL
----------------------------------------------------------------------
"""

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

"""
---------------------------------------------------------------
USE OHNN TO EXPLAIN PREDICTIONS
---------------------------------------------------------------
"""

ohnn_model = obj.oneHotRuleNN(input_size=13,
                              num_neurons=4,
                              num_rules=8,
                              final_activation=tof.Sign,
                              rule_weight_constraint=tof.getMax,
                              out_weight_constraint=tof.forcePositive,
                              dtype=torch.float32)

#loss_fn = nn.KLDivLoss(reduction='sum')
loss_fn = nn.SoftMarginLoss(reduction='sum')

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
                    learning_rate=1e-5,
                    display_log='epoch')

#%%


fig1 = sns.lineplot(trainer.history['train_loss'])
plt.show()

#%%

fig2 = sns.lineplot(trainer.history['train_accuracy'])
plt.show()

#%%

grafun.ruleWeightHistoryPlot(trainer.observer_history['rule_weight'])

#%%

grafun.ruleHistoryPlot(trainer.observer_history, num_rule=1)


#%%

grafun.neuronHistoryPlot(trainer.observer_history, num_rule=0, num_neuron=0)

#%%

linear_model = grafun.obtainLinearFormula(trainer)


#%%

"""
---------------------------------------------------------------
USE OHNN TO PREDICT
---------------------------------------------------------------
"""

ohnn_model = obj.oneHotRuleNN(input_size=13,
                              num_neurons=4,
                              num_rules=8,
                              final_activation=tof.Sign,
                              rule_weight_constraint=tof.getMax,
                              out_weight_constraint=tof.forcePositive,
                              dtype=torch.float32)

#loss_fn = nn.KLDivLoss(reduction='sum')
loss_fn = nn.SoftMarginLoss(reduction='sum')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('GPU')
else:
    device = torch.device('cpu')
    print('CPU')


trainer = mt.modelTrainer(model=ohnn_model,
                          loss_fn=loss_fn,
                          optimizer=AdamW)

trainer.train_model(train_data_loader=train_loader,
                    num_epochs=10000,
                    device=device,
                    learning_rate=1e-5,
                    display_log='epoch')

#%%

# test performances

trainer.eval_model(test_loader, device='cpu')


#%%

fig1 = sns.lineplot(trainer.history['train_loss'])
plt.show()

#%%

fig2 = sns.lineplot(trainer.history['train_accuracy'])
plt.show()

#%%

grafun.ruleWeightHistoryPlot(trainer.observer_history['rule_weight'])

#%%

grafun.ruleHistoryPlot(trainer.observer_history, num_rule=0)


#%%

grafun.neuronHistoryPlot(trainer.observer_history, num_rule=0, num_neuron=0)


#%%

linear_model = grafun.obtainLinearFormula(trainer)

#%%

out_linear = torch.matmul(torch.tensor(x_test, dtype=torch.float32), linear_model.unsqueeze(-1))
pred_linear = tof.Sign()(out_linear).squeeze()

loss = loss_fn(pred_linear, torch.tensor(y_test, dtype=torch.float32))/len(pred_linear)

accuracy = (pred_linear == torch.tensor(y_test, dtype=torch.float32)).sum().item()/len(pred_linear)

print(loss.item())
print(accuracy)

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