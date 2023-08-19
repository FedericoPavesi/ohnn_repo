# TEST ON DATASETS

## 1. DATASET HEARTH

SI utilizza un SVM per fare previsioni sul dataset hearth, il quale raggiunge un'accuratezza circa dell'83%. Sulle previsioni del modello SVM, si testa il modello OHNN con 8 regole da 4 neuroni, per 10,000 epoche. Di seguito sono riportati alcuni grafici sulla storia dell'addestramento del modello

__immagine 1: LOSS HISTORY__
[!alt text](https://github.com/FedericoPavesi/ohnn_repo/blob/main/hearh_ohnn_loss_history.png)

__immagine 2: ACCURACY HISTORY__
[!alt text](https://github.com/FedericoPavesi/ohnn_repo/blob/main/hearth_ohnn_accuracy_history.png)

__immagine 3: RULE WEIGHT HISTORY__
[!alt text](https://github.com/FedericoPavesi/ohnn_repo/blob/main/rule_weight_history.png)

Si noti l'importanza del LeakyReLU nel backrpopagation step

__immagine 4: RULE 2, NEURON 3 HISTORY__
[!alt text](https://github.com/FedericoPavesi/ohnn_repo/blob/main/Neuron_3_rule_2_history_weight_change.png)

Esempio cambio di peso e di rispettiva variabile in una regola
