# OHNN: ONE HOT NEURAL NETWORK

## 1. Introduzione

Questa rete si basa sull'imposizione di costrizioni ai pesi dei neuroni di una rete neurale in modo da mantenere la possibilità di una semplice interpretazione dei pesi, una volta addestrato il modello. La struttura è come segue:

![alt text](https://github.com/FedericoPavesi/ohnn_repo/blob/main/ohnn_network_explanation.png)

Ci sono $K$ blocchi densi che ricevono in input i dati, ognuno ha il medesimo numero di neuroni $N$. Ogni neurone ha $m$ pesi (uno per ogni feature di input dei dati), di questi tuttavia, solo quello con il valore assoluto più alto viene conservato, mentre gli altri sono impostati a zero. Quindi, per ogni blocco, ogni neurone seleziona una feature e vi assegna un peso. A questo punto, per ogni blocco, tutti i neuroni sono sommati in modo da ottenere una regola del tipo:

Blocco $k$ ha __regola__

$r_k = w_{k,1} * a_{\beta_1} + w_{k,2} * a_{\beta_2} + ... + w_{k,N} * a_{\beta_N}$

Dove $a_{\beta_n}$ indica la feature selezionata dal neurone $n$, associata al rispettivo peso $w_{k,n}$.

Ogni regola proveniente da $K$ blocchi viene poi pesata da un singolo peso, chiamato $W_{f_k}$ per la regola $k$. Assegnato il peso ad ogni regola, si sommano tutte le regole:

$Out = W_{f_1} * r_1 + W_{f_2} * r_2 + ... + W_{f_K} * r_K$

A questo punto il risultato $Out$ viene passato attraverso una sigmoide, per ottenere una previsione binaria.

$Previsione = 1/(1 + e^{Out})$



## 2. Blocco denso

Il blocco denso è impostato con N neuroni, per cui, definita con $W$ la matrice dei pesi, essa avrà dimensione $m \times N$, con $m$ numero di feature di input. Ogni neurone nel forward pass, invece di avere $m$ pesi con valori differenti, avrà il peso con valore assoluto maggiore che conserva il suo valore, metre gli altri $m-1$ pesi vengono impostati a zero. L'ouput del singolo neurone da:

$w_1 * a_1 + w_2 * a_2 + ... + w_m * a_m$

diventa

$w_i*a_i$

Dove $i$ corrisponde al peso con valore assoluto maggiore. Si noti tuttavia che durante la backpropagation questa modifica viene ignorata, per cui è possibile che a un certo punto dell'addestramento un peso che supera in valore assoluto il peso $i$, prenda il posto del peso $i$, mentre quest'ultimo diventa zero. 

__Possibili cambiamenti__:
  - utilizzo di una funzione che, oltre a selezionare un solo peso, forzi il valore tra -1 e +1. 

## 3. Aggregazione del blocco denso: regola

Attraverso l'aggregazione dell'output del blocco denso, è possibile ottenere una sorta di regola del tipo:

$r_k = w_{k,1} * a_{\beta_1} + w_{k,2} * a_{\beta_2} + ... + w_{k,N} * a_{\beta_N}$

Dove $a_{\beta_n}$ indica la feature selezionata dal neurone $n$, associata al rispettivo peso $w_{k,n}$.

__Possibili cambiamenti__:
  - Si possono sperimentare differenti funzioni di aggregazione.

## 4. Peso delle regole

Per come è strutturato il modello, si otterranno $K$ regole. Ad ogni regola, si vuole assegnare un peso, chiamato $W_{f_k}$ per la regola $k$. Ogni regola viene inizializzata ad un numero maggiore di zero, ed è forzata, nel forward pass, ad un valore positivo o nullo (ovvero, nel caso durante l'addestramento diventi negativa, il valore viene fermato a zero). Questo garantisce l'interpretabilità dei pesi delle regole. 

$Out = W_{f_1} * r_1 + W_{f_2} * r_2 + ... + W_{f_K} * r_K$

Si noti che durante la backpropagation il gradiente è leggermente modificato, per cui se durante il forward pass è come se i pesi passassero per una ReLU, durante la backpropagation l'output è il gradiente di una LeakyReLU. Questo garantisce che, una volta che il peso viene azzerato, possa in futuro tornare positivo poichè il suo valore continua a mutare.


## 5. Attivazione finale

Questa resta la parte più sensibile del modello, se da un lato l'interpretabilità in casi di regressione è conservata, nel caso di classificazione binaria è sicuramente più complicata (trattandosi di log-odds) e nel caso si utilizzi una softmax per classificazione multi-label, questa viene completamente persa.

__Possibili cambiamenti__:
  - Trovare delle funzioni di attivazione finale (o di perdita) che permettano di conservare l'interpretabilità del modello.

## 6. Limitazioni

Il principale limite del modello è la forte linearità, difatti se prendiamo l'equazione di $Out$

$Out = W_{f_1} * r_1 + W_{f_2} * r_2 + ... + W_{f_K} * r_K$

ed esplicitiamo le regole

$Out = W_{f_1} * (w_{1,1} * a_{\beta_1} + ... + w_{1,N} * a_{\beta_N}) + ... + W_{f_K} * (w_{K,1} * a_{\gamma_1} + ... + w_{K,N} * a_{\gamma_N})$

Si nota immediatamente che, moltiplicando ogni elemento per il peso assegnato alla regola ed eventualmente raccogliendo le features identiche (per cui $a_{\beta_s}$ è la medesima feature di $a_{\gamma_t}$), si ottiene un'equazione lineare.

Tuttavia questo è anche il grande punto di forza della rete, poichè un'equazione lineare è facilmente interpretabile

