# OHNN: ONE HOT NEURAL NETWORK

## Introduzione

Questa rete si basa sull'imposizione di costrizioni ai pesi dei neuroni di una rete neurale in modo da mantenere la possibilità di una semplice interpretazione dei pesi, una volta addestrato il modello. La struttura è come segue:

Ci sono $K$ blocchi densi che ricevono in input i dati, ognuno ha il medesimo numero di neuroni $N$. Ogni neurone ha $m$ pesi (uno per ogni feature di input dei dati), di questi tuttavia, solo quello con il valore assoluto più alto viene conservato, mentre gli altri sono impostati a zero. Quindi, per ogni blocco, ogni neurone seleziona una feature e vi assegna un peso. A questo punto, per ogni blocco, tutti i neuroni sono sommati in modo da ottenere una regola del tipo:

Blocco $k$ ha regola

$r_k = w_{k,1} * a_{\beta_1} + w_{k,2} * a_{\beta_2} + ... + w_{k,N} * a_{\beta_N}$

Dove $a_{\beta_n}$ indica la feature selezionata dal neurone $n$, associata al rispettivo peso $w_{k,n}$.