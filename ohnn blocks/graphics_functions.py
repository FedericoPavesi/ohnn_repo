import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def ruleWeightHistoryPlot(history, style='whitegrid', figsize=(10,6), title='Rule weight history'):
    array = torch.cat(history).detach().numpy()

    # Create a Seaborn lineplot
    sns.set(style=style)  # Set the style if you prefer

    plt.figure(figsize=figsize)  # Set the figure size

    # Loop through each column in the array and plot it as a line
    for i in range(array.shape[1]):
        sns.lineplot(x=range(array.shape[0]), y=array[:, i], label=f"rule {i + 1}")

    plt.xlabel("Num epoch")
    plt.ylabel("Value")
    plt.title(title)

    plt.show()


def ruleHistoryPlot(history, num_rule, style='whitegrid', figsize=(10,6), threshold=1e-4):
    if isinstance(num_rule, int):
        num_rule = 'rule_' + str(num_rule)

    array = np.array([t.detach().numpy() for t in history[num_rule]])

    # Create a Seaborn lineplot
    sns.set(style=style)  # Set the style if you prefer

    plt.figure(figsize=figsize)  # Set the figure size

    for neuron in range(array.shape[1]):
        non_zero_check = np.any(abs(array[:, neuron, :]) > threshold, axis=0)
        columns_with_non_zero = np.where(non_zero_check)[0]
        for col in columns_with_non_zero:
            sns.lineplot(x=range(array.shape[0]), y=array[:, neuron, col],
                         label=f'Neuron number {neuron + 1}, feature number {col + 1}')

    plt.xlabel("Num epoch")
    plt.ylabel("Value")
    plt.title(num_rule + ' history')

    plt.show()

def neuronHistoryPlot(history, num_rule, num_neuron, style='whitegrid', figsize=(10,6)):
    if isinstance(num_rule, int):
        num_rule = 'rule_' + str(num_rule)
    array = np.array([t.detach().numpy() for t in history[num_rule]])

    array = array[:, num_neuron, :]

    # Create a Seaborn lineplot
    sns.set(style=style)  # Set the style if you prefer

    plt.figure(figsize=figsize)  # Set the figure size

    # Loop through each column in the array and plot it as a line
    for i in range(array.shape[1]):
        sns.lineplot(x=range(array.shape[0]), y=array[:, i], label=f'weight {i + 1}')

    plt.xlabel("Num epoch")
    plt.ylabel("Value")
    plt.title("Neuron " + str(num_neuron) + ' ' + num_rule + ' ' + ' history')

    plt.show()


def obtainLinearFormula(trainer, dtype=torch.float32):
    num_rules = len(trainer.model.hidden_rules)
    num_features = trainer.model.hidden_rules[0].linear.in_features

    pre_out = torch.zeros((num_rules, num_features), dtype=dtype)

    for num_rule in range(pre_out.size()[0]):
        wel = trainer.model.hidden_rules[num_rule].linear.weight.data

        val, pos = torch.max(abs(wel), axis=-1)

        val = wel[torch.arange(wel.shape[0]), pos]

        rul_w = trainer.model.rule_weight.linear.weight.data[0][num_rule]

        val *= rul_w

        for p, v in zip(pos, val):
            pre_out[num_rule, p] += v

    return pre_out.sum(axis=0)
