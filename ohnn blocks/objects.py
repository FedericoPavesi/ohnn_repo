import torch
import torch.nn as nn
from fede_code import torch_functions as tof

"""
WEIGHT CONSTRAINED LINEAR

A custom linear module in which we can insert a function
which constraints weights
"""

class weightConstrainedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, weight_function=None, force_positive_init=False, dtype=torch.float64):
        super(weightConstrainedLinear, self).__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.weight_function = weight_function
        self.dtype = dtype
        self.force_positive_init = force_positive_init
        if force_positive_init: #forces positive weights initialization
            with torch.no_grad():
                self.weight.copy_(torch.abs(self.weight))

    def forward(self, input):
        assert input.dtype == self.dtype, ("Expected input dtype", self.dtype, "but got", input.dtype)
        if self.weight_function is not None:
            self.weight.data = self.weight_function(self.weight.data.T).T
        out = nn.functional.linear(input, self.weight, self.bias)
        return out

"""
HIDDEN DENSE

Hidden dense unit of the model: 
 1. Takes the input, creates a rule selecting the feature 
 with the highest magnitude weight for each neuron.
 2. Weights the rule with a parameter (a constrained one-neuron
 weight, forced to have all parameters with same, positive value)
"""

class hiddenDense(nn.Module):
    def __init__(self, input_size, num_neurons, weight_constraint=None, bias=False, force_positive_init=False, dtype=torch.float64):
        super(hiddenDense, self).__init__()
        self.input_size = input_size
        if weight_constraint is not None:
            self.weight_constraint = weight_constraint() #remember not to call it when passing arguments
        else:
            self.weight_constraint = None
        self.force_positive_init = force_positive_init
        self.linear = weightConstrainedLinear(input_size, num_neurons, weight_function=self.weight_constraint,
                                              bias=bias, force_positive_init=force_positive_init, dtype=dtype)

    def forward(self, x):
        x = self.linear(x)
        x = x.sum(dim=1) #row sum
        return x


"""
RULE'S WEIGHT DENSE

Defines the rule's weight dense network

NOTE: it's very similar to hidden dense but the last line of forward, 
for clarity is kept separate but merging is quite easy
"""

class ruleDense(nn.Module):
    def __init__(self, input_size, num_neurons, weight_constraint=None, bias=False, force_positive_init=True, dtype=torch.float64):
        super(ruleDense, self).__init__()
        self.input_size = input_size
        if weight_constraint is not None:
            self.weight_constraint = weight_constraint() #remember not to call it when passing arguments
        else:
            self.weight_constraint = None
        self.force_positive = force_positive_init
        self.linear = weightConstrainedLinear(input_size, num_neurons, weight_function=self.weight_constraint,
                                              bias=bias, force_positive_init=force_positive_init, dtype=dtype)

    def forward(self, x):
        x = self.linear(x)
        return x



"""
ONE HOT RULE NEURAL NETWORK


"""

class oneHotRuleNN(nn.Module):
    def __init__(self, input_size, num_neurons, num_rules, final_activation, rule_weight_constraint=None, out_weight_constraint=None,
                 hidden_bias=False, out_bias=False, force_positive_hidden_init=False, force_positive_out_init=True, dtype=torch.float64):
        """
        :param num_rules: Number of rules created by the network (number of hidden dense units)
        :param num_neurons: Number of neurons used during rule creation
        :param rule_weight_constraint: Weight constraint in rule's hidden dense units
        :param out_weight_constraint:  Weight constraint for out layer, i.e. rule weight
        :param hidden_bias: Whether hidden units have bias, default False, WARNING: setting to True might disrupt interpretability
        :param out_bias: Whether out unit has bias, default False, WARNING: setting to True might disrupt interpretability
        :param dtype: Input dtype, suggested torch.float32 or torch.float64, default torch.float64+
        :param force_positive_hidden_init: forces weights of hidden dense layers (rules) to be initialize as positive
        :param force_positive_out_init: forces weights of out dense layer (rule weight) to be initialized as positive
        """
        super(oneHotRuleNN, self).__init__()
        self.num_rules = num_rules
        self.num_neurons = num_neurons
        self.force_positive_hidden_init = force_positive_hidden_init
        self.force_positive_out_init = force_positive_out_init

        # hidden rules
        self.hidden_rules = nn.ModuleList([
            hiddenDense(input_size=input_size, num_neurons=num_neurons, weight_constraint=rule_weight_constraint,
                        bias=hidden_bias, force_positive_init=force_positive_hidden_init, dtype=dtype) for _ in range(num_rules)
        ])

        # rules' weight
        self.rule_weight = ruleDense(input_size=num_rules, num_neurons=1, weight_constraint=out_weight_constraint,
                                     bias=out_bias, force_positive_init=force_positive_out_init, dtype=dtype)

        self.final_activation = final_activation()

    def forward(self, x):
        # Input passes through all hidden rule dense modules, each output
        # is then summed up and concatenated
        # --> rule_outputs is n_obs x n_rules
        rule_outputs = []
        for r in range(self.num_rules):
            rule_outputs.append(self.hidden_rules[r](x).unsqueeze(1))
        rule_outputs = torch.cat(rule_outputs, dim=-1)

        # rules pass trough a dense layer in which a positive weight to each rule is assigned
        # --> out is n_obs x 1 (we are still limited to binary classification)
        out = self.rule_weight(rule_outputs)

        out = self.final_activation(out)

        return out




"""
TEST ONE-HOT RULE NN
"""

class oneHotRuleNN_test(nn.Module):
    def __init__(self, input_size, num_neurons, num_rules, final_activation, rule_weight_constraint=None, out_weight_constraint=None,
                 hidden_bias=False, out_bias=False, force_positive_hidden_init=False, force_positive_out_init=True, dtype=torch.float64):
        """
        :param num_rules: Number of rules created by the network (number of hidden dense units)
        :param num_neurons: Number of neurons used during rule creation
        :param rule_weight_constraint: Weight constraint in rule's hidden dense units
        :param out_weight_constraint:  Weight constraint for out layer, i.e. rule weight
        :param hidden_bias: Whether hidden units have bias, default False, WARNING: setting to True might disrupt interpretability
        :param out_bias: Whether out unit has bias, default False, WARNING: setting to True might disrupt interpretability
        :param dtype: Input dtype, suggested torch.float32 or torch.float64, default torch.float64+
        :param force_positive_hidden_init: forces weights of hidden dense layers (rules) to be initialize as positive
        :param force_positive_out_init: forces weights of out dense layer (rule weight) to be initialized as positive
        """
        super(oneHotRuleNN_test, self).__init__()
        self.num_rules = num_rules
        self.num_neurons = num_neurons
        self.force_positive_hidden_init = force_positive_hidden_init
        self.force_positive_out_init = force_positive_out_init

        # hidden rules
        self.linear = nn.Linear(input_size, num_rules, bias = False)
        self.out = nn.Linear(num_rules, 1)

        self.final_activation = final_activation()

    def forward(self, x):
        # Input passes through all hidden rule dense modules, each output
        # is then summed up and concatenated
        # --> rule_outputs is n_obs x n_rules
        x = self.linear(x)
        # rules pass trough a dense layer in which a positive weight to each rule is assigned
        # --> out is n_obs x 1 (we are still limited to binary classification)
        x = self.out(x)

        x = self.final_activation(x)

        return x





