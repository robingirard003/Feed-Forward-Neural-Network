import numpy as np
import matplotlib as mpl
import pandas as pd
import os

path = r"C:\Users\robin\Documents\Erasmus School of Economics\Courses\Machine Learning\Assignement"
os.chdir(path)

np.random.seed(2026)


X_test = pd.read_csv("X_test.csv")
X_trn = pd.read_csv("X_trn.csv")
Y_trn = pd.read_csv("Y_trn.csv")

X_test = np.array(X_test, dtype=float)
X_trn = np.array(X_trn, dtype=float)
Y_trn = np.array(Y_trn, dtype=float)

# Splitting the training data into training and validation sets

Y_trn = Y_trn.flatten()

Y_trn_0 = np.flatnonzero(Y_trn == 0)
Y_trn_1 = np.flatnonzero(Y_trn == 1)

Y_trn_0 = np.random.permutation(Y_trn_0)
Y_trn_1 = np.random.permutation(Y_trn_1)

n_train_0 = int(0.8 * len(Y_trn_0))
n_train_1 = int(0.8 * len(Y_trn_1))

n_test_0 = len(Y_trn_0) - n_train_0
n_test_1 = len(Y_trn_1) - n_train_1

training_idx = np.concatenate((Y_trn_0[:n_train_0], Y_trn_1[:n_train_1]))
training_idx = np.random.permutation(training_idx)
test_idx = np.concatenate((Y_trn_0[n_train_0:n_train_0 + n_test_0], Y_trn_1[n_train_1:n_train_1 + n_test_1]))
test_idx = np.random.permutation(test_idx)

X_training, X_val = X_trn[training_idx,:], X_trn[test_idx,:]
Y_training, Y_val = Y_trn[training_idx], Y_trn[test_idx] # Y is now one dimensional array so we index it with one index, instead of two.

X_train_scaled = X_training.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()
mu = []
std = []

# Normalization of the data using z-score normalization (normal distribution)
for i in range(X_train_scaled.shape[1]):
    mu_column, std_column = np.mean(X_train_scaled[:, i]), np.std(X_train_scaled[:, i])
    mu.append(mu_column)
    std.append(std_column)
    # mu and sigma of each column of X_train_scaled

    if std_column == 0:
        raise ValueError("Standard deviation is zero for feature index {}".format(i))
    
    X_train_scaled[:, i] = (X_train_scaled[:, i] - mu_column) / std_column
    X_val_scaled[:, i] = (X_val_scaled[:, i] - mu_column) / std_column
    X_test_scaled[:, i] = (X_test_scaled[:, i] - mu_column) / std_column

#Architecture of the neural network
layer_sizes = [23, 16, 8, 1]
size_W = [[layer_sizes[i], layer_sizes[i+1]] for i in range(len(layer_sizes)-1)]

weights = [np.random.normal(0, np.sqrt(2/size[0]), (size[0], size[1])) for size in size_W] 
weights[-1] = np.random.normal(0, 1/np.sqrt(size_W[-1][0]), (size_W[-1][0], size_W[-1][1]))
# Returns list of weights matrices
# The weights are initialized using He initialization for the hidden layers 
# and Xavier initialization for the output layer.

biases = [np.zeros((1, size[1])) for size in size_W]
# The biases are initialized to zero for all layers.

# Forward pass
def Relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def forward_pass(X, weights, biases):
    A_list = [X]
    Z_list = []
    for i in range(len(weights)):
        Z_l = np.dot(A_list[i], weights[i]) + biases[i]
        Z_list.append(Z_l)
        if i < len(weights) - 1:
            A = Relu(Z_l)
        else:
            A = sigmoid(Z_l)
        A_list.append(A)
    return (A_list, Z_list)

print(forward_pass(X_train_scaled, weights, biases))

def weighted_loss(Y_true, Y_pred, weight: tuple):
    Y_true = Y_true.reshape(-1 , 1)
    Y_pred = Y_pred.reshape(-1, 1)
    assert len(Y_true) == len(Y_pred), "Length of Y_true and Y_pred must be the same"
    assert len(Y_true) > 0 and len(Y_pred) > 0, "Y_true and Y_pred must not be empty"
    a, b = weight 
    # a corresponds to the penalty afflicted to the scored if a true default is missed
    # b corresponds to the penalty afflicted to the score if a faulse default is raised
    Y_true_0 = np.flatnonzero(Y_true == 0) #Filters all the indices where we have a true non-default
    Y_true_1 = np.flatnonzero(Y_true == 1) #Filters all the indices where we have a true default
    sum  = np.sum(np.square(Y_true[Y_true_1] - Y_pred[Y_true_1])) * a + np.sum(np.square(Y_true[Y_true_0] - Y_pred[Y_true_0])) * b
    # The loss is calculated by summing the squared differences between the true and predicted values,
    # weighted by the specified penalties for true defaults and non-defaults.
    return sum/len(Y_true)

# Briar score is a specific case of the Asymetric Economic Loss function with weights equal to 1

def derivative_weighted_loss(Y_true, Y_pred, weight: tuple):
    Y_true = Y_true.reshape(-1 , 1)
    Y_pred = Y_pred.reshape(-1, 1)
    assert len(Y_true) == len(Y_pred), "Length of Y_true and Y_pred must be the same"
    assert len(Y_true) > 0 and len(Y_pred) > 0, "Y_true and Y_pred must not be empty"
    a, b  = weight
    Y_true_0 = np.flatnonzero(Y_true == 0) #Filters all the indices where we have a true non-default
    Y_true_1 = np.flatnonzero(Y_true == 1) #Filters all the indices where we have a true default
    derivative = np.zeros_like(Y_true)
    derivative[Y_true_0] = -2 * b * (Y_true[Y_true_0] - Y_pred[Y_true_0]) / len(Y_true)
    derivative[Y_true_1] = -2 * a * (Y_true[Y_true_1] - Y_pred[Y_true_1]) / len(Y_true)
    return derivative

print(weighted_loss(Y_training, forward_pass(X_train_scaled, weights, biases)[0][-1], (1, 1)))
print(weighted_loss(Y_training, Y_training.reshape(-1,1), (1,1)))
print(weighted_loss(Y_training, 0.5 * np.ones_like(Y_training.reshape(-1 , 1)), (1,1)))
print(derivative_weighted_loss(Y_training, forward_pass(X_train_scaled, weights, biases)[0][-1], (1, 1)))


# Backpropagation
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1-s)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Exit layer
def backward_pass(Y_true, A_list, Z_list, weights, biases, weight: tuple):
    Y_true = Y_true.reshape(-1 , 1)
    Y_pred = A_list[-1]
    assert len(Y_true) > 0 and len(Y_pred) > 0, "Y_true and Y_pred must not be empty"
    a, b  = weight
    dW_list = [None] * len(weights)
    dbiases_list = [None] * len(biases)
    assert len(dW_list) == len(dbiases_list), "Length of dW_list and dbiases_list must be the same"
    dL_dA_last = derivative_weighted_loss(Y_true, Y_pred, weight)
    dL_dZ_last = dL_dA_last * sigmoid_derivative(A_list[-1])
    dL_dZ_courant = dL_dZ_last

    dL_dW_last = np.dot(A_list[-2].T, dL_dZ_last)
    dL_db_last = np.sum(dL_dZ_last, axis=0, keepdims=True)

    dW_list[-1] =dL_dW_last
    dbiases_list[-1] = dL_db_last

    for i in range(len(weights)-2, -1, -1):
        dL_dA_courant = np.dot(dL_dZ_courant, weights[i+1].T)
        dL_dZ_courant = dL_dA_courant * relu_derivative(Z_list[i])
        dW_list[i] = np.dot(A_list[i].T, dL_dZ_courant)
        dbiases_list[i] = np.sum(dL_dZ_courant, axis=0, keepdims=True)

    return dW_list, dbiases_list

# Update parameters

def update_params(weights, biases, dW_list, db_list, learning_rate):
    dW_list_copy = [np.copy(dW) for dW in weights]
    dbiases_list_copy = [np.copy(b) for b in biases]
    for i in range (len(weights)):
        dW_list_copy[i] -= learning_rate * dW_list[i]
        dbiases_list_copy[i] -= learning_rate * db_list[i]
    return dW_list_copy, dbiases_list_copy



w_original = [np.array([[1.0, 2.0]])]
b_original = [np.array([[0.0, 0.0]])]
dW = [np.array([[1.0, 1.0]])]
db = [np.array([[1.0, 1.0]])]

w_new, b_new = update_params(w_original, b_original, dW, db, 0.1)
print("w_original a-t-il changé ?", w_original[0])   # doit rester [[1.0, 2.0]]
print("w_new :", w_new[0])                           # doit être [[0.9, 1.9]]


# Mini-batch gradien descent
def train(X_train, Y_train, X_val, Y_val, layer_sizes, weight_tuple, learning_rate, batch_size, max_epochs, patience):
    size_W = [[layer_sizes[i], layer_sizes[i+1]] for i in range(len(layer_sizes)-1)]
    weights = [np.random.normal(0, np.sqrt(2/size[0]), (size[0], size[1])) for size in size_W] 
    weights[-1] = np.random.normal(0, 1/np.sqrt(size_W[-1][0]), (size_W[-1][0], size_W[-1][1]))
    biases = [np.zeros((1, size[1])) for size in size_W]

    perte_validation = []
    best_val_loss = np.inf
    epochs_sans_amelioration = 0
    best_weights = None
    best_biases = None

    for epoch in range(max_epochs):
        permutation = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[permutation]
        Y_train_shuffled = Y_train[permutation]

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            Y_batch = Y_train_shuffled[i:i+batch_size]

            A_list, Z_list = forward_pass(X_batch, weights, biases)
            dW_list, dbiases_list = backward_pass(Y_batch, A_list, Z_list, weights, biases, weight_tuple)
            dW_list_copy, dbiases_list_copy = update_params(weights, biases, dW_list, dbiases_list, learning_rate)
            weights, biases = dW_list_copy, dbiases_list_copy

        A_list_val_scaled, Z_list_val_scaled= forward_pass(X_val, weights, biases)
        perte_validation.append(weighted_loss(Y_val, A_list_val_scaled[-1], weight_tuple))

        
        if perte_validation[-1] < best_val_loss:
            best_val_loss = perte_validation[-1]
            best_weights = [np.copy(w) for w in weights]
            best_biases = [np.copy(b) for b in biases]
            epochs_sans_amelioration = 0
        else:
            epochs_sans_amelioration += 1
        if epochs_sans_amelioration >= patience:
            print(f"Early stopping at epoch {epoch}")
            return best_weights, best_biases, perte_validation

    return best_weights, best_biases, perte_validation



# Hyperparameters tuning
layer_sizes_list = [[23,16,8,1], [23,32,16,1], [23,16,1]]
learning_rates_list = [0.01, 0.05, 0.1]
batch_size_list = [64, 128, 256]
weight_tuple = (1, 1) #Briar score's weights

def hyperparameter_search(X_train, Y_train, X_val, Y_val, layer_sizes_list, learning_rates_list, batch_size_list, weight_tuple, max_epochs=1000, 
                          patience=10):
    best_val_loss = np.inf
    best_hyperparams = None
    best_weights = None
    best_biases = None
    best_perte_validation = None

    for layer_sizes in layer_sizes_list:
        for learning_rate in learning_rates_list:
            for batch_size in batch_size_list:
                np.random.seed(2026)
                print(f"Training with layer_sizes={layer_sizes}, learning_rate={learning_rate}, batch_size={batch_size}")
                weights, biases, perte_validation = train(X_train, Y_train, X_val, Y_val, layer_sizes, weight_tuple, learning_rate, batch_size, 
                                                          max_epochs, patience)
                val_loss = min(perte_validation)
                print(f"Validation loss: {val_loss}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_hyperparams = (layer_sizes, learning_rate, batch_size)
                    best_weights = weights
                    best_biases = biases
                    best_perte_validation = perte_validation

    return best_hyperparams, best_weights, best_biases, best_perte_validation

#print(hyperparameter_search(X_train_scaled, Y_training, X_val_scaled, Y_val, layer_sizes_list, learning_rates_list, batch_size_list, weight_tuple))


# Training 2 models with hyperparameters: ([23, 16, 1], 0.1, 128)
# Brier score model with weight_tuple = (1, 1)
best_weights_brier, best_biases_brier, best_perte_validation_brier = train(X_train_scaled, Y_training, X_val_scaled, Y_val, [23, 16, 1],
                                                                          (1, 1), learning_rate=0.1, batch_size=128, max_epochs=1000, patience=10) 
# Asymmetric Economic Loss model with weight_tuple = (3, 1)
weight_tuple = (3, 1)
best_weights_ael, best_biases_ael, best_perte_validation_ael = train(X_train_scaled, Y_training, X_val_scaled, Y_val, [23, 16, 1],
                                                                          (3, 1), learning_rate=0.1, batch_size=128, max_epochs=1000, patience=10) 
print("Brier score model: ", best_weights_brier, best_biases_brier, best_perte_validation_brier)
print("Asymmetric Economic Loss model: ", best_weights_ael, best_biases_ael, best_perte_validation_ael)


# Evaluation and comparison

def evaluate(Y_true, Y_pred_proba, threshold=0.5):
    Y_true = Y_true.reshape(-1)
    Y_pred_proba = Y_pred_proba.reshape(-1)
    vp = np.sum((Y_true == 1) & (Y_pred_proba >= threshold))
    fp = np.sum((Y_true == 0) & (Y_pred_proba >= threshold))
    vn = np.sum((Y_true == 0) & (Y_pred_proba < threshold))
    fn = np.sum((Y_true == 1) & (Y_pred_proba < threshold))
    taux_fn = fn/(fn + vp) if (fn + vp) > 0 else 0
    taux_fp = fp/(fp + vn) if (fp + vn) > 0 else 0
    perte = weighted_loss(Y_true, Y_pred_proba, (1, 1))
    return {"taux_fn": taux_fn, "taux_fp": taux_fp, "vp": vp, "fp": fp, "vn": vn, "fn": fn, "perte": perte}

print("Résultats de l'évaluation pour le modèle Brier:")
print(evaluate(Y_val, forward_pass(X_val_scaled, best_weights_brier, best_biases_brier)[0][-1]))
# On obtient un taux de faux négatifs de 65% et un taux de faux positifs de 5%
print("Résultats de l'évaluation pour le modèle Asymmetric Economic Loss:")
print(evaluate(Y_val, forward_pass(X_val_scaled, best_weights_ael, best_biases_ael)[0][-1]))
# On obtient un taux de faux négatifs de 43% et un taux de faux positifs de 16%


# Final prediction on X_test
# Evaluation of the Brier score model
Y_test_pred_brier = forward_pass(X_test_scaled, best_weights_brier, best_biases_brier)[0][-1]
# Evaluation of the Asymmetric Economic Loss model
Y_test_pred_ael = forward_pass(X_test_scaled, best_weights_ael, best_biases_ael)[0][-1]

Y_test_pred = np.concatenate((Y_test_pred_brier, Y_test_pred_ael), axis=1)
np.save("predictions.npy", Y_test_pred)
#np.savetxt("predictions_lisibles.csv", Y_test_pred, delimiter=",") # Fichier lisible depuis excel

print(Y_test_pred.shape)
print(Y_test_pred)
np.load("predictions.npy")