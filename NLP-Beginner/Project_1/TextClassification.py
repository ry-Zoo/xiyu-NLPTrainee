
import math
import re
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pandas as pd
# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.3


#X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)



def load_TrainingData(path):     #loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path, sep='\t', header=0)
    feature_names  = np.array(list(D.columns.values))
    #print(D[['PhraseId','Phrase']])
    X_train_byword=[]
    for sentence in list(D['Phrase']):
        re.sub(r'[,.\/-`﹑•＂^…‘’“”〝〞~\∕|¦‖—　()〈〉﹞﹝「」‹›〖〗】【»«』『〕〔》《}{\]\[﹐¸﹕︰﹔;！¡？¿﹖﹌﹏﹋＇´ˊˋ-―﹫@︳︴_¯＿￣﹢+﹦=﹤‐<­˜~﹟#﹩$﹠&﹪%﹡*﹨\\﹍﹉﹎﹊ˇ︵︶︷︸︹︿﹀︺︽︾_ˉ﹁﹂﹃﹄︻︼]', "", sentence)
        sentence_ls=sentence.split(" ")
        X_train_byword.append(sentence_ls)
    X_train = np.array(X_train_byword)
    Y_train = np.array(list(D['Sentiment']));
    return  X_train, Y_train, feature_names

def load_TestingData(path):     #loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path, sep='\t', header=0)
    X_test_byword=[]
    for sentence in list(D['Phrase']):
        re.sub(r'[,.\/-`﹑•＂^…‘’“”〝〞~\∕|¦‖—　()〈〉﹞﹝「」‹›〖〗】【»«』『〕〔》《}{\]\[﹐¸﹕︰﹔;！¡？¿﹖﹌﹏﹋＇´ˊˋ-―﹫@︳︴_¯＿￣﹢+﹦=﹤‐<­˜~﹟#﹩$﹠&﹪%﹡*﹨\\﹍﹉﹎﹊ˇ︵︶︷︸︹︿﹀︺︽︾_ˉ﹁﹂﹃﹄︻︼]', "", sentence)
        sentence_ls=sentence.split(" ")
        X_test_byword.append(sentence_ls)
    #X_test_byword=[setence.split(" ") for setence in list(D['Phrase'])]
    X_test=np.array(X_test_byword)
    X_test_PhraseID=np.array(list(D['PhraseId']))
    return  X_test,X_test_PhraseID


X_train, Y_train, feature_names = load_TrainingData('./train.tsv')
X_test,X_test_PhraseID = load_TestingData('./test.tsv')

X_Data=np.concatenate((X_train, X_test),axis=0)
corpus_words = sorted(list({word for phrase in X_Data for word in phrase}))
#print(corpus_words)
word2Ind = {word:i+1 for i, word in enumerate(corpus_words)}
all_word_count=0
word_count_dict={}
doc_flag=False
for phrase in X_train:
    doc_flag=True
    for word in phrase:
        all_word_count+=1
        if word not in word_count_dict.keys():
            word_count_dict[word]=[1,1]
        else:
            if doc_flag==True:
                word_count_dict[word][0]+=1 
                word_count_dict[word][1]+=1
                doc_flag==False
            else:
                word_count_dict[word][0]+=1 
#print(len(word_count_dict))
idf_dic={}
for k in word_count_dict.keys():
    idf_dic[k]=(word_count_dict[k][0]/all_word_count)*(math.log(int(len(X_train))/word_count_dict[k][1]))


idf_ls=[]
for k,v in idf_dic.items():
    idf_ls.append([k,v])
idf_ls.sort(key=lambda x:x[1])
idf_sel_dic={}
for block in  idf_ls[100:151]:
    idf_sel_dic[block[0]]=block[1]


X_Train_list=[phrase for phrase in  X_train]

#输入序列大小
maxWordCount= 50
#print(maxWordCount)

#输入序列化
Embedding_X_Train=[]
for P in X_Train_list:
    P_ls=[]
    for W in range(0,maxWordCount):
        #print(P[W])
        try:
            #print(word2Ind[P[W]])
            P_ls.append(idf_dic[P[W]])
        except:
            P_ls.append(0)
            
    Embedding_X_Train.append(P_ls)

nEmbedding_X_Train=np.array(Embedding_X_Train)
#print(nEmbedding_X_Train.shape)

std=nEmbedding_X_Train.std(axis=0)
mean=nEmbedding_X_Train.mean(axis=0)
X_norm = (nEmbedding_X_Train - mean) / std

y_one_hot=np.identity(5)
Yt_train=[]
for y_enum in range(0,len(Y_train)):
    if Y_train[y_enum]==0:
        Yt_train.append(y_one_hot[0])
    elif Y_train[y_enum]==1:
        Yt_train.append(y_one_hot[1])   
    elif Y_train[y_enum]==2:
        Yt_train.append(y_one_hot[2])  
    elif Y_train[y_enum]==3:
        Yt_train.append(y_one_hot[3])  
    elif Y_train[y_enum]==4:
        Yt_train.append(y_one_hot[4]) 
nEmbedding_Y_Train=np.array(Yt_train)
nEmbedding_Y_Train.reshape((len(Yt_train)),5)

 

X_train, X_test, y_train, y_test = train_test_split(X_norm, nEmbedding_Y_Train, test_size=TEST_SIZE, random_state=42)
#print(y_train)
#print(y_train.shape)



NN_ARCHITECTURE = [
    {"input_dim": 50, "output_dim": 51, "activation": "relu"},
    {"input_dim": 51, "output_dim": 51, "activation": "relu"},
    {"input_dim": 51, "output_dim": 51, "activation": "relu"},
    {"input_dim": 51, "output_dim": 51, "activation": "relu"},
    {"input_dim": 51, "output_dim": 5, "activation": "sigmoid"},
]




def init_layers(nn_architecture, seed = 99):
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values




def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;




def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    # calculation of the input value for the activation function
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    # selection of activation function
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
        
    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr




def full_forward_propagation(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr
        
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory




def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)




def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_




def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()





def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    # number of examples
    m = A_prev.shape[1]
    
    # selection of activation function
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    
    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    
    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr





def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    
    # number of examples
    m = Y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)
    
    # initiation of gradient descent algorithm
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values



def update(params_values, grads_values, nn_architecture, learning_rate):

    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values;




def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    # initiation of neural net parameters
    params_values = init_layers(nn_architecture, 2)
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    cost_history = []
    accuracy_history = []
    
    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        
        # calculating metrics and saving them in history
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        # updating model state
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
        if(i % 50 == 0):
            #print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            
            if(verbose):
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            
            if(callback is not None):
                callback(i, params_values)
            
    return params_values



     
params_values = train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], y_train.shape[1]))), NN_ARCHITECTURE, 100, 0.01)

Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values, NN_ARCHITECTURE)
#输出验证结果
Y_test_hat_output=Y_test_hat.T
print(Y_test_hat_output)
res_y=[]
for y in Y_test_hat_output:
    y=list(y)
    res_y.append(y.index(max(y)))
print(res_y)

# 验证集准确率
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], y_test.shape[1]))))
print("Test set accuracy: {:.2f} ".format(acc_test))

