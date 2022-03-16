#encoding:utf-8

import tensorflow as tf
import numpy as np
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode

#---------------------------Function: Prediction function-----------------------------
def network(inputs,shapes,num_entity,lstm_dim=100,
            initializer=tf.truncated_normal_initializer):
    """
    Function: Receive the feature data of a batch of samples and calculate the output value of the network
    :param char: int, id of chars a tensor of shape 2-D [None,None] Number of batches * length of sentences per batch
    :param bound: int, a tensor of shape 2-D [None,None]
    :param flag: int, a tensor of shape 2-D [None,None]
    :param shapes: Word vector shape dictionary
    :param lstm_dim: Number of neurons
    :param num_entity: Number of entity tags
    :param initializer: Initialization function
    :return
    """
    #--------------------------------------------------
    #Feature Embedding: Convert all feature ids into a fixed-length vector
    #--------------------------------------------------
    embedding = []
    keys = list(shapes.keys())
    print("Network Input:", inputs)
    #{'char':<tf.Tensor 'char_inputs_10:0' shape=(?, ?) dtype=int32>,
    print("Network Shape:", keys) 
    #['char', 'bound', 'flag']
    
    #Convert features into word vectors
    for key in keys:   #char
        with tf.variable_scope(key+'_embedding'):
            #Get information
            lookup = tf.get_variable(
                name = key + '_embedding',
                shape = shapes[key],
                initializer = initializer
            )
            #Word vector mapping [None,None,100] Each word maps a 100-dimensional vector inputs corresponding to each word
            embedding.append(tf.nn.embedding_lookup(lookup, inputs[key]))
    print("Network Embedding:", embedding)
    #[<tf.Tensor 'char_embedding_14:0' shape=(?, ?, 100) dtype=float32>,
    
    #Splicing word vector shape[None,None,char_dim+bound_dim+flag_dim]
    embed = tf.concat(embedding,axis=-1)  #Splicing on the last dimension    -1
    print("Network Embed:", embed, '\n')
    #Tensor("concat:0", shape=(?, ?, 270), dtype=float32) 
    
    #lengths: Calculate the actual length of each sentence of the input 'inputs' (the padding content is not counted)
    #The subscript of the padding value PAD is 0,
    #so the total length minus the number of PAD is the actual length, which improves the operation efficiency
    sign = tf.sign(tf.abs(inputs[keys[0]]))             #char length
    lengths = tf.reduce_sum(sign, reduction_indices=1)  #Second dimension
    
    #Get padding sequence length
    num_time = tf.shape(inputs[keys[0]])[1]
    print(sign, lengths, num_time)
    #Tensor("Sign:0", shape=(?, ?), dtype=int32) 
    #Tensor("Sum:0", shape=(?,), dtype=int32) 
    #Tensor("strided_slice:0", shape=(), dtype=int32)
    
    #-----------------------------------------------------------------------
    #Recurrent Neural Network Coding: Two-Layer Bidirectional Networks
    #-----------------------------------------------------------------------
    #Layer1
    with tf.variable_scope('BiLSTM_layer1'):
        lstm_cell = {}
        #Layer1 forward backward
        for name in ['forward','backward']:
            with tf.variable_scope(name):           #Set name
                lstm_cell[name] = rnn.BasicLSTMCell(
                    lstm_dim                        #Number of neurons
                )     
        #BiLSTM 2 LSTMs (100 neurons each)
        outputs1,finial_states1 = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell['forward'],
            lstm_cell['backward'],
            embed,
            dtype = tf.float32,
            sequence_length = lengths
        )
    #Splicing forward LSTM and backward LSTM outputs
    outputs1 = tf.concat(outputs1,axis=-1)  #b,L,2*lstm_dim
    print('Network BiLSTM-1:', outputs1)
    #Tensor("concat_1:0", shape=(?, ?, 200), dtype=float32)
    
    #Layer2
    with tf.variable_scope('BiLSTM_layer2'):
        lstm_cell = {}
        #Layer1 forward backward
        for name in ['forward','backward']:
            with tf.variable_scope(name):           #set name
                lstm_cell[name] = rnn.BasicLSTMCell(
                    lstm_dim                        #number of neurons
                )
        #BiLSTM
        outputs,finial_states = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell['forward'],
            lstm_cell['backward'],
            outputs1,                                #Whether to utilize the first layer network
            dtype = tf.float32,
            sequence_length = lengths
        )
    #Final Results [batch_size,maxlength,2*lstm_dim]
    result = tf.concat(outputs,axis=-1)
    print('Network BiLSTM-2:', result)
    #Tensor("concat_2:0", shape=(?, ?, 200), dtype=float32)
    
    #--------------------------------------------------
    #Output fully connected map
    #--------------------------------------------------
    #Convert to a two-dimensional matrix and then perform the multiplication operation [batch_size*maxlength,2*lstm_dim]
    result = tf.reshape(result, [-1,2*lstm_dim])
    
    #First mapping Matrix multiplication
    with tf.variable_scope('project_layer1'):
        #Weight
        w = tf.get_variable(
            name = 'w',
            shape = [2*lstm_dim,lstm_dim],
            initializer = initializer
        )
        #bias
        b = tf.get_variable(
            name = 'b',
            shape = [lstm_dim],
            initializer = tf.zeros_initializer()
        )
        #Activation function relu
        result = tf.nn.relu(tf.matmul(result,w)+b)
    print("Dense-1:",result)
    #Tensor("project_layer1/Relu:0", shape=(?, 100), dtype=float32)
    
    #Second mapping Matrix multiplication
    with tf.variable_scope('project_layer2'):
        #Weight
        w = tf.get_variable(
            name = 'w',
            shape = [lstm_dim,num_entity],
            initializer = initializer
        )
        #bias
        b = tf.get_variable(
            name = 'b',
            shape = [num_entity],
            initializer = tf.zeros_initializer()
        )
        #Activation function relu The last layer is not activated
        result = tf.matmul(result,w)+b
    print("Dense-2:",result)
    #Tensor("project_layer2/add:0", shape=(?, 31), dtype=float32)
    
    #Convert to 3D
    result = tf.reshape(result, [-1,num_time,num_entity])
    print('Result:', result, "\n")
    #Tensor("Reshape_1:0", shape=(?, ?, 31), dtype=float32)
    
    #[batch_size,max_length,num_entity]
    return result,lengths

#-----------------------------Function: Define the model---------------------------
class Model(object):
    
    #---------------------------------------------------------
    #Initialization
    def __init__(self, dict_, lr=0.0001):
        #Calculate the number of features by dict.pkl
        self.num_char = len(dict_['word'][0])
        self.num_bound = len(dict_['bound'][0])
        self.num_flag = len(dict_['flag'][0])
        self.num_entity = len(dict_['label'][0])
        print('model init:', self.num_char, self.num_bound, self.num_flag,
              self.num_, self.num_pinyin, self.num_entity)
        
        #Character maps to vector dimensions
        self.char_dim = 100
        self.bound_dim = 20
        self.flag_dim = 50
        
        #Shape is expressed as [num, dim] number of rows (number) * number of columns (vector dimension)
        
        #Set the dimension of the LSTM and the number of neurons
        self.lstm_dim = 100
        
        #Learning rate
        self.lr = lr
        
        #Save the initialization dictionary
        self.map = dict_
      
        #---------------------------------------------------------
        #Define placeholder to receive data [None,None] batch  sentence length
        self.char_inputs = tf.placeholder(dtype=tf.int32,shape=[None,None],name='char_inputs')
        self.bound_inputs = tf.placeholder(dtype=tf.int32,shape=[None,None],name='bound_inputs')
        self.flag_inputs = tf.placeholder(dtype=tf.int32,shape=[None,None],name='flag_inputs')
        self.targets = tf.placeholder(dtype=tf.int32,shape=[None,None],name='targets')    #Target true value
        self.global_step = tf.Variable(0,trainable=False)  #Can't train for counting
                
        #---------------------------------------------------------
        #Pass to the network Calculate the model output value
        #Parameters: input word, boundary, part of speech -> network converts word vector and calculates
        #Returns: network output value, true length of each sentence
        self.logits,self.lengths = self.get_logits(
            self.char_inputs,
            self.bound_inputs,
            self.flag_inputs,
        )
        
        #---------------------------------------------------------
        #Calculate the loss
        #Parameters: model output value, true label sequence, length (without counting padding)
        #Returns: loss value
        self.cost = self.loss(
            self.logits,
            self.targets,
            self.lengths
        )
        print("Cost:", self.cost)
        
        #---------------------------------------------------------
        #Optimizer optimization using gradient truncation
        with tf.variable_scope('optimizer'):
            opt = tf.train.AdamOptimizer(self.lr)      #Learning rate
            #Calculate the derivative value of all loss functions
            grad_vars = opt.compute_gradients(self.cost)
            #grad_vars Record the derivative of each set of parameters and itself
            clip_grad_vars = [[tf.clip_by_value(g,-5,5),v] for g,v in grad_vars]
            #Use the truncated gradient to update the parameters.
            # This method automatically increments the global_step parameter by 1 each time it is applied.
            self.train_op = opt.apply_gradients(clip_grad_vars, self.global_step)
            print("Optimizer:", self.train_op)
            
        #Save the model. Keep the last 5 models
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=5)
        
    #---------------------------------------------------------
    #Define the network. Receive batches of samples
    def get_logits(self,char,bound,flag,pinyin): 
        """
        Function: Receive the feature data of a batch of samples and calculate the output value of the network
        :param char: int, id of chars a tensor of shape 2-D [None,None]
        :param bound: int, a tensor of shape 2-D [None,None]
        :param flag: int, a tensor of shape 2-D [None,None]
        :return: 返回3-d tensor [batch_size,max_length,num_entity]
        """
        #Define dictionary parameters
        shapes = {}
        shapes['char'] = [self.num_char,self.char_dim]
        shapes['bound'] = [self.num_bound,self.bound_dim]
        shapes['flag'] = [self.num_flag,self.flag_dim]
        print("shapes:", shapes, '\n')
        #{'char': [1663, 100], 'bound': [5, 20], 'flag': [56, 50],      
        
        #Input parameters
        inputs = {}
        inputs['char'] = char
        inputs['bound'] = bound
        inputs['flag'] = flag

        
        #return network(char,bound,flag,pinyin,shapes)
        return network(inputs,shapes,lstm_dim=self.lstm_dim,num_entity=self.num_entity)

    #--------------------------Function: define the loss CRF model-------------------------
    #Parameters: model output value、true label sequence、length (without counting padding)
    def loss(self,result,targets,lengths):
        #Get length
        b = tf.shape(lengths)[0]              #True Length This value has only one dimension
        num_steps = tf.shape(result)[1]       #With padding
        print("Loss lengths:", b, num_steps)
        print("Loss Inputs:", result)
        print("Loss Targets:", targets)
        
        #Transition matrix
        with tf.variable_scope('crf_loss'):
            # Take 'log' is equivalent to probability close to 0
            small = -1000.0
            
            #Initial state
            start_logits = tf.concat(
                [small*tf.ones(shape=[b,1,self.num_entity]),tf.zeros(shape=[b,1,1])],
                axis = -1   #Two matrices are merged in the last dimension
            )

            #X Stitching
            pad_logits = tf.cast(small*tf.ones([b,num_steps,1]),tf.float32)
            logits = tf.concat([result, pad_logits], axis=-1)
            logits = tf.concat([start_logits,logits], axis=1)
            print("Loss Logits:", logits)
            
            #Y Stitching
            targets = tf.concat(
                [tf.cast(self.num_entity*tf.ones([b,1]),tf.int32),targets],
                axis = -1
            )
            print("Loss Targets:", targets)

            #Calculate
            self.trans = tf.get_variable(
                name = 'trans',
                #The initial probability 'start' add 1
                shape = [self.num_entity+1,self.num_entity+1],
                initializer = tf.truncated_normal_initializer()
            )

            #Loss Calculates the log-likelihood of the CRF
            log_likehood, self.trans = crf_log_likelihood(
                inputs = logits,
                tag_indices = targets,
                transition_params = self.trans,
                sequence_lengths = lengths         #True sample length
            )
            print("Loss loglikehood:", log_likehood)
            print("Loss Trans:", self.trans)
            
            #Returns the mean of all samples
            return tf.reduce_mean(-log_likehood)
       
    #--------------------------Function: run step by step-------------------------
    #Parameters: session, batch, training prediction
    def run_step(self,sess,batch,is_train=True):
        if is_train:
            feed_dict = {
                self.char_inputs : batch[0],
                self.bound_inputs : batch[2],
                self.flag_inputs : batch[3],
                self.targets : batch[1]
            }
            #Training computation loss
            _,loss = sess.run([self.train_op,self.cost], feed_dict=feed_dict)
            return loss
        else:
            feed_dict = {
                self.char_inputs : batch[0],
                self.bound_inputs : batch[2],
                self.flag_inputs : batch[3],

            }
            #Test calculation results
            logits,lengths = sess.run([self.logits, self.lengths], feed_dict=feed_dict)
            return logits,lengths
    
    #--------------------------Function: Decode to get id-------------------------
    #Parameters: model output value, true length, transition matrix (for decoding)
    def decode(self,logits,lengths,matrix):
        #Keep the path with the highest probability
        paths = []
        small = -1000.0
        #Decoding each sample
        start = np.asarray([[small]*self.num_entity+[0]])
        
        #Get the score of each sentence and the true length of the sample
        for score,length in zip(logits,lengths):
            score = score[:length]   #Take only output of valid characters
            pad = small*np.ones([length,1])
            #Stitching
            logits = np.concatenate([score,pad],axis=-1)
            logits = np.concatenate([start,logits],axis=0)
            #Decode
            path,_ = viterbi_decode(logits,matrix)
            paths.append(path[1:])
        
        #The 'path' gets the 'id' and needs to be converted into the corresponding entity label
        return paths
        
    #--------------------------Function: Predictive Analytics-------------------------
    #Parameters: session, batch
    def predict(self,sess,batch):
        results = []
        #Get the transition matrix
        matrix = self.trans.eval()
        
        #Get model results Execute tests
        logits, lengths = self.run_step(sess, batch, is_train=False)
        
        #Call the decode function to get the 'paths'
        paths = self.decode(logits, lengths, matrix)
        
        #View words and corresponding tags
        chars = batch[0]
        for i in range(len(paths)):
            #Get the ture length of the i-th sentence
            length = lengths[i]
            #True word in sentence i
            chars[i][:length]
            #The 'ID' is converted into the corresponding word for each word
            #map['word'][1] is  dictionary
            string = [self.map['word'][1][index] for index in chars[i][:length]]
            #Get tag
            tags = [self.map['label'][0][index] for index in paths[i]]

            result = [k for c,t in zip(string,tags)]
            results.append(result)
            
        #Get predicted value
        return results