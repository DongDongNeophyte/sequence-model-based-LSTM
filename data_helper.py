import os
import pandas as pd
import numpy as np

def sequencein(filepath,index):
    #print ("sequencein")
    data = pd.read_table(filepath+str(index),sep='\t',delim_whitespace=True)
    #fin=open(filepath)
    #matr1=[]
    #matr=np.mat(matr1)
    #line = fin.readline()
    #seq = fin.readline()
    #index=0
    #while seq and seq!="\n":
        #seq = fin.readline()
        #seq = seq.strip('\n')
        #print("len:",len(seq)) 
        #nums =seq.split("\t")
        #nums = [int(x) for x in nums ]
        #if index==0:
          # matr=np.mat(nums)
           #print("matr shape:", matr.shape)
           #index=1
        #else:
           #print("numshape:", np.mat(nums).shape) 
           #matr=np.r_[matr,np.mat(nums)]
        #matr1.append(np.array(nums))
        #matr1.append(nums)
        #seq=fin.readline()
    #matr=np.mat(matr1)
    #fin.close()
    #print ("read table is over")
    #data = data.fillna(0)
    matr = data.values
    #print("sequence shape:", matr.shape)
    return matr

def labelin(filepath,index):
    data = pd.read_table(filepath+str(index),sep='\t',delim_whitespace=True)
    matr = data.values
    #print("labels shape:", matr.shape)
    return matr

def lengthin(filepath, index, n_inputs):
    data = pd.read_table(filepath+str(index),sep='\t',delim_whitespace=True)
    matr = data.values
    #print (n_inputs)
    matr = matr[:, 0] * 4
    matr = matr/n_inputs
    #print("length shape:", matr.shape)
    return matr

class DataSet(object):
    def __init__(self, sequence, length, labels,y):
        self.num_examples = sequence.shape[0]
        sequence = sequence.astype(np.float32)
        self.sequence = np.array(sequence)
        self.length = np.array(length)
        self.labels = np.array(labels)
        self.y = np.array(y)
        self.epochs_completed = 0
        self.index_in_epoch = 0
        def sequence(self):
            return self.sequence
        def length(self):
            return self.length
        def labels(self):
            return self.labels
        def y(self):
            return self.y
        def num_examples(self):
            return self.num_examples
        def epochs_completed(self):
            return self.epochs_completed
    def next_batch(self,train_dir,index,n_inputs):
        if index!=0:
          train_labels = labelin(train_dir+"/trainlabels",index)
          train_y = labelin(train_dir+"/trainy",index)
          train_length = lengthin(train_dir+"/trainlength",index, n_inputs)
          train_sequence = sequencein(train_dir+"/trainsequence",index)
          self.sequence = np.array(train_sequence)
          self.length = np.array(train_length)
          self.labels = np.array(train_labels)
          self.y = np.array(train_y)
        return self.sequence, self.length, self.labels,self.y
    def Next_batch(self,batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            self.epochs_completed += 1
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.sequence = self.sequence[perm]
            self.length = self.length[perm]
            self.labels = self.labels[perm]
            self.y = self.y[perm]
            start = 0
            self.index_in_epoch = batch_size
        end = self. index_in_epoch
        return self.sequence[start:end], self.length[start:end], self.labels[start:end],self.y[start:end]

def read_data(train_dir, n_inputs):
    class DataSets(object):
        pass
    datasets = DataSets()
    test_labels = labelin(train_dir+"/testlabels",0)
    test_y = labelin(train_dir+"/testy",0)
    train_labels = labelin(train_dir+"/trainlabels",0)
    train_y = labelin(train_dir+"/trainy",0)
    validation_labels = labelin(train_dir+"/validationlabels",0)
    validation_y = labelin(train_dir+"/validationy",0)
    test_length = lengthin(train_dir+"/testlength",0, n_inputs)
    train_length = lengthin(train_dir+"/trainlength",0, n_inputs)
    validation_length = lengthin(train_dir+"/validationlength",0, n_inputs)
    test_sequence = sequencein(train_dir+"/testsequence",0)
    train_sequence = sequencein(train_dir+"/trainsequence",0)
    validation_sequence = sequencein(train_dir+"/validationsequence",0)
    datasets.train = DataSet(train_sequence, train_length, train_labels,train_y )
    datasets.validation = DataSet(validation_sequence, validation_length, validation_labels,validation_y)
    datasets.test = DataSet(test_sequence, test_length, test_labels,test_y)
    return datasets
