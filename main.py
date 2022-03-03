from operator import index
from re import S, X
from tabnanny import verbose
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch

import torch_utils

class MNISTGeneratorTrainer(object):
  def __init__(self,lr=0.0001,batch_size=50,verbose=False):
    self.batch_size=batch_size
    self.epsilon = 1e-9
    self.verbose = verbose
    (self.X_train, self.Y_train, 
    self.X_val, self.Y_val,
    self.X_test, self.Y_test,
    self.num_classes) = self.load_dataset()
    self.class_net = self.build_classifier_network()
    self.class_optim = torch.optim.Adam(self.class_net.parameters(), lr=lr)
    self.gen_net = self.build_generator_network()
    self.gen_optim = torch.optim.Adam(self.gen_net.parameters(), lr=lr)

  def one_hot(self, Y_arr, n_classes) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.Tensor(Y_arr).to(torch.int64), num_classes=n_classes)

  def load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, int]:
    mnist_train = MNIST(root='./data', train=True, download=True,
      transform=None)
    mnist_test = MNIST(root='./data', train=False, download=True,
      transform=None)
    
    X_train, Y_train = zip(*mnist_train)
    X_test, Y_test = zip(*mnist_test)

    num_classes = np.max(Y_train)+2
    
    X_train = torch.Tensor(np.array([np.array(X_i) for X_i in X_train]))
    Y_train = self.one_hot(Y_train, n_classes=num_classes)
    
    # Split into val and train
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.8)

    
    X_test = torch.Tensor([np.array(X_i) for X_i in X_test])
    Y_test = self.one_hot(Y_test, n_classes=num_classes)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, num_classes
  
  def build_classifier_network(self):
    return torch.nn.Sequential(
      # First conv2d with 15 filters
      torch_utils.MNISTFlatten(in_channels=1,out_channels=20,kernel_size=4,
        padding=0),
      # Sigmoid activation
      torch.nn.Sigmoid(),
      # Print shape
      torch_utils.Passthrough(verbose=self.verbose),
      
      # Pooling layer
      torch.nn.MaxPool2d(kernel_size=2),
      torch_utils.Passthrough(verbose=self.verbose),

      # Conv2d layer
      torch.nn.Conv2d(in_channels=20,out_channels=10,kernel_size=2),
      # Sigmoid activation
      torch.nn.Sigmoid(),
      torch_utils.Passthrough(verbose=self.verbose),
      
      # Linear layer and print shape
      torch.nn.Flatten(),
      torch.nn.Linear(1210,256),
      torch.nn.Sigmoid(),
      torch.nn.Linear(256,self.num_classes),
      torch.nn.Softmax(-1)
    )
  def build_generator_network(self):
    return torch.nn.Sequential(
      torch_utils.RandomizedLinear(in_features=self.num_classes,out_features=256,random_features=3),
      torch.nn.Sigmoid(),
      torch.nn.Linear(256,384),
      torch.nn.Sigmoid(),
      torch.nn.Linear(384,384),
      torch.nn.Sigmoid(),
      torch.nn.Linear(384,784),
      torch.nn.Sigmoid(),
      torch_utils.MNISTReshape()
    )
  def class_loss_and_accuracy(self, Y_preds: torch.Tensor, Y_labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        return (
          # Loss
          -(Y_labels.double()*torch.log(
                  (# Everything close to 0 is set to 0
                  Y_preds*(Y_preds >= self.epsilon
                  
                  # Add epsilon to everything that is 0
                  )+(Y_preds < self.epsilon)*self.epsilon

                  # Then everything close to 1 is set to 0
                  )*(Y_preds <= 1.-self.epsilon

                  # And epsilon is added to everything that is 0
                  )+(Y_preds > 1.-self.epsilon)*(1.-self.epsilon)
              )
          ).sum(1).mean(0),

          # Accuracy
          (Y_preds.argmax(1) == Y_labels.argmax(1)).double().mean()
        )

  def classifier_train_loop(self,X_train: torch.Tensor, Y_train: torch.Tensor, batch_size: int):
    # For i in Number of Batches
    for i in range((X_train.shape[0]//batch_size)+1):
      start_batch, end_batch = i*batch_size, np.min([(i+1)*batch_size, X_train.shape[0]])
      Y_pred = self.class_net.forward(X_train[start_batch:end_batch])
      loss, acc = self.class_loss_and_accuracy(Y_pred, Y_train[start_batch:end_batch])
      loss.backward()
      self.class_optim.step()
      self.class_optim.zero_grad()

    loss_val, acc_val = self.class_loss_and_accuracy(
      self.class_net.forward(self.X_val),
      self.Y_val
    )
    return loss_val, acc_val
  
  def train_classifier(self, n_epochs):
    losses, accs = [], []
    
    for i in range(n_epochs):
      loss_val, acc_val = self.classifier_train_loop(self.X_train,
        self.Y_train, self.batch_size)
      losses.append(loss_val); accs.append(acc_val)
      print(f"loss: {loss_val}")
      print(f"acc: {acc_val}")
      plt.plot(np.arange(i+1),accs)
      plt.pause(0.05)
      plt.show(block=False)
  
  def gan_step(self, idx_epoch, batch_size, n_gen=2000):
    

    train_size=0.8
    #generate random one-hot tensors
    Y_gen_train, Y_gen_val, Y_clf_train, Y_clf_val = train_test_split(
      # One-hot labels to use for the generator's loss computation(labeled 0-9)
      self.one_hot(torch.randint(low=0,high=9,size=(n_gen,1)).to(torch.float),self.num_classes),
      # One-hot labels to use for the classifier's loss computation
      # (labeled fake, i.e. index is 10)
      self.one_hot(torch.ones(size=(n_gen,1))*(self.num_classes-1),
        self.num_classes),
      train_size=train_size
    )
    
    #fake images for train data
    X_fakes_train:torch.Tensor = self.gen_net.forward(Y_gen_train).detach()

    #concatenate fake and real images & labels
    X_train = torch.cat([self.X_train, X_fakes_train], 0)
    Y_train_clf = torch.cat([self.Y_train, Y_clf_train.view(int(n_gen*train_size),self.num_classes)],0)
    Y_train_gen = torch.cat([self.Y_train, Y_gen_train.view(int(n_gen*train_size),self.num_classes)],0)
    n_train = X_train.shape[0]
    #shuffle data
    idx_train = torch.randperm(n_train)
    X_train = torch.index_select(X_train,0,idx_train)
    Y_train_clf:torch.Tensor = torch.index_select(Y_train_clf,0,idx_train)
    Y_train_gen:torch.Tensor = torch.index_select(Y_train_gen,0,idx_train)
    
    clf_loss_sum = 0
    clf_loss_avg = 0
    gen_loss_sum = 0
    gen_loss_avg = 0
    #train on the shuffled mixed data
    for i in range(X_train.shape[0]//batch_size):
      start_batch, end_batch = i*batch_size, np.min([(i+1)*batch_size, X_train.shape[0]])
      Y_fake_gen_labels=self.one_hot(
        torch.randint(low=0,high=9,size=(batch_size,1)).to(torch.float),
        self.num_classes
      )

      Y_fake_clf_labels=self.one_hot(
        torch.ones(size=(batch_size,1))*(self.num_classes-1),
        self.num_classes
      )
      
      n=end_batch-start_batch
      idxlist=torch.randperm(n)
      X_fakes_batch=self.gen_net.forward(Y_fake_gen_labels)

      batch_X=torch.cat([X_train[start_batch:end_batch],X_fakes_batch])
      batch_X = torch.index_select(batch_X,dim=0,index=idxlist)

      batch_Y_gen=torch.cat([
        Y_train_gen[start_batch:end_batch].view(size=(batch_size,1,self.num_classes)),
        Y_fake_gen_labels
      ])
      batch_Y_gen=torch.index_select(batch_Y_gen,dim=0,index=idxlist)
      
      batch_Y_clf=torch.cat([
        Y_train_clf[start_batch:end_batch].view(size=(batch_size,1,self.num_classes)),
        Y_fake_clf_labels
      ])
      batch_Y_clf=torch.index_select(batch_Y_clf,dim=0,index=idxlist)

      Y_pred:torch.Tensor = self.class_net.forward(batch_X)
      # compute classifier loss
      clf_loss, clf_acc = self.class_loss_and_accuracy(Y_pred,batch_Y_clf.view(batch_size,self.num_classes))
      
      clf_loss_sum += clf_loss.data.item()
      clf_loss_avg = clf_loss_sum/(i+1)
      
      # backpropagation for the classifier
      clf_loss.backward(retain_graph=True)
      self.class_optim.step()
      self.class_optim.zero_grad()
      
      Y_pred:torch.Tensor = self.class_net.forward(batch_X)
      # compute generator loss
      gen_loss, gen_acc = self.class_loss_and_accuracy(Y_pred,Y_fake_gen_labels.view(batch_size,self.num_classes))
      #print(gen_loss)
      gen_loss_sum += gen_loss.data.item()
      gen_loss_avg = gen_loss_sum/(i+1)
      
      # backpropagation for the generator
      gen_loss.backward()
      self.gen_optim.step()
      self.gen_optim.zero_grad()
    
    plt.pause(0.05)
    plt.show(block=False)

    print(f"loss generator: {gen_loss_avg}")
    print(f"loss classifier: {clf_loss_avg}")

    #plt.imshow(self.gen_net.forward(
    #  self.one_hot(torch.Tensor([8]),self.num_classes).view(1,1,self.num_classes)
    #).detach().view(size=(28,28)))
    #plt.pause(0.05)
    #plt.show(block=False)

    # Run validation
    Y_pred_val = self.class_net.forward(self.X_val)
    loss_val, acc_val = self.class_loss_and_accuracy(Y_pred_val, self.Y_val)
    return loss_val, acc_val
    

  def train_gan(self, n_epochs):
    losses, accs = [], []
    for i in range(n_epochs):

      # Train classifier
      loss_val, acc_val = self.classifier_train_loop(self.X_train,
        self.Y_train, self.batch_size)
      # Store loss and accuracy
      losses.append(loss_val); accs.append(acc_val)

      print(f"loss: {loss_val}")
      print(f"acc: {acc_val}")
      plt.plot(np.arange((i*2)+1),accs)
      plt.pause(0.05)
      plt.show(block=False)

      # Train classifier and generator
      loss_val, acc_val = self.gan_step(i,self.batch_size)
      
      # Store loss and accuracy
      losses.append(loss_val); accs.append(acc_val)

      print(f"loss: {loss_val}")
      print(f"acc: {acc_val}")
      plt.plot(np.arange((i+1)*2),accs)
      plt.pause(0.05)
      plt.show(block=False)
      
      
      
with torch.autograd.anomaly_mode.set_detect_anomaly(True):
  trainer = MNISTGeneratorTrainer(verbose=False)  
  #trainer.train_classifier(5)
  trainer.train_gan(30)