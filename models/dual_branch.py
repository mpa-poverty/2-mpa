'''
Concatenation of two separately trained networks
merged on their last layer. 
'''

import torch 
import torchvision
from torchgeo import models
from utils import utils, transfer_learning as tl
from base import BaseModel
from data import data_loader


class DualBranch(BaseModel):
   
   def __init__(self, first_branch, second_branch, config):
      super().__init__()
      self.first_branch = first_branch
      self.second_branch = second_branch
      self.config = config

      self.first_branch.fc = Identity()
      self.second_branch.fc = Identity()
      self.regressor = torch.nn.Linear(
         self.first_branch.out_features+self.second_branch.out_features, 
         self.config.out_features
      )
      
      #  Data loader, Loss and Optimizer from the config file 
      self.data_loader = utils.configure_data_loader(config=config)
      self.loss = utils.configure_loss(self.config)
      self.optimizer = utils.configure_optimizer(self.config)
      
      # CUDA flag
      self.is_cuda = torch.cuda.is_available()
      if self.is_cuda and not self.config.cuda:
         self.logger.info("WARNING: You have a CUDA device. You can enable CUDA in the config file.")
      self.cuda = self.is_cuda & self.config.cuda 

      # Counters
      self.current_epoch = 0
      self.current_iteration = 0


   def forward(self, x1, x2):
      head1 = self.first_branch(x1)
      head2 = self.second_branch(x2)
      x = torch.cat( (head1, head2), dim=1 )
      return self.regressor(torch.nn.functional.relu(x))


   def load_checkpoints(self, filename):
      path = self.config.checkpoint_path
      self.model = torch.load(path)


   def save_checkpoints(self, epoch):
      path = self.config.checkpoint_path
      torch.save(self.model, path)


   def train_one_epoch(self):
      self.model.train()
      for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1
  
  
   def validate(self):
      self.model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
         for data, target in self.data_loader.test_loader:
               data, target = data.to(self.device), target.to(self.device)
               output = self.model(data)
               test_loss += self.loss(output, target, size_average=False).item()  # sum up batch loss
               pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
               correct += pred.eq(target.view_as(pred)).sum().item()

      test_loss /= len(self.data_loader.test_loader.dataset)
      self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
         test_loss, correct, len(self.data_loader.test_loader.dataset),
         100. * correct / len(self.data_loader.test_loader.dataset)))


   def train(self):
      for epoch in range(1, self.config.n_epoch + 1):
         self.train_one_epoch()
         self.validate()