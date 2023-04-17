import torch 
import torchvision
import sklearn

from utils import utils, transfer_learning as tl


class ResNet18(BaseModel):
   
   def __init__(self, config, train_set, val_set, test_set):
      super().__init__(self)
      self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
      self.config = config
      # Adapt model head and tail weights
      self.model = tl.update_last_layer(self.model, self.config['out_features'])
      self.model = tl.update_first_layer(
         self.model, 
         in_channels=self.config['in_channels'], 
         weights_init=self.config['weights_init'],
         scaling=self.config['scaling']
      )
      
      #  Data loader, Loss and Optimizer from the config file 
      self.loss = utils.configure_loss(self)
      self.optimizer = utils.configure_optimizer(self)
      # CUDA flag
      self.is_cuda = torch.cuda.is_available()
      if self.is_cuda and not self.config['cuda']:
         print("WARNING: You have a CUDA device. You can enable CUDA in the config file.")
      self.device = 'cuda' if self.is_cuda and self.config['cuda'] else 'cpu'

      # Counters
      self.current_epoch = 0
      self.current_iteration = 0

      # Data Loader
      self.train_loader = torch.utils.data.DataLoader(
          train_set, 
          batch_size=self.config['batch_size'], 
          shuffle=True,
          num_workers=2,
          pin_memory=True
      )
      self.val_loader = torch.utils.data.DataLoader(
          val_set,
          batch_size=self.config['batch_size'],
          shuffle=True,
          num_workers=2,
          pin_memory=True
      )
      self.test_loader = torch.utils.data.DataLoader(
          test_set,
          batch_size=1,
          shuffle=True,
          num_workers=2,
          pin_memory=True
      )


   def forward(self, x):
      self.model(x)


   def load_checkpoints(self, filename):
      # TOFIX
      # with open(filename, 'r') as f:
      # map_location='cpu'
      # if self.cuda:
      #       map_location='cuda'
      # self.model = self.model.load_state_dict(torch.load(filename),strict=False)#, map_location=map_location)
      return
   

   def save_checkpoints(self):
      path = self.config['checkpoint_path']
      torch.save(self.model, path)


   def train_one_epoch(self):
      self.model = self.model.to(self.device)
      self.model.train()
      if self.config['freeze']:
         # Freeze all parameters
         for param in self.model.parameters():
            param.requires_grad = False
         # Unfreeze last layer
         for param in self.model.fc.parameters():
            param.requires_grad = True
      for batch_idx, couple in enumerate(self.train_loader):
            data, target = couple['tile'].float(), couple['value'].float()
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target.view(-1,1))
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))
  
  
   def validate(self):
      self.model = self.model.to(self.device)
      self.model.eval()
      val_loss = 0
      with torch.no_grad():
         for couple in self.val_loader:
               data, target = couple['tile'].float(), couple['value'].float()
               data, target = data.to(self.device), target.to(self.device)
               output = self.model(data)
               val_loss += self.loss(output, target.view(-1,1)).item()  # sum up batch loss

      val_loss /= len(self.val_loader.dataset)
      print('\nTest set: Average loss: {:.4f}', val_loss)


   def train(self):
      for epoch in range(1, self.config["n_epochs"] + 1):
         self.current_epoch = epoch
         self.train_one_epoch()
         self.validate()
   
   def eval(self):
      self.model = self.model.to('cpu')
      self.model.eval()
      inputs = []
      outputs = []
      with torch.no_grad():
         for couple in self.test_loader:
               data, target = couple['tile'].float(), couple['value'].float()
               data, target = data.to('cpu'), target.to('cpu')
               output = self.model(data)
               outputs.append( output.numpy() )
               print
               inputs.append( target )
      return inputs, outputs