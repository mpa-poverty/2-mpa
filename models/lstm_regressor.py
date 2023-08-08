# @MDC, MARBEC, 2023

import torch


class LSTMRegressor( torch.nn.Module ):
    """Custom LSTM Regressor Network, 
       that takes time-series as inputs

    Args:
        ms (torch.nn.Module): first branch cnn
        nl (torch.nn.Module): second branch cnn
    """

    def __init__(self, msnl, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMRegressor, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, 
                     hidden_size=hidden_size, 
                     num_layers=num_layers,
                     batch_first=True)
        self.msnl = msnl
        in_features=self.msnl.fc.in_features
        self.msnl.fc = torch.nn.Identity()
        self.fc = torch.nn.Linear(hidden_size+in_features, output_size)
        
        
    def forward(self, p, ms, nl):
        # Pass input through the LSTM layer
        lstm_out, _ = self.lstm(p)
        # Take the output of the last time step
        last_lstm_output = lstm_out[:, -1, :]
        msnl_out = self.msnl(ms,nl)
        # Pass through the fully connected layer
        output = self.fc(torch.cat((last_lstm_output, msnl_out),dim=1))
        return output
