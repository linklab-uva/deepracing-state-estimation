from preprocessing import transform_data_for_rnn
from data_scraper import scrape_udp_data
import torch
import torch.nn as nn
import numpy as np

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class StateEstimationDataset(torch.utils.data.Dataset):
    def __init__(self, motion_dir, udp_filename, buffer_size):
        scrape_udp_data(motion_dir, filename=udp_filename)
        self.X_data, self.y_data = transform_data_for_rnn(udp_filename, buffer_size)
        self.X_data = torch.from_numpy(self.X_data).to(device)
        self.y_data = torch.from_numpy(self.y_data).to(device)
    def __len__(self):
        return(self.X_data.shape[0])
    def __getitem__(self, idx):
        x = self.X_data[idx]
        y = self.y_data[idx]
        return x, y
    def split(self, percent):
        split_id = int(len(self)* 0.8)
        return torch.utils.data.random_split(self, [split_id, (len(self) - split_id)])

class StateEstimationNet(nn.Module):
    def __init__(self, buffer_size, input_shape, hidden_dim, n_layers, batch_size, drop_prob=0.0002):
        super().__init__()
        self.output_size = 3 * buffer_size
        self.input_shape = input_shape
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_shape, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.output_size)

    def forward(self, x, hidden):
        x = x.float()
        x = torch.unsqueeze(x, 1)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_out)
        out = out.view(self.batch_size, -1)
        return out, hidden
    
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(device))
        return hidden

    def compile(self, loss_function, optimizer):
        self.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer

    def fit(self, epochs, train_data_loader, val_data_loader, model_path='state_dict.pt'):
        valid_loss_min = np.Inf
        self.train()
        for i in range(epochs):
            h = self.init_hidden()
            for inputs, labels in train_data_loader:
                h = tuple([e.data for e in h])
                inputs, labels = inputs.to(device), labels.to(device)
                self.zero_grad()
                output, h = self.forward(inputs, h)
                loss = self.loss_function(output.squeeze(), labels.float())
                loss.backward()
                self.optimizer.step()

            val_h = self.init_hidden()
            val_losses = []
            self.eval()
            for inp, lab in val_data_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out, val_h = self.forward(inp, val_h)
                val_loss = self.loss_function(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())
            self.train()
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(self.state_dict(), model_path)
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)
            print("Epoch: {}/{}...".format(i+1, epochs),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(np.mean(val_losses)))
    
    def predict(self, data_loader):
        test_losses = []
        predictions = []
        ground_truth = []

        h = self.init_hidden()
        self.eval()
        for inputs, labels in data_loader:
            h = tuple([each.data for each in h])
            inputs, labels = inputs.to(device), labels.to(device)
            output, h = self.forward(inputs, h)
            test_loss = self.loss_function(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())
            predictions.append(output.squeeze())
            ground_truth.append(labels.cpu())
        print("Loss: {:.3f}".format(np.mean(test_losses)))
        return predictions, ground_truth