import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size=2,
        lstm_layers=2,
        lstm_hidden=32,
        n_output=1,
        bidirectional=False,
    ):
        super(LSTMClassifier, self).__init__()
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self._n_output = n_output

        self.rnn = nn.LSTM(
            input_size,
            self.lstm_hidden,
            self.lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, 1)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 1)
            self.lin = nn.Linear(self.lstm_hidden + 2, 1)
            self.lin = nn.Sequential(
                nn.BatchNorm1d(self.lstm_hidden + input_size),
                nn.Linear(self.lstm_hidden + input_size, 10),
                nn.ReLU(),
                nn.Linear(10, n_output),
            )

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return (
                Variable(
                    torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden),
                    requires_grad=True,
                ),
                Variable(
                    torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden),
                    requires_grad=True,
                ),
            )
        else:
            return (
                Variable(
                    torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden),
                    requires_grad=True,
                ).to(device),
                Variable(
                    torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden),
                    requires_grad=True,
                ).to(device),
            )

    def forward(self, x):
        device = x.device

        batch_size, timesteps, C = x.size()
        self.hidden = self.init_hidden(batch_size, device)

        r_in = x.view(batch_size, timesteps, -1).contiguous()
        r_out, states = self.rnn(r_in, self.hidden)
        r_out = torch.cat([r_out, r_in], 2)
        r_out = r_out.view(batch_size * timesteps, -1)
        out = self.lin(r_out)
        out = out.view(batch_size, timesteps, self._n_output)
        if not self.training:
            out = out[:, :, 0]

        return out


if __name__ == "__main__":
    model = LSTMClassifier(1, 1, 256).cuda()
    data = torch.rand((1, 64, 512, 8, 8)).cuda()
    out = model(data)
    print(out.shape)
