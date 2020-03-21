import argparse
import base64
import contextlib
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
import matplotlib.pyplot as plt

import dns.message
import dns.rdatatype

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def bytes_to_tensor(b, max_len):
    tensor = torch.from_numpy(np.frombuffer(b, dtype=np.uint8))
    padded = F.pad(tensor, [0, max_len - len(tensor)])
    return padded

def tensor_to_bytes(tensor: torch.Tensor):
    return bytes(tensor.byte())

class DNSDataset(Dataset):
    def __init__(self, filename):
        query_max_len = 0
        response_max_len = 0
        data = []
        with open(filename, 'r') as fp, contextlib.closing(tqdm.tqdm(total=100_000)) as t:
            for line in fp:
                query, response = json.loads(line)
                query, response = base64.b64decode(query), base64.b64decode(response)
                if len(query) > query_max_len:
                    query_max_len = len(query)
                if len(response) > response_max_len:
                    response_max_len = len(response)
                data.append((query, response))
            t.update()
        self.query_max_len = query_max_len
        self.response_max_len = response_max_len
        self._data = self._data_as_tensor(data)

    def _data_as_tensor(self, data):
        tensor_data = []
        for query, response in tqdm.tqdm(data):
            query = bytes_to_tensor(query, self.query_max_len)
            response = bytes_to_tensor(response, self.response_max_len)
            tensor_data.append((query, response))
        return tensor_data

    def __getitem__(self, index):
        return self._data[index][0], self._data[index][1]

    def __len__(self):
        return len(self._data)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, in_char, hidden_state):
        output = self.embedding(in_char).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden_state)
        return self.out(output[0]), hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)

class DNSModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.conv1d = nn.Conv1d(in_features, in_features, kernel_size=1)

    def forward(self, query):
        # result = self.linear(query)
        # return self.relu(result)
        result = self.conv1d(query.unsqueeze(dim=-1))
        result = self.relu(result)
        return self.linear(result.squeeze(dim=-1))

def train_rnn():
    pass

def train_gpt2():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile')
    args = parser.parse_args()

    dataset = DNSDataset(args.datafile)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    criterion = nn.MSELoss()
    # model = DNSModel(dataset.query_max_len, dataset.response_max_len)
    model = RNN(input_size=dataset.query_max_len,
                output_size=dataset.response_max_len,
                hidden_size=512, n_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    losses = []
    t = tqdm.trange(num_epochs, leave=True)
    for epoch in t:
        epoch_losses = []
        for x, y_truth in train_loader:
            optimizer.zero_grad()
            hidden = model.init_hidden()
            y_hat = model(x.float(), hidden)
            loss = criterion(y_hat, y_truth.float())
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_mean_loss = np.mean(epoch_losses)
        losses.append(epoch_mean_loss)
        t.update()
        t.set_description('Mean loss: {}'.format(epoch_mean_loss))

        qnames = ['google.com', ]  # 'example.com', 'dohjs.org', 'asdf.com', 'cdn.kimbo.net', 'a.b.c.e.dns-oarc.net']
        for qname in qnames:
            qtype = dns.rdatatype.A
            q = dns.message.make_query(qname, qtype)
            wire = q.to_wire()
            q_id = dns.message.from_wire(wire).id
            inp = bytes_to_tensor(wire, dataset.query_max_len)
            output = model(inp.float().unsqueeze(dim=0)).squeeze(dim=0)
            try:
                msg = dns.message.from_wire(tensor_to_bytes(output), ignore_trailing=True)
                r_id = msg.id
                print('query id: {}, response id: {}'.format(q_id, r_id))
            except Exception as e:
                print('ERROR: ', e)
        torch.save(model, 'model.pt')
    plt.plot(losses, label='Train loss')
    plt.title('Average loss per epoch')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
