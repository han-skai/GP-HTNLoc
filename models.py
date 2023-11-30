import torch
import torch.nn as nn
# import torchvision.models as models
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes, norm=True, scale=True):
        super(Net, self).__init__()
        # self.extractor = Extractor(input_size=51, output_size=256)
        self.extractor = BiLSTM()
        self.classifier = Classifier(num_classes)
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.norm = norm
        self.scale = scale

    def forward(self, x):
        x = self.extractor(x)
        if self.norm:
            x = self.l2_norm(x)
        if self.scale:
            x = self.s * x
        x = self.classifier(x)
        return x

    def extract(self, x):
        x = self.extractor(x)
        x = self.l2_norm(x)
        return x

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))


class Extractor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Extractor, self).__init__()
        self.fc1 = nn.Linear(input_size, 96)
        self.fc2 = nn.Linear(96, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(51, 300, 4, bidirectional=True)
        self.linear = nn.Linear(300 * 2, 256)

    def forward(self, X):
        '''
        :param X: [batch_size, seq_len]
        :return:
        '''

        X = X.view(len(X), 1, -1)  # Change the original 2D [a, b] to 3D [a, 1, b] x=(16,1,51)
        output, (final_hidden_state, final_cell_state) = self.lstm(
            X)  # output shape: [batch_size, seq_len=1,n_hidden * 2]
        output = output.transpose(0, 1)  # output : [seq_len=1, batch_size, n_hidden * num_directions(=2)]
        output = output.squeeze(0)  # [batch_size, n_hidden * num_directions(=2)]
        output = self.linear(output)
        return output


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x1 = self.fc(x)
        x = torch.sigmoid(x1)
        return x


class Transfer(nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()
        self.transfor = nn.Linear(256, 256, bias=False)
        self.transfor.weight.data = self.transfor.weight.data.float()

    def forward(self, x):
        x = self.transfor(x)
        return x
