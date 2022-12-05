import torch

def conv1d_same(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    return torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                            padding=(kernel_size - 1) // 2, bias=bias, dilation=dilation)

class ResBlock(torch.nn.Module):
    def __init__(self, conv, n_hidden_ch, kernel_size,
                 batchnorm=False, dropout=0.0,
                 activation=torch.nn.ReLU(True), res_scale=0.1):

        super(ResBlock, self).__init__()

        bias = not batchnorm
        layers = []
        for i in range(2):
            layers.append(conv(n_hidden_ch, n_hidden_ch, kernel_size, bias=bias))
            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(n_hidden_ch))
            if dropout:
                layers.append(torch.nn.Dropout(dropout))
            if i == 0:
                layers.append(activation)

        self.body = torch.nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Resnet(torch.nn.Module):
    def __init__(self, batchnorm=True, dropout=0.5):
        super(Resnet, self).__init__()
        self.batchnorm = batchnorm
        self.bias = not self.batchnorm

        self.n_res_blocks = 1
        self.n_hidden_ch = 512
        self.kernel_size = 7
        self.activation = torch.nn.ReLU(True)
        self.res_scaling = 0.1
        self.n_input_ch = 1
        self.n_output_ch = 1

        head = [conv1d_same(self.n_input_ch, self.n_hidden_ch, self.kernel_size)]

        body = [ResBlock(conv1d_same, self.n_hidden_ch, self.kernel_size, activation=self.activation, 
                         res_scale=self.res_scaling, batchnorm=self.batchnorm, dropout=dropout)
                for _ in range(self.n_res_blocks)]

        tail = [conv1d_same(self.n_hidden_ch, self.n_output_ch, 1)]

        self.model = torch.nn.Sequential(*head, *body, *tail)

    def forward(self, x):
        input_ = x
        x = self.model(x)
        x += input_
        return x