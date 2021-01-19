import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MODEL_PATH


class Network(nn.Module):
    def __init__(self, CNN_layers, CNN_layers_output, num_classes):
        super(Network, self).__init__()

        self.layers = []
        self.layers = nn.Sequential(*CNN_layers)

        self.linear_label = nn.Linear(CNN_layers_output, num_classes, bias=False)

    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])

        label_output = self.linear_label(output)
        label_output = label_output / torch.norm(self.linear_label.weight, dim=1)

        return label_output


def load_models():
    state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model = state["model"]
    # Set the pytroch model into evaluation mode
    model.eval()

    return model