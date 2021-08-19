import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    """
    Implements the FiLM layer
    https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16528/16646
    """

    def __init__(self, in_features, out_features, cnn_mode=False):
        super(FiLM, self).__init__()

        self.cnn_mode = cnn_mode

        self.out_features = out_features
        self.affine_transformation = nn.Linear(in_features=in_features, out_features=2 * self.out_features)

    def forward(self, cond_data, cnn_activations):
        """
        this model assumes to have h_c and f_c as affine maps
        """
        gamma, beta = torch.split(self.affine_transformation(cond_data), self.out_features, 1)

        # for cnns
        # gamma = gamma.reshape(gamma.shape[0], gamma.shape[1], 1, 1)
        # beta = beta.reshape(beta.shape[0], beta.shape[1], 1, 1)

        if self.cnn_mode:
            # gamma = gamma.reshape(gamma.shape[0], gamma.shape[1], 1, 1)
            # beta = beta.reshape(beta.shape[0], beta.shape[1], 1, 1)

            # contex and choices are stacked. so are gamme beta
            # gamma = gamma.repeat(2, 1, 1, 1)
            # beta = beta.repeat(2, 1, 1, 1)

            gamma = gamma.reshape(cnn_activations.shape[0], 32, 1, 1)
            beta = beta.reshape(cnn_activations.shape[0], 32, 1, 1)

            assert (gamma.size()) == (cnn_activations.shape[0], cnn_activations.shape[1], 1, 1)
            assert (beta.size()) == (cnn_activations.shape[0], cnn_activations.shape[1], 1, 1)

        return gamma * cnn_activations + beta


class FiLMBlock(nn.Module):
    """
    Implements the FiLM layer
    https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16528/16646
    """

    def __init__(self, d_condition, in_features, out_features, cnn_mode=False, wren=False):
        super().__init__()

        self.cnn_mode = cnn_mode
        self.out_features = out_features
        self.affine_transformation = nn.Linear(in_features=d_condition, out_features=2 * out_features)

        if self.cnn_mode:
            self.cnn = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(3, 3), stride=1,
                                 padding=1)
            self.batchnorm = nn.BatchNorm2d(out_features)

        else:
            self.linear = nn.Linear(in_features=in_features, out_features=out_features)
            self.batchnorm = nn.BatchNorm1d(out_features)

        if wren:
            print("WREN MODE")
            out_features = 16 * out_features

        self.film = FiLM(d_condition, out_features, self.cnn_mode)

    def forward(self, input):
        if self.cnn_mode:
            x = self.cnn(input["input"])

        else:
            x = self.linear(input["input"])

        x = self.batchnorm(x)
        x = F.relu(self.film(input["aux"], x))

        # return input
        return {"input": x, "aux": input["aux"]}