import torch
import torch.nn as nn
from .Transformer import Encoder

class WReNTransformer(nn.Module):
    def __init__(self, config_transformer, N, d_classifier=[256, 256, 256, 13], use_delimiter=False,
                 old_transformer_model=False):
        super().__init__()
        self.N = N
        self.d_model = config_transformer["d_model"]
        self.n_labels = d_classifier[-1]
        self.use_delimiter = use_delimiter

        # should only be set by function
        self.attention_maps = False

        if use_delimiter:
            self.N += 1

        self.encoder_transformer = Encoder(
            self.N,
            d_model=config_transformer["d_model"],
            n_layers=config_transformer["n_layers"],
            h=config_transformer["h"],
            d_ff=config_transformer["d_ff"],
            d_v=config_transformer["d_v"],
            d_k=config_transformer["d_k"],
            d_q=config_transformer["d_q"],
            dropout_dot_product=config_transformer["dropout"],
            dropout_fc=config_transformer["dropout"],
            dropout_pos_ff=config_transformer["dropout"],
            old_transformer_model=old_transformer_model
        )

        d_classifier[0] = config_transformer["d_model"]

        theta_features = [config_transformer["d_model"] for i in range(5)]

        self.g_theta_transformer = nn.Sequential(*[layer for i in range(len(theta_features) - 1) for layer in
                                                   [nn.Linear(in_features=theta_features[i],
                                                              out_features=theta_features[i + 1]),
                                                    nn.ReLU()]])

        layers = [nn.Linear(in_features=d_classifier[0], out_features=d_classifier[1]), nn.ReLU(),
                  nn.Dropout(p=0.5),
                  nn.Linear(in_features=d_classifier[1], out_features=d_classifier[2]), nn.ReLU(),
                  nn.Linear(in_features=d_classifier[2], out_features=d_classifier[3])
                  ]

        self.classifier_transformer = nn.Sequential(*layers)

    def get_attention_maps_data(self):
        attentions = self.encoder_transformer.get_attention_maps_data()

        return attentions

    def return_attention_maps(self, state):
        self.attention_maps = state
        self.encoder_transformer.return_attention_maps(state)

    def forward(self, chi, i_iteration=None):
        """
        encoded_panel_sequence --> encoder_transformer --> MLP each sequence element + sum them up --> classifier --> [RETURN] logits
        :return: logits
        """

        if self.attention_maps == True:
            self.encoder_transformer.set_iteration(i_iteration)

        x = self.encoder_transformer(chi)
        assert (self.N, self.d_model) == x.shape[1:]

        x = torch.sum(self.g_theta_transformer(x), dim=1)
        assert (self.d_model,) == x.shape[1:]

        logits = self.classifier_transformer(x)
        assert (self.n_labels,) == logits.shape[1:]

        return logits