import torch
import torch.nn as nn

from .RelationNetwork import RelationNetwork
from .FiLM import FiLMBlock
from .WReNTransformer import WReNTransformer


class WReN(nn.Module):
    def __init__(self,
                 use_film = False,
                 n_film_layers = 2,
                 triples = False,
                 both = False,
                 film_cnn = False,
                 classifier_extra_layers = 0,
                 embedding_siz = 256,
                 ctx_only = False,
                 config_transformer = None,
                 use_delimiter = True,
                 old_transformer_model = False,
                 theta_dim = 512):

        super(WReN, self).__init__()

        if config_transformer is not None:
            self.use_transformer = True

        else:
            self.use_transformer = False

        self.use_delimiter = use_delimiter

        self.film_cnn = film_cnn
        self.classifier_extra_layers = classifier_extra_layers

        self.use_film = use_film

        self.n_context = 8
        self.n_choices = 8

        self.labels_context = nn.Parameter(torch.zeros(self.n_context, 9), requires_grad = False)
        self.labels_choices = nn.Parameter(torch.zeros(self.n_choices, 9), requires_grad = False)

        self.size_aux_data = 12
        self.out_dim_choice = 1

        for i in range(self.n_context):
            self.labels_context[i, i] = 1

        for i in range(self.n_choices):
            self.labels_choices[i, -1] = 1

        self.h = 160
        self.w = 160
        self.out_channels = 32

        # cnn, h_out = h_in/2, w_out = w_in/2 per layer
        num_cnn_layer = 4
        kernel_size = 3
        stride = 2
        padding = int((kernel_size - 1) / 2)
        channels = [1] + [self.out_channels] * num_cnn_layer
        self.cnn = nn.Sequential(*[layer for i in range(len(channels) - 1) for layer in
                                   [nn.Conv2d(channels[i], channels[i + 1], kernel_size, stride, padding),
                                    nn.BatchNorm2d(channels[i + 1]),
                                    nn.ReLU()]])

        if self.use_transformer:
            self.embedding_size = config_transformer["d_model"]

        else:
            self.embedding_size = embedding_siz

        # + 1 --> panel tagging
        downsize_by = stride ** num_cnn_layer
        self.RNN_embedding = nn.Linear(
            in_features = int(self.h / downsize_by) * int(self.w / downsize_by) * self.out_channels + 9,
            out_features = self.embedding_size)

        # Relation Network
        self.theta_dim = theta_dim
        self.n_labels = 13

        if self.use_film:
            print("USING film")
            # its stacked to 32*8 = 256
            self.out_dim_choice = 32
            self.n_labels = self.out_dim_choice + self.size_aux_data

            self.dim_film = self.n_choices * self.out_dim_choice

            self.film_layers = nn.Sequential(
                *[FiLMBlock(self.size_aux_data, self.dim_film, self.dim_film, cnn_mode = False, wren = False) for i in range(n_film_layers)])

            self.film_classifier = nn.Sequential(*[
                nn.Dropout(p = 0.2),
                nn.Linear(in_features = self.dim_film, out_features = int(self.dim_film / 2.)),
                nn.ReLU(),
                nn.Linear(in_features = int(self.dim_film / 2.), out_features = 8)
            ])

        if self.film_cnn:
            self.film_cnn_layers = nn.Sequential(
                *[FiLMBlock(self.size_aux_data, self.out_channels, self.out_channels, cnn_mode = True, wren = True) for
                  i in
                  range(1)])

        if config_transformer is not None:
            print("USING TRANSFORMER")

            if self.use_delimiter:
                self.delimiter = nn.Parameter(torch.randn(config_transformer["d_model"]), requires_grad = True)

            self.transformer = WReNTransformer(config_transformer, self.n_context + 1, use_delimiter = use_delimiter,
                                               old_transformer_model = old_transformer_model)
        else:
            self.RN = RelationNetwork(self.embedding_size, self.n_context + 1, self.theta_dim, self.n_labels,
                                      triples = triples, both = both, classifier_extra_layers = classifier_extra_layers,
                                      ctx_only = ctx_only)

    def get_panel_embeddings_stack(self, context, choices, meta_targets = None):
        # HAVE TO DO THIS DYNAMIC
        batch_size = context.shape[0]
        labels_context = self.labels_context.repeat(batch_size, 1)
        labels_choices = self.labels_choices.repeat(batch_size, 1)

        labels = torch.cat((labels_context, labels_choices), dim = 0)

        assert self.n_context == self.n_choices

        context_choices = torch.cat((context.reshape(batch_size * self.n_context, 1, self.h, self.w),
                                     choices.reshape(batch_size * self.n_context, 1, self.h, self.w)), dim = 0)
        assert tuple(context_choices.size()) == (batch_size * 2 * self.n_context, 1, self.h, self.w)

        panel_features = self.cnn(context_choices)
        assert tuple(panel_features.size()) == (
            batch_size * 2 * self.n_context, self.out_channels, int(self.h / 2 ** 4), int(self.w / 2 ** 4))

        if self.film_cnn:
            filmed_params = self.film_cnn_layers({"aux": meta_targets, "input": panel_features})

            panel_features = filmed_params["input"]
            aux = filmed_params["aux"]

        panel_features = panel_features.reshape(batch_size * 2 * self.n_context, -1)
        assert tuple(panel_features.size()) == (
            batch_size * 2 * self.n_context, int(self.h / 2 ** 4) * int(self.w / 2 ** 4) * self.out_channels)

        panel_features = torch.cat((panel_features, labels), dim = 1)

        assert tuple(panel_features.size()) == (
            batch_size * 2 * self.n_context,
            int(self.h / 2 ** 4) * int(self.w / 2 ** 4) * self.out_channels + self.n_context + 1)

        panel_embedding = self.RNN_embedding(panel_features)
        assert tuple(panel_embedding.size()) == (batch_size * 2 * self.n_context, self.embedding_size)

        panel_embedding = panel_embedding.reshape(batch_size * 2, self.n_context, -1)

        context = panel_embedding[:batch_size, :, :]
        choices = panel_embedding[batch_size:, :, :]

        return context, choices

    def get_panel_embeddings(self, panels, label = None):

        if label is not None:
            panel_label = torch.zeros(self.batch_size, panels.shape[1] + 1)

            if torch.cuda.is_available():
                panel_label = panel_label.cuda()

            panel_label[:, label] = 1

        for i in range(panels.shape[1]):

            panel = panels[:, i, :, :, ].unsqueeze(1)
            assert tuple(panel.size()) == (self.batch_size, 1, self.h, self.w)

            # create feature maps and put them all in one dimension
            panel_features = self.cnn(panel)
            assert tuple(panel_features.size()) == (
                self.batch_size, self.out_channels, int(self.h / 2 ** 4), int(self.w / 2 ** 4))

            # prepare for linear layer
            panel_features = panel_features.reshape(self.batch_size, -1)
            assert tuple(panel_features.size()) == (
                self.batch_size, int(self.h / 2 ** 4) * int(self.w / 2 ** 4) * self.out_channels)

            # apply feature tagging to encode the position of each panel: paper: tags 1 - 9

            if label is None:
                panel_label = torch.zeros(self.batch_size, panels.shape[1] + 1)

                if torch.cuda.is_available():
                    panel_label = panel_label.cuda()

                panel_label[:, i] = 1

            panel_features = torch.cat((panel_features, panel_label), dim = 1)

            assert tuple(panel_features.size()) == (
                self.batch_size, int(self.h / 2 ** 4) * int(self.w / 2 ** 4) * self.out_channels + 9)

            # apply embedding
            panel_embedding = self.RNN_embedding(panel_features).unsqueeze(1)
            assert tuple(panel_embedding.size()) == (self.batch_size, 1, self.embedding_size)

            # collect embeddings of all panels
            if i == 0:
                panel_embeddings = panel_embedding

            else:
                panel_embeddings = torch.cat((panel_embeddings, panel_embedding), dim = 1)

        assert tuple(panel_embeddings.size()) == (self.batch_size, panels.shape[1], self.embedding_size)

        return panel_embeddings

    def forward(self, images_context, images_choices, meta_targets = None):
        """
        images is a tensor holding n choices x 8 context + 1 choice panels with respect to their pixels. That is,
        (Batch, n choices, n context + 1 (choice panel), h, w).
        :param images:
        :return: logits of the class labels + logits for class labels over meta labels (12 bit possible choices): logits_targets, logits_meta_targets
        """

        self.batch_size = images_context.shape[0]

        assert tuple(images_context.size()) == (self.batch_size, self.n_context, self.h, self.w)
        assert tuple(images_choices.size()) == (self.batch_size, self.n_choices, self.h, self.w)

        # labels 0 - 7
        # features_ctx = self.get_panel_embeddings(images_context)

        if self.film_cnn:
            assert tuple(meta_targets.size()) == (self.batch_size, 12)
            features_ctx, features_choices = self.get_panel_embeddings_stack(images_context, images_choices,
                                                                             meta_targets)

        else:
            features_ctx, features_choices = self.get_panel_embeddings_stack(images_context, images_choices)

        for choice in range(self.n_choices):
            if self.use_delimiter:
                chi = torch.cat((features_ctx,
                                 self.delimiter.unsqueeze(0).unsqueeze(1).repeat((self.batch_size, 1, 1), 0),
                                 features_choices[:, choice, :].unsqueeze(1)), dim = 1)

                assert tuple(chi.size()) == (self.batch_size, self.n_context + 2, self.embedding_size)

            else:
                chi = torch.cat((features_ctx,
                                 features_choices[:, choice, :].unsqueeze(1)), dim = 1)

                assert tuple(chi.size()) == (self.batch_size, self.n_context + 1, self.embedding_size)

            if self.use_transformer:
                logits = self.transformer(chi, i_iteration = choice)

            else:
                logits = self.RN(chi)

            assert tuple(logits.size()) == (self.batch_size, self.n_labels)

            # logits for answer panels
            # logits pass through a softmax layer [logits_0, ..., logits_7]

            logits_target = logits[:, :self.out_dim_choice]
            assert tuple(logits_target.size()) == (self.batch_size, self.out_dim_choice)

            # logits of meta targets pass through sigmoid: logits_final_meta = sum_i^n_answers logits_meta_target
            # logits_meta_target = logits[:, 1:]

            logits_meta_target = logits[:, self.out_dim_choice:]
            assert tuple(logits_meta_target.size()) == (self.batch_size, self.n_labels - self.out_dim_choice)

            # collect logits

            if choice == 0:
                logits_targets = logits_target
                logits_meta_targets = logits_meta_target

            else:
                logits_targets = torch.cat((logits_targets, logits_target), dim = 1)
                logits_meta_targets += logits_meta_target

        assert tuple(logits_meta_targets.size()) == (self.batch_size, 12)

        if self.use_film:
            assert tuple(logits_targets.size()) == (self.batch_size, self.dim_film)

            if meta_targets is not None:
                filmed_params = self.film_layers({"aux": meta_targets, "input": logits_targets})

            else:
                filmed_params = self.film_layers({"aux": logits_meta_targets, "input": logits_targets})

            logits_targets = filmed_params["input"]
            logits_meta_targets = filmed_params["aux"]

            logits_targets = self.film_classifier(logits_targets)

        assert tuple(logits_targets.size()) == (self.batch_size, self.n_choices)

        return logits_targets, logits_meta_targets
