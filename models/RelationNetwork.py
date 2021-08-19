import torch
import torch.nn as nn

class RelationNetwork(nn.Module):
    def __init__(self, embedding_size, n_features, theta_dim, n_labels, triples=False, both=False,
                 classifier_extra_layers=0, ctx_only=False):
        super(Relation_Network, self).__init__()

        self.ctx_only = ctx_only

        # dimension of each input object x_i in Chi
        self.embedding_size = embedding_size

        # number of features => n_features*n_features combinations are considered in the sum
        self.n_features = n_features

        self.theta_dim = theta_dim
        theta_features = [self.theta_dim for i in range(5)]

        self.triples = triples

        if self.triples:
            print("RELATION NETWORK > USING TRIPLES")
            theta_features[0] = int(theta_features[0] + theta_features[0] / 2.)

        # f_phi
        self.n_labels = n_labels
        phi_features = [512, 256, 256, self.n_labels]

        if classifier_extra_layers > 0:
            phi_features = [512, 256, 256]

            for layer in range(classifier_extra_layers):
                phi_features.append(256)

            phi_features.append(self.n_labels)

        assert phi_features[-1] == self.n_labels

        self.g_theta = nn.Sequential(*[layer for i in range(len(theta_features) - 1) for layer in
                                       [nn.Linear(in_features=theta_features[i], out_features=theta_features[i + 1]),
                                        nn.ReLU()]])

        # both = triples and tuples
        self.both = both
        if self.both:
            assert self.triples == True
            print("using triples and pairs")
            # for tuples
            theta_features = [self.theta_dim for i in range(5)]
            self.g_theta_both = nn.Sequential(
                *[layer for i in range(len(theta_features) - 1) for layer in [nn.Linear(in_features=theta_features[i],
                                                                                        out_features=theta_features[
                                                                                            i + 1]),
                                                                              nn.ReLU()]])

            # network to join triples and tuples
            theta_features_merge = [self.theta_dim for i in range(3)]
            theta_features_merge[0] = 2 * self.theta_dim

            self.g_theta_merge = nn.Sequential(*[layer for i in range(len(theta_features_merge) - 1) for layer in
                                                 [nn.Linear(in_features=theta_features_merge[i],
                                                            out_features=theta_features_merge[i + 1]),
                                                  nn.ReLU()]])

        layers = [nn.Linear(in_features=phi_features[0], out_features=phi_features[1]), nn.ReLU(), nn.Dropout(p=0.5)]

        extra_layers = []

        for i in range(1, len(phi_features) - 1):
            extra_layers += [nn.Linear(in_features=phi_features[i], out_features=phi_features[i + 1]), nn.ReLU()]

        # get rid of last relu (logits needed)
        layers += extra_layers[:-1]

        self.f_phi = nn.Sequential(*layers)

    def get_triples(self, chi):
        """
        get triples. reverse versions are also included: panel_1, panel_2, panel_3 (reverse) --> panel_3, panel_2, panel_1

        2x rows = 6
        2x columns = 6
        2x diagonals = 4
        =14 combinations

        :param chi:
        :return:
        """

        batch_size = chi.shape[0]
        n_panels = chi.shape[1]  # =9
        d_embedding = chi.shape[2]

        assert tuple(chi.size()) == (batch_size, self.n_features, self.embedding_size)
        # TODO reduce unsqueeze functions
        # rows and columns
        for i in range(3):
            triple_column = torch.cat((chi[:, 0 + i, :].unsqueeze(1),
                                       chi[:, 3 + i, :].unsqueeze(1),
                                       chi[:, 6 + i, :].unsqueeze(1)), dim=2)

            triple_row = torch.cat((chi[:, 0 + i * 3, :].unsqueeze(1),
                                    chi[:, 1 + i * 3, :].unsqueeze(1),
                                    chi[:, 2 + i * 3, :].unsqueeze(1)), dim=2)

            triple_column_reverse = torch.cat((chi[:, 6 + i, :].unsqueeze(1),
                                               chi[:, 3 + i, :].unsqueeze(1),
                                               chi[:, 0 + i, :].unsqueeze(1)), dim=2)

            triple_row_reverse = torch.cat((chi[:, 2 + i * 3, :].unsqueeze(1),
                                            chi[:, 1 + i * 3, :].unsqueeze(1),
                                            chi[:, 0 + i * 3, :].unsqueeze(1)), dim=2)

            if i == 0:
                triples = torch.cat((triple_column, triple_column_reverse, triple_row, triple_row_reverse), dim=1)

            else:
                triples = torch.cat((triples, triple_column, triple_column_reverse, triple_row, triple_row_reverse),
                                    dim=1)

            assert tuple(triples.size()) == (batch_size, (i + 1) * 4, d_embedding * 3)

        # diagonals
        diag_1 = torch.cat((chi[:, 0, :].unsqueeze(1), chi[:, 4, :].unsqueeze(1), chi[:, 8, :].unsqueeze(1)), dim=2)
        diag_2 = torch.cat((chi[:, 2, :].unsqueeze(1), chi[:, 4, :].unsqueeze(1), chi[:, 6, :].unsqueeze(1)), dim=2)

        diag_1_reverse = torch.cat((chi[:, 8, :].unsqueeze(1), chi[:, 4, :].unsqueeze(1), chi[:, 0, :].unsqueeze(1)),
                                   dim=2)
        diag_2_reverse = torch.cat((chi[:, 6, :].unsqueeze(1), chi[:, 4, :].unsqueeze(1), chi[:, 2, :].unsqueeze(1)),
                                   dim=2)

        triples = torch.cat((triples, diag_1, diag_1_reverse, diag_2, diag_2_reverse), dim=1)

        assert tuple(triples.size()) == (batch_size, 16, d_embedding * 3)

        return triples

    def get_pairs(self, chi):

        batch_size = chi.shape[0]
        n_panels = chi.shape[1]
        d_embedding = chi.shape[2]

        if self.ctx_only:

            assert tuple(chi.size()) == (batch_size, self.n_features - 1, self.embedding_size)

        else:
            assert tuple(chi.size()) == (batch_size, self.n_features, self.embedding_size)

        for i in range(n_panels):
            if i == 0:
                chi_rep = chi[:, i, :].unsqueeze(1).repeat(1, n_panels, 1)

            else:
                chi_rep = torch.cat((chi_rep, chi[:, i, :].unsqueeze(1).repeat(1, n_panels, 1)), dim=1)

        chi = chi.repeat(1, n_panels, 1)

        pairs = torch.cat((chi_rep, chi), dim=2)

        assert tuple(pairs.size()) == (batch_size, n_panels ** 2, d_embedding * 2)

        return pairs

    def forward(self, chi):
        """
        :param chi: chi = {x1, ..., xn} where x_i is a feature vector
        :return: logits vector of class labels
        """

        # TODO tagging is missung
        self.batch_size = chi.size()[0]

        if self.triples:
            pairs = self.get_triples(chi)

        else:
            pairs = self.get_pairs(chi)

        obj_sum = torch.sum(self.g_theta(pairs), dim=1)

        if self.both:
            pairs = self.get_pairs(chi)
            obj_sum_both = torch.sum(self.g_theta_both(pairs), dim=1)

            obj_sum_merge = torch.cat((obj_sum, obj_sum_both), dim=1)
            assert (obj_sum_merge.size()) == (self.batch_size, self.theta_dim * 2)

            obj_sum = self.g_theta_merge(obj_sum_merge)

        assert tuple(obj_sum.size()) == (self.batch_size, self.theta_dim)

        logits = self.f_phi(obj_sum)
        assert tuple(logits.size()) == (self.batch_size, self.n_labels)

        return logits
