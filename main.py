import argparse

import torch
import utility.functions as util
from utility.Metrics import Metrics_WReN as Metrics
from datasets import PGM_PT, PGMType, PGM
from optimisation.losses import WReN_Loss
from models import WReN

import os

parser = argparse.ArgumentParser(description = 'Additional settings for loading, saving and configurations of ARNe')
parser.add_argument('--load_checkpoint', type = str, nargs = 1, metavar = 'TIMESTAMP',
                    help = 'Timestamp of EXPERIMENT to load.')

parser.add_argument('--experiment', type = str, nargs = 1, metavar = 'EXPERIMENT', help = 'Name of experiment.')

# transformer
parser.add_argument("--transformer", action = "store_true", default = True,
                    help = "use the encoder of the transformer instead of RNs")
parser.add_argument("--d_model", type = int, default = 512, help = "input size of the model")
parser.add_argument("--h", type = int, default = 10, help = "number of attention heads")
parser.add_argument("--n_layers", type = int, default = 6, help = "length of the transformer encoder")
parser.add_argument("--d_ff", type = int, help = "length of the transformer encoder")
parser.add_argument("--d_att", type = int, default = 64, help = "attention dimensions")
parser.add_argument("--d_q_k", type = int, help = "attention dimensions: query and key")
parser.add_argument("--d_v", type = int, help = "attention dimension: values")
parser.add_argument("--dropout_transformer", type = float, default = 0.1, help = "transformer dropout")

parser.add_argument("--patience", type = int, default = 3, help = "early stopping patience")
parser.add_argument('--lr', type = float, default = 0.00005, nargs = 1,
                    help = 'set a non default (json str below) learning rate')
parser.add_argument("--lr_scheduler", action = "store_true", help = "use lr scheduler of the transformer")
parser.add_argument('--warmup_steps', type = int, default = 4000, help = 'set a non default warmup steps')
parser.add_argument('--lr_restore_overwrite', action = "store_true",
                    help = "apply learning rate decay directly after restoring")
parser.add_argument("--lr_decay", default = 0.75, type = float, help = "apply lr decay")
parser.add_argument('--lr_decay_threshold', default = 0.6, type = float,
                    help = 'set a non default thershold for lr decay')
parser.add_argument("--only_loss_target", action = "store_true",
                    help = "only consider the loss of targets. for ablations only")


def main(args):
    json_file = "default_config.json"
    config = util.load_config_file(json_file)

    use_lr_scheduler = False
    lr_scheduler = None
    if args.lr_scheduler:
        use_lr_scheduler = True
        print("USING LR SCHEDULER")
        warmup_steps = args.warmup_steps
        config["experiment"]["name"] += "_lr_scheduler"
        config["experiment"]["name"] += "_warmup_steps_" + str(warmup_steps) + "_"

    config_transformer = None
    if args.transformer:
        print("using transformer")

        config["experiment"]["name"] += "_transformer"

        d_model = 512
        n_layers = 6
        h = 8

        d_ff = 2048
        d_att = int(d_model / h)

        d_v = d_att
        d_k = d_att
        d_q = d_att

        if args.d_model is not None:
            print("Using Transformer d_model {}".format(args.d_model))
            d_model = args.d_model

        if args.n_layers is not None:
            print("Using Transformer n_layers {}".format(args.n_layers))
            n_layers = args.n_layers

        if args.h is not None:
            print("Using Transformer h {}".format(args.h))
            h = args.h

        if args.d_ff is not None:
            print("Using Transformer d_ff {}".format(args.d_ff))
            d_ff = args.d_ff

        if args.d_att is not None:
            print("Using Transformer d_att {}".format(args.d_att))
            d_att = args.d_att

            d_v = d_att
            d_k = d_att
            d_q = d_att

        if args.d_q_k is not None:
            d_q = args.d_q_k
            d_k = args.d_q_k

        if args.d_v is not None:
            d_v = args.d_v

        assert d_q == d_k

        config_transformer = {
            "d_model": d_model,
            "n_layers": n_layers,
            "h": h,
            "d_ff": d_ff,
            "d_v": d_v,
            "d_k": d_k,
            "d_q": d_q,
            "dropout": args.dropout_transformer,
        }

        print(config_transformer)
        config["experiment"]["name"] += "_" + "d_model_" + str(config_transformer["d_model"]) + "_" + "n_layers_" + str(
            config_transformer["n_layers"]) + "_" + "h_" + str(config_transformer["h"]) + "_" + "d_ff_" + str(
            config_transformer["d_ff"]) + "_" + "d_v_" + str(config_transformer["d_v"]) + "_" + "d_k_" + str(
            config_transformer["d_k"]) + "_" + "d_q_" + str(config_transformer["d_q"])

        config["transformer"] = config_transformer

    config["experiment"]["name"] += "_dropout_" + str(args.dropout_transformer)

    # optional name suffix
    if args.experiment is not None:
        config["experiment"]["name"] += "_" + args.experiment[0]

    if args.load_checkpoint is not None:
        config["experiment"]["checkpoint"]["load"] = True
        config["experiment"]["checkpoint"]["timestamp"] = args.load_checkpoint[0]

        config["experiment"]["name"] += "_classifier_extra_layers"

    if args.lr is not None:
        # config["experiment"]["name"] += "_lr_" + str(args.lr[0])
        # print("USING CUSTOM LEARNING RATE")
        # config["optimisation"]["optimiser"]["learning_rate"] = args.lr[0]
        config["experiment"]["name"] += "_lr_" + str(args.lr)
        print("USING CUSTOM LEARNING RATE")
        config["optimisation"]["optimiser"]["learning_rate"] = args.lr

    use_lr_decay = False
    if args.lr_decay is not None:
        use_lr_decay = True
        config["experiment"]["name"] += "_lr_decay_" + str(args.lr_decay) + "_"
        config["optimisation"]["learning_rate_decay"]["rate"] = args.lr_decay

    if args.lr_restore_overwrite:
        config["experiment"]["name"] += "_lr_restore_overwrite_"

    lr_decay_threshold = 0.5
    if args.lr_decay_threshold is not None:
        lr_decay_threshold = args.lr_decay_threshold
        config["experiment"]["name"] += "_lr_decay_threshold_" + str(lr_decay_threshold)

    use_only_loss_target = False
    if args.only_loss_target:
        print("ABLATION: Only use target losses")
        use_only_loss_target = True

        config["experiment"]["name"] += "_target_loss_only"

    config["experiment"]["name"] += "_meta_targets"

    print(config["experiment"]["name"])

    timestamp = util.create_timestamp()
    dirs = util.create_dirs(timestamp, config)

    util.copy_config_file(config, dirs, timestamp)

    # dataloader_train = util.get_dataloader(PGM_PT(PGMType.train, config["dataset"]["data_path"]), None)
    # dataloader_val = util.get_dataloader(PGM_PT(PGMType.val, config["dataset"]["data_path"]), None)
    # dataloader_test = util.get_dataloader(PGM_PT(PGMType.test, config["dataset"]["data_path"]), None)

    dataloader_train = util.get_dataloader(PGM("train", config["dataset"]["data_path"]), None)
    dataloader_val = util.get_dataloader(PGM("val", config["dataset"]["data_path"]), None)
    dataloader_test = util.get_dataloader(PGM("test", config["dataset"]["data_path"]), None)

    model = WReN(config_transformer = config_transformer) # use Transformer encoder
    # model = WReN() # use RN

    model = util.move_to_devices(model)

    criterion = WReN_Loss(beta = 10)

    metrics = Metrics(
        model,
        len(dataloader_train.dataset),
        len(dataloader_val.dataset),
        config["optimisation"]["optimiser"]["learning_rate"],
        config,
        timestamp,
        len_dataset_test = len(dataloader_test.dataset),
        test = True,
    )

    epoch_start = 1
    ema = None

    # restore
    if args.load_checkpoint is not None:
        print("loading checkpoint")

        if use_lr_scheduler:
            model, optimiser, ema, epoch_start, metrics, lr_scheduler = util.restore(model, metrics, ema, dirs, config,
                                                                                     use_lr_scheduler = use_lr_scheduler)

        else:
            model, optimiser, ema, epoch_start, metrics = util.restore(model, metrics, ema, dirs, config,
                                                                       use_lr_scheduler = use_lr_scheduler)

            lr_scheduler = None

        if args.lr_restore_overwrite:
            lr_prev = optimiser.param_groups[0]["lr"]
            lr_decay = optimiser.param_groups[0]["lr"] * config["optimisation"]["learning_rate_decay"]["rate"]
            optimiser.param_groups[0]["lr"] = lr_decay

            assert optimiser.param_groups[0]["lr"] == lr_decay
            print("changed learning rate from {} to {}".format(lr_prev, lr_decay))
    else:

        if not use_lr_scheduler:
            # default
            optimiser = torch.optim.Adam(model.parameters(), lr = config["optimisation"]["optimiser"]["learning_rate"])

        if use_lr_scheduler:
            # paper attention is all you need
            optimiser = torch.optim.Adam(model.parameters(), betas = (0.9, 0.98), eps = 1e-09)
            lr_scheduler = util.Learning_Rate_Scheduler(config_transformer["d_model"], warmup_steps, optimiser)

    epoch_stop = epoch_start + config["model"]["n_epochs"]

    early_stopping = util.Early_Stopping(args.patience, timestamp, dirs)

    # TODO move this to the restore function!
    if args.load_checkpoint is not None:
        early_stopping.acc_best = metrics.data["val"]["accuracies"]["targets"][
            max(metrics.data["val"]["accuracies"]["targets"].keys())]

    # TODO UNDO
    # criterion = criterion.cuda()

    ema = None
    j = 0
    for epoch in range(epoch_start, epoch_stop):
        print("Epoch Started")
        model.train()
        metrics.train(epoch)

        for i, input_data in enumerate(dataloader_train):
            print("i %d" % (i))
            images, correct_labels, correct_meta_labels, idxs = input_data

            if torch.cuda.device_count() > 0:
                images, correct_labels, correct_meta_labels = images.cuda(), correct_labels.cuda(), correct_meta_labels.cuda()

            # cast on gpu for better performance
            images_context, images_choices, correct_labels, correct_meta_labels = PGM_PT.cast_data(images,
                                                                                                   correct_labels,
                                                                                                   correct_meta_labels)

            logits_labels, logits_meta_labels = model(images_context, images_choices, correct_meta_labels)

            optimiser.zero_grad()

            loss_targets, loss_meta, loss_total = criterion(logits_labels, correct_labels, logits_meta_labels,
                                                            correct_meta_labels)

            if use_only_loss_target:
                loss_targets.backward()

            else:
                loss_total.backward()

            if use_lr_scheduler:
                assert lr_scheduler is not None
                lr_scheduler.update_lr()

            optimiser.step()

            metrics.update(logits_labels, logits_meta_labels, correct_labels, correct_meta_labels, loss_targets,
                           loss_meta, loss_total)
            metrics.print_status()

        model.eval()
        metrics.eval(epoch)

        with torch.no_grad():
            for j, input_data in enumerate(dataloader_val):
                images, correct_labels, correct_meta_labels, idxs = input_data

                if torch.cuda.device_count() > 0:
                    images, correct_labels, correct_meta_labels = images.cuda(), correct_labels.cuda(), correct_meta_labels.cuda()

                # cast on gpu for better performance
                images_context, images_choices, correct_labels, correct_meta_labels = PGM_PT.cast_data(images,
                                                                                                       correct_labels,
                                                                                                       correct_meta_labels)

                logits_labels, logits_meta_labels = model(images_context, images_choices, correct_meta_labels)

                loss_targets, loss_meta, loss_total = criterion(logits_labels, correct_labels, logits_meta_labels,
                                                                correct_meta_labels)

                metrics.update(logits_labels, logits_meta_labels, correct_labels, correct_meta_labels, loss_targets,
                               loss_meta, loss_total)
                metrics.print_status()

        early_stopping.check(model, optimiser, ema, metrics, epoch, lr_scheduler)

        if early_stopping.stop:
            print("Stopped training by early stopping.\n Best Epoch {} | Val <Acc> {}".format(early_stopping.epoch_best,
                                                                                              early_stopping.acc_best))
            break

        if use_lr_decay:
            util.check_LR(config, metrics, optimiser, epoch, thr_1 = lr_decay_threshold)

    if early_stopping.stop:
        # restore past model
        config_restore = config
        config_restore["experiment"]["checkpoint"]["timestamp"] = timestamp

        model, optimiser, ema, epoch_start, metrics = util.restore(model, metrics, ema, dirs, config_restore)
        epoch = max(metrics.data[list(metrics.data.keys())[0]]["accuracies"]["targets"])

    print("Testing")
    model.eval()
    metrics.test(epoch)

    with torch.no_grad():
        for j, input_data in enumerate(dataloader_test):
            images, correct_labels, correct_meta_labels, idxs = input_data

            if torch.cuda.device_count() > 0:
                images, correct_labels, correct_meta_labels = images.cuda(), correct_labels.cuda(), correct_meta_labels.cuda()

            # cast on gpu for better performance
            images_context, images_choices, correct_labels, correct_meta_labels = PGM_PT.cast_data(images,
                                                                                                   correct_labels,
                                                                                                   correct_meta_labels)

            logits_labels, logits_meta_labels = model(images_context, images_choices, correct_meta_labels)

            loss_targets, loss_meta, loss_total = criterion(logits_labels, correct_labels, logits_meta_labels,
                                                            correct_meta_labels)

            metrics.update(logits_labels, logits_meta_labels, correct_labels, correct_meta_labels, loss_targets,
                           loss_meta, loss_total)
            metrics.print_status()

    util.save_checkpoint(epoch, model, optimiser, metrics, dirs, timestamp, ema, lr_scheduler)


if __name__ == "__main__":
    main(parser.parse_args())
