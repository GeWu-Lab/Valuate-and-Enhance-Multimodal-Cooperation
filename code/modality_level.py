import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.dataloader import AV_KS_Dataset, AV_KS_Dataset_modality_level
from models.models import AVClassifier
import random

sample_nums = []


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


sample_nums = []
part_a = []
part_v = []


### key parameters:
## --warmup: warm up epochs. Default 5 epochs.
## --part_ratio: the percentage of sunset to estimate modality preference ($Z$ in the modality-level method in the paper). Default 0.2, which means the number of Z is 20% samples.



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="KineticSound")
    parser.add_argument("--model", default="resnet18", type=str, choices=["resnet18"])
    parser.add_argument(
        "--modulation", default="modality", type=str, choices=["sample", "modality"]
    )
    parser.add_argument("--compare", default="none", type=str, choices=["none"])
    parser.add_argument("--n_classes", default=31, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--loader", default=165, type=int)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--part_ratio", default=0.2, type=float, help="part ratio")
    parser.add_argument("--alpha", default=1, type=float, help="alpha")

    parser.add_argument("--optimizer", default="adam", type=str, choices=["sgd", "adam"])
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--lr_decay_step", default=30, type=int, help="where learning rate decays"
    )
    parser.add_argument(
        "--lr_decay_ratio", default=0.1, type=float, help="decay coefficient"
    )

    parser.add_argument("--train", action="store_true", help="turn on train mode")
    parser.add_argument(
        "--log_path",
        default="log_modality",
        type=str,
        help="path to save tensorboard logs",
    )

    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--gpu_ids", default="2, 3", type=str, help="GPU ids")

    return parser.parse_args()


def execute_modulation(args, model, device, dataloader, log_name, epoch):
    n_classes = args.n_classes

    contribution = {}
    softmax = nn.Softmax(dim=1)
    cona = 0.0
    conv = 0.0

    with torch.no_grad():
        model.eval()

        for step, (image, spec, label, index) in tqdm(enumerate(dataloader)):
            image = image.to(device)
            spec = spec.to(device)
            # label = label.to(device)
            a, v, out = model(spec.float(), image.float())

            out_v = model.module.exec_drop(a, v, drop="audio")
            out_a = model.module.exec_drop(a, v, drop="visual")

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i, item in enumerate(label):
                all = prediction[i].cpu().data.numpy()
                index_all = np.argmax(all)
                v = pred_v[i].cpu().data.numpy()
                index_v = np.argmax(v)
                a = pred_a[i].cpu().data.numpy()
                index_a = np.argmax(a)

                value_all = 0.0
                value_a = 0.0
                value_v = 0.0
                if index_all == label[i]:
                    value_all = 2.0
                if index_v == label[i]:
                    value_v = 1.0
                if index_a == label[i]:
                    value_a = 1.0

                contrib_a = (value_a + value_all - value_v) / 2.0
                contrib_v = (value_v + value_all - value_a) / 2.0
                cona += contrib_a
                conv += contrib_v

                contribution[int(index[i])] = (contrib_a, contrib_v)

    cona /= len(dataloader.dataset)
    conv /= len(dataloader.dataset)

    if not os.path.exists(args.log_path + "/" + log_name):
        os.mkdir(args.log_path + "/" + log_name)
    if not os.path.exists(args.log_path + "/" + log_name + "/contribution"):
        os.mkdir(args.log_path + "/" + log_name + "/contribution")
    np.save(
        args.log_path + "/" + log_name + "/contribution/" + str(epoch) + ".npy",
        contribution,
    )
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("now train epoch, cona and conv: ", cona, conv)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    if epoch >= args.warmup - 1:
        part_cona = 0.0
        part_conv = 0.0
        num = int(len(dataloader.dataset) * args.part_ratio)
        choice = np.random.choice(len(dataloader.dataset), num)
        print(len(choice))
        for i in choice:
            contri_a, contri_v = contribution[i]
            part_cona += contri_a
            part_conv += contri_v
        part_cona /= num
        part_conv /= num
        part_a.append(part_cona)
        part_v.append(part_conv)

        gap_a = 1.0 - part_cona
        gap_v = 1.0 - part_conv

        part_difference = abs(gap_a - gap_v) / 3 * 2 * args.alpha
        print("part_p:", part_difference)

        train_dataset = AV_KS_Dataset_modality_level(
            mode="train",
            loader=args.loader,
            contribution_a=part_cona,
            contribution_v=part_conv,
            alpha=args.alpha,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    return cona, conv, train_dataloader


def train_epoch(args, epoch, model, device, dataloader, optimizer):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training ... ")

    _loss = 0

    for step, (image, spec, label, index, drop) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        image = image.to(device)
        spec = spec.to(device)
        label = label.to(device)
        drop = drop.to(device)
        a, v, out = model(spec.float(), image.float(), drop)

        loss = criterion(out, label)
        loss.backward()

        optimizer.step()
        _loss += loss.item()

    sample_nums.append(len(dataloader.dataset))

    return _loss / len(dataloader)


def warmup_epoch(args, epoch, model, device, dataloader, optimizer):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Warm up ... ")

    _loss = 0

    for step, (image, spec, label, index) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        image = image.to(device)
        spec = spec.to(device)
        label = label.to(device)
        a, v, out = model(spec.float(), image.float())

        loss = criterion(out, label)
        loss.backward()

        optimizer.step()
        _loss += loss.item()

    return _loss / len(dataloader)


def valid(args, model, device, dataloader, epoch, log_name):
    softmax = nn.Softmax(dim=1)
    print("testing...")
    n_classes = args.n_classes

    cri = nn.CrossEntropyLoss()
    _loss = 0

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        for step, (image, spec, label, index) in tqdm(enumerate(dataloader)):
            image = image.to(device)
            spec = spec.to(device)
            label = label.to(device)
            a, v, out = model(spec.float(), image.float())

            prediction = softmax(out)
            loss = cri(out, label)
            _loss += loss.item()

            for i, item in enumerate(label):
                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0

    return sum(acc) / sum(num)


def main():
    cona_all = []
    conv_all = []
    args = get_arguments()
    print(args)
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device("cuda:0")

    model = AVClassifier(args)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    train_dataset = AV_KS_Dataset(mode="train", loader=args.loader)
    train_val_dataset = AV_KS_Dataset(mode="train", loader=args.loader)
    test_dataset = AV_KS_Dataset(mode="test", loader=args.loader)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    train_val_dataloader = DataLoader(
        train_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16
    )

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4,
            amsgrad=False,
        )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, args.lr_decay_ratio
    )

    if args.train:
        best_acc = 0.0

        for epoch in range(args.epochs):
            print("Epoch: {}: ".format(epoch))
            writer_path = os.path.join(args.log_path)
            if not os.path.exists(writer_path):
                os.mkdir(writer_path)
            log_name = (
                "{}_{}_{}_{}_{}_epochs{}_batch{}_lr{}_en{}_warmup{}_alpha{}".format(
                    args.compare,
                    args.optimizer,
                    args.dataset,
                    args.modulation,
                    args.model,
                    args.epochs,
                    args.batch_size,
                    args.learning_rate,
                    args.part_ratio,
                    args.warmup,
                    args.alpha,
                )
            )

            if epoch < args.warmup:
                batch_loss = warmup_epoch(
                    args, epoch, model, device, train_dataloader, optimizer
                )
            else:
                batch_loss = train_epoch(
                    args, epoch, model, device, train_dataloader, optimizer
                )

            if epoch >= args.warmup - 1:
                cona, conv, train_dataloader = execute_modulation(
                    args, model, device, train_val_dataloader, log_name, epoch
                )
            else:
                cona, conv, _ = execute_modulation(
                    args, model, device, train_val_dataloader, log_name, epoch
                )
            cona_all.append(cona)
            conv_all.append(conv)
            scheduler.step()
            acc = valid(args, model, device, test_dataloader, epoch, log_name)

            if acc > best_acc:
                best_acc = float(acc)

                model_name = "{}_best_model_of_{}_{}_{}_{}_epochs{}_batch{}_lr{}_en{}_warmup{}_alpha{}_{}_{}.pth".format(
                    args.compare,
                    args.optimizer,
                    args.dataset,
                    args.modulation,
                    args.model,
                    args.epochs,
                    args.batch_size,
                    args.learning_rate,
                    args.part_ratio,
                    args.warmup,
                    args.alpha,
                    args.dynamic,
                    args.method,
                )

                saved_dict = {
                    "saved_epoch": epoch,
                    "acc": acc,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }

                save_dir = os.path.join(args.log_path, model_name)

                torch.save(saved_dict, save_dir)
                print("The best model has been saved at {}.".format(save_dir))
                print("Loss: {:.4f}, Acc: {:.4f}".format(batch_loss, acc))

            else:
                print(
                    "Loss: {:.4f}, Acc: {:.4f}, Best Acc: {:.4f}".format(
                        batch_loss, acc, best_acc
                    )
                )


if __name__ == "__main__":
    main()
