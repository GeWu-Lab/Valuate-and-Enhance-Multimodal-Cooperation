import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets.dataloader import AV_KS_Dataset
from models.models import AVClassifier
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='KineticSound')
    parser.add_argument('--model', default='resnet18', type=str, choices=['resnet18'])
    parser.add_argument('--modulation', default='none', type=str, choices=['none', 'sample', 'modality'])
    parser.add_argument('--compare', default='none', type=str, choices=['none'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--n_classes', default=31, type=int)
    parser.add_argument('--encoder_lr_decay', default=1.0, type=float, help='decay coefficient')
    parser.add_argument('--loader', default=158, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=30, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--train', action='store_true', help='turn on train mode')
    parser.add_argument('--log_path', default='log_model', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='2, 3', type=str, help='GPU ids')

    return parser.parse_args()



def train_epoch(args, epoch, model, device, dataloader, optimizer):
  criterion = nn.CrossEntropyLoss()

  model.train()
  print("Start training ... ")

  _loss = 0
  _loss_a = 0
  _loss_v = 0

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
    print('testing...')
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
  args = get_arguments()
  print(args)
  print('Now compare method, %s'% args.compare)

  setup_seed(args.random_seed)
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
  gpu_ids = list(range(torch.cuda.device_count()))

  device = torch.device('cuda:0')


  model = AVClassifier(args)

  model.to(device)
  model = torch.nn.DataParallel(model, device_ids=gpu_ids)
  model.cuda()


  train_dataset = AV_KS_Dataset(mode='train',loader=args.loader)
  train_val_dataset = AV_KS_Dataset(mode='train',loader=args.loader)
  test_dataset = AV_KS_Dataset(mode='test',loader=args.loader)
  

  
  train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16,pin_memory=True)    
  train_val_dataloader = DataLoader(train_val_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=16,pin_memory=True)                                
  test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)

  if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
  elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)

  scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

  if args.train:
    best_acc = 0.0

    for epoch in range(args.epochs):

      print('Epoch: {}: '.format(epoch))
      writer_path = os.path.join(args.log_path)
      if not os.path.exists(writer_path):
        os.mkdir(writer_path)
      log_name = '{}_{}_{}_{}_{}_epochs{}_batch{}_lr{}_en{}'.format(args.compare,args.optimizer,  args.dataset, args.modulation, args.model, args.epochs, args.batch_size, args.learning_rate, args.encoder_lr_decay)


      batch_loss = train_epoch(args, epoch, model, device, train_dataloader, optimizer)
      scheduler.step()
      acc= valid(args, model, device, test_dataloader, epoch, log_name)


      if acc > best_acc:
        best_acc = float(acc)

        model_name = '{}_best_model_of_{}_{}_{}_{}_epochs{}_batch{}_lr{}_en{}.pth'.format(args.compare,args.optimizer,  args.dataset, args.modulation, args.model, args.epochs, args.batch_size, args.learning_rate, args.encoder_lr_decay)
            
        saved_dict = {'saved_epoch': epoch,
                      'acc': acc,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict()}

        save_dir = os.path.join(args.log_path, model_name)

        torch.save(saved_dict, save_dir)
        print('The best model has been saved at {}.'.format(save_dir))
        print("Loss: {:.4f}, Acc: {:.4f}".format(batch_loss, acc))

      else:
        print("Loss: {:.4f}, Acc: {:.4f}, Best Acc: {:.4f}".format(batch_loss, acc, best_acc))


if __name__ == "__main__":
  main()