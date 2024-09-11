import os
import torch
import pickle

import numpy as np
import torch.backends.cudnn as cudnn

from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.auto import tqdm
# from torchviz import make_dot
from losses import MultiTaskLoss, CoxLoss
from datasets import RadDataset
from models import FusionModelBi, Model
from utils import *
from parameters import parse_args
import scipy.io
import time as timetime
# from monai.networks.nets import DenseNet121,HighResNet,SEResNext50

import matplotlib.pyplot as plt 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def one_epoch(args, split, model, optim, loader, criterion):
    if split == "train":
        model.train()
    else:
        model.eval()
    total = 0
    sum_loss = 0
    all_preds_grade = []
    all_preds_hazard = []
    all_grade = []
    all_time = []
    all_event = []
    all_ID = []

    device = 'cuda:0'
    for i, (mod1, mod2, grade, time, event, ID) in enumerate(loader):
        if i%1==0:
            print(f"Sample {i}/{len(loader)}")
            # Display the four samples for each region. Just run it for a single batch and then exit the run to look at the saved images
            # PT_image, LN_image = mod1[0], mod2[0]
            # print("saving patient", ID[0], "to folder")
            # for i in range(4):
            #     plt.imsave(os.path.join(args.savedir, str(i)+"_PT.png"), PT_image[i,0,:,:])
            #     plt.imsave(os.path.join(args.savedir, str(i)+"_LN.png"), LN_image[i,0,:,:])
            
            # print("-----------------------")
        
        model = model.to(device)
        
        mod1, mod2, grade, time, event = mod1.to(device), mod2.to(device), grade.to(device), time.to(device), event.to(device)
        batch = mod1.shape[0]

        pred = model(mod1, mod2)
        
        if args.batch_size==1:
            if args.task == "multitask":
                pred_grade, pred_hazard = pred
            elif args.task == "classification":
                pred_grade, pred_hazard = pred[0], torch.empty(1)
            elif args.task == "survival":
                pred_grade, pred_hazard = torch.empty(1), pred[0]
            else:
                raise NotImplementedError(
                    f'task method {args.task} is not implemented')
        else:
            if args.task == "multitask":
                pred_grade, pred_hazard = pred
            elif args.task == "classification":
                pred_grade, pred_hazard = pred.squeeze(), torch.empty(1)
            elif args.task == "survival":
                pred_grade, pred_hazard = torch.empty(1), pred.squeeze()
            else:
                raise NotImplementedError(
                    f'task method {args.task} is not implemented')
        loss_task = criterion(args.task, pred_grade, pred_hazard, grade, time, event)
        loss = loss_task

        
        if split == 'train':
            optim.zero_grad()
            loss.backward()
            optim.step()

        total += batch
        sum_loss += batch * (loss.item())
        all_preds_grade.append(pred_grade)
        all_preds_hazard.append(pred_hazard)
        all_grade.append(grade)
        all_time.append(time)
        all_event.append(event)
        all_ID.append(ID)

    all_grade = torch.concat(all_grade)
    all_time = torch.concat(all_time)
    all_event = torch.concat(all_event)

    if args.task == "classification" :
        all_preds_grade = torch.concat(all_preds_grade)
        return sum_loss / total, (all_preds_grade, None, all_grade, all_time, all_event, all_ID)
    elif args.task == "multitask":
        all_preds_grade = torch.concat(all_preds_grade)
        all_preds_hazard = torch.concat(all_preds_hazard)
        return sum_loss / total, (all_preds_grade, all_preds_hazard, all_grade, all_time, all_event, all_ID)
    else: 
        all_preds_hazard = torch.concat(all_preds_hazard)
        return sum_loss / total, (None, all_preds_hazard, all_grade, all_time, all_event, all_ID)
    
def test(args, device):
    model_name = args.fusion_type+'_'+args.task+'_'+str(args.n_epochs)+'_'+str(args.lr)
    criterion = MultiTaskLoss()
    data_test = extract_csv(os.path.join(
        args.dataroot, "data_table_test.csv"))
    
    checkpoint = torch.load(os.path.join(args.checkpoints_dir, args.exp_name, model_name, f'{model_name}_best_val_cindex.pt'))

    # Create an instance of the model
    model = Model(args)

    # Extract the 'epoch' from the loaded checkpoint
    saved_epoch = checkpoint['epoch']

    # Print or use the extracted epoch
    print(f"The model is saved on epoch: {saved_epoch}")

    # Load the model state from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    

    test_set = RadDataset(data_test, args.dataroot, train_flag=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)

    
    train_loss = checkpoint['train_loss']
    train_preds = checkpoint['train_pred']
    val_loss = checkpoint['val_loss']
    val_preds = checkpoint['val_pred']
    test_loss, test_preds = one_epoch(args, "test", model, None, test_loader, criterion)

    ci_train, _ = compute_metrics(args, train_preds)
    ci_val, _ = compute_metrics(args, val_preds)
    ci_test, _ = compute_metrics(args, test_preds)

    print(
        f"[Final] Apply model to training set: Loss = {train_loss}, C-Index = {ci_train}")
    print(
        f"[Final] Apply model to validation set: Loss = {val_loss}, C-Index = {ci_val}")
    print(
        f"[Final] Apply model to test set: Loss = {test_loss}, C-Index = {ci_test}")

    pickle.dump(train_preds, open(os.path.join(args.checkpoints_dir, args.exp_name, model_name, 'pred_train.pkl'), 'wb'))
    pickle.dump(val_preds, open(os.path.join(args.checkpoints_dir, args.exp_name, model_name, 'pred_val.pkl'), 'wb'))
    pickle.dump(test_preds, open(os.path.join(args.checkpoints_dir, args.exp_name, model_name, 'pred_test.pkl'), 'wb'))
    


def train_model(args, data_train, data_val, model, criterion, optim, scheduler, device):

    model_name = args.fusion_type+'_'+args.task+'_'+str(args.n_epochs)+'_'+str(args.lr)

    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    np.random.seed(42)

    train_set = RadDataset(
        data_train, args.dataroot)
    
    val_set = RadDataset(data_val, args.dataroot, train_flag=False)



    train_loader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    

    metric_logger = {'train': {'loss': [], 'cindex': []},
                     'val': {'loss': [], 'cindex': []}}

    best_val_cindex = float('-inf')  # Initialize to negative infinity
    # cudnn.deterministic = True
    for epoch in tqdm(range(args.epoch_count, args.niter+args.n_epochs+1)):

        print(device)
        loss, preds = one_epoch(args,
                                "train", model, optim, train_loader, criterion)
        scheduler.step()
        vloss, vpreds = one_epoch(args,
                                    "val", model, None, val_loader, criterion)

        if epoch % args.print_freq == 0:
            print(f"epoch {epoch}")

            lr_tmp = get_lr(optim)
            print(f"Learning rate in current epoch: {lr_tmp}")

            ci_train, _ = compute_metrics(args, preds)
            metric_logger['train']['loss'].append(loss)
            metric_logger['train']['cindex'].append(ci_train)

            print(f"Training loss = {loss}")
            print(f"Train C-index (survival) = {ci_train}")

            ci_val, _ = compute_metrics(args, vpreds)
            metric_logger['val']['loss'].append(vloss)
            metric_logger['val']['cindex'].append(ci_val)

            print(f"Validation loss = {vloss}")
            print(f"Val C-index (survival) = {ci_val}")

            if (epoch > 5) and (ci_val > best_val_cindex):
                best_val_cindex = ci_val

                torch.save({
                'args': args,
                'epoch': epoch,
                'model_state_dict': model.cpu().state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'metrics': metric_logger,
                'train_loss': loss,
                'train_pred': preds,
                'val_loss': vloss,
                'val_pred': vpreds},
                os.path.join(args.checkpoints_dir, args.exp_name, model_name, f'{model_name}_best_val_cindex.pt'))


    return model, optim, metric_logger


def train_val(args, device):
    criterion = MultiTaskLoss()
    data_train = extract_csv(os.path.join(
        args.dataroot, "data_table_train.csv"))
    
    data_val = extract_csv(os.path.join(
        args.dataroot, "data_table_val.csv"))
    
    # torch.cuda.manual_seed_all(42)
    # torch.manual_seed(42)
    # np.random.seed(42)
    model = Model(args)
    model.to(device)
    
    optim = define_optimizer(args, model)
    scheduler = define_scheduler(args, optim)
    print(model)
    print("Number of Trainable Parameters: %d" %
          count_parameters(model))
    print("Optimizer Type:", args.optimizer_type)
    print("Activation Type:", args.act_type)



    model, optim, metric_logger = train_model(
        args, data_train, data_val, model, criterion, optim, scheduler, device)
    
    return metric_logger



if __name__ == '__main__':
    args = parse_args()
    root = args.dataroot
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    np.random.seed(42)
    metric_logger = train_val(args, device)
    test(args, device)

    model_name = args.fusion_type+'_'+args.task+'_'+str(args.n_epochs)+'_'+str(args.lr)


    # Save results for train, validation, and test sets
    save_results_to_mat("train", args, model_name)
    save_results_to_mat("val", args, model_name)
    save_results_to_mat("test", args, model_name)

    
    
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plotting the training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(args.epoch_count, args.niter + args.n_epochs + 1),
            metric_logger['train']['loss'], label='Train')
    plt.plot(range(args.epoch_count, args.niter + args.n_epochs + 1),
            metric_logger['val']['loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting the training C-index
    plt.subplot(1, 2, 2)
    plt.plot(range(args.epoch_count, args.niter + args.n_epochs + 1),
            metric_logger['train']['cindex'], label='Train')
    plt.plot(range(args.epoch_count, args.niter + args.n_epochs + 1),
            metric_logger['val']['cindex'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('C-Index')
    plt.title('Training and Validation C-Index')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()