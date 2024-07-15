import torch
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import utils
import copy

#from losses import *

def produce_data_target(sample,inv_or_forw):
    if inv_or_forw == 'forward_gamma' or inv_or_forw == 'forward_radiation':
        data, target = sample[0], sample[1]
    elif inv_or_forw == 'inverse':
        data, target = (sample[0], sample[1]), sample[2]
    elif inv_or_forw == 'inverse_forward':
        data, target = (sample[0], sample[1]), sample[0]
    elif inv_or_forw == 'inverse_forward_GammaRad':
        data, target = (sample[0], sample[1]), (sample[0], utils.downsample_radiation(sample[1], rates=[4, 2]))
    return data,target
def train_model_single_epoch(model, loss_fn, optimizer, train_loader, epoch,inv_or_forw, grad_accumulation_step = None, clip_norm=1 ):
    model.train()

    running_loss = 0
    num_batches_per_print = 100
    for idx, sample in enumerate(train_loader):
        data, target = produce_data_target(sample,inv_or_forw)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)  # average loss over batch
        running_loss += loss.item()
        if grad_accumulation_step is not None:
            loss = loss / grad_accumulation_step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm )
        if grad_accumulation_step is not None:
            if (idx + 1) % grad_accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()


        if (idx+1) % num_batches_per_print == 0:
            avg_loss = running_loss / num_batches_per_print
            print('TRAIN Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (idx+1) * train_loader.batch_size,
                                                                           len(train_loader.dataset),
                                                                           100. * (idx+1) * train_loader.batch_size / len(
                                                                               train_loader.dataset), avg_loss))
            running_loss = 0
    if grad_accumulation_step is not None:
        if grad_accumulation_step > 1 and (idx + 1) % grad_accumulation_step != 0:
            optimizer.step()
    return avg_loss


def evaluate_model(model, loss_fn, data_loader, set,inv_or_forw, return_output = False):
    model.eval()
    all_losses = np.array([])
    with torch.no_grad():
        for idx, sample in enumerate(data_loader):
            data, target = produce_data_target(sample,inv_or_forw)
            output = model(data)
            if return_output:
                if data_loader.batch_size != 1:
                    return output,target
                else:
                    if idx == 0:
                        all_output0,all_output1 = output[0],output[1]
                        all_target0,all_target1 = target[0],target[1]
                    else:
                        all_output0 = torch.cat((all_output0, output[0]), dim=0)
                        all_output1 = torch.cat((all_output1, output[1]), dim=0)
                        all_target0 = torch.cat((all_target0, target[0]), dim=0)
                        all_target1 = torch.cat((all_target1, target[1]), dim=0)
                        if idx == len(data_loader)-1:
                            return (all_output0,all_output1),(all_target0,all_target1)
            loss = loss_fn(output, target).item()
            all_losses = np.append(all_losses, loss)
    avg_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    print('\n{} set: Average loss: {:.4f} +- {:.4f}'.format(set, avg_loss, std_loss))
    return avg_loss, std_loss




def run_model(model, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs, schedule_step, gamma,inv_or_forw,grad_accumulation_step = None,clip_norm = 1):
    train_losses, train_losses_std, val_losses, val_losses_std = [], [], [], []
    # before training:
    val_loss, val_loss_std = evaluate_model(model, loss_fn, val_loader, set='Validation',inv_or_forw=inv_or_forw)
    best_loss = val_loss
    train_loss, train_loss_std = evaluate_model(model, loss_fn, train_loader, set='Train',inv_or_forw=inv_or_forw)
    train_losses.append(train_loss)
    train_losses_std.append(train_loss_std)
    val_losses.append(val_loss)
    val_losses_std.append(val_loss_std)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=schedule_step, gamma=gamma)
    print('Starting training')
    for epoch in range(1, epochs + 1):
        print('Epoch:', epoch, 'LR:', scheduler.get_last_lr()[0])
        _ = train_model_single_epoch(model, loss_fn, optimizer, train_loader, epoch,inv_or_forw, grad_accumulation_step,clip_norm)
        scheduler.step()
        val_loss, val_loss_std = evaluate_model(model, loss_fn, val_loader, set='Validation',inv_or_forw=inv_or_forw)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
        train_loss, train_loss_std = evaluate_model(model, loss_fn, train_loader, set='Train',inv_or_forw=inv_or_forw)
        train_losses.append(train_loss)
        train_losses_std.append(train_loss_std)
        val_losses.append(val_loss)
        val_losses_std.append(val_loss_std)
    test_loss, _ = evaluate_model(model, loss_fn, test_loader, set='Test',inv_or_forw=inv_or_forw)
    return train_losses, train_losses_std, val_losses, val_losses_std, test_loss, best_state_dict, best_loss
