# -----------------------------------------------------------------------------------------------------
import torch as tt
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .common import save_state, load_state
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------------------------------
import datetime
now = datetime.datetime.now


class QuantiyMonitor:
    """ Monitors a quantity overtime to check if it improves(decrease) after a given patience. """

    def __init__(self, name, patience, delta, verbose=False):
        assert(patience>0) # patience should be positive
        assert(delta>0) # delta should be positive
        self.patience, self.delta, self.verbose = patience, delta, verbose
        self.name = name
        self.reset()
    
    def reset(self, initial=None):
        self.last = (tt.inf if initial is None else initial)
        self.best = self.last
        self.counter = 0
        self.best_epoch = -1

    def __call__(self, current, epoch=-1):
        self.last = current
        if self.best == tt.inf: 
            self.best=self.last
            self.best_epoch = epoch
            if self.verbose: print(f'|~|\t{self.name} Set to [{self.best}] on epoch {epoch}')
        else:
            delta = self.best - current # `loss` has decreased if `delta` is positive
            if delta > self.delta:
                # loss decresed more than self.delta
                if self.verbose: print(f'|~|\t{self.name} Decreased by [{delta}] on epoch {epoch}') # {self.best} --> {current}, 
                self.best = current
                self.best_epoch = epoch
                self.counter = 0
            else:
                # loss didnt decresed more than self.delta
                self.counter += 1
                if self.counter >= self.patience: 
                    if self.verbose: print(f'|~| Stopping on {self.name} = [{current}] @ epoch {epoch} | best value = [{self.best}] @ epoch {self.best_epoch}')
                    return True # end of patience
        return False

class Trainer:

    def train(model, training_data=None, validation_data=None, 
                epochs=0, batch_size=0, shuffle=None, validation_freq=0, 
                criterion_type=None, criterion_args={}, optimizer_type=None, optimizer_args={}, lrs_type=None, lrs_args={},
                record_batch_loss=False, early_stop_train=None, early_stop_val=None, checkpoint_freq=0, save_path=None,
                save_state_only=False, verbose=0, plot=0):

        assert epochs>0, f'Epochs should be at least 1'
        assert batch_size>0, f'Batch Size should be at least 1'
        assert training_data is not None, f'Training data not provided'
        
        if validation_data is not None: 
            assert validation_freq>0, f'Validation frequency should be at least 1'
        
        
        assert criterion_type is not None, f'Loss Criterion data not provided'
        assert optimizer_type is not None, f'Optimizer not provided'

        criterion = criterion_type(**criterion_args)
        optimizer = optimizer_type(model.parameters(), **optimizer_args)

        do_validation = ((validation_freq>0) and (validation_data is not None))
        do_checkpoint = ((checkpoint_freq>0) and (save_path))

        start_time=now()
        if verbose: print('Start Training @ {}'.format(start_time))

        loss_history = []
        training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
        if verbose: 
            print(f'Training samples: [{len(training_data)}]')
            print(f'Training batches: [{len(training_data_loader)}]')

        if validation_data is not None: 
            val_loss_history=[]
            validation_data_loader = DataLoader(validation_data, batch_size=len(validation_data), shuffle=False)
            if verbose: 
                print(f'Validation samples: [{len(validation_data)}]')
                print(f'Validation batches: [{len(validation_data_loader)}]')

        lr_history = []
        lrs = (lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs) if lrs_type is None else lrs_type(optimizer, **lrs_args))
        early_stop=False
        if verbose: print('-------------------------------------------')
        for epoch in range(1, epochs+1):
            if verbose>1: print(f'[+] Epoch {epoch} of {epochs}')
            lr_history.append(lrs.get_last_lr())
            model.train()
            batch_loss=[]
            for i,(X,Y) in enumerate(training_data_loader, 0):
                optimizer.zero_grad()
                P = model(X)
                loss = criterion(P, Y)
                loss.backward()
                optimizer.step()
                loss_value = loss.item()
                batch_loss.append(loss_value)
                if verbose>2: print(f'[++] Batch {i+1}\t Loss: {loss_value}')
            lrs.step()
            mean_batch_loss = np.mean(batch_loss)
            if verbose>1: print(f'(-)\tTraining Loss: {mean_batch_loss}')
            if record_batch_loss:
                loss_history.extend(batch_loss)
            else:
                loss_history.append(mean_batch_loss)
            if early_stop_train is not None: early_stop=early_stop_train(mean_batch_loss, epoch)
                
            if do_checkpoint:
                if (epoch%checkpoint_freq==0):
                    if save_state_only:
                        save_state(save_path, model)
                    else:
                        tt.save(save_path, model)
                    if verbose: print(f'(-)\tCheckpoint created on epoch {epoch}')


            if do_validation:
                if (epoch%validation_freq==0):
                    model.eval()
                    with tt.no_grad():
                        for iv,(Xv,Yv) in enumerate(validation_data_loader, 0):
                            Pv = model(Xv)
                            vloss = criterion(Pv, Yv).item()
                            val_loss_history.append(vloss)
                    if verbose>1: print(f'(-)\tValidation Loss: {vloss}')
                    if early_stop_val is not None: early_stop=early_stop_val(vloss, epoch)

            if early_stop: 
                if verbose: print(f'[~] Early-Stopping on epoch {epoch}')
                break
        # end for epochs...................................................

        if save_path: 
            if save_state_only:
                save_state(save_path, model)
            else:
                tt.save(save_path, model)
            if verbose: print(f'[*] Saved @ {save_path}')
        if verbose: print('-------------------------------------------')
        end_time=now()
        if verbose: print('End Training @ {}, Elapsed Time: [{}]'.format(end_time, end_time-start_time))

        history = {
            'lr':       lr_history, 
            'loss':     loss_history,
            'val_loss': (val_loss_history if do_validation else [])
            }
        if plot:
            plt.figure(figsize=(12,6))
            plt.title('Training Loss')
            plt.plot(history['loss'],color='tab:red', label='train_loss')
            plt.legend()
            plt.show()
            plt.close()
            if validation_data is not None:
                plt.figure(figsize=(12,6))
                plt.title('Validation Loss')
                plt.plot(history['val_loss'],color='tab:orange', label='val_loss')
                plt.legend()
                plt.show()
                plt.close()
            plt.figure(figsize=(12,6))
            plt.title('Learning Rate')
            plt.plot(history['lr'],color='tab:green', label='learning_rate')
            plt.legend()
            plt.show()
            plt.close()
            
        return history


