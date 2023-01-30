#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/ktorch/utils.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    'QuantiyMonitor', 'Trainer', 
]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
import numpy as np
from torch.utils.data import DataLoader
from .common import save_state #, load_state
import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class QuantiyMonitor:
    """ Monitors a quantity overtime to check if it improves (decreases) after a given patience. 
    Quantity is checked on each call to :func:`~known.ktorch.utils.QuantiyMonitor.check`. 
    The ``__call__`` methods implements the ``check`` method. Can be used to monitor loss for early stopping.
    
    :param name: name of the quantity to be monitored
    :param patience: number of calls before the monitor decides to stop
    :param delta: the amount by which the monitored quantity should decrease to consider an improvement
    """

    def __init__(self, name:str, patience:int, delta:float) -> None:
        r"""
        :param name: name of the quantity to be monitored
        :param patience: number of calls before the monitor decides to stop
        :param delta: the amount by which the monitored quantity should decrease to consider an improvement
        """
        assert(patience>0) # patience should be positive
        assert(delta>0) # delta should be positive
        self.patience, self.delta = patience, delta
        self.name = name
        self.reset()
    
    def reset(self, initial=None):
        r""" Resets the monitor's state and starts at a given `initial` value """
        self.last = (tt.inf if initial is None else initial)
        self.best = self.last
        self.counter = 0
        self.best_epoch = -1

    def __call__(self, current, epoch=-1, verbose=False) -> bool:
        return self.check(current, epoch, verbose)
        
    def check(self, current, epoch=-1, verbose=False) -> bool:
        r""" Calls the monitor to check the current value of monitored quality
        
        :param current: the current value of quantity
        :param epoch:   optional, the current epoch (used only for verbose)
        :param verbose: if `True`, prints monitor status when it changes

        :returns: `True` if the quanity has stopped improving, `False` otherwise.
        """
        self.last = current
        if self.best == tt.inf: 
            self.best=self.last
            self.best_epoch = epoch
            if verbose: print(f'|~|\t{self.name} Set to [{self.best}] on epoch {epoch}')
        else:
            delta = self.best - current # `loss` has decreased if `delta` is positive
            if delta > self.delta:
                # loss decresed more than self.delta
                if verbose: print(f'|~|\t{self.name} Decreased by [{delta}] on epoch {epoch}') # {self.best} --> {current}, 
                self.best = current
                self.best_epoch = epoch
                self.counter = 0
            else:
                # loss didnt decresed more than self.delta
                self.counter += 1
                if self.counter >= self.patience: 
                    if verbose: print(f'|~| Stopping on {self.name} = [{current}] @ epoch {epoch} | best value = [{self.best}] @ epoch {self.best_epoch}')
                    return True # end of patience
        return False

class Trainer:
    r""" Holds a model, compiles it and trains/tests/evaluates it multiple times """
    
    def __init__(self, model, optimizer=None, criterion=None, save_state_only=False) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_state_only=save_state_only
        self.clear_results()

    def compile(self, optimizerF, optimizerA, criterionF, criterionA):
        if optimizerA is None: optimizerA={}
        if criterionA is None: criterionA={}
        self.optimizer = optimizerF(self.model.parameters(), **optimizerA)
        self.criterion = criterionF(**criterionA)

    def on_training_start(self, epochs):
        # can create a lr_scheduler here
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        # can step lr_scheduler, self.early_stop, chekpoint
        pass

    def on_training_end(self, epochs):
        pass

    def on_early_stop(self, epoch): 
        pass

    def fit_epoch(self, data_loader):
        self.model.train()
        batch_loss=[]
        data_iter = iter(data_loader)
        while True:
            try:
                X, Y = next(data_iter)
                self.optimizer.zero_grad()
                P = self.model(X)
                #if P.shape!=Y.shape: print(f'!!!! {P.shape}, {Y.shape}')
                loss = self.criterion(P, Y)
                loss.backward()
                self.optimizer.step()
                loss_value = loss.item()
                batch_loss.append(loss_value)
            except StopIteration: break
        return np.array(batch_loss)

    @tt.no_grad()
    def eval_epoch(self, data_loader):
        self.model.eval()
        batch_loss=[]
        data_iter = iter(data_loader)
        while True:
            try:
                X, Y = next(data_iter)
                P = self.model(X)
                #if P.shape!=Y.shape: print(f'!!!! {P.shape}, {Y.shape}')
                loss_value = self.criterion(P, Y).item()
                batch_loss.append(loss_value)
            except StopIteration: break
        return np.array(batch_loss)

    def clear_results(self):
        self.train_loss_history=[]
        self.train_mean_loss_history=[]
        self.val_loss_history=[]
        self.val_mean_loss_history=[]
        self.val_loss_epochs=[]

    def save_results(self, path:str, mean_only=False):
        if not path.lower().endswith('.npz'): path = f'{path}.npz'
        if mean_only:
            np.savez(path, 
                    train_mean_loss_history=np.array(self.train_mean_loss_history),
                    val_mean_loss_history=np.array(self.val_mean_loss_history),
                    val_loss_epochs=np.array(self.val_loss_epochs),
            )
        else:
            np.savez(path, 
                    train_loss_history=np.array(self.train_loss_history),
                    train_mean_loss_history=np.array(self.train_mean_loss_history),
                    val_loss_history=np.array(self.val_loss_history),
                    val_mean_loss_history=np.array(self.val_mean_loss_history),
                    val_loss_epochs=np.array(self.val_loss_epochs),
            )

    def train(self,
            training_data, validation_data,
            epochs,
            batch_size,
            shuffle,
            validation_freq,
            save_path,
            clear_results=True,
            verbose=0
            ):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # assertions
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        assert self.model is not None, f'Model not available'
        assert self.criterion is not None, f'Criterion not available'
        assert self.optimizer is not None, f'Optimizer not available'
        assert training_data is not None, f'Training data not provided'
        assert epochs>0, f'Epochs should be at least 1'
        assert batch_size>0, f'Batch Size should be at least 1'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # additional checks and flags
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        do_validation = ((validation_freq>0) and (validation_data is not None))
        if validation_data is not None: 
            if validation_freq<=0: 
                print(f'[!] Validation data is provided but frequency is not set, Validation will not be performed')
                validation_freq=1
        else:
            if validation_freq>0: 
                print(f'[!] Validation frequency is set but data is not provided, Validation will not be performed')
                validation_freq=1

        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Data
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
        if verbose: print(f'Training on [{len(training_data)}] samples in [{len(training_data_loader)}] batches')

        if do_validation: 
            validation_data_loader = DataLoader(validation_data, batch_size=len(validation_data), shuffle=False)
            if verbose: print(f'Validating on [{len(validation_data)}] samples in [{len(validation_data_loader)}] batches')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Temporary Variables
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        self.early_stop = False
        if clear_results: self.clear_results()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Training Loop
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        start_time=now()
        if verbose: 
            print('Start Training @ {}'.format(start_time))
            print('-------------------------------------------')
        self.on_training_start(epochs)
        for epoch in range(1, epochs+1):
            if verbose>1: print(f'[+] Epoch {epoch} of {epochs}')
            self.on_epoch_start(epoch)
            train_loss = self.fit_epoch(training_data_loader)
            mean_train_loss = np.mean(train_loss)
            self.train_loss_history.append(train_loss)
            self.train_mean_loss_history.append(mean_train_loss)
            if verbose>1: print(f'(-)\tTraining Loss: {mean_train_loss}')
            

            if do_validation and (epoch%validation_freq==0):
                #self.on_val_begin(epoch)
                val_loss = self.eval_epoch(validation_data_loader) 
                mean_val_loss = np.mean(val_loss)
                self.val_loss_history.append(val_loss)
                self.val_mean_loss_history.append(mean_val_loss)
                self.val_loss_epochs.append(epoch-1)
                if verbose>1: print(f'(-)\tValidation Loss: {mean_val_loss}')
                #self.on_val_end(epoch)

            self.on_epoch_end(epoch)
            if self.early_stop: 
                if verbose: print(f'[~] Early-Stopping on epoch {epoch}')
                self.on_early_stop(epoch)
                break
        # end for epochs...................................................
        self.on_training_end(epochs)
        if save_path: 
            if self.save_state_only:
                save_state(self.model, save_path)
                if verbose: print(f'[*] Saved State @ {save_path}')
            else:
                tt.save(self.model, save_path)
                if verbose: print(f'[*] Saved Model @ {save_path}')

        if verbose: print('-------------------------------------------')
        end_time=now()
        if verbose:
            print(f'Final Training Loss: [{self.train_mean_loss_history[-1]}]')
            if do_validation: print(f'Final Validation Loss: [{self.val_mean_loss_history[-1]}]') 
            print('End Training @ {}, Elapsed Time: [{}]'.format(end_time, end_time-start_time))
        return

    def evaluate(self, testing_data, batch_size=None):
        if batch_size is None: batch_size=len(testing_data)
        testing_data_loader=DataLoader(testing_data, batch_size=batch_size, shuffle=False)
        print(f'Testing on [{len(testing_data)}] samples in [{len(testing_data_loader)}] batches')
        test_loss = self.eval_epoch(testing_data_loader)
        mean_test_loss = np.mean(test_loss)
        print(f'Testing Loss: {mean_test_loss}') 
        return mean_test_loss, test_loss

    def plot_results(self, title='Loss', color='black', figsize=(12,6), ylim=None, val_marker=None):
        fig = plt.figure(figsize=figsize)
        if ylim is not None: 
            if hasattr(ylim, '__len__'): # is a tuple
                plt.ylim(ylim)
            else:
                plt.ylim(0, abs(ylim))

        plt.title(title)
        plt.plot(self.train_mean_loss_history, color=color, label='train_loss')
        if self.val_mean_loss_history: 
            plt.plot(self.val_loss_epochs, self.val_mean_loss_history, color=color,  label='val_loss', linestyle='dotted')
            if val_marker is not None: plt.scatter(self.val_loss_epochs, self.val_mean_loss_history, color=color,  marker=val_marker)
        plt.legend()
        plt.show()
        plt.close()
        return fig

    @staticmethod
    def plot_load_results(path, title='Loss', color='black', figsize=(12,6), ylim=None, val_marker=None):
        fig = plt.figure(figsize=figsize)
        if ylim is not None: 
            if hasattr(ylim, '__len__'): # is a tuple
                plt.ylim(ylim)
            else:
                plt.ylim(0, abs(ylim))

        if not path.lower().endswith('.npz'): path = f'{path}.npz'
        res = np.load(path) 
        #train_loss_history=res['train_loss_history']
        train_mean_loss_history=res['train_mean_loss_history']
        #val_loss_history=res['val_loss_history']
        val_mean_loss_history=res['val_mean_loss_history']
        val_loss_epochs=res['val_loss_epochs']
        res.close()

        plt.title(title)
        plt.plot(train_mean_loss_history, color=color, label='train_loss')
        if len(val_mean_loss_history)>0: 
            plt.plot(val_loss_epochs, val_mean_loss_history, color=color,  label='val_loss', linestyle='dotted')
            if val_marker is not None: plt.scatter(val_loss_epochs, val_mean_loss_history, color=color,  marker=val_marker)
        plt.legend()
        plt.show()
        plt.close()
        return fig

# @staticmethod
# def train(model, training_data=None, validation_data=None, testing_data=None,
#             epochs=0, batch_size=0, shuffle=None, validation_freq=0, 
#             criterion_type=None, criterion_args={}, optimizer_type=None, optimizer_args={}, lrs_type=None, lrs_args={},
#             record_batch_loss=False, early_stop_train=None, early_stop_val=None, checkpoint_freq=0, save_path=None,
#             save_state_only=False, verbose=0, plot=0, loss_plot_start=0):
#     assert model is not None, f'Model not provided'
#     assert criterion_type is not None, f'Criterion not provided'
#     assert optimizer_type is not None, f'Optimizer not provided'
#     criterion=criterion_type(**criterion_args)
#     optimizer=optimizer_type(model.parameters(), **optimizer_args)
#     lrs=(lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs) \
#         if lrs_type is None else lrs_type(optimizer, **lrs_args))
#     return __class__.train_(model, training_data, validation_data, testing_data,
#             epochs, batch_size, shuffle, validation_freq, 
#             criterion, optimizer, lrs, record_batch_loss, 
#             early_stop_train, early_stop_val, checkpoint_freq, save_path,
#             save_state_only, verbose, plot, loss_plot_start)

# @staticmethod
# def train_(model, training_data=None, validation_data=None, testing_data=None,
#             epochs=0, batch_size=0, shuffle=None, validation_freq=0, 
#             criterion=None, optimizer=None, lrs=None,
#             record_batch_loss=False, early_stop_train=None, early_stop_val=None, checkpoint_freq=0, save_path=None,
#             save_state_only=False, verbose=0, plot=0, loss_plot_start=0):
#     assert model is not None, f'Model not provided'
#     assert criterion is not None, f'Loss Criterion not provided'
#     assert optimizer is not None, f'Optimizer not provided'

#     assert epochs>0, f'Epochs should be at least 1'
#     assert batch_size>0, f'Batch Size should be at least 1'
#     assert training_data is not None, f'Training data not provided'
    
#     do_validation = ((validation_freq>0) and (validation_data is not None))
#     if validation_data is not None: 
#         if validation_freq<=0: 
#             print(f'[!] Validation data is provided but frequency is not set, Validation will not be performed')
#             validation_freq=1
#     else:
#         if validation_freq>0: 
#             print(f'[!] Validation frequency is set but data is not provided, Validation will not be performed')
#             validation_freq=1

#     do_checkpoint = ((checkpoint_freq>0) and (save_path))

#     start_time=now()
#     if verbose: print('Start Training @ {}'.format(start_time))

#     loss_history = []
#     training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
#     if verbose: 
#         print(f'Training samples: [{len(training_data)}]')
#         print(f'Training batches: [{len(training_data_loader)}]')

#     if validation_data is not None: 
#         val_loss_history=[]
#         validation_data_loader = DataLoader(validation_data, batch_size=len(validation_data), shuffle=False)
#         if verbose: 
#             print(f'Validation samples: [{len(validation_data)}]')
#             print(f'Validation batches: [{len(validation_data_loader)}]')

#     lr_history = []
#     lrs = (lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs) if lrs is None else lrs)
#     early_stop=False
#     if verbose: print('-------------------------------------------')
#     for epoch in range(1, epochs+1):
#         if verbose>1: print(f'[+] Epoch {epoch} of {epochs}')
#         lr_history.append(lrs.get_last_lr())
#         model.train()
#         batch_loss=[]
#         for i,(X,Y) in enumerate(training_data_loader, 0):
#             optimizer.zero_grad()
#             P = model(X)
#             loss = criterion(P, Y)
#             loss.backward()
#             optimizer.step()
#             loss_value = loss.item()
#             batch_loss.append(loss_value)
#             if verbose>2: print(f'[++] Batch {i+1}\t Loss: {loss_value}')
#         lrs.step()
#         mean_batch_loss = np.mean(batch_loss)
#         if verbose>1: print(f'(-)\tTraining Loss: {mean_batch_loss}')
#         if record_batch_loss:
#             loss_history.extend(batch_loss)
#         else:
#             loss_history.append(mean_batch_loss)
#         if early_stop_train is not None: early_stop=early_stop_train(mean_batch_loss, epoch, verbose>1)
            
#         if do_checkpoint:
#             if (epoch%checkpoint_freq==0):
#                 if save_state_only:
#                     save_state( model, save_path)
#                 else:
#                     tt.save(model, save_path )
#                 if verbose>1: print(f'(-)\tCheckpoint created on epoch {epoch}')


#         if do_validation:
#             if (epoch%validation_freq==0):
#                 model.eval()
#                 with tt.no_grad():
#                     for iv,(Xv,Yv) in enumerate(validation_data_loader, 0):
#                         Pv = model(Xv)
#                         vloss = criterion(Pv, Yv).item()
#                         val_loss_history.append(vloss)
#                 if verbose>1: print(f'(-)\tValidation Loss: {vloss}')
#                 if early_stop_val is not None: early_stop=early_stop_val(vloss, epoch, verbose>1)

#         if early_stop: 
#             if verbose: print(f'[~] Early-Stopping on epoch {epoch}')
#             break
#     # end for epochs...................................................

#     if save_path: 
#         if save_state_only:
#             save_state( model, save_path)
#         else:
#             tt.save(model, save_path)
#         if verbose: print(f'[*] Saved @ {save_path}')
#     if verbose: print('-------------------------------------------')
#     end_time=now()
#     if verbose:
#         print(f'Final Training Loss: [{loss_history[-1]}]')
#         if do_validation: print(f'Final Validation Loss: [{val_loss_history[-1]}]') 
#         print('End Training @ {}, Elapsed Time: [{}]'.format(end_time, end_time-start_time))

#     if (testing_data is not None):
#         testing_data_loader = DataLoader(testing_data, batch_size=len(testing_data), shuffle=False)
#         model.eval()
#         with tt.no_grad():
#             for iv,(Xv,Yv) in enumerate(testing_data_loader, 0):
#                 Pv = model(Xv)
#                 tloss = criterion(Pv, Yv).item()
#         if verbose: 
#             print(f'Testing samples: [{len(testing_data)}]')
#             print(f'Testing batches: [{len(testing_data_loader)}]')
#             print(f'Testing Loss: [{tloss}]') 
#     else:
#         tloss=None

#     history = {
#         'lr':       lr_history, 
#         'loss':     loss_history,
#         'val_loss': (val_loss_history if do_validation else [None]),
#         'test_loss': tloss,
#         }
#     if plot:
#         plt.figure(figsize=(12,6))
#         plt.title('Training Loss')
#         plt.plot(history['loss'][loss_plot_start:],color='tab:red', label='train_loss')
#         plt.legend()
#         plt.show()
#         plt.close()
#         if do_validation:
#             plt.figure(figsize=(12,6))
#             plt.title('Validation Loss')
#             plt.plot(history['val_loss'],color='tab:orange', label='val_loss')
#             plt.legend()
#             plt.show()
#             plt.close()
#         plt.figure(figsize=(12,6))
#         plt.title('Learning Rate')
#         plt.plot(history['lr'],color='tab:green', label='learning_rate')
#         plt.legend()
#         plt.show()
#         plt.close()
        
#     return history



# """
# m = nn.LogSoftmax(dim=-1)
# loss = nn.NLLLoss()
# # input is of size N x C = 3 x 5
# input = tt.zeros(5, 3, requires_grad=False)
# ins = m(input)
# print(input, '\n', ins, ins.shape)
# # each element in target has to have 0 <= value < C
# target = tt.tensor([1, 0, 2, 0, 1])
# print('target', target.shape, target)
# output = loss(ins, target)
# print(output)
# """