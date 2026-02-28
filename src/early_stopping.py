import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', 
                 trace_func=print, save_best_only=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_best_only = save_best_only

    def __call__(self, val_loss, model, optimizer=None, epoch=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer=None, epoch=None):
        '''Sauvegarde le modèle'''
        if self.verbose:
            self.trace_func(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        
        # Sauvegarder plus d'informations
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss
