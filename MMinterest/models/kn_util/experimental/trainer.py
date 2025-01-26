import weakref
class HookBase:
    def __init__(self) -> None:
        pass
    
    def 

class TrainerBase:
    def __init__(self) -> None:
        pass

    def register_hook(self, hook):
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)
    
    def run_step(self):
        raise NotImplementedError
    
    def train(self):
        iter = self.iter
        
    
    
    def state_dict(self):
        """Return a dict containing the state of the trainer."""
        ret = dict(iteration=self.iter)
        # load state_dict from hooks
        for h in self._hooks:
            hs = h.state_dict()
            if hs:
                hook_name = type(h).__qualname__
                if hook_name in ret:
                    continue
                ret[hook_name] = hs
        return ret


class Trainer:
    def __init__(self, model, optimizer, dataloader, lr_scheduler=None, grad_scalar=None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.lr_scheduler = lr_scheduler
        self.grad_scalar = grad_scalar
    
    def run_step(self):
        pass
    
    def state_dict(self):
        model.state_dict