class Agent:
    def reset(self):
        raise NotImplementedError
    
    def get_action(self, state, step):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    
    def save(self, save_dir, epoch):
        raise NotImplementedError
    
    def load(self, load_dir, epoch):
        raise NotImplementedError
    