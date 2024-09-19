class DummyPlayer:
    def __init__(self):
        self.rolled_once = False  
    
    def select_action(self):
        if not self.rolled_once:
            self.rolled_once = True
            return 1 
        else:
            return 0

    def reset(self):
        self.rolled_once = False

