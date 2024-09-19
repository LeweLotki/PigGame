import torch

class PigGame:
    def __init__(self):
        self.dices = torch.tensor([0, 0], dtype=torch.float32)
        self.current_stack = torch.tensor([0, 0], dtype=torch.float32)
        self.permanent_stack = torch.tensor([0, 0], dtype=torch.float32)
        self.current_player = 0
        self.done = False

    def reset(self):
        self.dices = torch.tensor([0, 0], dtype=torch.float32)
        self.current_stack = torch.tensor([0, 0], dtype=torch.float32)
        self.permanent_stack = torch.tensor([0, 0], dtype=torch.float32)
        self.current_player = 0
        self.done = False
        return self.get_observation()

    def roll_dice(self):
        self.dices = torch.randint(1, 7, (2,), dtype=torch.float32)
    
    def get_observation(self):
        return torch.cat((self.dices, self.current_stack, self.permanent_stack, torch.tensor([self.current_player], dtype=torch.float32)))

    def step(self, action):
        reward = 0 
        if action == 1: 
            self.roll_dice()
            if 6 in self.dices:  
                self.current_stack[self.current_player] = 0
                self.switch_turn()
            else:
                self.current_stack[self.current_player] += torch.sum(self.dices)

            if (self.permanent_stack[self.current_player] - self.permanent_stack[1 - self.current_player] > 0):
                reward = 1
            else:
                reward = -1

        elif action == 0:  
            self.permanent_stack[self.current_player] += self.current_stack[self.current_player]
            reward = 1 if self.current_stack[self.current_player] > 0 else -5
            self.current_stack[self.current_player] = 0
            self.switch_turn()
        
        self.done = self.permanent_stack[self.current_player] >= 100
        return self.get_observation(), reward, self.done

    def switch_turn(self):
        self.current_player = 1 - self.current_player

