import numpy as np

class PigGame:
    def __init__(self):
        """
        Initialize the PigGame environment.
        - dices: [dice1, dice2] storing the values of the rolled dice.
        - current_stack: Current round points for both players, initialized to zero.
        - permanent_stack: Total points for both players that cannot be lost.
        - current_player: 0 for Player 1, 1 for Player 2.
        - done: Flag indicating if the game has ended.
        """
        self.dices = np.array([0, 0])
        self.current_stack = np.array([0, 0])
        self.permanent_stack = np.array([0, 0])
        self.current_player = 0
        self.done = False

    def reset(self):
        """
        Reset the environment to start a new game.
        """
        self.dices = np.array([0, 0])
        self.current_stack = np.array([0, 0])
        self.permanent_stack = np.array([0, 0])
        self.current_player = 0
        self.done = False
        return self.get_observation()

    def roll_dice(self):
        """
        Simulate rolling two dice by assigning a new NumPy array
        with random integers between 1 and 6.
        """
        self.dices = np.random.randint(1, 7, size=2)
    
    def get_observation(self):
        """
        Return the current observation vector, consisting of:
        - Dice values (self.dices)
        - Current stack for both players (self.current_stack)
        - Permanent stack for both players (self.permanent_stack)
        - Current player's turn (self.current_player)
        """
        return np.concatenate((self.dices, self.current_stack, self.permanent_stack, [self.current_player]))

    def step(self, action):
        """
        Execute one step in the environment based on the action:
        - Action 0: Pass
        - Action 1: Roll the dice
        
        Returns:
        - observation: New state of the game as an observation vector.
        - reward: Scalar reward for the action.
        - done: Whether the episode (game) has ended.
        """
        if action == 1:  # Roll the dice
            self.roll_dice()
            if 6 in self.dices:  # If either dice is a 6, reset current stack
                self.current_stack[self.current_player] = 0
                reward = -50  # Penalty for rolling a 6
                self.switch_turn()
            else:
                self.current_stack[self.current_player] += np.sum(self.dices)
                reward = np.sum(self.dices) 
        elif action == 0:  # Pass
            self.permanent_stack[self.current_player] += self.current_stack[self.current_player]
            reward = -100 if self.current_stack[self.current_player] == 0 else self.permanent_stack[self.current_player] 
            self.current_stack[self.current_player] = 0
            self.switch_turn()
        
        # Check if the current player has won
        self.done = self.permanent_stack[self.current_player] >= 100
        return self.get_observation(), reward, self.done

    def switch_turn(self):
        """
        Switch the current player turn.
        """
        self.current_player = 1 - self.current_player

