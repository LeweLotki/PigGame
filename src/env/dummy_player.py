class DummyPlayer:
    """
    Dummy player that rolls the dice once and then passes.
    """
    def __init__(self):
        self.rolled_once = False  # Track whether the dummy has rolled once in this round

    def select_action(self):
        if not self.rolled_once:
            self.rolled_once = True
            return 1  # Roll the dice (action 1)
        else:
            return 0  # Pass (action 0)

    def reset(self):
        """
        Reset the state for the next round.
        """
        self.rolled_once = False

