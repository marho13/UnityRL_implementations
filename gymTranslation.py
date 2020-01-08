import numpy as np
class checker:
    def __init__(self, method):
        self.method = method

    def translationNeeds(self):
        if self.method == "DQN":
            actionDict = self.actionRecreator()
            actionLen = len(actionDict)
            return actionDict, actionLen

        elif self.method == "DDPG":
            obsSize = [96, 96, 3]
            return obsSize

        else:
            pass

    def actionRecreator(self): #fix the number order
        dicty = {0: np.array([-1.0, 1.0, 0]), 1: np.array([-1.0, 0.5, 0]), 2: np.array([-0.75, 1.0, 0]),
                 3: np.array([-0.5, 1.0, 0]), 4: np.array([-0.25, 1.0, 0]), 5: np.array([0.0, 1.0, 0]),
                 6: np.array([0.0, 0.5, 0]), 7: np.array([0.25, 1.0, 0]), 8: np.array([0.5, 1.0, 0]),
                 9: np.array([0.75, 1.0, 0]), 10: np.array([1.0, 1.0, 0]),
                 11: np.array([1.0, 0.5, 0]), 12: np.array([0.0, 0.0, 1.0])}
        return dicty