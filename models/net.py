class Net:
    def __init__(self):
        self.net = None

    def train(self, save=True):
        raise NotImplemented

    def load(self):
        raise NotImplemented

    def prepare(self):
        raise NotImplemented
