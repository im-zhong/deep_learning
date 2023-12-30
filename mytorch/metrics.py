# 2023/12/30
# zhangzhong

class Metrics:
    def __init__(self):
        pass

    def __call__(self, y_hat, y):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def to_dict(self):
        return self.__dict__

    def from_dict(self, d):
        self.__dict__.update(d)


class Evaluator:
    def clear(self):
        raise NotImplementedError

    def __call__(self, y_hat, y):
        raise NotImplementedError

    def summary(self) -> Metrics:
        raise NotImplementedError

    def eval_batch(self, y_hat, y):
        raise NotImplementedError
