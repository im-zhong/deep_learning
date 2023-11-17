# 2023/9/10
# zhangzhong

from mytorch import utils


class SimpleClass(utils.ParametersToAttributes):
    def __init__(self, a, b, c):
        super().__init__()
        self.make_parameters_be_attributes()


def test_ParametersToAttributes():
    s = SimpleClass(1, 'hello', 0.2)
    assert hasattr(s, 'a')
    assert hasattr(s, 'b')
    assert hasattr(s, 'c')

    # assert s.a == 1
    # assert s.b == 'hello'
    # assert s.c == 0.2
    assert getattr(s, 'a') == 1
    assert getattr(s, 'b') == 'hello'
    assert getattr(s, 'c') == 0.2
