# by using pytest, you can not use relative import, that way pytest can not find its parent
# in other words, what you could only use is absolute import, quite anoying
from module.vision import lenet
# import sys
# print(sys.path)

# 原来如此，我们的tests也必须是一个python package!
# 也就是tests/__init__.py这个文件必须存在 这样pytest才能正确的识别我们的test文件的查找路径 也就是sys.path


def test_mylenet():
    lenet.my_lenet()
