# Deep Learning

My Deep Learning Codes.

## Structure

1. basic deep learning library, mydl
2. every single other networks, use a separate folder, and group by them by categories, such as CNN, GPT, Diffusion, etc
3. how would that be? I don't want to use PYTHONPATH, so how to structure my code?
4. dl/mytorch, ßdl/modules/CNN, LeNet, Transformers, ..., whereis my tests? it should be close to my code
5. 因为python的路径寻找规则，当我们使用vscode直接运行当前文件的时候，python解释器会从python脚本的路径开始查找其他的模块，这当然找不到，所以我们的解决方案就是使用pytest来运行所有的程序，这样的话所有的程序都是写成单元测试的样子，这种方式非常有效！

## TODO:
1. 还差data data_module module 以及对应的测试文件
