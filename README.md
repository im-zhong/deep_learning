# Deep Learning

My Deep Learning Codes.

## Structure

1. basic deep learning library, mydl
2. every single other networks, use a separate folder, and group by them by categories, such as CNN, GPT, Diffusion, etc
3. how would that be? I don't want to use PYTHONPATH, so how to structure my code?
4. dl/mytorch, dl/modules/CNN, LeNet, Transformers, ..., whereis my tests? it should be close to my code
5. 因为python的路径寻找规则，当我们使用vscode直接运行当前文件的时候，python解释器会从python脚本的路径开始查找其他的模块，这当然找不到，所以我们的解决方案就是使用pytest来运行所有的程序，这样的话所有的程序都是写成单元测试的样子，这种方式非常有效！

## TODO

- [x] 还差 cnn 和一些ipynb
- [x] make sure all the tests are passes when we push our code to the github. but how? github action?
- [ ] 等cnn也弄完之后 我们需要进行下一步的重构，这就包括更加合理的架构 更加合理的命名 更加清晰的注释等
- [x] 单元测试花费太多时间，削减数据集的数量，进一步减少测试时间 将测试时间控制在数秒以内
- [x] d2l书上的layer_summary这个函数挺好的，看一下怎么实现的
- [ ] trainer需要重构 我们需要考虑test_dataloader, 还有更合理的模型存档方式，比如loss变小，正确率提高在存档 而不是每次都存档

## 我们不禁要问一个问题，在什么地方应用type hint最为合适？

1. 显而易见的是函数题的签名应该做type hint
2. 函数题内部需要做吗？比如我生命一个新的变量，调用函数返回一个变量？个人认为在可以轻松的由类型系统作出推断的场景，都不应该再使用type hint，比如调用函数的返回值，调用构造函数构造对象，而显示的声明接下来要使用的变量，且无法做初始化的时候，就应该做type hint

## TIPS

1. put all the datasets under the data/ folder
2. 分层训练，这样就可以训练大模型了
3. 为了在jupyter notebook里面使用我们编写的package，需要在根目录下提供一个.env文件并在内部指定PYTHONPATH=your-root-dir [https://github.com/microsoft/vscode-jupyter/issues/9436]
4. jupyter notebook对git非常不友好 垃圾格式 不用了 不要再用了 球球了 全是错误 不会自动加载新代码
5. start tensorboard: `conda deactivate` back to base env; then `tmux`; then `conda activate ml`; then `tensorboard --logdir=runs --bind_all`; then `ctrl+b d` to detach the tmux session

## packages

1. pytorch
2. pip install tensorboard numpy matplotlib tqdm pytest torchinfo
