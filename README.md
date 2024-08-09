# Deep Learning

My Deep Learning Codes.

## Structure

1. basic deep learning library, mydl
2. every single other networks, use a separate folder, and group by them by categories, such as CNN, GPT, Diffusion, etc
3. how would that be? I don't want to use PYTHONPATH, so how to structure my code?
4. dl/mytorch, dl/modules/CNN, LeNet, Transformers, ..., whereis my tests? it should be close to my code
5.

因为python的路径寻找规则，当我们使用vscode直接运行当前文件的时候，python解释器会从python脚本的路径开始查找其他的模块，这当然找不到，所以我们的解决方案就是使用pytest来运行所有的程序，这样的话所有的程序都是写成单元测试的样子，这种方式非常有效！

## TODO

- [ ] 还差 scratch/quick_test.ipynb
- [x] make sure all the tests are passes when we push our code to the github. but how? github action?
- [x] 等cnn也弄完之后 我们需要进行下一步的重构，这就包括更加合理的架构 更加合理的命名 更加清晰的注释等
- [x] 单元测试花费太多时间，削减数据集的数量，进一步减少测试时间 将测试时间控制在数秒以内
- [x] d2l书上的layer_summary这个函数挺好的，看一下怎么实现的
- [x] trainer需要重构 我们需要考虑test_dataloader, 还有更合理的模型存档方式，比如loss变小，正确率提高在存档 而不是每次都存档
- [x] 修改模型保存时json文件的loss计算方式，应该保存平均的而不是全部的和，accuracy换成error rate 这样所有的曲线就都是下降的了
- [x] 实现TOP-1 TOP-5 error rate on TrainerV2
- [x] 实现WarmUpCosine Learning Scheduler
- [x] 神经网络的每一层都要有名字，但是有一些网络的层我们是用sequential或者其他自己实现的module实现的，这些东西能有自己的名字吗？
- [x] 把MaskedLanguageModel实现一个吧，是现成两个是可以开关的方式，如果还是不行的，就调试一下书上的代码吧，看看到底是哪里出了问题
- [x] 额外实现一个BERTTrainer 主要是accuracy等等的计算和其他的模型太不一样了, 然后利用pytorch的DistributedDataParallel来实现并行训练, 不对 不应该重写，应该仔细思考不同的模型之间训练时的共同点和差异，比如这次是accuracy_batch和summary_write不同，那么我们应该提供一种方法 让不同的模型提供自己的方法 而其他的部分仍然复用 这才是最合理的方式
- [x] 重构WikiTextDataset，修复不同的数据集使用不同的字典的bug，优化流程
- [x] 增加训练集的accuracy的功能 更加全面的展示数据 更有助于分析模型和实现
- [] test模型和train模型要分开，要让测试总是可以很快的很简单的运行，而不是每次都要调整参数

## DataMining Ideas
1. 根据AlexNet和ResNet在做一个数据集的分类，数据处理，实现，调参都可以参考他们
2. 然后根据AlexNet的思想做一个以图搜图的分类，基本的实现思路是：用预训练好的模型做特征提取，然后把所有测试集的图片都放在经过预训练模型输出的向量全部保存起来，然后系统读取一张图片，经过特征提取之后，遍历所有的测试集的图片，然后输出top-5向量距离最近的图片，当然这个向量距离可以有多种计算方式，比如1范数，2范数等等。
3. 还有一个idea是如果給我们训练好的模型一个不在测试类别里面的图片，模型能否正确的识别出来呢？这又如何体现在向量距离上呢？也就是我们的以图搜图的应用能否识别不存在的类别并输出“不认识这个图片”，而不是硬给出一些图片。
4. 模型的选择：ResNet18, ResNet34, ResNet50, 我们可以根据论文上的结构重新实现一下，毕竟和李沐书上的不太一样。
5. 可以像AlexNet那样输出第一层的kernel，因为他们都是人眼可以看懂的东西，看看他们到底长什么样子

## 我们不禁要问一个问题，在什么地方应用type hint最为合适？

1. 显而易见的是函数题的签名应该做type hint
2. 函数题内部需要做吗？比如我生命一个新的变量，调用函数返回一个变量？个人认为在可以轻松的由类型系统作出推断的场景，都不应该再使用type
   hint，比如调用函数的返回值，调用构造函数构造对象，而显示的声明接下来要使用的变量，且无法做初始化的时候，就应该做type hint

## TIPS

1. put all the datasets under the data/ folder
2. 分层训练，这样就可以训练大模型了
3. 为了在jupyter
   notebook里面使用我们编写的package，需要在根目录下提供一个.env文件并在内部指定PYTHONPATH=your-root-dir [https://github.com/microsoft/vscode-jupyter/issues/9436]
4. jupyter notebook对git非常不友好 垃圾格式 不用了 不要再用了 球球了 全是错误 不会自动加载新代码
5. start tensorboard: `conda deactivate` back to base env; then `tmux`; then `conda activate ml`;
   then `tensorboard --logdir=runs --bind_all`; then `ctrl+b d` to detach the tmux session
6. 模型中的所有层都要有一个名字，这样方便对模型进行分析
7. 直接连接得到的向量不适合作为输出层的输入，最好在接一个隐藏层，多做一层的映射

## packages

1. pip install torch torchvision torchaudio
2. pip install tensorboard numpy matplotlib tqdm pytest torchsummary ipykernel tsnecuda SciPy
3. TODO: 安装tsnecuda还挺麻烦的 需要装一些别的库 https://github.com/CannyLab/tsne-cuda
