# 2023/9/9
# zhangzhong

# 3.2.4 Training
# TODO: add gpu support for Trainer

# 原来如此 epoch epochs batch batches


from mytorch import config
from tqdm import tqdm
# import torch.utils.tensorboard as tb
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from torch import device, nn, Tensor
import os.path


# TODO: 我有一个想法，对于我们训练过程中的每一个中间过程都应该保存下来
# 而且应该保存对应的训练信息，我们后期挑选模型应该用的到 而不是重新训练


class Trainer:
    """The base class for training models with data."""

    def __init__(self, *, model, loss_fn, optimizer, num_epochs, train_dataloader, val_dataloader, scheduler=None, num_gpus=0, gradient_clip_val=0, is_test: bool = True, device: device = torch.device('cpu')):
        assert num_gpus == 0, 'No GPU support yet'
        # 用这个函数 类型检查系统会complain 所以还是不要用了
        # 你也不知道你引进来了什么
        # self.make_parameters_be_attributes()
        self.num_epochs = num_epochs
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.num_gpus = num_gpus
        self.gradient_clip_val = gradient_clip_val
        self.writer = SummaryWriter()
        self.model_file_prefix = 'snapshots'
        self.is_test = is_test
        self.max_batch: int = 10

        # if torch.cuda.is_available():
        #     self.device = torch.device(config.conf['device'])
        # else:
        #     self.device = torch.device('cpu')
        self.device = device

        # 我的理解是 trainer也不能够使用cuda
        # 我们应该遵循一条原则，就是默认的情况下，大家都在cpu上
        # 但是我们可以显示的指定设备，但是这些显示的指定的代码应该只有一处！

    # def prepare_data(self, data):
    #     self.train_dataloader = data.train_dataloader()
    #     self.val_dataloader = data.val_dataloader()
    #     # 这个是说 整个数据集 被分成了多少个batch
    #     self.num_train_batches = len(self.train_dataloader)
    #     self.num_val_batches = (len(self.val_dataloader)
    #                             if self.val_dataloader is not None else 0)

    # def prepare_model(self, model):
    #     # 你这个给整的双向依赖呀
    #     # model.trainer = self
    #     # board我应该会用tensorboard
    #     # model.board.xlim = [0, self.max_epochs]
    #     self.model = model

    # def fit(self, model, data):
    #     self.prepare_data(data)
    #     self.prepare_model(model)
    #     self.optim = model.configure_optimizers()
    #     self.epoch = 0
    #     self.train_batch_idx = 0
    #     self.val_batch_idx = 0
    #     for self.epoch in range(self.max_epochs):
    #         self.fit_epoch()

    def cross_entropy_accuracy(self, logits: Tensor, labels: Tensor) -> tuple[int, int]:
        '''
        logits: shape = (batch_size, num_labels)
        labels: shape = (batch_size,)

        step 1. logits: (batch_size, num_labels) -> (batch_size,)
            也就是没行选出一个最大的值的下标 predict_labels = torch.argmax(logits, dim=1)

        step 2: 然后和labels进行比较 predict_labels == labels
        为了最后计算的准确率，我们需要返回判断准确的样本的个数 最后由外部累计所有的batch的和，然后再计算准确率
        '''
        batch_size, num_labels = logits.shape
        predict_labels = torch.argmax(logits, dim=1)
        # torch.sum(): Returns the sum of all elements in the input tensor.
        return int((predict_labels == labels).int().sum()), batch_size

    def save_model(self, filename: str):
        os.makedirs(self.model_file_prefix, exist_ok=True)
        filename = os.path.join(self.model_file_prefix, filename)
        torch.save(self.model, filename)

    def load_model(self, filename: str):
        filename = os.path.join(self.model_file_prefix, filename)
        self.model = torch.load(filename)

    # def fit_epoch():
    #     raise NotImplementedError
    # TODO: 我们希望training可以自动保存
    def train(self, tag, calculate_accuracy: bool = False):

        # send model parameters to device
        # TODO: 不太确定如果我的参数不使用 torch.Parameter来定义的话 能不能正确的被to(device)呢??
        # 目前看来是不能的，还是得用torch.Parameter来定义
        self.model.to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()
            # TODO: computing the loss for every minibatch on the GPU and reporting it back to the usr on the command line
            # or logging it in a NumPy array will trigger a global interpreter lock which stalls all GPUs
            # so we need to compute and store the loss on GPU!
            train_losses = torch.tensor(0.0, device=self.device)
            # dataloader的每次遍历都应该进行一次shuffle
            # for X, y in tqdm(self.train_dataloader):
            current_batch = 0
            for batch in tqdm(self.train_dataloader):
                # if we in test mode, we only do several batch
                if self.is_test:
                    current_batch += 1
                    if current_batch > self.max_batch:
                        break

                X = list(batch[:-1])
                y = batch[-1]
                # X = X.to(self.device)
                for i, _ in enumerate(X):
                    X[i] = X[i].to(self.device)
                y = y.to(self.device)
                y_hat = self.model(*X)
                # backward propagation is kicked off when we call .backward()
                # on the loss tensor.
                # Autograd then calculates and stores the gradients
                # for each model parameter in the parameter's .grad attribute.
                loss = self.loss_fn(y_hat, y)
                train_losses += loss
                loss.backward()

                # TODO:DONE
                # 在这里我们才能够给到optimizer我们的参数
                # self.optimizer.set_parameters(self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad()

            # after every epoch, update the learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # 每个epoch结束之后进行一次validation
            self.model.eval()
            val_losses = torch.tensor(0.0, device=self.device)
            accuracyes = 0
            num_examples = 0
            current_batch = 0
            for batch in tqdm(self.val_dataloader):
                # if we in test mode, we only do several batch
                if self.is_test:
                    current_batch += 1
                    if current_batch > self.max_batch:
                        break

                X = list(batch[:-1])
                y = batch[-1]
                # X = X.to(self.device)
                for i, _ in enumerate(X):
                    X[i] = X[i].to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    y_hat = self.model(*X)
                    loss = self.loss_fn(y_hat, y)
                    val_losses += loss

                    if calculate_accuracy:
                        accuracy, num_batch = self.cross_entropy_accuracy(
                            y_hat, y)
                        accuracyes += accuracy
                        num_examples += num_batch

            tag_scalar_dict = {'train': float(train_losses) / len(self.train_dataloader),
                               'val': val_losses / len(self.val_dataloader)}
            if calculate_accuracy:
                tag_scalar_dict.update(
                    {'accuracy': float(accuracyes) / float(num_examples)})
            self.writer.add_scalars(tag, tag_scalar_dict, epoch)

            # 不行 必须每个epoch结束都保存一下我的模型 我才安心
            self.save_model(tag)

        # automatically save trained model
        # self.save_model(tag)
