# 2023/9/9
# zhangzhong

# 3.2.4 Training
# TODO: add gpu support for Trainer

# 原来如此 epoch epochs batch batches


from dataclasses import dataclass
from unittest import result
from sympy import S
from tqdm import tqdm
# import torch.utils.tensorboard as tb
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from torch import device, nn, Tensor
import os.path
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import Module
from torch.utils.data import DataLoader
import json
import torchinfo
from mytorch import utils
from module.nlp.bert import BERTOutput
from typing import Any
from mytorch.data.wikitext2v2 import WikiText2Label


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


@dataclass
class Result:
    epoch: int = 0
    train_loss: float = 0
    val_loss: float = 0
    val_accuracy: float = 0
    test_loss: float = 0
    test_accuracy: float = 0
    val_top1_error_rate: float = 0
    val_top5_error_rate: float = 0
    test_top1_error_rate: float = 0
    test_top5_error_rate: float = 0

    def __lt__(self, other: "Result") -> bool:
        return self.val_accuracy < other.val_accuracy and self.test_accuracy < other.test_accuracy

    def __gt__(self, other: "Result") -> bool:
        # 我们是不应该通过test_accuracy来保存的 因为我们根本就不知道
        # 还有就是我们需要固定随机数种子 不然每次重新训练都会导致val_accuracy暴涨 这根本没有任何意义
        # return self.val_accuracy > other.val_accuracy or self.test_accuracy > other.test_accuracy
        return self.val_accuracy > other.val_accuracy

    def to_dict(self) -> dict:
        return self.__dict__

    @staticmethod
    def from_dict(d: dict) -> "Result":
        return Result(**d)


class TrainerV2:
    '''
        only for pytorch's model, only for cross entropy
    '''

    def __init__(self, *,
                 model: Module,
                 loss_fn: Module,
                 optimizer: Optimizer,
                 num_epochs: int,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 scheduler: LRScheduler | None = None,
                 device: device = torch.device('cpu'),
                 prefix: str = 'snapshots') -> None:
        self.num_epochs = num_epochs
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.scheduler = scheduler
        self.writer = SummaryWriter()
        self.prefix = prefix
        self.device = device

    def summary(self, tag: str) -> None:
        with open(file=self.summary_path(tag=tag), mode='w') as fp:
            self.model.eval()
            fp.write(str(self.model))
            fp.write('\n\n')
            x, _ = next(iter(self.train_dataloader))
            stats = torchinfo.summary(
                model=self.model, input_size=x.shape, device='cpu')
            fp.write(str(stats))

    def folder_path(self, tag: str) -> str:
        return os.path.join(self.prefix, tag)

    def model_path(self, tag: str) -> str:
        return os.path.join(self.folder_path(tag=tag), f'{tag}.model')

    def result_path(self, tag: str) -> str:
        return os.path.join(self.folder_path(tag=tag), f'{tag}.json')

    def summary_path(self, tag: str) -> str:
        return os.path.join(self.folder_path(tag=tag), f'{tag}.summary')

    def load_model(self, tag: str) -> nn.Module:
        path = self.model_path(tag=tag)
        return torch.load(path) if os.path.exists(path) else self.model

    def save_model(self, tag: str, result: Result) -> None:
        best_result = self.load_result(tag=tag)
        if result > best_result:
            self.save_result(tag=tag, result=result)
            torch.save(obj=self.model, f=self.model_path(tag=tag))

    def open_results(self, tag: str) -> list[Result]:
        file = self.result_path(tag=tag)
        results: list[Result] = []
        if os.path.exists(file):
            with open(file=file, mode='r') as fp:
                for d in json.loads(s=fp.read()):
                    results.append(Result.from_dict(d=d))
        return results

    def load_result(self, tag: str) -> Result:
        results = self.open_results(tag=tag)
        return Result() if len(results) == 0 else results[-1]

    def save_result(self, tag: str, result: Result) -> None:
        results = self.open_results(tag=tag)
        results.append(result)
        with open(file=self.result_path(tag=tag), mode='w') as fp:
            text: list[dict] = [result.to_dict() for result in results]
            fp.write(json.dumps(obj=text, indent=4))

    def accuracy_batch(self, logits: Any, labels: Any) -> tuple[int, int]:
        '''
        logits: shape = (batch_size, num_labels)
        labels: shape = (batch_size,)

        step 1. logits: (batch_size, num_labels) -> (batch_size,)
            也就是没行选出一个最大的值的下标 predict_labels = torch.argmax(logits, dim=1)

        step 2: 然后和labels进行比较 predict_labels == labels
        为了最后计算的准确率，我们需要返回判断准确的样本的个数 最后由外部累计所有的batch的和，然后再计算准确率
        '''
        # TODO: 都怪BERT bert一次训练有两个任务 所以理论上应该有两个accuracy
        # 但是现在先不考虑这个了
        # logits是什么
        if isinstance(logits, BERTOutput):
            mlm = logits.mlm
            nsp = logits.nsp
            mlm_mask = logits.mlm_mask
            assert isinstance(labels, WikiText2Label)
            mlm_labels = labels.mlm
            nsp_accuracy, nsp_batch_size = self.accuracy_batch(
                logits.nsp, labels.nsp)
            # mlm_accuracy, mlm_batch_size = self.accuracy_batch(
            #     logits.mlm, labels.mlm.flatten())
            # 卧槽 太麻烦了 mlm的正确率还要剔除一部分元素
            mlm_predict_labels = logits.mlm.argmax(dim=1)
            mlm_batch_size, _ = logits.mlm.shape
            mlm_accuracy = int(((mlm_predict_labels == labels.mlm.flatten()
                                 ).int() * logits.mlm_mask.flatten()).sum())
            mlm_batch_size = int(mlm_mask.sum())
            return nsp_accuracy + mlm_accuracy, nsp_batch_size + mlm_batch_size

        assert isinstance(logits, Tensor)
        assert isinstance(labels, Tensor)
        predict_labels: Tensor = logits.argmax(dim=1)
        # torch.sum(): Returns the sum of all elements in the input tensor.
        batch_size, _ = logits.shape
        return int((predict_labels == labels).int().sum()), batch_size

    def error_rate_batch(self, logits: Tensor, labels: Tensor) -> tuple[int, int, int]:
        batch_size, _ = logits.shape
        top1_errors = utils.top1_error_rate(logits=logits, labels=labels)
        top5_errors = utils.top5_error_rate(logits=logits, labels=labels)
        return int(top1_errors), int(top5_errors), batch_size

    def train_epoch(self, dataloader: DataLoader) -> float:
        # set to training mode
        self.model.train()
        # computing the loss for every minibatch on the GPU and reporting it back to the usr on the command line
        # or logging it in a NumPy array will trigger a global interpreter lock which stalls all GPUs
        # so we need to compute and store the loss on GPU!
        training_loss: Tensor = torch.tensor(
            data=0, dtype=torch.float32, device=self.device)

        x: Tensor
        y: Tensor
        for x, y in tqdm(dataloader):
            # send data to device
            x = x.to(device=self.device)
            y = y.to(device=self.device)

            # forward pass
            y_hat: Tensor = self.model(x)
            loss: Tensor = self.loss_fn(y_hat, y)
            training_loss += loss
            # Computes the gradient
            loss.backward()

            # Performs a single optimization step (parameter update).
            self.optimizer.step()
            # clear the gradients
            self.optimizer.zero_grad()

        # after every epoch, update the learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        return float(training_loss / len(dataloader))

    def eval_epoch(self, dataloader: DataLoader) -> tuple[float, float, float, float]:
        # set to evaluation mode
        self.model.eval()
        val_loss: Tensor = torch.tensor(
            data=0, dtype=torch.float32, device=self.device)
        # TODO: this may hurt performance, do some tests, if it hurts performance, move it to gpu
        accuracy: int = 0
        num_examples: int = 0
        top1_errors: int = 0
        top5_errors: int = 0

        # ugly type hint
        x: Tensor
        y: Tensor
        for x, y in tqdm(dataloader):
            x = x.to(device=self.device)
            y = y.to(device=self.device)

            # Disabling gradient calculation is useful for inference
            with torch.no_grad():
                y_hat: Tensor = self.model(x)
                loss: Tensor = self.loss_fn(y_hat, y)
                val_loss += loss
                # calculate accuracy, only for cross entropy loss, classification
                accuracy_batch, num_batch = self.accuracy_batch(
                    logits=y_hat, labels=y)
                # TODO: 有朝一日再打开 傻逼BERT啊
                # top1_errors_batch, top5_errors_batch, _ = self.error_rate_batch(
                #     logits=y_hat, labels=y)
                accuracy += accuracy_batch
                # top1_errors += top1_errors_batch
                # top5_errors += top5_errors_batch
                num_examples += num_batch
        return float(val_loss / len(dataloader)), float(accuracy) / float(num_examples), float(top1_errors) / float(num_examples), float(top5_errors) / float(num_examples)

    def train(self, tag: str, summary: bool = True) -> None:
        # 1. 判断 snapshots/{tag} 文件夹是否存在，如果不存在则创建
        path = self.folder_path(tag=tag)
        os.makedirs(name=path, exist_ok=True)
        if summary:
            self.summary(tag=tag)

        self.model = self.load_model(tag=tag)
        # send model parameters to device
        self.model = self.model.to(self.device)

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(self.train_dataloader)
            val_loss, val_accuracy, val_top1_error_rate, val_top5_error_rate = self.eval_epoch(
                self.val_dataloader)
            test_loss, test_accuracy, test_top1_error_rate, test_top5_error_rate = self.eval_epoch(
                self.test_dataloader)

            # write training result to tensorboard
            tag_scalar_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'val_accuracy': val_accuracy,
                'test_accuracy': test_accuracy,
                'val_top1_error_rate': val_top1_error_rate,
                'val_top5_error_rate': val_top5_error_rate,
                'test_top1_error_rate': test_top1_error_rate,
                'test_top5_error_rate': test_top5_error_rate
            }
            self.writer.add_scalars(
                main_tag=tag, tag_scalar_dict=tag_scalar_dict, global_step=epoch)

            # save model based on result
            result = Result.from_dict(d=tag_scalar_dict)
            self.save_model(tag=tag, result=result)
