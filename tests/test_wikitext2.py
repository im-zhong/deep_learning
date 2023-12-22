# 2023/12/21
# zhangzhong

from mytorch.data.wikitext2 import WikiText2, WikiText2Example
from torch.utils.data import DataLoader
import torch
from torch import Tensor


def test_wikitext2() -> None:
    wiki = WikiText2(root='datasets/wikitext-2', split='train')
    print(len(wiki))
    batch_size = 128
    dataloader = wiki.get_dataloader(batch_size=batch_size, num_workers=0)
    for batch in dataloader:
        # print(type(batch))
        # 但是实际情况是什么？
        # 实际情况是我们的batch返回的东西会被视作两部分 x, y
        # 这是trainer的代码决定的，然后 y_hat = model(x)
        # 然后loss = loss_fn(y_hat, y)
        # 所以为了和trainer保持一致，我们必须包装返回的batch
        # 只要我们的模型认识这种数据类型就可以了
        # 因为我们目前实现的都是 supervised learning, 所以必然返回一对数据
        examples: WikiText2Example
        labels: Tensor
        examples, labels = batch
        print(labels)
        # print(examples.sentences.shape)
        # print(examples.segments.shape)
        # print(labels.shape)
        # print(examples.sentences)
        # print(examples.segments)
        # print(examples.valid_lens)
        # print(labels)
        # valid_lens should be zero
        # 我知道了，最后一个batch的数据长度可能不足batch_size
        # for i in range(len(labels)):
        #     assert torch.all(
        #         examples.sentences[i, examples.valid_lens[i]:] == 0)
        assert torch.all(examples.sentences[examples.mask] == 0)
