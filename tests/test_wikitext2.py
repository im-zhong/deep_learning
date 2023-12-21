# 2023/12/21
# zhangzhong

from mytorch.data.wikitext2 import WikiText2
from torch.utils.data import DataLoader


def test_wikitext2():
    wiki = WikiText2(root='datasets/wikitext-2', split='train')
    print(len(wiki))
    train_dataloader = DataLoader(wiki, batch_size=4, shuffle=True)
    for batch in train_dataloader:
        print(type(batch))
        break
