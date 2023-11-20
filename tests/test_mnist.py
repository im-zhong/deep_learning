# 2023/11/20
# zhangzhong


from mytorch.data import mnist


def test_FashionMNIST():
    batch_size = 32
    ds = mnist.FashionMNISTDataset()
    train_dataloader = ds.get_train_dataloader(batch_size=32)

    for imgs, labels in train_dataloader:
        print(imgs.shape)
        print(labels.shape)
