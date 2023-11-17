from mytorch.data import linear
import torch


def test_SyntheticLinearRegressionData():
    w = torch.tensor([2, -3.4])
    b = torch.tensor(4.2)
    assert w.shape == (2,)
    assert b.shape == ()

    num = 1000
    train_data = linear.SyntheticLinearRegressionData(w, b, num=num)
    assert len(train_data) == num

    batch_size = 32
    train_dataloader = train_data.get_dataloader(
        batch_size=batch_size, shuffle=True)
    assert len(train_dataloader) == 32

    for X, y in train_dataloader:
        assert X.shape == (batch_size, len(w)) or X.shape == (
            num % batch_size, len(w))
        assert y.shape == (batch_size, 1) or y.shape == (num % batch_size, 1)
