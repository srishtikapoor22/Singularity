from data.mnist import get_mnist_loaders

def test_mnist_shapes():
    train_loader, test_loader = get_mnist_loaders(batch_size=32)
    X, y = next(iter(train_loader))
    assert X.shape == (32, 1, 28, 28)
    assert y.shape[0] == 32
