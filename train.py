import os
import argparse

import torch
from torchvision import datasets, transforms

import utils


def main(seed):
    # set device and random seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.set_seed(seed)

    # set up LeNet model
    model = utils.get_lenet()
    model.to(device)

    # set up MNIST data loader for training
    data_path = os.path.abspath('/mnt/qb/hennig/data/')
    train_set = datasets.MNIST(
        data_path, train=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True)

    train(model, train_loader, device)


def train(model, train_loader, device):
    model.train()
    
    # Set loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    N = len(train_loader.dataset)
    for epoch in range(30):
        train_loss = 0.
        for X, y in train_loader:
            f = model(X.to(device))
            loss = loss_fn(f, y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss * len(X)

        train_loss = train_loss.item() / N
        print(f'Epoch {epoch+1} -- loss: {train_loss:.3f}')

    models_path = os.path.abspath('./models')
    os.makedirs(models_path, exist_ok=True)
    model_path = os.path.join(models_path, 'lenet_mnist_seed{}')
    torch.save(model.state_dict(), model_path.format(seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    seed = parser.parse_args().seed

    main(seed)
