import os

import torch
from torchvision import datasets, transforms

from laplace import Laplace
import utils

import warnings
warnings.filterwarnings('ignore')


def main():
    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set up MNIST data loaders
    data_path = os.path.abspath('/mnt/qb/hennig/data/')
    train_set = datasets.MNIST(
        data_path, train=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=512, shuffle=True)
    test_set = datasets.MNIST(
        data_path, train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=False)

    # assumes pretrained models
    models = utils.load_pretrained_models(device)

    # get components of mixture of Laplace approximations
    mola_components = get_mola(models, train_loader)

    ensemble_metrics = test('ensemble', models, test_loader, device)
    mola_metrics = test('mola', mola_components, test_loader, device)

    print('Ensemble results:')
    print(', '.join([f'{k}: {v:.4f}' for k, v in ensemble_metrics.items()]))
    print()
    print('MoLA results:')
    print(', '.join([f'{k}: {v:.4f}' for k, v in mola_metrics.items()]))


def get_mola(models, train_loader):
    components = list()
    for model in models:
        la = Laplace(model, 'classification')
        la.fit(train_loader)
        la.optimize_prior_precision()
        components.append(la)
    return components


@torch.no_grad()
def test(prediction_mode, components, test_loader, device):
    loss_fn = torch.nn.NLLLoss()

    all_y_true = list()
    all_y_prob = list()
    for X, y in test_loader:
        all_y_true.append(y)

        # set uniform mixture weights
        K = len(components)
        mixture_weights = torch.ones(K, device=device) / K
        y_prob = mixture_model_pred(
            components, X.to(device), mixture_weights, prediction_mode)
        all_y_prob.append(y_prob)

    # aggregate predictive distributions and true labels
    all_y_prob = torch.cat(all_y_prob, dim=0).cpu()
    labels = torch.cat(all_y_true, dim=0)

    # compute some metrics: negative log-likelihood, accuracy, mean confidence, Brier score, and ECE
    metrics = {}
    assert all_y_prob.sum(-1).mean() == 1, '`all_y_prob` are logits but probs. are required'
    c, preds = torch.max(all_y_prob, 1)
    metrics['nll'] = loss_fn(all_y_prob.log(), labels).item()
    metrics['acc'] = (labels == preds).float().mean().item()
    metrics['conf'] = c.mean().item()
    metrics['brier'] = utils.get_brier_score(all_y_prob, labels)
    metrics['ece'], _ = utils.get_calib(all_y_prob, labels)

    return metrics


def mixture_model_pred(components, x, mixture_weights, prediction_mode='mola',
                       pred_type='glm', link_approx='probit', n_samples=100):
    out = 0.  # out will be a tensor
    for model, pi in zip(components, mixture_weights):
        if prediction_mode == 'mola':
            out_prob = model(x, pred_type=pred_type, link_approx=link_approx, n_samples=n_samples)
        elif prediction_mode == 'ensemble':
            model.eval()
            out_prob = model(x).detach().softmax(1)
        else:
            raise ValueError('prediction_mode needs to be mola or ensemble')
        out += pi * out_prob
    return out


if __name__ == "__main__":
    main()
