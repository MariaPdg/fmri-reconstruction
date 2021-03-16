import os
import math
import torch
import random
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

from torch import nn, no_grad
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import torchvision


class EarlyStopping(object):
    """
    Perform early stopping during training when validation loss does not change during specified number of epochs
    https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    """
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):

        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


class VoxelLoss(nn.Module):
    """
    Calculates loss in voxel space, see R. Beliy - "From voxels to pixels and back: Self-supervision in natural-image
    reconstruction from fMRI" https://arxiv.org/abs/1907.02431
    """

    def __init__(self, alpha, device='cpu'):
        """
        @param alpha: float value
            Weight for similarity
        @param device: used device
        """

        super(VoxelLoss, self).__init__()

        self.alpha = alpha
        self.loss1 = nn.MSELoss()
        self.loss2 = nn.CosineSimilarity(dim=1)
        # self.loss2 = PearsonCorrelation()
        self.device = device
        self.name = 'VoxelLoss'

    def forward(self, y_pred, y_true):

        return (self.loss1(y_pred, y_true) + (1 - self.loss2(y_pred, y_true).mean())).to(self.device)


class ImageLoss(nn.Module):
    """
    Calculates loss in image space, see R. Beliy - "From voxels to pixels and back: Self-supervision in natural-image
    reconstruction from fMRI" https://arxiv.org/abs/1907.02431
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):

        super(ImageLoss, self).__init__()
        self.name = 'ImageLoss'
        self.mean = mean
        self.std = std
        self.device = device
        self.vgg_model = models.vgg19(pretrained=True).features[:9].eval().to(self.device)
        for layer in self.vgg_model:
            layer.requires_grad = False

    def forward(self, y_pred, y_true):
        """
        Pixel loss
        @param y_pred: tensor [batch_size x channels x width x height]
            Predicted image
        @param y_true: tensor [batch_size x channels x width x height]
            Ground truth image
        @return: float value
        """

        y_pred = norm_image_prediction(y_pred)
        mse_loss = nn.MSELoss()

        return mse_loss(y_pred, y_true)

    def vgg_loss(self, y_pred, y_true, conv_layer='conv1'):
        """
        Feature loss with VGG19 network
        @param y_pred: tensor [batch_size x channels x width x height]
            Predicted image
        @param y_true: tensor [batch_size x channels x width x height]
            Ground truth image
        @param conv_layer:  defaukt: 'conv1' or 'conv2'
        @return: float value
        """

        try:
            if conv_layer == 'conv1':
                features = self.vgg_model[:4]
            elif conv_layer == 'conv2':
                features = self.vgg_model[:9]
            else:
                raise ValueError

        except ValueError as error:
            print('Wrong layer value', error)

        y_pred = norm_image_prediction(y_pred)
        y_pred_conv = features(y_pred)
        y_true_conv = features(y_true)

        mse_loss = nn.MSELoss()
        rmse_loss = torch.sqrt(mse_loss(y_pred_conv, y_true_conv))
        return rmse_loss

    def vgg_cosine_loss(self, y_pred, y_true):

        # specify the vgg layers to include in loss
        layers = torch.tensor([3, 8, 13, 17, 22]) + 1
        sum_cos_loss = 0
        cos_loss = nn.CosineSimilarity()
        vgg_model = models.vgg19(pretrained=True).features.eval().to(self.device)
        for layer in vgg_model:
            layer.requires_grad = False

        for layer in layers:
            layer.requires_grad = False
            features = vgg_model[:layer]
            y_pred_conv = features(y_pred)
            y_true_conv = features(y_true)
            sum_cos_loss += - cos_loss(y_pred_conv, y_true_conv).mean()

        return sum_cos_loss.mean()


class NormalizeCosSimilarity(nn.Module):
    """
    Calculates similarity where predicted and gt images are normalized with mean and std
    (default values are chosen for the pretrained model)
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        super(NormalizeCosSimilarity, self).__init__()

        self.cosine_similarity = nn.CosineSimilarity()
        self.mean = mean
        self.std = std

    def forward(self, y_pred, y_true):

        try:
            # data validation
            if torch.min(y_pred) < 0.0 or torch.max(y_pred) > 1.0:
                y_pred = denormalize_image(y_pred, mean=self.mean, std=self.std).detach().clone()
                if torch.min(y_pred) < 0.0 or torch.max(y_pred) > 1.0:
                    raise ValueError

            if torch.min(y_true) < 0.0 or torch.max(y_true) > 1.0:
                y_true = denormalize_image(y_true, mean=self.mean, std=self.std).detach().clone()
                if torch.min(y_true) < 0.0 or torch.max(y_true) > 1.0:
                    raise ValueError

        except ValueError as error:
            print('Image values in Cosine Similarity must be between 0 and 1 or normalized with mean and std', error)


        return self.cosine_similarity(y_pred, y_true).mean()


def norm_image_prediction(pred, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalizes predicted image

    @param pred: tensor [batch_size x channels x width x height]
            Predicted image
    @param mean: mean values for 3 channels
    @param std: std values  for 3 channels
    @return: normalized predicted image
    """
    norm_img = pred.detach().clone()  # deep copy of tensor
    for i in range(3):
        norm_img[:, i, :, :] = (norm_img[:, i, :, :] - mean[i]) / std[i]

    return norm_img


def denormalize_image(pred, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    denorm_img = pred.detach().clone()  # deep copy of tensor
    for i in range(3):
        denorm_img[:, i, :, :] = denorm_img[:, i, :, :] * std[i] + mean[i]

    return denorm_img


def total_variation_loss(x):

    a = torch.sqrt(torch.abs(x[:, :, :-1, :-1] - x[:, :, 1:, :-1]))
    b = torch.sqrt(torch.abs(x[:, :, :-1, :-1] - x[:, :, :-1, 1:]))

    return torch.mean(torch.pow(a + b, 1.25))


def total_variation_l1(x):

    a = torch.abs(x[:, :, :-1, :-1] - x[:, :, 1:, :-1])
    b = torch.abs(x[:, :, :-1, :-1] - x[:, :, :-1, 1:])

    return torch.mean(a + b)


def total_variation_l2(x):

    a = torch.pow(x[:, :, :-1, :-1] - x[:, :, 1:, :-1], 2)
    b = torch.pow(x[:, :, :-1, :-1] - x[:, :, :-1, 1:], 2)

    return torch.mean(torch.sqrt(a + b))


class PearsonCorrelation(nn.Module):

    """
    Calculates Pearson Correlation Coefficient
    """

    def __init__(self):
        super(PearsonCorrelation, self).__init__()

    def forward(self, y_pred, y_true):
        """
        @param y_pred: tensor [batch_size x channels x width x height]
            Predicted image
        @param y_true: tensor [batch_size x channels x width x height]
            Ground truth image
        @return: float
            Pearson Correlation Coefficient
        """

        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        loss = cost.mean()

        return loss


class StructuralSimilarity(nn.Module):

    """
    Structural Similarity Index Measure (mean of local SSIM)
    see Z. Wang "Image quality assessment: from error visibility to structural similarity"

    Calculates the SSIM between 2 images, the value is between -1 and 1:
     1: images are very similar;
    -1: images are very different

    Adapted from https://github.com/pranjaldatta/SSIM-PyTorch/blob/master/SSIM_notebook.ipynb
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(StructuralSimilarity, self).__init__()
        self.mean = mean
        self.std = std

    def gaussian(self, window_size, sigma):

        """
        Generates a list of Tensor values drawn from a gaussian distribution with standard
        diviation = sigma and sum of all elements = 1.

        @param window_size: 11 from the paper
        @param sigma: standard deviation of Gaussian distribution
        @return: list of values, length = window_size
        """

        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel=1):

        """
        @param window_size: 11 from the paper
        @param channel: 3 for RGB images
        @return: 4D window with size [channels, 1, window_size, window_size]

        """
        # Generates an 1D tensor containing values sampled from a gaussian distribution.
        _1d_window = self.gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
        # Converts it to 2D
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        # Adds extra dimensions to convert to 4D
        window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

        return window

    def forward(self, img1, img2, val_range=255, window_size=11, window=None, size_average=True, full=False):

        """
        Calculating Structural Similarity Index Measure

        @param img1: torch.tensor
        @param img2: torch.tensor
        @param val_range: 255 for RGB images
        @param window_size: 11 from the paper
        @param window: created with create_window function
        @param size_average: if True calculates the mean
        @param full: if true, return result and contrast_metric
        @return: value of SSIM
        """

        pad = window_size // 2

        try:
            _, channels, height, width = img1.size()
        except:
            channels, height, width = img1.size()

        # if window is not provided, init one
        if window is None:
            real_size = min(window_size, height, width)  # window should be atleast 11x11
            window = self.create_window(real_size, channel=channels).to(img1.device)

        # calculating the mu parameter (locally) for both images using a gaussian filter
        # calculates the luminosity params
        mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        # now we calculate the sigma square parameter
        # Sigma deals with the contrast component
        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

        # Some constants for stability
        C1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
        C2 = (0.03) ** 2

        contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        contrast_metric = torch.mean(contrast_metric)

        numerator1 = 2 * mu12 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

        if size_average:
            result = ssim_score.mean()
        else:
            result = ssim_score.mean(1).mean(1).mean(1)

        if full:
            return result, contrast_metric

        return result


def save_image(images_dir, outputs, ground_truth, number, idx_epoch, norm=False):

    """
    Function to save images during training

    @param images_dir: directory where the images should be saved
    @param outputs: torch.tensor [batch_size x channels x width x height]
        Network outputs
    @param ground_truth: torch.tensor [batch_size x channels x width x height]
        Ground truth images
    @param number: integer
        Number of images to save
    @param idx_epoch: integer
        index of the epoch to include it in the caption
    @return:
    """
    if norm:
        ground_truth = denormalize_image(ground_truth)
        outputs = denormalize_image(outputs)
    for idx in range(number):
        plt.imshow(ground_truth.permute(0, 2, 3, 1)[idx].cpu().detach().numpy())
        gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + str(idx))
        plt.savefig(gt_dir)
        plt.imshow(outputs.permute(0, 2, 3, 1)[idx].cpu().detach().numpy())
        output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + str(idx))
        plt.savefig(output_dir)


def save_image_grid(images_dir, outputs, ground_truth, idx_epoch, size=5):

        if idx_epoch == 0:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(ground_truth[: size * size].cpu().detach(), nrow=size, normalize=True).permute(1, 2, 0))
            gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
            plt.savefig(gt_dir)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(outputs[: size * size].cpu().detach(), nrow=size, normalize=True).permute(1, 2, 0))
        output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
        plt.savefig(output_dir)


def evaluate(model, dataloader, norm=True, mean=None, std=None, mode=None, path=None, save=False):
    """
    Calculate metrics for the dataset specified with dataloader

    @param model: model for evaluation
    @param dataloader: DataLoader object
    @param norm: normalization
    @param mean: mean of the dataset
    @param std: standard deviation of the dataset
    @param mode: 'bold' or None
    @param path: path to save images
    @param save: True if save images, otherwise False
    @return: mean PCC, mean SSIM, MSE, mean IS (inseption score)
    """

    pearson_correlation = PearsonCorrelation()
    structural_similarity = StructuralSimilarity()
    mse_loss = nn.MSELoss()
    ssim = 0.0
    pcc = 0.0
    mse = 0.0
    gt_path = path + '/ground_truth'
    out_path = path + '/out'
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for batch_idx, data_batch in enumerate(dataloader):
        model.eval()

        with no_grad():
            if mode == 'bold':
                data_target = Variable(data_batch['image'], requires_grad=False).cpu().detach()
            else:
                data_target = data_batch.cpu().detach()
            out = model(data_batch)
            out = out.data.cpu()
            if save:
                for i, im in enumerate(out):
                    torchvision.utils.save_image(im, fp=out_path + '/' + str(batch_idx * i) + '.png', normalize=True)
                for i, im in enumerate(data_target):
                    torchvision.utils.save_image(im, fp=gt_path + '/' + str(batch_idx * i) + '.png', normalize=True)
        if norm and mean is not None and std is not None:
            data_target = denormalize_image(data_target, mean=mean, std=std)
            out = denormalize_image(out, mean=mean, std=std)
        pcc += pearson_correlation(out, data_target)
        ssim += structural_similarity(out, data_target)
        mse += mse_loss(out, data_target)

    mean_pcc = pcc / (batch_idx + 1)
    mean_ssim = ssim / (batch_idx + 1)
    mse_loss = mse / (batch_idx + 1)

    return mean_pcc, mean_ssim, mse_loss


def objective_assessment(model, dataloader, mode=None):
    """
    Calculates objective score of the predictions

    @param model: network for evaluation
    @param dataloader: DataLoader object
    @param mode: 'bold' or none
    @return: objective score - percentage of correct predictions
    """

    pearson_correlation = PearsonCorrelation()
    structural_similarity = StructuralSimilarity()
    true_positives = torch.tensor([0, 0])

    for batch_idx, data_batch in enumerate(dataloader):
        model.eval()

        with no_grad():
            if mode == 'bold':
                data_target = Variable(data_batch['image'], requires_grad=False).cpu().detach()
            else:
                data_target = data_batch.cpu().detach()
            out = model(data_batch)
            out = out.data.cpu()

            for idx, image in enumerate(out):
                numbers = list(range(0, len(out)))
                numbers.remove(idx)
                rand_idx = random.choice(numbers)
                score_rand = pearson_correlation(image, data_target[rand_idx])
                score_gt = pearson_correlation(image, data_target[idx])
                if score_gt > score_rand:
                    true_positives[0] += 1
                image_for_ssim = torch.unsqueeze(image, 0)
                target_gt_for_ssim = torch.unsqueeze(data_target[idx], 0)
                target_rand_for_ssim = torch.unsqueeze(data_target[rand_idx], 0)
                score_rand = structural_similarity(image_for_ssim, target_rand_for_ssim)
                score_gt = structural_similarity(image_for_ssim, target_gt_for_ssim)
                if score_gt > score_rand:
                    true_positives[1] += 1

    objective_score = true_positives.float() / ((batch_idx + 1) * len(out))

    return objective_score


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):

    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
    """
    from torchvision.models.inception import inception_v3
    import numpy as np
    from scipy.stats import entropy

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)


if __name__ == '__main__':

    image_loss = ImageLoss()
    image_loss.vgg_loss(None, None)
