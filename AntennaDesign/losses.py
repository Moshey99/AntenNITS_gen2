import torch.nn as nn
import torch
import numpy as np
import pytorch_msssim
from AntennaDesign.utils import *


class multiloss(nn.Module):
    def __init__(self, objective_num, losses_fns):
        super(multiloss, self).__init__()
        self.objective_num = objective_num
        self.log_var = nn.Parameter(torch.zeros(self.objective_num), requires_grad=True)
        self.losses_fns = losses_fns

    def forward(self, output, target):
        loss = 0
        for i in range(len(self.losses_fns)):
            loss_fn = self.losses_fns[i]
            precision = torch.exp(-self.log_var[i])
            loss += precision * loss_fn(output, target) + self.log_var[i]
        return loss


class CircularLoss(nn.Module):
    def forward(self, y_true, y_pred):
        delta_theta = y_pred - y_true
        cos_delta_theta = torch.cos(delta_theta)
        loss = 1 - cos_delta_theta
        return loss.mean()


# class CircularLoss(nn.Module):
#     def forward(self, y_true, y_pred):
#         delta_theta = torch.atan2(torch.sin(y_pred - y_true), torch.cos(y_pred - y_true))
#         loss = torch.abs(delta_theta)
#         return loss.mean()


class gamma_loss(nn.Module):
    def __init__(self, delta=1.0, smoothness_weight=0):
        super(gamma_loss, self).__init__()
        self.delta = delta
        self.smooth_loss = self.mag_smoothness_loss(smoothness_weight)

    class mag_smoothness_loss(nn.Module):
        def __init__(self, weight=0):
            super(gamma_loss.mag_smoothness_loss, self).__init__()
            self.weight = weight

        def forward(self, pred_mag):
            second_order_absdiff = torch.abs(pred_mag[:, 2:] - 2 * pred_mag[:, 1:-1] + pred_mag[:, :-2])
            return self.weight * second_order_absdiff.mean()

    def forward(self, gamma, target):
        gamma_magnitude = gamma[:, :gamma.shape[1] // 2]
        smooth_loss_mag = self.smooth_loss(gamma_magnitude)
        gamma_phase = gamma[:, gamma.shape[1] // 2:]
        target_magnitude = target[:, :target.shape[1] // 2]
        target_phase = target[:, target.shape[1] // 2:]
        gamma_x, gamma_y = gamma_magnitude * torch.cos(gamma_phase), gamma_magnitude * torch.sin(gamma_phase)
        target_x, target_y = target_magnitude * torch.cos(target_phase), target_magnitude * torch.sin(target_phase)
        diff = torch.abs(gamma_x - target_x) + torch.abs(gamma_y - target_y)
        loss = torch.where(diff < self.delta,
                           0.5 * (torch.square(gamma_x - target_x) + torch.square(gamma_y - target_y)),
                           self.delta * (diff - 0.5 * self.delta)).mean()
        return loss + smooth_loss_mag


class gamma_loss_dB(nn.Module):
    def __init__(self, mag_smooth_weight=0, phase_smooth_weight=0):
        super(gamma_loss_dB, self).__init__()
        self.phase_loss = CircularLoss()
        self.dB_magnitude_loss = nn.HuberLoss()
        self.smooth_loss_mag = self.mag_smoothness_loss(mag_smooth_weight)
        self.smooth_loss_phase = self.phase_smoothness_loss(phase_smooth_weight)

    class mag_smoothness_loss(nn.Module):
        def __init__(self, weight=0):
            super(gamma_loss_dB.mag_smoothness_loss, self).__init__()
            self.weight = weight

        def forward(self, pred_mag):
            second_order_absdiff = torch.abs(pred_mag[:, 2:] - 2 * pred_mag[:, 1:-1] + pred_mag[:, :-2])
            return self.weight * second_order_absdiff.mean()

    class phase_smoothness_loss(nn.Module):
        def __init__(self, weight=0):
            super(gamma_loss_dB.phase_smoothness_loss, self).__init__()
            self.weight = weight

        def forward(self, y_pred):
            circular_diff = y_pred[:, 1:] - y_pred[:, :-1]
            # Adjust circular differences for values close to a full circle
            circular_diff = torch.where(torch.abs(circular_diff) > torch.pi,
                                        2 * torch.pi - torch.abs(circular_diff), circular_diff)
            circular_diff_second_order = circular_diff[:, 1:] - circular_diff[:, :-1]
            circular_diff_second_order = torch.where(torch.abs(circular_diff_second_order) > torch.pi,
                                                     2 * torch.pi - torch.abs(circular_diff_second_order),
                                                     circular_diff_second_order)
            circular_loss = torch.abs(circular_diff_second_order).mean()
            return self.weight * circular_loss

    def forward(self, pred, target):
        pred_magnitude = 10 * torch.log10(pred[:, :pred.shape[1] // 2])  # expecting pred in linear scale, convert to dB
        smooth_loss_mag = self.smooth_loss_mag(pred_magnitude)
        pred_phase = pred[:, pred.shape[1] // 2:]
        smooth_loss_phase = self.smooth_loss_phase(pred_phase)
        target_magnitude = target[:, :target.shape[1] // 2]  # expecting target in dB
        target_phase = target[:, target.shape[1] // 2:]
        mag_loss, phase_loss = self.dB_magnitude_loss(pred_magnitude, target_magnitude), self.phase_loss(pred_phase,
                                                                                                         target_phase)
        loss = mag_loss + phase_loss
        return loss + smooth_loss_mag + smooth_loss_phase


class radiation_loss_dB(nn.Module):
    def __init__(self, mag_loss='combined', rad_phase_factor=1.):
        super(radiation_loss_dB, self).__init__()
        self.phase_loss = CircularLoss()
        if mag_loss == 'combined':
            self.dB_magnitude_loss = self.huber_msssim_combined_mag_loss()
        elif mag_loss == 'huber':
            self.dB_magnitude_loss = nn.HuberLoss()
        self.rad_phase_factor = rad_phase_factor

    def forward(self, pred, target):
        pred_magnitude = pred[:, :pred.shape[1] // 2]  # expecting pred in dB
        pred_phase = pred[:, pred.shape[1] // 2:]
        target_magnitude = target[:, :target.shape[1] // 2]  # expecting target in dB
        target_phase = target[:, target.shape[1] // 2:]
        loss = self.dB_magnitude_loss(pred_magnitude, target_magnitude) + self.rad_phase_factor * self.phase_loss(
            pred_phase, target_phase)
        return loss

    class huber_msssim_combined_mag_loss(nn.Module):
        def __init__(self):
            super(radiation_loss_dB.huber_msssim_combined_mag_loss, self).__init__()
            self.huber_loss = nn.HuberLoss()
            self.msssim_loss = pytorch_msssim.MSSSIM()

        def forward(self, pred, target):
            huber_loss = self.huber_loss(pred, target)
            msssim_loss = self.msssim_loss(pred, target)
            return 0.85 * huber_loss + 0.15 * msssim_loss


class GammaRad_loss(nn.Module):
    def __init__(self, gamma_loss_fn=None, radiation_loss_fn=None, lamda=1.0, rad_phase_fac=0.0, geo_weight=1e-4, euc_weight=1e-2):
        super(GammaRad_loss, self).__init__()
        if gamma_loss_fn is None:
            self.gamma_loss_fn = Euclidean_Gamma_Loss()  # gamma_loss_dB()
        if radiation_loss_fn is None:
            self.radiation_loss_fn = radiation_loss_dB(mag_loss='huber', rad_phase_factor=rad_phase_fac)
        self.lamda = lamda
        self.euc_weight = euc_weight

    def forward(self, pred, target):
        gamma_pred, radiation_pred, geo_pred = pred
        gamma_target, radiation_target = target
        gamma_loss = self.gamma_loss_fn(gamma_pred, gamma_target)
        radiation_loss = self.radiation_loss_fn(radiation_pred, radiation_target)
        #print(f'Gamma Loss: {gamma_loss}, Radiation Loss: {radiation_loss}')
        loss = gamma_loss + self.lamda * radiation_loss
        return loss

    def geometry_loss(self, geo):
        return self.geo_weight * torch.mean(geo)


class Euclidean_Gamma_Loss(nn.Module):
    def __init__(self):
        super(Euclidean_Gamma_Loss, self).__init__()

    def forward(self, gamma_pred, gamma_target):
        sep_gamma = gamma_pred.shape[1] // 2
        gamma_pred_mag, gamma_pred_phase = gamma_pred[:, :sep_gamma], gamma_pred[:, sep_gamma:]
        gamma_target_mag, gamma_target_phase = gamma_target[:, :sep_gamma], gamma_target[:, sep_gamma:]

        gamma_pred_euc = gamma_pred_mag * torch.exp(1j * gamma_pred_phase)
        gamma_target_mag_lin = gamma_mag_to_linear(gamma_target_mag)
        gamma_target_euc = gamma_target_mag_lin * torch.exp(1j * gamma_target_phase)
        gamma_loss = torch.abs(gamma_pred_euc - gamma_target_euc).mean()
        return gamma_loss


class Euclidean_Radiation_Loss(nn.Module):
    def __init__(self):
        super(Euclidean_Radiation_Loss, self).__init__()

    def forward(self, radiation_pred, radiation_target):
        sep_radiation = radiation_pred.shape[1] // 2
        rad_pred_mag, rad_pred_phase = radiation_pred[:, :sep_radiation], radiation_pred[:, sep_radiation:]
        rad_target_mag, rad_target_phase = radiation_target[:, :sep_radiation], radiation_target[:, sep_radiation:]

        rad_pred_mag_lin = radiation_mag_to_linear(rad_pred_mag)
        rad_target_mag_lin = radiation_mag_to_linear(rad_target_mag)
        rad_pred_euc = rad_pred_mag_lin * torch.exp(1j * rad_pred_phase)
        rad_target_euc = rad_target_mag_lin * torch.exp(1j * rad_target_phase)
        rad_loss = torch.abs(rad_pred_euc - rad_target_euc).mean()
        return rad_loss


class Euclidean_GammaRad_Loss(nn.Module):
    def __init__(self, lamda=1.):
        super(Euclidean_GammaRad_Loss, self).__init__()
        self.lamda = lamda
        self.gamma_loss = Euclidean_Gamma_Loss()
        self.rad_loss = Euclidean_Radiation_Loss()

    def forward(self, pred, target):
        gamma_pred, radiation_pred, geo_pred = pred
        gamma_target, radiation_target = target
        gamma_loss = self.gamma_loss(gamma_pred, gamma_target)
        radiation_loss = self.rad_loss(radiation_pred, radiation_target)
        loss = gamma_loss + self.lamda * radiation_loss
        return loss

