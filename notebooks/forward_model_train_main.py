from models.forward_GammaRad import forward_GammaRad
from losses import GammaRad_loss

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'../AntennaDesign_data/newdata_dB.npz')
    parser.add_argument('--forward_model_path_gamma', type=str, default=r'checkpoints/forward_gamma_smoothness_0.001_0.0001.pth')
    parser.add_argument('--forward_model_path_radiation', type=str, default=r'checkpoints/forward_radiation_huberloss.pth')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--gamma_schedule', type=float, default=0.95, help='gamma decay rate')
    parser.add_argument('--step_size', type=int, default=9, help='step size for gamma decay')
    parser.add_argument('--grad_accumulation_step', type=int, default=None, help='gradient accumulation step. Should be None if HyperNet is not used')
    parser.add_argument('--inv_or_forw', type=str, default='inverse_forward_GammaRad',
    help='architecture name, to parse dataset correctly. options: inverse, forward_gamma, forward_radiation, inverse_forward_gamma, inverse_forward_GammaRad')
    parser.add_argument('--rad_range', type=list, default=[-55,5], help='range of radiation values for scaling')
    parser.add_argument('--GammaRad_lambda', type=float, default=1.0, help='controls the influence of radiation in GammaRad loss')
    parser.add_argument('--rad_phase_factor', type=float, default=1.0, help='controls the influence of radiations phase in GammaRad loss')
    parser.add_argument('--mag_smooth_weight', type=float, default=1e-3, help='controls the influence of gamma magnitude smoothness')
    parser.add_argument('--phase_smooth_weight', type=float, default=1e-3, help='controls the influence of gamma phase smoothness')
    parser.add_argument('--geo_weight', type=float, default=1e-3, help='controls the influence of geometry loss')
    parser.add_argument('--checkpoint_path', type=str, default=r'checkpoints/inverseforward_bigger_data.pth')
    return parser.parse_args()