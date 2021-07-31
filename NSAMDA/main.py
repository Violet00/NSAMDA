import warnings

from train import Train
from utils import plot_auc_curves, plot_prc_curves


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    fprs, tprs, auc, precisions, recalls, prc = Train(directory='data/HMDD V2.0',
                                                      epochs=500,
                                                      attn_size=64,
                                                      num_heads=8,
                                                      out_dim=64,
                                                      feat_drop=0.5,
                                                      attn_drop=0.,
                                                      slope=0.3,
                                                      lr=0.001,
                                                      wd=1e-6,
                                                      random_seed=2021,
                                                      cuda=False,
                                                      model_type='HGATMDA')

    plot_auc_curves(fprs, tprs, auc, directory='roc_result', name='test_auc')
    plot_prc_curves(precisions, recalls, prc, directory='roc_result', name='test_prc')