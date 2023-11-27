from sacred import Experiment
from sacred.observers import FileStorageObserver
import logging
from algorithms import gwl, conealign, grasp as grasp, regal, eigenalign, NSD, isorank2 as isorank, netalign, klaus, sgwl,Grampa,GraspB,GrampaS, ALHPA, ALHPA_QR, Grampa_k5, Grampa_k10, Grampa_k15, Grampa_k20, Grampa_k_halfN,ALHPA_k2, ALHPA_k5, ALHPA_k10, ALHPA_k15, ALHPA_k20, ALHPA_k_halfN, ALHPA_QR_k2, ALHPA_QR_k5, ALHPA_QR_k10, ALHPA_QR_k8, ALHPA_QR_k20, ALHPA_QR_k_halfN,GrampaL,ALHPA_S

ex = Experiment("ex")

ex.observers.append(FileStorageObserver('runs'))

# create logger
logger = logging.getLogger('e')
logger.setLevel(logging.INFO)
logger.propagate = False

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

ex.logger = logger


_GW_args = {
    'opt_dict': {
        'epochs': 1,
        'batch_size': 1000000,
        'use_cuda': False,
        'strategy': 'soft',
        # 'strategy': 'hard',
        # 'beta': 0.1,
        'beta': 1e-1,
        'outer_iteration': 400,  # M
        'inner_iteration': 1,  # N
        'sgd_iteration': 300,
        'prior': False,
        'prefix': 'results',
        'display': False
    },
    'hyperpara_dict': {
        'dimension': 90,
        # 'loss_type': 'MSE',
        'loss_type': 'L2',
        'cost_type': 'cosine',
        # 'cost_type': 'RBF',
        'ot_method': 'proximal'
    },
    # 'lr': 0.001,
    'lr': 1e-3,
    # 'gamma': 0.01,
    # 'gamma': None,
    'gamma': 0.8,
    # 'max_cpu': 20,
    # 'max_cpu': 4
}

_SGW_args = {
    'ot_dict': {
        'loss_type': 'L2',  # the key hyperparameters of GW distance
        'ot_method': 'proximal',
         #'beta': 0.025,#euroroad
         #'beta': 0.025,#netscience,eurorad,arenas
        #'beta': 0.1,#dense ex fb, socfb datasets
        'beta': 0.025,# 0.025-0.1 depends on degree
        # outer, inner iteration, error bound of optimal transport
        'outer_iteration': 2000,  # num od nodes
        'iter_bound': 1e-10,
        'inner_iteration': 2,
        'sk_bound': 1e-30,
        'node_prior': 1000,
        'max_iter': 4,  # iteration and error bound for calcuating barycenter
        'cost_bound': 1e-26,
        'update_p': False,  # optional updates of source distribution
        'lr': 0,
        'alpha': 0
    },
    "mn": 1,  # gwl
    # "mn": 1,  # s-gwl-3
    # "mn": 2,  # s-gwl-2
    # "mn": 3,  # s-gwl-1
    'max_cpu': 20,
}

_CONE_args = {
    'dim': 512,  # clipped by Src[0] - 1
    'window': 10,
    'negative': 1.0,
    'niter_init': 10,
    'reg_init': 1.0,
    'nepoch': 5,
    'niter_align': 10,
    'reg_align': 0.05,
    'bsz': 10,
    'lr': 1.0,
    'embsim': 'euclidean',
    'alignmethod': 'greedy',
    'numtop': 10
}

_GRASP_args = {
    'laa': 2,
    'icp': False,
    'icp_its': 3,
    'q': 100,
    'k': 20,
    #'n_eig': Src.shape[0] - 1
    'n_eig': 100,
    'lower_t': 1.0,
    'upper_t': 50.0,
    'linsteps': True,
    'base_align': True
}

_REGAL_args = {
    'attributes': None,
    'attrvals': 2,
    'dimensions': 128,  # useless
    'k': 10,            # d = klogn
    'untillayer': 2,    # k
    'alpha': 0.01,      # delta
    'gammastruc': 1.0,
    'gammaattr': 1.0,
    'numtop': 10,
    'buckets': 2
}

_LREA_args = {
    'iters': 40,
    'method': "lowrank_svd_union",
    'bmatch': 3,
    'default_params': True
}

_NSD_args = {
    'alpha': 0.8,
    'iters': 20
}

_ISO_args = {
    'alpha': 0.9,
    'tol': 1e-12,
    'maxiter': 100,
    'lalpha': 10000,
    'weighted': True
}

_NET_args = {
    'a': 1,
    'b': 2,
    'gamma': 0.95,
    'dtype': 2,
    'maxiter': 100,
    'verbose': True
}

_KLAU_args = {
    'a': 1,
    'b': 1,
    'gamma': 0.4,
    'stepm': 25,
    'rtype': 2,
    'maxiter': 100,
    'verbose': True
}
_Grampa_args = {
    # 'eta': 0.2
    'eta': 0.2,
    'lalpha': 10000,
    'lap': True
}
_GrampaS_args = {
    'eta': 0.2,
    'rsc': 0.5,
    'ki': True,
    'lap': True,
    'lalpha': 10000,
    'weighting_scheme': 'rcut',
    'n_comp': 10,
}
_ALHPA_args = {
    'rsc': 0.5,
    'n_comp': 10,
}
_ALHPA_qr_args = {
    'rsc': 0.5,
    'n_comp': 10,
}
_GRASPB_args = {
    'laa': 3,
    'icp': True,
    'icp_its': 3,
    'q': 20,
    'k': 20,
    #'n_eig': Src.shape[0] - 1
    #n_eig': 100,
    'lower_t': 0.1,
    'upper_t': 50.0,
    'linsteps': True,
    'ba_': True,
    'corr_func': 3,
    'k_span':40

}
_Grampa_k15_args = {
    'eta': 0.2,
    'lalpha': 10000,
    'lap': True
}
_Grampa_k20_args = {
    'eta': 0.2,
    'lalpha': 10000,
    'lap': True
}
_Grampa_k5_args = {
    'eta': 0.2,
    'lalpha': 10000,
    'lap': True
}
_Grampa_k10_args = {
    'eta': 0.2,
    'lalpha': 10000,
    'lap': True
}
Grampa_k_halfN_args = {
    'eta': 0.2,
    'lalpha': 10000,
    'lap': True
}

_ALHPA_args_tau_5p = {
    'rsc': 0.05,
    'n_comp': 10,
}

_ALHPA_args_tau_10p = {
    'rsc': 0.1,
    'n_comp': 10,
}
_ALHPA_args_tau_20p = {
    'rsc': 0.2,
    'n_comp': 10,
}
_ALHPA_args_tau_30p = {
    'rsc': 0.3,
    'n_comp': 10,
}
_ALHPA_args_tau_40p = {
    'rsc': 0.4,
    'n_comp': 10,
}
_ALHPA_args_tau_60p = {
    'rsc': 0.6,
    'n_comp': 10,
}
_ALHPA_args_tau_70p = {
    'rsc': 0.7,
    'n_comp': 10,
}   
_ALHPA_args_tau_100p = {
    'rsc': 1,
    'n_comp': 10,
}

_GrampaL_args = {
    'rsc': 1,
}

_algs = [
    (gwl, _GW_args, [3], "GW"),
    (conealign, _CONE_args, [-3], "CONE"),
    (grasp, _GRASP_args, [-3], "GRASP"),
    (regal, _REGAL_args, [-3], "REGAL"),
    (eigenalign, _LREA_args, [3], "LREA"),
    (NSD, _NSD_args, [30], "NSD"),

    (isorank, _ISO_args, [3], "ISO"),
    (netalign, _NET_args, [3], "NET"),
    (klaus, _KLAU_args, [3], "KLAU"),
    (sgwl, _SGW_args, [3], "SGW"),
    (Grampa, _Grampa_args, [3], "GRAMPA"),
    (GraspB, _GRASPB_args, [-96], "GRASPB"),
    (GrampaS, _GrampaS_args, [4], f'ALHPA_'),
    (ALHPA, _ALHPA_args, [4], f'ALHPA'),
    (ALHPA_QR, _ALHPA_qr_args, [4], f'ALHPA_QR'),

    (Grampa_k5, _Grampa_k5_args, [3], f'Grampa_k5'),
    (Grampa_k10, _Grampa_k10_args, [3], f'Grampa_k10'),
    (Grampa_k15, _Grampa_k15_args, [3], f'Grampa_k15'),
    (Grampa_k20, _Grampa_k20_args, [3], f'Grampa_k20'),
    (Grampa_k_halfN, Grampa_k_halfN_args, [3], f'Grampa_k_halfN'),

    (ALHPA_k2, _ALHPA_args, [4], f'ALHPA_k2'),
    (ALHPA_k5, _ALHPA_args, [4], f'ALPHA_k5'),
    (ALHPA_k10, _ALHPA_args, [4], f'ALPHA_k10'),
    (ALHPA_k15, _ALHPA_args, [4], f'ALPHA_k15'),
    (ALHPA_k20, _ALHPA_args, [4], f'ALPHA_k20'),
    (ALHPA_k_halfN, _ALHPA_args, [4], f'ALPHA_k_halfN'),

    (ALHPA_QR_k2, _ALHPA_qr_args, [4], f'ALPHA_QR_k2'),
    (ALHPA_QR_k5, _ALHPA_qr_args, [4], f'ALPHA_QR_k5'),
    (ALHPA_QR_k8, _ALHPA_qr_args, [4], f'ALPHA_QR_k8'),
     (ALHPA_QR_k10, _ALHPA_qr_args, [4], f'ALPHA_QR_k10'),
    (ALHPA_QR_k20, _ALHPA_qr_args, [4], f'ALPHA_QR_k20'),
    (ALHPA_QR_k_halfN, _ALHPA_qr_args, [4], f'ALPHA_QR_k_halfN'),

    (ALHPA, _ALHPA_args_tau_5p, [4], f'ALHPA_tau_5p'), #32
    (ALHPA, _ALHPA_args_tau_10p, [4], f'ALHPA_tau_10p'), #33
    (ALHPA, _ALHPA_args_tau_20p, [4], f'ALHPA_tau_20p'), #34
    (ALHPA, _ALHPA_args_tau_30p, [4], f'ALHPA_tau_30p'), #35
    (ALHPA, _ALHPA_args_tau_40p, [4], f'ALHPA_tau_40p'), #36
    (ALHPA, _ALHPA_args_tau_60p, [4], f'ALHPA_tau_60p'), #37
    (ALHPA, _ALHPA_args_tau_70p, [4], f'ALHPA_tau_70p'), #38
    (ALHPA, _ALHPA_args_tau_100p, [4], f'ALHPA_tau_100p'), #39

    (ALHPA_k5, _ALHPA_args_tau_5p, [4], f'ALPHA_k5_tau_5p'), #40
    (ALHPA_k5, _ALHPA_args_tau_10p, [4], f'ALPHA_k5_tau_10p'), #41
    (ALHPA_k5, _ALHPA_args_tau_20p, [4], f'ALPHA_k5_tau_20p'), #42
    (ALHPA_k5, _ALHPA_args_tau_30p, [4], f'ALPHA_k5_tau_30p'), #43
    (ALHPA_k5, _ALHPA_args_tau_40p, [4], f'ALPHA_k5_tau_40p'), #44
    (ALHPA_k5, _ALHPA_args_tau_60p, [4], f'ALPHA_k5_tau_60p'), #45
    (ALHPA_k5, _ALHPA_args_tau_70p, [4], f'ALPHA_k5_tau_70p'), #46
    (ALHPA_k5, _ALHPA_args_tau_100p, [4], f'ALPHA_k5_tau_100p'), #47

    (ALHPA_QR, _ALHPA_args_tau_5p, [4], f'ALPHA_QR_tau_5p'), #48
    (ALHPA_QR, _ALHPA_args_tau_10p, [4], f'ALPHA_QR_tau_10p'), #49
    (ALHPA_QR, _ALHPA_args_tau_20p, [4], f'ALPHA_QR_tau_20p'), #50
    (ALHPA_QR, _ALHPA_args_tau_30p, [4], f'ALPHA_QR_tau_30p'), #51
    (ALHPA_QR, _ALHPA_args_tau_40p, [4], f'ALPHA_QR_tau_40p'), #52
    (ALHPA_QR, _ALHPA_args_tau_60p, [4], f'ALPHA_QR_tau_60p'), #53
    (ALHPA_QR, _ALHPA_args_tau_70p, [4], f'ALPHA_QR_tau_70p'), #54
    (ALHPA_QR, _ALHPA_args_tau_100p, [4], f'ALPHA_QR_tau_100p'), #55

    (ALHPA_QR_k5, _ALHPA_args_tau_5p, [4], f'ALHPA_QR_k5_tau_5p'), #56
    (ALHPA_QR_k5, _ALHPA_args_tau_10p, [4], f'ALHPA_QR_k5_tau_10p'), #57
    (ALHPA_QR_k5, _ALHPA_args_tau_20p, [4], f'ALHPA_QR_k5_tau_20p'), #58
    (ALHPA_QR_k5, _ALHPA_args_tau_30p, [4], f'ALHPA_QR_k5_tau_30p'), #59
    (ALHPA_QR_k5, _ALHPA_args_tau_40p, [4], f'ALHPA_QR_k5_tau_40p'), #60
    (ALHPA_QR_k5, _ALHPA_args_tau_60p, [4], f'ALHPA_QR_k5_tau_60p'), #61
    (ALHPA_QR_k5, _ALHPA_args_tau_70p, [4], f'ALHPA_QR_k5_tau_70p'), #62
    (ALHPA_QR_k5, _ALHPA_args_tau_100p, [4], f'ALHPA_QR_k5_tau_100p'), #63

    (GrampaL, _GrampaL_args , [4], "GRAMPA_L"), #64

    (ALHPA_S, _ALHPA_args, [4], f'ALHPA_S'), #65
]

_acc_names = [
    "acc",
    "EC",
    "ICS",
    "S3",
    "jacc",
    "mnc",
]
