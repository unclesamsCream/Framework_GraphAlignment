from . import ex, _algs, _CONE_args, _GRASP_args, _GW_args, _ISO_args, _KLAU_args, _LREA_args, _NET_args, _NSD_args, _REGAL_args,_Grampa_args,_GrampaS_args, _ALHPA_args, _ALHPA_qr_args
from generation import generate as gen
from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, grasp2 as grasp, isorank2 as isorank,Grampa,GraspB,GrampaS
import networkx as nx
import numpy as np

# mprof run workexp.py with playground run=[1,2,3,4,5] iters=2 win

def generate_sbm(n, p, q, k):
    '''
    parameters:
    n: number of nodes to be generated.
    p: probability of forming an edge within same cluster.
    q: probability of forming an edge to another cluster.
    k: number of clusters.
    '''
    q = q / (k-1)
    pmat = np.zeros(shape=(k, k)) + q
    for i in range(k):
        pmat[i,i] = p

    sizes = [n // k for _ in range(k)]
    G = nx.stochastic_block_model(sizes, pmat)
    return G

def aaa(vals, dist_type=0):
    g = []
    for val in vals:
        if dist_type == 0:
            dist = np.random.randint(15, 21, val)
        if dist_type == 1:
            dist = nx.utils.powerlaw_sequence(val, 2.5)
            dist = np.array(dist)
            dist = dist.round()
            dist += 1
            dist = dist.tolist()
        if dist_type == 2:
            dist = np.random.normal(10, 1, val)
            # dist = np.random.normal(val, 1, 2**14)
        if dist_type == 3:
            dist = np.random.poisson(lam=1, size=val)
            dist = np.array(dist)
            dist += 1
            dist = dist.tolist()

        dist = [round(num) for num in dist]
        usum = sum(dist)
        if usum % 2 == 1:
            max_value = max(dist)
            max_index = dist.index(max_value)
            dist[max_index] = dist[max_index]-1
        G2 = nx.configuration_model(dist, nx.Graph)
        G2.remove_edges_from(nx.selfloop_edges(G2))
        g.append((lambda x: x, (G2,)))
    return g
    # normald = np.random.normal(10, 2, 1000) make it 1 for standard


def ggg(vals):
    return [str(x) for x in vals]


@ex.named_config
def scaling():

    # Greedied down
    _algs[0][2][0] = 2
    _algs[1][2][0] = -2
    _algs[2][2][0] = -2
    _algs[3][2][0] = -2
    _algs[4][2][0] = 2
    _algs[5][2][0] = 2
    _algs[6][2][0] = 2

    _GW_args["max_cpu"] = 40
    # _CONE_args["dim"] = 1000
    _CONE_args["dim"] = 256
    _GRASP_args["n_eig"] = 256
    _ISO_args["alpha"] = 0.9
    _ISO_args["lalpha"] = 100000  # full dim

    run = [1, 2, 3, 4, 5, 6]

    iters = 5

    tmp = [
        # 2**i for i in range(10, 14)
        # 2 ** 15,
        # 2 ** 16,
        # 2 ** 17,
        # 10, 100, 1000, 10000
    ]

    # graphs = aaa(tmp, dist_type=0)
    # xlabel = "kdist"
    # graphs = aaa(tmp, dist_type=1)
    # xlabel = "powerlaw"
    # graphs = aaa(tmp, dist_type=2)
    xlabel = "normal"
    # graphs = aaa(tmp, dist_type=3)
    # xlabel = "poisson"
    graphs = []

    graph_names = ggg(tmp)

    noises = [
        # 0.00,
        0.01,
        # 0.02,
        # 0.03
        # 0.04,
    ]

    s_trans = (2, 1, 0, 3)


def alggs(tmp):
    alg, args, mtype, algname = _algs[tmp[0]]
    return [
        # (alg, {**args, **update}, mtype, f"{algname}{list(update.values())[0]}") for update in tmp[1]
        (alg, {**args, **update}, mtype, str(list(update.values())[0])) for update in tmp[1]
    ]

# def args_gen():
#     alg, args, mtype, algname = _algs[tmp[0]]
#     return [
#         # (alg, {**args, **update}, mtype, f"{algname}{list(update.values())[0]}") for update in tmp[1]
#         (alg, {**args, **update}, mtype, str(list(update.values())[0])) for update in tmp[1]
#     ]

def mod_arg(tmp):
    alg, args, mtype, algname = _algs[tmp[0]]
    return [
        # (alg, {**args, **update}, mtype, f"{algname}{list(update.values())[0]}") for update in tmp[1]
        (alg, {**args, **update}, mtype, str(list(update.values())[0])) for update in tmp[1]
    ]

@ex.named_config
def socials_small():
    run = [
        12,
        13,
    ]

    iters = 5

    graph_names = [             # n     / e
        "soc-hamsterster",      # 2.4K  / 16.6K / disc - 400
        "socfb-Bowdoin47",      # 2.3K  / 84.4K / disc - only 2
        "socfb-Hamilton46",     # 2.3K  / 96.4K / disc - only 2
        "socfb-Haverford76",    # 1.4K  / 59.6K / connected
        "socfb-Swarthmore42",   # 1.7K  / 61.1K / disc - only 2
        "soc-facebook",         # 4k    / 87k   / connected
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        0.05,
        0.10,
    ]

@ex.named_config
def tuning():

    tmp = [
        13,  # ALHPA
        [
            # {'k': x} for x in [3, 5, 10, 15, 20]
            # {'rsc': x} for x in [2500, 4000, 6000, 8000]
            # {'weighting_scheme': x} for x in ['size', 'rcut', 'ncut']
            {'n_comp': x} for x in [10, 20, 30]
        ]
    ]
    tmp2 = [
        14, # ALHPA_QR
        [
            {'n_comp': x} for x in [10, 20, 30]
        ]
    ]

    _algs[:] = alggs(tmp) + alggs(tmp2)

    run = list(range(len(tmp[1])))

    iters = 10

    graph_names = [             # n     / e
        "in-arenas",            # 1.1k  / 5.4k  / connected
        "socfb-Haverford76",    # 1.4K  / 59.6K / connected
        "soc-facebook",         # 4k    / 87k   / connected
        "ca-GrQc",              # 4.2k  / 13.4K / connected - (5.2k  / 14.5K)?
        "inf-power",            # 4.9K  / 6.6K  / connected

        "bio-dmela",            # 7.4k  / 25.6k / connected

        # "arenas-pgp",            # 10.68k / 24.316K / connected
        # "CA-AstroPh",           # 18k   / 195k  / connected
        # "socfb-Cornell5",         # 18.6K / 79K / connected,
        # "socfb-BU10"              # 19.6K / 637.5K / connected
        # # "fb-wosn",                # 63.4K / 817K / connected
        
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        0.05,
        # 0.10,
    ]

    # s_trans = (0, 2, 1, 3)
    # xlabel = list(tmp[1][0].keys())[0]

@ex.named_config
def tuning13():

    tmp = [
        13,  # ALHPA
        [
            # {'k': x} for x in [3, 5, 10, 15, 20]
            # {'rsc': x} for x in [2500, 4000, 6000, 8000]
            # {'weighting_scheme': x} for x in ['size', 'rcut', 'ncut']
            {'n_comp': x} for x in [10, 30, 50]
        ]
    ]

    _algs[:] = alggs(tmp)

    run = list(range(len(tmp[1])))

    iters = 10

    graph_names = [             # n     / e
        "in-arenas",            # 1.1k  / 5.4k  / connected
        "socfb-Haverford76",    # 1.4K  / 59.6K / connected
        "socfb-Swarthmore42",   # 1.7K  / 61.1K / disc - only 2
        "soc-facebook",         # 4k    / 87k   / connected
        "ca-GrQc",              # 4.2k  / 13.4K / connected - (5.2k  / 14.5K)?
        "inf-power",            # 4.9K  / 6.6K  / connected

        # "bio-dmela",            # 7.4k  / 25.6k / connected

        # "arenas-pgp",            # 10.68k / 24.316K / connected
        # "CA-AstroPh",           # 18k   / 195k  / connected
        # "socfb-Cornell5",         # 18.6K / 79K / connected,
        # "socfb-BU10"              # 19.6K / 637.5K / connected
        # # "fb-wosn",                # 63.4K / 817K / connected
        
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        0.05,
        # 0.10,
    ]

    # s_trans = (0, 2, 1, 3)
    # xlabel = list(tmp[1][0].keys())[0]

@ex.named_config
def tuning14():

    tmp = [
        14,  # ALHPA QR
        [
            # {'k': x} for x in [3, 5, 10, 15, 20]
            # {'rsc': x} for x in [2500, 4000, 6000, 8000]
            # {'weighting_scheme': x} for x in ['size', 'rcut', 'ncut']
            {'n_comp': x} for x in [10, 30, 50]
        ]
    ]

    _algs[:] = alggs(tmp)

    run = list(range(len(tmp[1])))

    iters = 10

    graph_names = [             # n     / e
        "in-arenas",            # 1.1k  / 5.4k  / connected
        "socfb-Haverford76",    # 1.4K  / 59.6K / connected
        "socfb-Swarthmore42",   # 1.7K  / 61.1K / disc - only 2
        "soc-facebook",         # 4k    / 87k   / connected
        "ca-GrQc",              # 4.2k  / 13.4K / connected - (5.2k  / 14.5K)?
        "inf-power",            # 4.9K  / 6.6K  / connected        
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        0.05,
        # 0.10,
    ]

    # s_trans = (0, 2, 1, 3)
    # xlabel = list(tmp[1][0].keys())[0]


def namess(tmp):
    return [name[-15:] for name in tmp[1]]
def namessgt(tmp):
    return [name[-15:] for name in tmp[2]]

def graphss(tmp):
    return [
        (lambda x:x, [[
            tmp[0],
            target,
            None
        ]]) for target in tmp[1]
    ]

def graphss1(tmp):
    x=len(tmp[2])
    return [
        (lambda x:x, [[
            tmp[0],
            tmp[1][i],
            tmp[2][i]
        ]]) for i in range(x)
            
    ]

@ex.named_config
def real_noisetest():

   
    tmp = [
        "data/real world/arenas/arenas_orig.txt",
        [
            f"data/real world/arenas/noise_level_10/edges_{i}.txt" for i in [
                 1]
        ],
        [
            f"data/real world/arenas/noise_level_10/gt_{i}.txt" for i in [
               1]
        ]

    ]

    #xlabel = "CA-AstroPh"
    xlabel = "arenas"
    graph_names = namess(tmp)
    graphs = graphss1(tmp)
    print(graphs)
    run=[11]
    iters = 1

    noises = [
        1.0
    ]

    s_trans = (2, 1, 0, 3)

    # (g,alg,acc,n,i)
    # s_trans = (3, 1, 2, 0, 4)




@ex.named_config
def real_noise():

    #tmp = [
    #    "data/real world/contacts-prox-high-school-2013/contacts-prox-high-school-2013_100.txt",
    #    [
    #        f"data/real world/contacts-prox-high-school-2013/contacts-prox-high-school-2013_{i}.txt" for i in [
    #            99, 95, 90, 80]
    #    ]
    #]
    #xlabel = "high-school-2013"

    # tmp = [
    #     "data/real world/mamalia-voles-plj-trapping/mammalia-voles-plj-trapping_100.txt",
    #     [
    #         f"data/real world/mamalia-voles-plj-trapping/mammalia-voles-plj-trapping_{i}.txt" for i in [
    #             99, 95, 90, 80]
    #     ]
    # ]
    # xlabel = "mammalia-voles"

    tmp = [
        "data/real world/MultiMagna/yeast0_Y2H1.txt",
        [
             f"data/real world/MultiMagna/yeast{i}_Y2H1.txt" for i in [
                 5, 10, 15, 20, 25]
        ]
    ]
    xlabel = "yeast_Y2H1"
    #tmp = [
    #    "data/real world/arenas/arenas_orig.txt",
    #    [
    #        f"data/real world/arenas/noise_level_0/edges_{i}.txt" for i in [
    #            1, 2, 3, 4,5]
    #    ],
    #    [
    #        f"data/real world/arenas/noise_level_0/gt_{i}.txt" for i in [
    #            1, 2, 3, 4,5]
    #    ]

    #]
    #xlabel = "yeast_Y2H1"

    graph_names = namess(tmp)
    #graphs = graphss1(tmp)
    graphs = graphss(tmp)
    print(graphs)
    run=[10]
    iters = 1

    noises = [
        1.0
    ]

    #s_trans = (2, 1, 0, 3)

    # (g,alg,acc,n,i)
    # s_trans = (3, 1, 2, 0, 4)


def rgraphs(gnames):
    return [
        (gen.loadnx, (f"data/{name}.txt",)) for name in gnames
    ]


@ex.named_config
def fb():

    run = [
        # 10,
        # 12, 
        13,
        # 14,
        # 10,
    ]

    iters = 10

    graph_names = [             # n     / e
        "soc-facebook",         # 4k    / 87k   / connected
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        # 0.05,
    ]

@ex.named_config
def real():

    run = [
        12, 
        13,
        # 14,
        # 10,
    ]
    # _ALHPA_args['n_comp'] = 100
    # _ALHPA_mt_args['rsc'] = 0.2
    # _ALHPA_args['n_comp'] = 10
    # _ALHPA_args['rsc'] = 0.5
    # _GrampaS_args['n_comp'] = 10
    # _GrampaS_args['rsc'] = 0.5
    iters = 3

    graph_names = [             # n     / e
        # "ca-netscience",       # 379   / 914   / connected
        #n "voles",
        # "high-school",
        # "yeast"
        # "bio-celegans",         # 453   / 2k    / connected
        # "in-arenas",            # 1.1k  / 5.4k  / connected
        # # # "inf-euroroad",         # 1.2K  / 1.4K  / disc - 200
        "soc-facebook",         # 4k    / 87k   / connected
        "inf-power",            # 4.9K  / 6.6K  / connected
        # "ca-GrQc",              # 4.2k  / 13.4K / connected - (5.2k  / 14.5K)?
        # "bio-dmela",            # 7.4k  / 25.6k / connected

        # # # "soc-hamsterster",      # 2.4K  / 16.6K / disc - 400
        # # # "socfb-Bowdoin47",      # 2.3K  / 84.4K / disc - only 2
        # # # "socfb-Hamilton46",     # 2.3K  / 96.4K / disc - only 2
        # "socfb-Haverford76",    # 1.4K  / 59.6K / connected
        # # "socfb-Swarthmore42",   # 1.7K  / 61.1K / disc - only 2

        # "ca-Erdos992",          # 6.1K  / 7.5K  / disc - 100 + 1k disc nodes
        # "arenas-pgp",            # 10.68k / 24.316K / connected
        # "CA-AstroPh",           # 18k   / 195k  / connected
        # "socfb-Cornell5",         # 18.6K / 79K / connected,
        # "socfb-BU10"              # 19.6K / 637.5K / connected
        # # "fb-wosn",                # 63.4K / 817K / connected
        
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        # 0.01,
        # 0.02,
        # 0.03,
        # 0.04,
        0.05,
        # 0.00,
        # 0.05,
        # 0.10,
        # 0.15,
        # 0.20,
        #0.25,
    ]
@ex.named_config
def cornell_n0():
    run = [
        # 12, 
        # 13,
        14,
        # 10,
    ]
    iters = 1
    graph_names = [             # n     / e
        # "CA-AstroPh",           # 18k   / 195k  / connected
        "socfb-Cornell5",         # 18.6K / 79K / connected,
        # "socfb-BU10"              # 19.6K / 637.5K / connected
        # # "fb-wosn",                # 63.4K / 817K / connected
    ]
    graphs = rgraphs(graph_names)
    noises = [
        0.00,
    ]
@ex.named_config
def cornell_n05():
    run = [
        # 12, 
        # 13,
        14,
        # 10,
    ]
    iters = 1
    graph_names = [             # n     / e
        # "CA-AstroPh",           # 18k   / 195k  / connected
        "socfb-Cornell5",         # 18.6K / 79K / connected,
        # "socfb-BU10"              # 19.6K / 637.5K / connected
        # # "fb-wosn",                # 63.4K / 817K / connected
    ]
    graphs = rgraphs(graph_names)
    noises = [
        0.05,
    ]
@ex.named_config
def cornell_n10():
    run = [
        # 12, 
        13,
        14,
        # 10,
    ]
    iters = 1
    graph_names = [             # n     / e
        # "CA-AstroPh",           # 18k   / 195k  / connected
        "socfb-Cornell5",         # 18.6K / 79K / connected,
        # "socfb-BU10"              # 19.6K / 637.5K / connected
        # # "fb-wosn",                # 63.4K / 817K / connected
    ]
    graphs = rgraphs(graph_names)
    noises = [
        0.05,
    ]

@ex.named_config
def cons_large():

    run = [
        12, 
        13,
        # 10,
    ]
    _ALHPA_args['n_comp'] = 10
    _ALHPA_args['rsc'] = 0.5
    _GrampaS_args['n_comp'] = 10
    _GrampaS_args['rsc'] = 0.5
    iters = 10

    graph_names = [             # n     / e
        "arenas-pgp",            # 10.68k / 24.316K / connected
        "CA-AstroPh",           # 18k   / 195k  / connected
        "socfb-Cornell5",         # 18.6K / 79K / connected,
        "socfb-BU10"              # 19.6K / 637.5K / connected
        # "fb-wosn",                # 63.4K / 817K / connected        
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        # 0.01,
        # 0.02,
        # 0.03,
        # 0.04,
        0.05,
        # 0.00,
        # 0.05,
        0.10,
        # 0.15,
        # 0.20,
        #0.25,
    ]


@ex.named_config
def mt_test():

    run = [
        # 12, 
        13,
        14,
        # 10,
    ]
    _ALHPA_args['n_comp'] = 10
    _ALHPA_args['rsc'] = 0.5
    _ALHPA_mt_args['n_comp'] = 10
    _ALHPA_mt_args['rsc'] = 0.5
    iters = 1

    graph_names = [             # n     / e
        # "in-arenas",            # 1.1k  / 5.4k  / connected
        # "socfb-Haverford76",    # 1.4K  / 59.6K / connected
        # "soc-facebook",         # 4k    / 87k   / connected
        # "ca-GrQc",              # 4.2k  / 13.4K / connected - (5.2k  / 14.5K)?
        # "inf-power",            # 4.9K  / 6.6K  / connected

        # "bio-dmela",            # 7.4k  / 25.6k / connected --- stupid graph!

        # "arenas-pgp",            # 10.68k / 24.316K / connected
        # "CA-AstroPh",           # 18k   / 195k  / connected
        "socfb-Cornell5",         # 18.6K / 79K / connected,
        # "socfb-BU10"              # 19.6K / 637.5K / connected
        # # "fb-wosn",                # 63.4K / 817K / connected
        
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.05,
        # 0.10,
    ]

@ex.named_config
def cons_small():

    run = [
        # 12,        
        13,
        14,
        10,
    ]
    iters = 10

    graph_names = [             # n     / e
        "in-arenas",            # 1.1k  / 5.4k  / connected
        "socfb-Haverford76",    # 1.4K  / 59.6K / connected
        "soc-facebook",         # 4k    / 87k   / connected
        "ca-GrQc",              # 4.2k  / 13.4K / connected - (5.2k  / 14.5K)?
        "inf-power",            # 4.9K  / 6.6K  / connected
        # "bio-dmela",            # 7.4k  / 25.6k / connected

        # "arenas-pgp",            # 10.68k / 24.316K / connected
        # "CA-AstroPh",           # 18k   / 195k  / connected
        # "socfb-Cornell5",         # 18.6K / 79K / connected,
        # "socfb-BU10"              # 19.6K / 637.5K / connected
        # # "fb-wosn",                # 63.4K / 817K / connected
        
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        # 0.01,
        # 0.02,
        # 0.03,
        # 0.04,
        0.05,
        0.10,
        # 0.15,
        # 0.20,
        #0.25,
    ]

@ex.named_config
def small_size():
    run = [
        # 12,        
        13,
        14,
        # 10,
    ]
    iters = 10

    graph_names = [                # n     / e
        "in-arenas",               # 1.1k  / 5.4k  / connected
        # "inf-euroroad_lcon",   # 1k / 1.6k
        "inf-euroroad",
        "socfb-Haverford76",       # 1.4K  / 59.6K / connected
        "socfb-Swarthmore42",      # 1.7K  / 61.1K / disc - only 2
        # "soc-hamsterster_lcon", # 2k / 16k
        "soc-hamsterster",
        "socfb-Bowdoin47",         # 2.3K  / 84.4K / disc - only 2
        "socfb-Hamilton46",        # 2.3K  / 96.4K / disc - only 2        
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        0.05,
        0.10,
    ]

@ex.named_config
def medium_size(): 
    run = [
        # 12,        
        13,
        # 14,
        # 10,
    ]
    _ALHPA_args['n_comp'] = 20
    
    iters = 5

    graph_names = [             # n     / e
        "inf-power",            # 4.9K  / 6.6K  / connected
        "ca-GrQc",              # 4.2k  / 13.4K / connected - (5.2k  / 14.5K)?
        "bio-dmela",            # 7.4k  / 25.6k / connected
        "soc-facebook",         # 4k    / 87k   / connected
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        0.05,
        0.10,
    ]        

@ex.named_config
def large_size():
    run = [
        # 12,        
        13,
        14,
        # 10,
    ]
    iters = 5

    graph_names = [             # n     / e
        "arenas-pgp",            # 10.68k / 24.316K / connected
        "CA-AstroPh",           # 18k   / 195k  / connected
        "socfb-Cornell5",         # 18.6K / 79K / connected,
        "socfb-BU10"              # 19.6K / 637.5K / connected
        # "fb-wosn",                # 63.4K / 817K / connected
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        0.05,
        0.10,
    ]

@ex.named_config
def fb_wosn():
    run = [
        13,
        # 10,
    ]
    iters = 1

    graph_names = [             # n     / e
        "fb-wosn",                # 63.4K / 817K / connected
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        0.05,
    ]            
    
@ex.named_config
def synth_inflexion(): 
    iters = 5
    run = [
        10,
        13,
    ]
    graph_names = [
        "ER(500)",
        "ER(1000)",
        "ER(1500)",
        "ER(2000)",
        "ER(3000)",
        "ER(4000)",
    ]
    graphs = [
        (nx.gnp_random_graph, (500, 0.5)),
        (nx.gnp_random_graph, (1000, 0.5)),
        (nx.gnp_random_graph, (1500, 0.5)),
        (nx.gnp_random_graph, (2000, 0.5)),
        (nx.gnp_random_graph, (3000, 0.5)),
        (nx.gnp_random_graph, (4000, 0.5)),
    ]

    noises = [
        0.00,
    ]

@ex.named_config
def synth_ncomp(): 
    iters = 5
    tmp = [
        13,
        [
            {'n_comp': x} for x in [10, 20, 30]
        ]
    ]
    _algs[:] = alggs(tmp)
    run = list(range(len(tmp[1])))
    graph_names = [
        "SBM(6000k5)",
        "SBM(6000k10)",
        "SBM(6000k20)",
        "SBM(6000k30)",        
        "SBM(6000k50)",
    ]

    graphs = [
        (generate_sbm, (6000, .35, .15, 5)),
        (generate_sbm, (6000, .35, .15, 10)),
        (generate_sbm, (6000, .35, .15, 20)),
        (generate_sbm, (6000, .35, .15, 30)),
        (generate_sbm, (6000, .35, .15, 50)),
    ]

    noises = [
        0.00,
    ]
    

@ex.named_config
def synth_benchmarks(): 
    iters = 1
    run = [
        13,
        10,
    ]
    graph_names = [
        "ER",
        "SBMk4",
        "LFRk4",
        "SBMk9",
        "LFRk9",
    ]

    graphs = [
        (nx.gnp_random_graph, (500, 0.05)),
        # (generate_sbm, (6000, .35, .15, 4)),
        # (nx.LFR_benchmark_graph, (6000, 3, 2, 0.3, None, 4, None, 900, 2400, 1/10000000, 2000)), # 4 clusters 900-2400
        # (generate_sbm, (6000, .35, .15, 8)),
        # (nx.LFR_benchmark_graph, (6000, 3, 2, 0.3, None, 4, None, 450, 1500, 1/10000000, 2000)), # 9 clusters 450-1500
        
        # (nx.LFR_benchmark_graph, (6000, 3, 2, 0.3, 15, None, None, 100, 500, 1/10000000, 2000)) #  clusters 300-1200
        # (nx.LFR_benchmark_graph, (6000, 3, 2, 0.3, 15, None, None, 150, 800, 1/10000000, 2000)) #  clusters 300-1200
        # (nx.LFR_benchmark_graph, (6000, 3, 2, 0.3, 15, None, None, 300, 1200, 1/10000000, 2000)) # 12 clusters 300-1200
        # (nx.LFR_benchmark_graph, (6000, 3, 2, 0.3, 15, None, None, 450, 1500, 1/10000000, 2000)), # 9 clusters 450-1500
        # (nx.LFR_benchmark_graph, (6000, 3, 2, 0.3, 15, None, None, 600, 1800, 1/10000000, 2000)), # 6 clusters 600-1800
        # (nx.LFR_benchmark_graph, (6000, 3, 2, 0.3, 10, None, None, 900, 2400, 1/10000000, 2000)), # 4 clusters 900-2400

        # (nx.LFR_benchmark_graph, (10000, 3, 2, 0.3, 15, None, None, 750, 2000, 1/10000000, 2000)), # 13 clusters size 500-2000
        # (nx.LFR_benchmark_graph, (10000, 3, 2, 0.3, None, 10, None, 1500, 4000, 1/10000000, 2000)), # 4 clusters size 1500-4000
        # (nx.LFR_benchmark_graph, (15000, 3, 2, 0.3, 10, None, None, 750, 3000, 1/10000000, 2000)), #(n=6000, tau1=3, tau2=2, mu=0.3, average_degree=25, min_community=300, max_community=1200))
    ]

    noises = [
        0.00,
        # 0.01,
        # 0.02,
        # 0.03
    ]    
@ex.named_config
def synth_benchmarks_grampa():
    iters = 10
    run = [
        10,
        13
    ]

    graphs = [
        (nx.gnp_random_graph, (4000, p)) for p in [0.01, 0.05, 0.1, 0.25, 0.5]
    ]
    graph_names = [f'ER(p={p})' for p in [0.01, 0.05, 0.1, 0.25, 0.5]]

    noises = [
        0.00,
    ]    

@ex.named_config
def ncomps():
    run = [
        # 12, 
        13,
    ]
    # _GrampaS_args['n_comp'] = 10
    _ALHPA_args['n_comp'] = 100
    _ALHPA_args['n_comp'] = 100

    iters = 1

    graph_names = [             # n     / e
        "socfb-Cornell5",         # 18.6K / 79K / connected,
        # "socfb-BU10"              # 19.6K / 637.5K / connected
    ]
    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        # 0.01,
        # 0.02,
        # 0.03,
        # 0.04,
        # 0.05,
    ]

@ex.named_config
def rsc_small():
    run = [
        10, 
        # 12,        
        13,
    ]
    _ALHPA_args['rsc'] = 50
    iters = 10
    graph_names = [             # n     / e
        "in-arenas",               # 1.1k  / 5.4k  / connected
        "inf-euroroad",
        "socfb-Haverford76",       # 1.4K  / 59.6K / connected
        "socfb-Swarthmore42",      # 1.7K  / 61.1K / disc - only 2
        "soc-hamsterster",
        "socfb-Bowdoin47",         # 2.3K  / 84.4K / disc - only 2
        "socfb-Hamilton46",        # 2.3K  / 96.4K / disc - only 2        
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ]

@ex.named_config
def rsc_small2():
    run = [
        # 10, 
        # 12,        
        13,
    ]
    # _GrampaS_args['rsc'] = 50
    _ALHPA_args['rsc'] = 1000

    iters = 10

    graph_names = [             # n     / e
        "in-arenas",               # 1.1k  / 5.4k  / connected
        "inf-euroroad",
        "socfb-Haverford76",       # 1.4K  / 59.6K / connected
        "socfb-Swarthmore42",      # 1.7K  / 61.1K / disc - only 2
        "soc-hamsterster",
        "socfb-Bowdoin47",         # 2.3K  / 84.4K / disc - only 2
        "socfb-Hamilton46",        # 2.3K  / 96.4K / disc - only 2
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ]    

    
@ex.named_config
def rsc_large():
    tmp = [
        13,  # ALHPA
        [
            {'rsc': x} for x in [2500, 4000, 6000, 8000]
        ]
    ]
    _algs[:] = alggs(tmp)
    run = list(range(len(tmp[1])))

    iters = 5

    graph_names = [             # n     / e
        "socfb-Cornell5",       # 18.6K / 79K / connected,
        "socfb-BU10",           # 19.6K / 637.5K / connected
        "CA-AstroPh",           # 18k   / 195k  / connected
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
    ]

    

@ex.named_config
def socials():
    iters = 3

    run = [
        10,
        12,
        13,
    ]

    _ALHPA_args['rsc'] = 0 # np.sqrt(|V|)*40    

    graph_names = [               # n     / e
        "soc-facebook",           # 4k    / 87k   / connected
        "socfb-Cornell5",         # 18.6K / 79K / connected,
        "socfb-BU10"              # 19.6K / 637.5K / connected
        # "fb-wosn",                # 63.4K / 817K / connected
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        # 0.05,
        # 0.10,
    ]

@ex.named_config
def socials_large():
    iters = 10

    run = [
           13,
    ]

    _ALHPA_args['rsc'] = 0 # np.sqrt(|V|)*40    

    graph_names = [               # n     / e
        # "soc-facebook",           # 4k    / 87k   / connected
        # "socfb-Cornell5",         # 18.6K / 79K / connected,
        # "socfb-BU10"              # 19.6K / 637.5K / connected
        "fb-wosn",                # 63.4K / 817K / connected
    ]

    graphs = rgraphs(graph_names)

    noises = [
        0.00,
        0.05,
        0.10,
    ]
    
@ex.named_config
def synthetic():

    # use with 'mall'

    iters = 1
    run = [
        # 10, 
        # 12,
        13,
    ]
    graph_names = [
        # "ER(2000)",
        # "ER(4000)",
        # "ER(5000)",
        # "SBM(2000)",
        # "SBM(4000)",
        "SBM(4000k5_.5)",
        "SBM(4000k5_.5)",        
        "SBM(4000k5_.5)",
    ]

    graphs = [
        # (nx.gnp_random_graph, (2000, 0.5)),
        # (nx.gnp_random_graph, (4000, 0.5)),
        # (nx.gnp_random_graph, (5000, 0.5)),
        # (generate_sbm, (2000, .85, 5)),
        # (generate_sbm, (4000, .85, 5)),
        # (generate_sbm, (6000, .5, 5)),
        (generate_sbm, (4000, .35, .15, 4)),
        (generate_sbm, (4000, .4, .15, 5)),
        (generate_sbm, (4000, .5, .15, 5)),

    ]

    noises = [
        0.00,
        0.05,
        # 0.10
    ]


@ex.named_config
def tuned():
    _CONE_args["dim"] = 512
    _LREA_args["iters"] = 40
    _ISO_args["alpha"] = 0.9
    _ISO_args["lalpha"] = 100000  # full dim
    # _ISO_args["lalpha"] = 25


@ex.named_config
def test():

    graph_names = [
        "test1",
        "test2",
    ]

    graphs = [
        # (gen.loadnx, ('data/arenas.txt',)),
        (nx.gnp_random_graph, (50, 0.5)),
        (nx.barabasi_albert_graph, (50, 3)),
    ]

    run = [1, 3, 5]

    iters = 4

    noises = [
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
    ]
