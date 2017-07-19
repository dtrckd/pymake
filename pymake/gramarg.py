from functools import partial
from pymake.expe.gram import exp_append, exp_append_uniq

_gram = [
    #  Expe Settings -- Context-Free
    #
    #  * Are repeatable

    '--epoch', dict(
        nargs='*', action=partial(exp_append, _t=int),
        help='number for averaginf generative process'),

    '-c','--corpus', dict(
        nargs='*', dest='corpus', action=exp_append,
        help='ID of the frontend data.'),

    '-r','--random', dict(
        nargs='*', dest='corpus', action=exp_append,
        help='Random generation of synthetic frontend  data [uniforma|alternate|cliqueN|BA].'),

    '-m','--model', dict(
        nargs='*',dest='model', action=exp_append,
        help='ID of the model.'),

    '-n','--N', dict(
        nargs='*', action=exp_append, # str because keywords "all"
        help='Size of frontend data [int | all].'),

    '-k','--K', dict(
        nargs='*', action=partial(exp_append, _t=int),
        help='Latent dimensions'),

    '-i','--iterations', dict(
        nargs='*', action=partial(exp_append, _t=int),
        help='Max number of iterations for the optimization.'),

    '--repeat', dict(
        nargs='*', action=exp_append, #type=check_positive_integer,
        help='Index of tn nth repetitions/randomization of an design of experiments. Impact the outpout path as data/<bdir>/<refdir>/<repeat>/...'),

    '--hyper',  dict(
        dest='hyper', nargs='*', action=exp_append,
        help='type of hyperparameters optimization [auto|fix|symmetric|asymmetric]'),

    '--hyper-prior','--hyper_prior', dict(
        dest='hyper_prior', action=partial(exp_append_uniq, _t=float), nargs='*',
        help='Set paramters of the hyper-optimization [auto|fix|symmetric|asymmetric]'),

    '--testset-ratio', dict(
        dest='testset_ratio', nargs='*', action=partial(exp_append, _t=int),
        help='testset/learnset percentage for testing.'),

    '--homo', dict(
        nargs='*', action=exp_append,
        help='Centrality type (NotImplemented)'),

    '--alpha', dict(
        nargs='*', action=partial(exp_append, _t=float),
        help='First hyperparameter.'),
    '--gmma', dict(
        nargs='*', action=partial(exp_append, _t=float),
        help='Second hyperparameter.'),
    '--delta', dict(
        nargs='*', action=partial(exp_append, _t=float),
        help='Third hyperparameter.'),
    '--chunk', dict(
        nargs='*', action=partial(exp_append, _t=float),
        help='Chunk size for online learning.'),
    '--burnin', dict(
        nargs='*', action=partial(exp_append, _t=int),
        help='Number of samples used for burnin period.'),

    '-g', '--generative',dict(
        dest='_mode', action='store_const', const='generative'),
    '-p', '--predictive', dict(
        dest='_mode', action='store_const', const='predictive'),

]
