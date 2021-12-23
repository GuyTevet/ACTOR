from argparse import ArgumentParser  # noqa


def add_misc_options(parser):
    group = parser.add_argument_group('Miscellaneous options')
    group.add_argument("--expname", default="exps", help="general directory to this experiments, use it if you don't provide folder name")
    group.add_argument("--folder", help="directory name to save models")
    

def add_cuda_options(parser):
    group = parser.add_argument_group('Cuda options')
    group.add_argument("--cuda", dest='cuda', action='store_true', help="if we want to try to use gpu")
    group.add_argument('--cpu', dest='cuda', action='store_false', help="if we want to use cpu")
    group.set_defaults(cuda=True)


def add_experiment_options(parser):
    group = parser.add_argument_group('Experiment options')
    group.add_argument("--exp_name", default='no_exp', choices=['no_exp', 'naive_interp', 'diff_interp', 'smooth'], type=str, help="")
    group.add_argument("--exp_type", default='', type=str, help="")
    group.add_argument("--interp_sample_type", default='non-adaptive', type=str, help="")
    group.add_argument("--exp_param", default=1, type=int, help="1 means no exp")

    
def adding_cuda(parameters):
    import torch
    if parameters["cuda"] and torch.cuda.is_available():
        parameters["device"] = torch.device("cuda")
    else:
        parameters["device"] = torch.device("cpu")
        
    
