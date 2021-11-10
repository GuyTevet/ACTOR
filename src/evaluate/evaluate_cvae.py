from src.parser.evaluation import parser


def main():
    parameters, folder, checkpointname, epoch, niter = parser()

    dataset = parameters["dataset"]
    print(dataset)
    if dataset in ["ntu13", "humanact12"]:
        from src.evaluate.gru_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter, parameters['interp_ratio'], parameters['interp_type'])
    elif dataset in ["uestc"]:
        from .stgcn_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter, parameters['interp_ratio'], parameters['interp_type'])
    else:
        raise NotImplementedError("This dataset is not supported.")


if __name__ == '__main__':
    main()
