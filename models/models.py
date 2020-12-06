
def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'PFP':
        from .pfp import PFPModel
        model = PFPModel()
    elif opt.model == 'STG':
        from .stg import STGModel
        model = STGModel()
    elif opt.model == 'Final':
        from .crowdgan import CrowdganModel
        model = CrowdganModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model


