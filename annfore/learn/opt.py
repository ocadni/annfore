import torch

def make_opt(params, lr=1e-3, opt="adam", alpha=0.99,
             momentum=0.9, betas=(0.9, 0.999), weight_decay=0):
    if opt == "SGD":
        optimizer = torch.optim.SGD(params, lr=lr)
    elif opt == 'sgdm':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    elif opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, alpha=alpha)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=lr,
                                     betas=betas, weight_decay=weight_decay)
    
    elif opt == 'adam_amsgrad':
        optimizer = torch.optim.Adam(params,
                                     lr=lr,
                                     betas=betas,
                                    amsgrad=True)

    else:
        print("optimizer not found, setted Adam")
        optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    return optimizer

