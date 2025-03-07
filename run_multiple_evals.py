'''run evaluations on multiple models'''

import os

models_to_evaluate = [

    # model trained on BC1Song
    "UMSS_4s_bc1song",
    "unet_4s_bc1song",
    # "Sf-Sf_bc1song",
    # "Sft-Sft_bc1song",
    # "Sf-Sft_bc1song",
    # "W-Up_bc1song",
    
    # model trained on BCBQ
    "UMSS_4s_bcbq",
    "unet_4s_bcbq",
    # "Sf-Sf_bcbq",
    # "Sft-Sft_bcbq",
    # "Sf-Sft_bcbq",
    # "W-Up_bcbq",
]

eval_mode='default' # default evaluation
# eval_mode='fast' # fast evaluation



for tag in models_to_evaluate:
    
    if eval_mode=='original_paper':
        command=f"python eval.py --tag '{tag}' --f0-from-mix --test-set 'CSD'"

    elif eval_mode=='default':
        # mf0 extract with Crepe
        command=f"python eval.py --tag '{tag}' --test-set 'CSD' --show-progress --compute all"
        
        # mf0 extract with Cuesta et al. (model 3) model
        # command=f"python eval.py --tag '{tag}' --f0-from-mix --test-set 'CSD' --show-progress --compute all"
    
    elif eval_mode=='fast':
        command=f"python eval.py --tag '{tag}' --f0-from-mix --test-set 'CSD' --show-progress --compute SI-SDR_mask"

    else:
        raise ValueError("eval_mode not recognized. Please choose 'default' or 'fast'")

    print(command)
    os.system(command)
