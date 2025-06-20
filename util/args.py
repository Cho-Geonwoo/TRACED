# Created for the TRACED project
# Note: Author information anonymized for double-blind review.

def get_method_from_args(args):
    method = None
    # DR
    if (
        not args.use_plr
        and not args.no_exploratory_grad_updates
        and not args.use_editor
    ):
        method = "DR"
    elif args.use_traced:
        method = "TRACED"
    # PLR
    elif args.use_plr and not args.no_exploratory_grad_updates and not args.use_editor:
        method = "PLR"
    # PLR⊥
    elif args.use_plr and args.no_exploratory_grad_updates and not args.use_editor:
        method = "PLR⊥"
    # ACCEL
    elif args.use_plr and args.no_exploratory_grad_updates and args.use_editor:
        method = "ACCEL"
    else:
        method = "Unknown"
    return method


def args_to_dict(args):
    args_dict = vars(args)
    args_dict["env_name"] = args.env_name
    args_dict["use_plr"] = args.use_plr
    args_dict["use_reset_random_dr"] = args.use_reset_random_dr
    return args_dict
