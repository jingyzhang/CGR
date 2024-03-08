

def get_option_setter(trainer):
    """Return the static method <modify_commandline_options> of the model class."""
    return trainer.modify_commandline_options