from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class EarlyStoppingVerbose(EarlyStopping):
    """
        The default EarlyStopping callback's verbose mode is too verbose.
        This class outputs a message only when it's getting ready to stop. 
    """
    def _evalute_stopping_criteria(self, *args, **kwargs):
        should_stop, reason = super()._evalute_stopping_criteria(*args, **kwargs)
        if(should_stop):
            rank_zero_info(f"{reason}\n")

        return should_stop, reason
