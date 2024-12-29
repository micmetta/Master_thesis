from transformers import StoppingCriteria

# Custom class to stop generation when a stop token present in a list is generated
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        """
        stop_token_ids: List of token IDs that cause termination.
        """
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        """
        Checks whether the last generated token belongs to the termination token list.
        """
        return input_ids[0, -1].item() in self.stop_token_ids