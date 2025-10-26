import os
import sys

from ssbenchmark.ssmodels.base_model import Registries, SSMethod
from ssbenchmark.utils import canonicalize_smiles, split_reactions
import torch
import numpy as np

@Registries.ModelChoice.register(key="retrostar_mlp")
class retrostar_mlp(SSMethod):
    """Benchmarking Retrosynthesis with the MLP model from RetroStar.
    Adapted from: https://github.com/AustinT/syntheseus-retro-star-benchmark
    """
    def __init__(self, module_path=None):
        self.model_name = "retrostar_mlp"
        if module_path is not None:
            sys.path.insert(len(sys.path), os.path.abspath(module_path))

    def preprocess(self):
        pass

    def process_input(self):
        pass

    def preprocess_store(self):
        pass

    def process_output(self):
        pass

    def model_setup(self, use_gpu=False, **kwargs):
        
        model_checkpoint = kwargs["model_checkpoint"]
        template_file = kwargs["template_file"]
        self.expansion_topk = kwargs["expansion_topk"]

        print(f"Settings: Model: {model_checkpoint}, Template: {template_file}, Cut-off: {self.expansion_topk}")

        from syntheseus_retro_star_benchmark.original_code.mlp_inference import MLPModel
        device = -1

        print("Loading model RETROSTAR MLP")

        device = 0 if torch.cuda.is_available() else -1
        self.model = MLPModel(model_checkpoint, template_file, device=device)
        self.model.net.eval()  # ensure eval mode

    def _model_call(self, X):
        smiles_list = X
        assert len(smiles_list) == 1, "Only one molecule can be processed at a time"
        
        output_dict = self.model.run(smiles_list[0], topk=self.expansion_topk)
        if output_dict is not None:  # could be None if no reactions are possible
            reactants = output_dict["reactants"]
            scores = output_dict["scores"]
            scores = np.clip(np.asarray(scores), 1e-3, 1.0)  # done by original paper
            templates = output_dict["template"]
        else:
            reactants = []
            scores = []
            templates = []
        
        # azf expects a lists as return
        return [reactants], [scores] #, templates
    
    def model_call(self, X):
        return self._model_call(X)