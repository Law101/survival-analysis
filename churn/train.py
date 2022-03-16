import numpy as np
import pandas as pd
from pysurvival.utils import save_model, load_model
from pysurvival.utils.display import integrated_brier_score, create_risk_groups
from pysurvival.utils.metrics import concordance_index
from pathlib import Path


artifact_path = Path('/mnt/c/Users/Lawrence/Downloads/Lawrence/survival-analysis/artifact/model.zip')
class Trainer:

    def __init__(self, model):

        # Set Parameters
        self.model = model
        # self.eval_metric = eval_metric

    def train_step(self,  X, T, E):
        # fitting a survival forest model
        self.model.fit(
                X, 
                T, 
                E, 
                max_features="sqrt",    # number of features randomly chosen at each split (int, float, "sqrt", "log2", or "all")
                max_depth=5,            #
                min_node_size=20,       # minimum number of samples at leaf node
                alpha=0.05,             # significance threshold to allow splitting 
                seed=42,                # seed for random variable generator
                sample_size_pct=0.63,   # % of original samples used in each tree building
                num_threads=-1          # number of jobs to run; if -1, then all available cores will be used
                )

        # Save artifacts
        save_model(self.model, artifact_path)        
        return {"Model Finsihed Training"}


    def eval_step(self, artifact_path, X, T, E):
        # Load artifact
        model = load_model(artifact_path)
        # testing: model quality - prediction error measures by concordance index and Brier score
        ci = concordance_index(model, X, T, E)
        print("concordance index: {:.2f}".format(ci))

        ibs = integrated_brier_score(model, X, T, E, t_max=12)
        print("integrated Brier score: {:.2f}".format(ibs))

        return {"concordance index": "{:.2f}".format(ci),
                "integrated Brier score": "{:.2f}".format(ibs)}
        


