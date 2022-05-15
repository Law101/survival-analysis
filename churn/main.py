import pandas as pd
from pathlib import Path
from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
import train
import preprocess

# Define project base directory
def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent


def training(data, artifact_path):
    encoded_data = preprocess.one_hot_encoder(data)
    X_train, T_train, E_train, X_test, T_test, E_test = preprocess.data_splitter(encoded_data)
    # Define Model
    model = ConditionalSurvivalForestModel(num_trees=200)
    trainer = train.Trainer(model)
    trainer.train_step(X_train, T_train, E_train, artifact_path)
    # Evaluation
    scores = trainer.eval_step(X=X_test, T=T_test, E=E_test, artifact_path=artifact_path)
    return scores


if __name__ == '__main__':

    PROJECT_BASE = get_project_root()
    artifact_path = Path.joinpath(PROJECT_BASE, "artifact/model.zip")
    data_path = Path.joinpath(PROJECT_BASE, "data/customer_churn.csv")
    data = pd.read_csv(data_path)
    training(data, artifact_path)