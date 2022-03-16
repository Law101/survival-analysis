import pandas as pd
from pathlib import Path
from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
import train
import preprocess

artifact_path = Path('/mnt/c/Users/Lawrence/Downloads/Lawrence/survival-analysis/artifact/model.zip')

def training(data):
    encoded_data = preprocess.one_hot_encoder(data)
    X_train, T_train, E_train, X_test, T_test, E_test = preprocess.data_splitter(encoded_data)

    # Define Model
    model = ConditionalSurvivalForestModel(num_trees=200)

    trainer = train.Trainer(model)
    trainer.train_step(X_train, T_train, E_train)

    # Evaluation
    scores = trainer.eval_step(artifact_path=artifact_path, X=X_test, T=T_test, E=E_test)

    return scores

data_path = Path('/mnt/c/Users/Lawrence/Downloads/Lawrence/survival-analysis/data/customer_churn.csv')
print(data_path)

data = pd.read_csv(data_path)

training(data)