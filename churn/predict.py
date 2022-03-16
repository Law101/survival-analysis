from pysurvival.utils import load_model
    
def predict(X, artifact_path):
    # Load artifact
    model = load_model(artifact_path)
    # compute every customer's survival function and risk score for 4 chosen periods (active months)
    T = [1,3,6,12]
    for t in T:
        # survival function over time
        svf = model.predict_survival(X.values, t=t)     
        X["svf" + str(t)] = svf

    # risk score
    X["risk"] = model.predict_risk(X.values)

    return X