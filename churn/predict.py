from pysurvival.utils import load_model
    
def get_prediction(X, artifact_path):
    """_summary_

    Args:
        X (_type_): _description_
        artifact_path (_type_): _description_

    Returns:
        _type_: _description_
    """
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