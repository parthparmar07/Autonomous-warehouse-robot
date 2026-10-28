from src.deployment.policies import load_policy

def predict_actions(observations,masks=None,algorithm='qmix'):
    if masks is None:raise ValueError('Explicit action masks are required')
    model=load_policy(algorithm,len(observations))
    model.reset_hidden_states()
    return model.act(observations,masks)

def inference_source():
    return 'Versioned checkpoint; raw /predict is stateless. Use sessions for recurrent execution.'
