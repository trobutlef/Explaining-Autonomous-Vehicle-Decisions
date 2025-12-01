import numpy as np
import shap
import lime
import lime.lime_tabular
import torch

def explain_with_shap(model, input_data, background_data):
    """
    Compute SHAP values for tabular/state features using DeepExplainer or KernelExplainer.
    
    Args:
        model: PyTorch model (nn.Module)
        input_data: Single input instance (numpy array or torch tensor) [1, obs_dim]
        background_data: Background dataset for SHAP (numpy array or torch tensor) [N, obs_dim]
        
    Returns:
        shap_values: SHAP values for the input_data
        expected_value: The base value (expected output)
    """
    # Ensure inputs are torch tensors
    device = next(model.parameters()).device
    
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.as_tensor(input_data, dtype=torch.float32, device=device)
    else:
        input_tensor = input_data.to(device)
        
    if isinstance(background_data, np.ndarray):
        background_tensor = torch.as_tensor(background_data, dtype=torch.float32, device=device)
    else:
        background_tensor = background_data.to(device)

    # Use DeepExplainer for PyTorch models
    try:
        explainer = shap.DeepExplainer(model, background_tensor)
        shap_values = explainer.shap_values(input_tensor)
    except Exception as e:
        print(f"DeepExplainer failed, falling back to KernelExplainer: {e}")
        # Fallback to KernelExplainer (model agnostic)
        # KernelExplainer expects a function that returns numpy arrays
        def predict_fn(x):
            with torch.no_grad():
                t = torch.as_tensor(x, dtype=torch.float32, device=device)
                return model(t).cpu().numpy()
        
        explainer = shap.KernelExplainer(predict_fn, background_tensor.cpu().numpy())
        shap_values = explainer.shap_values(input_tensor.cpu().numpy())

    return shap_values, explainer.expected_value


def explain_with_lime(model, input_data, training_data, feature_names=None, class_names=None):
    """
    Compute LIME explanation for tabular/state features.
    
    Args:
        model: PyTorch model (nn.Module)
        input_data: Single input instance (numpy array) [obs_dim]
        training_data: Dataset to initialize LIME explainer (numpy array) [N, obs_dim]
        feature_names: List of feature names
        class_names: List of class (action) names
        
    Returns:
        exp: LIME explanation object
    """
    device = next(model.parameters()).device
    
    # LIME expects a function that takes numpy array and returns probabilities (or Q-values)
    def predict_fn(x):
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32, device=device)
            return model(t).cpu().numpy()

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data,
        feature_names=feature_names,
        class_names=class_names,
        mode='regression', # DQN outputs Q-values (regression), not probabilities
        discretize_continuous=True
    )
    
    # input_data should be 1D for LIME
    if input_data.ndim > 1:
        input_data = input_data.flatten()
        
    exp = explainer.explain_instance(
        input_data,
        predict_fn,
        num_features=len(input_data) if feature_names is None else len(feature_names)
    )
    
    return exp
