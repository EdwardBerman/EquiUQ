import numpy as np
import torch

import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from models import SO2MLP
import time

from escnn import gspaces
from escnn import nn
from escnn import group

def set_all_seeds(seed: int = 10): 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def beta_nll_loss(mean, variance, target, beta=1.0):
    """Compute beta-NLL loss

    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative
        weighting between data points, where `0` corresponds to
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * (variance.detach() ** beta)

    return loss.sum(axis=-1)

def mse_loss(mean, target):
    """Compute MSE loss

    :param mean: Predicted mean of shape B x D
    :param target: Target of shape B x D
    :returns: Loss per batch element of shape B
    """
    return 0.5 * ((target - mean) ** 2).sum(axis=-1)

def combined_loss(mean, variance, target, beta=1.0):
    return beta_nll_loss(mean, variance, target, beta) + mse_loss(mean, target)

def train_model(model, input_positions, vector_field, n_holdout, 
                minimum_epochs, maximum_epochs_no_improve, maximum_epochs, 
                batch_size, train_val_split, beta, save_dir):
    """
    Train a model for vector field regression.
    
    Args:
        model: The model to train
        model_type: Type of model ('correct_equivariant', 'incorrect_equivariant', 'mlp')
        key: JAX random key
        input_positions: Input positions (x,y coordinates)
        vector_field: Vector field values at each position
        n_holdout: Number of samples to hold out for testing
        minimum_epochs: Minimum number of epochs to train
        maximum_epochs_no_improve: Maximum number of epochs with no improvement before early stopping
        maximum_epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        train_val_split: Proportion of non-test data to use for training
        beta: Weight for the beta-NLL loss
        save_dir: Directory to save results
    """
    n_samples = input_positions.shape[0]
    
    # Split data into train, validation, and test sets
    n_test = n_holdout
    n_train_val = n_samples - n_test
    
    n_train = int(n_train_val * train_val_split)
    n_val = n_train_val - n_train
    
    perm = torch.randperm(n_samples)
    train_indices = perm[:n_train]
    val_indices = perm[n_train:n_train + n_val]
    test_indices = perm[n_train_val:]
    
    train_inputs = input_positions[train_indices]
    train_targets = vector_field[train_indices]
    
    val_inputs = input_positions[val_indices]
    val_targets = vector_field[val_indices]
    
    test_inputs = input_positions[test_indices]
    test_targets = vector_field[test_indices]
    
    best_val_loss = torch.finfo(torch.float32).max
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(maximum_epochs):
        train_loss = 0
        
        for i in range(0, n_train, batch_size):
            batch_indices = train_indices[i:i+batch_size]
            batch_inputs = train_inputs[batch_indices]
            batch_targets = train_targets[batch_indices]

            mean, variance = model(batch_inputs)

            loss = combined_loss(mean, variance, batch_targets)
            loss = loss.mean()
            train_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        
        train_loss /= np.ceil(n_train / batch_size)
        
        val_prediction_mu, val_prediction_sigma_sq = model(val_inputs)
        
        
        val_loss = combined_loss(val_prediction_mu, val_prediction_sigma_sq, val_targets).mean().item()
        
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        
        if epoch >= minimum_epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve > maximum_epochs_no_improve:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    test_prediction_mu, test_prediction_sigma_sq = model(test_inputs)
    test_mse = torch.mean((test_prediction_mu - test_targets) ** 2).item()
    test_nll = beta_nll_loss(test_prediction_mu, test_prediction_sigma_sq, test_targets).mean().item()
    
    print(f"Test MSE: {test_mse:.4f}, Test NLL: {test_nll:.4f}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    test_prediction_mu_np = np.array(test_prediction_mu_array)
    test_prediction_sigma_sq_np = np.array(test_prediction_sigma_sq_array)
    test_targets_np = np.array(test_targets_array)
    
    test_inputs_np = np.array(test_inputs)
   
    np.save(f"{save_dir}/test_prediction_mu_{model_type}.npy", test_prediction_mu_np)
    np.save(f"{save_dir}/test_prediction_sigma_sq_{model_type}.npy", test_prediction_sigma_sq_np)
    np.save(f"{save_dir}/test_targets_{model_type}.npy", test_targets_np)
    np.save(f"{save_dir}/test_inputs_{model_type}.npy", test_inputs_np)
    
    # Save training history
    np.save(f"{save_dir}/train_losses_{model_type}.npy", np.array(train_losses))
    np.save(f"{save_dir}/val_losses_{model_type}.npy", np.array(val_losses))
    
    return {
        'model_type': model_type,
        'test_mse': float(test_mse),
        'test_nll': float(test_nll),
        'test_predictions_mu': test_prediction_mu_np,
        'test_predictions_sigma_sq': test_prediction_sigma_sq_np
    }

def test_with_rotations(model, model_type, parameters, test_inputs, test_targets, n_rotations=8, save_dir=None):
    """
    Test the model with rotated inputs to analyze equivariance properties.
    
    Args:
        model: The trained model
        model_type: Type of model ('correct_equivariant', 'incorrect_equivariant', 'mlp')
        parameters: Model parameters
        test_inputs: Test inputs
        test_targets: Test targets
        n_rotations: Number of rotations to test
        save_dir: Directory to save results
    """
    rotation_angles = np.linspace(0, 2*np.pi, n_rotations, endpoint=False)
    mse_results = []
    nll_results = []
    calibration_results = []
    
    
    for angle in rotation_angles:
        # Create rotation matrix
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        if is_equivariant:
            # Extract arrays from IrrepsArray if needed
            if hasattr(test_inputs, 'array'):
                inputs_array = test_inputs.array[:, :2]  # Extract x,y components
                targets_array = test_targets.array[:, :2]  # Extract x,y components
            else:
                inputs_array = test_inputs[:, :2]
                targets_array = test_targets[:, :2]
                
            # Rotate inputs
            rotated_inputs = np.einsum('ij,nj->ni', rotation_matrix, inputs_array)
            
            rotated_inputs_padded = np.pad(rotated_inputs, ((0, 0), (0, 1)), mode="constant")
            rotated_inputs_irreps = IrrepsArray(e3nn.Irreps("1x1e"), rotated_inputs_padded)
            
            # get predictions
            pred_mu, pred_sigma_sq = model.apply(parameters, rotated_inputs_irreps)
            
            # extract arrays from predictions
            if hasattr(pred_mu, 'array'):
                pred_mu_array = pred_mu.array[:, :2]  # Use only x,y components
                pred_sigma_sq_array = pred_sigma_sq.array[:, :2]
            else:
                pred_mu_array = pred_mu[:, :2]
                pred_sigma_sq_array = pred_sigma_sq[:, :2]
            
            # rotate the targets appropriately for evaluation
            rotated_targets = np.einsum('ij,nj->ni', rotation_matrix, targets_array)
            
        else:
            # For MLP, we rotate the inputs and apply the inverse rotation to outputs
            rotated_inputs = np.einsum('ij,nj->ni', rotation_matrix, test_inputs)
            
            # Get predictions
            pred_mu, pred_sigma_sq = model.apply(parameters, rotated_inputs)
            
            # For non-equivariant models, we need to rotate the predictions back
            # to compare with the original targets
            pred_mu_array = np.einsum('ji,nj->ni', rotation_matrix, pred_mu)
            pred_sigma_sq_array = pred_sigma_sq  # Uncertainty doesn't need rotation
            
            # Original targets (no rotation needed)
            rotated_targets = test_targets
        
        # Calculate metrics
        mse = np.mean((pred_mu_array - rotated_targets) ** 2)
        nll = np.mean(((pred_mu_array - rotated_targets) ** 2 / (2 * pred_sigma_sq_array)) + 0.5 * np.log(pred_sigma_sq_array))
        
        # Calculate calibration metric (expected vs actual error)
        expected_error = np.sqrt(pred_sigma_sq_array)
        actual_error = np.abs(pred_mu_array - rotated_targets)
        calibration = np.mean(actual_error / (expected_error + 1e-6))
        
        mse_results.append(mse)
        nll_results.append(nll)
        calibration_results.append(calibration)
    
    # Save results
    if save_dir:
        np.save(f"{save_dir}/rotation_angles_{model_type}.npy", rotation_angles)
        np.save(f"{save_dir}/rotation_mse_{model_type}.npy", np.array(mse_results))
        np.save(f"{save_dir}/rotation_nll_{model_type}.npy", np.array(nll_results))
        np.save(f"{save_dir}/rotation_calibration_{model_type}.npy", np.array(calibration_results))
    
    return {
        'angles': rotation_angles,
        'mse': mse_results,
        'nll': nll_results,
        'calibration': calibration_results
    }

def visualize_predictions(test_inputs, test_targets, predictions_mu, predictions_sigma_sq, title, save_path=None):
    """
    Visualize vector field predictions with uncertainty.
    
    Args:
        test_inputs: Test input positions
        test_targets: True vector field
        predictions_mu: Predicted vector field
        predictions_sigma_sq: Predicted uncertainty
        title: Plot title
        save_path: Path to save the figure
    """
    # Extract arrays if needed
    if hasattr(test_inputs, 'array'):
        test_inputs_array = test_inputs.array
    else:
        test_inputs_array = test_inputs
        
    if hasattr(test_targets, 'array'):
        test_targets_array = test_targets.array
    else:
        test_targets_array = test_targets
        
    if hasattr(predictions_mu, 'array'):
        predictions_mu_array = predictions_mu.array
        predictions_sigma_sq_array = predictions_sigma_sq.array
    else:
        predictions_mu_array = predictions_mu
        predictions_sigma_sq_array = predictions_sigma_sq
    
    # Extract 2D components if inputs are in 3D
    if test_inputs_array.shape[1] > 2:
        test_inputs_array = test_inputs_array[:, :2]
    if test_targets_array.shape[1] > 2:
        test_targets_array = test_targets_array[:, :2]
    if predictions_mu_array.shape[1] > 2:
        predictions_mu_array = predictions_mu_array[:, :2]
    if predictions_sigma_sq_array.shape[1] > 2:
        predictions_sigma_sq_array = predictions_sigma_sq_array[:, :2]
    
    # Calculate uncertainty as standard deviation
    predictions_sigma = np.sqrt(predictions_sigma_sq_array)
    #  uncertainty = np.mean(predictions_sigma, axis=1)
    uncertainty = np.linalg.norm(predictions_sigma, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot ground truth
    ax[0].quiver(test_inputs_array[:, 0], test_inputs_array[:, 1], 
                test_targets_array[:, 0], test_targets_array[:, 1], 
                angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.8)
    ax[0].set_title('Ground Truth Vector Field')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].grid(True)
    ax[0].axis('equal')
    
    # Plot predictions with uncertainty
    scatter = ax[1].quiver(test_inputs_array[:, 0], test_inputs_array[:, 1], 
                         predictions_mu_array[:, 0], predictions_mu_array[:, 1], 
                         uncertainty, cmap='viridis', 
                         angles='xy', scale_units='xy', scale=1, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax[1])
    cbar.set_label('Uncertainty (σ)')
    ax[1].set_title(f'Predicted Vector Field with Uncertainty - {title}')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[1].grid(True)
    ax[1].axis('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_rotation_results(rotation_results, save_dir=None):
    """
    Plot results of rotation tests.
    
    Args:
        rotation_results: Dictionary of rotation test results for different models
        save_dir: Directory to save results
    """
    model_types = list(rotation_results.keys())
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    
    # Extract angles (they should be the same for all models)
    angles_deg = np.degrees(rotation_results[model_types[0]]['angles'])
    
    # Plot MSE
    for model_type in model_types:
        ax[0].plot(angles_deg, rotation_results[model_type]['mse'], 
                  label=model_type, marker='o')
    ax[0].set_title('MSE vs. Rotation Angle')
    ax[0].set_xlabel('Rotation Angle (degrees)')
    ax[0].set_ylabel('Mean Squared Error')
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot NLL
    for model_type in model_types:
        ax[1].plot(angles_deg, rotation_results[model_type]['nll'], 
                  label=model_type, marker='o')
    ax[1].set_title('NLL vs. Rotation Angle')
    ax[1].set_xlabel('Rotation Angle (degrees)')
    ax[1].set_ylabel('Negative Log-Likelihood')
    ax[1].legend()
    ax[1].grid(True)
    
    # Plot Calibration
    for model_type in model_types:
        ax[2].plot(angles_deg, rotation_results[model_type]['calibration'], 
                  label=model_type, marker='o')
    ax[2].axhline(y=1.0, color='r', linestyle='--', label='Perfect Calibration')
    ax[2].set_title('Uncertainty Calibration vs. Rotation Angle')
    ax[2].set_xlabel('Rotation Angle (degrees)')
    ax[2].set_ylabel('Actual Error / Expected Error')
    ax[2].legend()
    ax[2].grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/rotation_results_comparison.png")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    
    @dataclass
    class Options:
        dataset_type: str = "spiral"
        n_samples: int = 2000
        noise: float = 0.1
        hidden_dim: int = 32
        batch_size: int = 64
        beta: float = 1.0
        n_holdout: int = 200
        minimum_epochs: int = 10
        maximum_epochs_no_improve: int = 10
        maximum_epochs: int = 100
        train_val_split: float = 0.8
        n_rotations: int = 8
        data_dir: str = "../data/vector_field"
        results_dir: str = "../results/vector_field"
        random_seed: int = 42
    
    parser.add_arguments(Options, dest="options")
    args = parser.parse_args()
    
    # Extract arguments
    dataset_type = args.options.dataset_type
    n_samples = args.options.n_samples
    noise = args.options.noise
    hidden_dim = args.options.hidden_dim
    batch_size = args.options.batch_size
    beta = args.options.beta
    n_holdout = args.options.n_holdout
    minimum_epochs = args.options.minimum_epochs
    maximum_epochs_no_improve = args.options.maximum_epochs_no_improve
    maximum_epochs = args.options.maximum_epochs
    train_val_split = args.options.train_val_split
    n_rotations = args.options.n_rotations
    data_dir = args.options.data_dir
    results_dir = args.options.results_dir
    random_seed = args.options.random_seed
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    data_file_prefix = f"{data_dir}/input_positions_{dataset_type}_{n_samples}_samples_{noise}_noise"
    if os.path.exists(f"{data_file_prefix}.npy"):
        print(f"Loading existing dataset from {data_file_prefix}")
        input_positions = np.load(f"{data_file_prefix}.npy")
        vector_field = np.load(f"{data_dir}/vector_field_{dataset_type}_{n_samples}_samples_{noise}_noise.npy")
    else:
        print(f"Generating new dataset of type {dataset_type}")
        from make_vector_field_dataset import create_synthetic_vector_field
        input_positions, vector_field = create_synthetic_vector_field(n_samples, noise, dataset_type)
        
        np.save(f"{data_file_prefix}.npy", input_positions)
        np.save(f"{data_dir}/vector_field_{dataset_type}_{n_samples}_samples_{noise}_noise.npy", vector_field)
    
    SO2MLP = SO2MLP()
    
    print("\n--- Training SO2 Equivariant Model ---")
    correct_results = train_model(
        SO2MLP, input_positions, vector_field,
        n_holdout, minimum_epochs, maximum_epochs_no_improve, maximum_epochs,
        batch_size, train_val_split, beta, results_dir
    )
    
    
    print("\n--- Testing Models with Rotations ---")
    
    test_indices = np.arange(n_samples - n_holdout, n_samples)
    
    # Test data for each model type
    correct_test_inputs = np.load(f"{results_dir}/test_inputs_correct_equivariant.npy")
    correct_test_targets = np.load(f"{results_dir}/test_targets_correct_equivariant.npy")
    
    incorrect_test_inputs = np.load(f"{results_dir}/test_inputs_incorrect_equivariant.npy")
    incorrect_test_targets = np.load(f"{results_dir}/test_targets_incorrect_equivariant.npy")
    
    mlp_test_inputs = np.load(f"{results_dir}/test_inputs_mlp.npy")
    mlp_test_targets = np.load(f"{results_dir}/test_targets_mlp.npy")
    
    # Convert to IrrepsArray for equivariant models
    correct_test_inputs_irreps = IrrepsArray(input_irreps, correct_test_inputs)
    correct_test_targets_irreps = IrrepsArray(output_irreps, correct_test_targets)
    
    incorrect_test_inputs_irreps = IrrepsArray(input_irreps, incorrect_test_inputs)
    incorrect_test_targets_irreps = IrrepsArray(output_irreps, incorrect_test_targets)
    
    # Perform rotation tests
    correct_rotation_results = test_with_rotations(
        correct_model, 'correct_equivariant', 
        correct_results['parameters'],
        correct_test_inputs_irreps, correct_test_targets_irreps,
        n_rotations, results_dir
    )
    
    incorrect_rotation_results = test_with_rotations(
        incorrect_model, 'incorrect_equivariant', 
        incorrect_results['parameters'],
        incorrect_test_inputs_irreps, incorrect_test_targets_irreps,
        n_rotations, results_dir
    )
    
    mlp_rotation_results = test_with_rotations(
        mlp_model, 'mlp', 
        mlp_results['parameters'],
        mlp_test_inputs, mlp_test_targets,
        n_rotations, results_dir
    )
    
    # Create visualizations
    print("\n--- Creating Visualizations ---")
    
    # Load predictions for visualization
    SO2MLP_pred_mu = np.load(f"{results_dir}/test_prediction_mu_correct_equivariant.npy")
    SO2MLP_pred_sigma_sq = np.load(f"{results_dir}/test_prediction_sigma_sq_correct_equivariant.npy")
    
    
    # Visualize predictions
    visualize_predictions(
        correct_test_inputs, correct_test_targets,
        SO2MLP_pred_mu, SO2MLP_pred_sigma_sq,
        "Correct Equivariant Model",
        f"{results_dir}/equivariant_predictions.png"
    )
    
    
    rotation_results = {
        'Correct Equivariant': correct_rotation_results,
    }
    
    plot_rotation_results(rotation_results, results_dir)
    
