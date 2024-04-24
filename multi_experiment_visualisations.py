import plotly.graph_objs as go
from plotly.subplots import make_subplots
from SCOPE_experiment import SCOPE_experiment
from existing_experiments import existing_experiments

def viz_over_num_trajectories(base_params, num_trajectories):
  # Initialize lists to store values for IS, Train, and Test
  IS_bias_values = []
  Train_bias_values = []
  Test_bias_values = []
  IS_variance_values = []
  Train_variance_values = []
  Test_variance_values = []
  IS_mse_values = []
  Train_mse_values = []
  Test_mse_values = []

  for length in num_trajectories:
      # Create experiment instance and load data
      # test_experiment = SCOPE_experiment(1, 0.4, 1, 0.05, q_table, 0.99, i, 10000, 0.3, smallest_distance_to_deadend,0.1, [16], 0.001, 0.2, 0.00001, 0.00001, 1.0, 1.0, 300, 50, 0.0, torch.float64, "varying_num_trajectories", "/content/drive/MyDrive/Lifegate_experiments")
      params = base_params.copy()  # Make a copy to avoid modifying the base parameters
      params["num_trajectories"] = length  # Update the number of trajectories
      test_experiment = SCOPE_experiment(**params)
      # test_load = existing_experiments(test_experiment,"/content/drive/MyDrive/Lifegate_experiments")
      test_load = existing_experiments(test_experiment) 

      # Get epoch-specific values for IS, Train, and Test
      IS_bias, Train_bias, Test_bias, IS_variance, Train_variance, Test_variance, IS_mse, Train_mse, Test_mse = test_load.epoch_specific_values()

      # Append values to respective lists
      IS_bias_values.append(IS_bias)
      Train_bias_values.append(Train_bias)
      Test_bias_values.append(Test_bias)
      IS_variance_values.append(IS_variance)
      Train_variance_values.append(Train_variance)
      Test_variance_values.append(Test_variance)
      IS_mse_values.append(IS_mse)
      Train_mse_values.append(Train_mse)
      Test_mse_values.append(Test_mse)


  # Plot bias over trajectories
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=num_trajectories, y=IS_bias_values, mode='lines', name='IS Bias'))
  fig.add_trace(go.Scatter(x=num_trajectories, y=Train_bias_values, mode='lines', name='Train Bias'))
  fig.add_trace(go.Scatter(x=num_trajectories, y=Test_bias_values, mode='lines', name='Test Bias'))
  fig.update_layout(title='Bias over Trajectories', xaxis_title='Number of Trajectories', yaxis_title='Bias')
  fig.show()

  # Plot variance over trajectories
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=num_trajectories, y=IS_variance_values, mode='lines', name='IS Variance'))
  fig.add_trace(go.Scatter(x=num_trajectories, y=Train_variance_values, mode='lines', name='Train Variance'))
  fig.add_trace(go.Scatter(x=num_trajectories, y=Test_variance_values, mode='lines', name='Test Variance'))
  fig.update_layout(title='Variance over Trajectories', xaxis_title='Number of Trajectories', yaxis_title='Variance')
  fig.show()

  # Plot MSE over trajectories
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=num_trajectories, y=IS_mse_values, mode='lines', name='IS MSE'))
  fig.add_trace(go.Scatter(x=num_trajectories, y=Train_mse_values, mode='lines', name='Train MSE'))
  fig.add_trace(go.Scatter(x=num_trajectories, y=Test_mse_values, mode='lines', name='Test MSE'))
  fig.update_layout(title='MSE over Trajectories', xaxis_title='Number of Trajectories', yaxis_title='MSE')
  fig.show()

  # # Create subplots
  # fig = make_subplots(rows=3, cols=1, subplot_titles=("Bias over Trajectories", "Variance over Trajectories", "MSE over Trajectories"))

  # # Add traces for bias
  # fig.add_trace(go.Scatter(x=num_trajectories, y=IS_bias_values, mode='lines', name='IS Bias'), row=1, col=1)
  # fig.add_trace(go.Scatter(x=num_trajectories, y=Train_bias_values, mode='lines', name='Train Bias'), row=1, col=1)
  # fig.add_trace(go.Scatter(x=num_trajectories, y=Test_bias_values, mode='lines', name='Test Bias'), row=1, col=1)

  # # Add traces for variance
  # fig.add_trace(go.Scatter(x=num_trajectories, y=IS_variance_values, mode='lines', name='IS Variance'), row=2, col=1)
  # fig.add_trace(go.Scatter(x=num_trajectories, y=Train_variance_values, mode='lines', name='Train Variance'), row=2, col=1)
  # fig.add_trace(go.Scatter(x=num_trajectories, y=Test_variance_values, mode='lines', name='Test Variance'), row=2, col=1)

  # # Add traces for MSE
  # fig.add_trace(go.Scatter(x=num_trajectories, y=IS_mse_values, mode='lines', name='IS MSE'), row=3, col=1)
  # fig.add_trace(go.Scatter(x=num_trajectories, y=Train_mse_values, mode='lines', name='Train MSE'), row=3, col=1)
  # fig.add_trace(go.Scatter(x=num_trajectories, y=Test_mse_values, mode='lines', name='Test MSE'), row=3, col=1)

  # # Update layout with increased subplot height
  # fig.update_layout(title_text="Metrics over Trajectories", showlegend=True, height=1000)

  # # Show plot
  # fig.show()


def viz_over_train_set(base_params, train_set_sizes):
  # Initialize lists to store values for IS, Train, and Test
  IS_bias_values = []
  Train_bias_values = []
  Test_bias_values = []
  IS_variance_values = []
  Train_variance_values = []
  Test_variance_values = []
  IS_mse_values = []
  Train_mse_values = []
  Test_mse_values = []

  for size in train_set_sizes:
      # Create experiment instance and load data
      # test_experiment = SCOPE_experiment(1, 0.4, 1, 0.05, q_table, 0.99, i, 10000, 0.3, smallest_distance_to_deadend,0.1, [16], 0.001, 0.2, 0.00001, 0.00001, 1.0, 1.0, 300, 50, 0.0, torch.float64, "varying_train_set_sizes", "/content/drive/MyDrive/Lifegate_experiments")
      params = base_params.copy()  # Make a copy to avoid modifying the base parameters
      params["percent_to_estimate_phi"] = size  # Update the Percent of set for training phi
      test_experiment = SCOPE_experiment(**params)
      # test_load = existing_experiments(test_experiment,"/content/drive/MyDrive/Lifegate_experiments")
      test_load = existing_experiments(test_experiment)      

      # Get epoch-specific values for IS, Train, and Test
      IS_bias, Train_bias, Test_bias, IS_variance, Train_variance, Test_variance, IS_mse, Train_mse, Test_mse = test_load.epoch_specific_values()

      # Append values to respective lists
      IS_bias_values.append(IS_bias)
      Train_bias_values.append(Train_bias)
      Test_bias_values.append(Test_bias)
      IS_variance_values.append(IS_variance)
      Train_variance_values.append(Train_variance)
      Test_variance_values.append(Test_variance)
      IS_mse_values.append(IS_mse)
      Train_mse_values.append(Train_mse)
      Test_mse_values.append(Test_mse)


  # Plot bias over train set sizes
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=train_set_sizes, y=IS_bias_values, mode='lines', name='IS Bias'))
  fig.add_trace(go.Scatter(x=train_set_sizes, y=Train_bias_values, mode='lines', name='Train Bias'))
  fig.add_trace(go.Scatter(x=train_set_sizes, y=Test_bias_values, mode='lines', name='Test Bias'))
  fig.update_layout(title='Bias over train set sizes', xaxis_title='Percent of set for training phi', yaxis_title='Bias')
  fig.show()

  # Plot variance over train set sizes
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=train_set_sizes, y=IS_variance_values, mode='lines', name='IS Variance'))
  fig.add_trace(go.Scatter(x=train_set_sizes, y=Train_variance_values, mode='lines', name='Train Variance'))
  fig.add_trace(go.Scatter(x=train_set_sizes, y=Test_variance_values, mode='lines', name='Test Variance'))
  fig.update_layout(title='Variance over train set sizes', xaxis_title='Percent of set for training phi', yaxis_title='Variance')
  fig.show()

  # Plot MSE over train set sizes
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=train_set_sizes, y=IS_mse_values, mode='lines', name='IS MSE'))
  fig.add_trace(go.Scatter(x=train_set_sizes, y=Train_mse_values, mode='lines', name='Train MSE'))
  fig.add_trace(go.Scatter(x=train_set_sizes, y=Test_mse_values, mode='lines', name='Test MSE'))
  fig.update_layout(title='MSE over train set sizes', xaxis_title='Percent of set for training phi', yaxis_title='MSE')
  fig.show()