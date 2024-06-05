import plotly.graph_objs as go
from plotly.subplots import make_subplots
import torch 
import numpy as np
import os
from neural_net import NN_l1_l2_reg
import matplotlib.pyplot as plt




class existing_experiments(object):

    def __init__(self, experiment_instance):

      self.experiment_instance = experiment_instance
      self.folder_path = experiment_instance.folder_path

      if not self.experiment_exists():
          print("Experiment does not exist in the specified folder.")       
    
    def experiment_exists(self):
      # Check if experiment directory exists
      filename = self.experiment_instance.generate_file_name()
      # generate file path with folder and filename
      file_path = os.path.join(self.folder_path, f"{filename}.pt")
      return os.path.exists(file_path)

    # Individual Experiments
    '''
    Loading data
    '''
    def load_experiment_metrics(self):
        '''
        Load saved files and data from a saved file
        '''
        filename = self.experiment_instance.generate_file_name()
        # filename = self.generate_file_name()

        # generate file path with folder and filename
        file_path = os.path.join(self.folder_path, f"{filename}.pt")

        # Load data from the .pt file
        # loaded_data = torch.load(f"{self.folder_name}/{self.filename}.pt")
        # loaded_data = torch.load(f"{self.folder_path}/{filename}.pt")
        loaded_data = torch.load(file_path)

        # Access specific data from the loaded dictionary
        experiment_metrics = loaded_data["Experiment Metrics"]

        per_epoch = experiment_metrics["per_epoch"]
        model_weights = experiment_metrics["model_weights"]

        return per_epoch, model_weights

    def load_pi_b(self):
        filename = self.experiment_instance.generate_file_name()
        # Load data from the .pt file
        # loaded_data = torch.load(f"{self.folder_name}/{self.filename}.pt")
        # loaded_data = torch.load(f"{self.folder_path}/{filename}.pt")

        file_path = os.path.join(self.folder_path, f"{filename}.pt")
        loaded_data = torch.load(file_path)


        pi_b = loaded_data["pi_b"]

        return pi_b

    def load_on_policy_estimate(self):
       # Load data from the .pt file
        # loaded_data = torch.load(f"{self.folder_name}/{self.filename}.pt")
        filename = self.experiment_instance.generate_file_name()
        # loaded_data = torch.load(f"{self.folder_path}/{filename}.pt")
        file_path = os.path.join(self.folder_path, f"{filename}.pt")
        loaded_data = torch.load(file_path)

        on_policy_estimate = loaded_data["On Policy Estimate"]
        return on_policy_estimate

    def load_IS_estimate(self):
      filename = self.experiment_instance.generate_file_name()
      # loaded_data = torch.load(f"{self.folder_path}/{filename}.pt")
      file_path = os.path.join(self.folder_path, f"{filename}.pt")
      loaded_data = torch.load(file_path)

      IS_estimate = loaded_data['IS Estimate']['Estimate']
      IS_variance = loaded_data['IS Estimate']['Variance']
      return IS_estimate, IS_variance

    
    def load_multi_estimates(self):
      filename =  self.experiment_instance.generate_file_name()
      file_path = os.path.join(self.folder_path, f"{filename}.pt")
      # loaded_data = torch.load(file_path)

      IS_all_means, IS_all_variances, Train_means, Train_variances, Test_means, Test_variances = self.experiment_instance.get_multi_experiment()

      return IS_all_means, IS_all_variances, Train_means, Train_variances, Test_means, Test_variances
    
    '''
    Preprocessing
    '''

    def preprocess_epoch_metrics(self):
        per_epoch, _ = self.load_experiment_metrics()

        # Extract metrics
        # IS_mean = np.zeros(len(per_epoch))
        # IS_variance = np.zeros(len(per_epoch))
        Train_mean = np.zeros(len(per_epoch))
        Train_variance = np.zeros(len(per_epoch))
        Train_mse_loss = np.zeros(len(per_epoch))
        total_loss = np.zeros(len(per_epoch))
        Test_mean = np.zeros(len(per_epoch))
        Test_variance = np.zeros(len(per_epoch))

        for i, epoch_data in enumerate(per_epoch):
            # IS_mean[i] = epoch_data['IS_mean']
            # IS_variance[i] = epoch_data['IS_variance']
            Train_mean[i] = epoch_data['Train_mean']
            Train_variance[i] = epoch_data['Train_variance']
            Train_mse_loss[i] = epoch_data['Train_mse_loss']
            total_loss[i] = epoch_data['total_loss']
            Test_mean[i] = epoch_data['Test_mean']
            Test_variance[i] = epoch_data['Test_variance']

        return Train_mean, Train_variance, Train_mse_loss, total_loss, Test_mean, Test_variance


    '''
    Calculations
    '''
    def calculate_bias(self, estimate):
      on_policy_estimate = self.load_on_policy_estimate()
      bias = estimate - on_policy_estimate
      return bias

    def calculate_mse(self, variance, estimate):
      bias = self.calculate_bias(estimate)
      mse = variance + bias**2
      return mse
    

    def calculate_multi_bias(self, multi_estimates):
       on_policy_estimate = self.load_on_policy_estimate()
       biases = [estimate - on_policy_estimate for estimate in multi_estimates]

       return biases
    
    def calculate_multi_mse(self, multi_variances, multi_estimates):
       biases = self.calculate_multi_bias(multi_estimates)
       multi_mse = [multi_variances[i] + biases[i]**2 for i in range(len(multi_estimates))]

       return multi_mse


    def get_multi_values(self):
       
       IS_all_means, IS_all_variances, Train_means, Train_variances, Test_means, Test_variances = self.load_multi_estimates()

       IS_bias = self.calculate_multi_bias(IS_all_means)
       Train_bias = self.calculate_multi_bias(Train_means)
       Test_bias = self.calculate_multi_bias(Test_means)
       IS_variance = IS_all_variances
       Train_variance = Train_variances
       Test_variance = Test_variances
       IS_mse = self.calculate_multi_mse(IS_variance, IS_all_means)
       Train_mse = self.calculate_multi_mse(Train_variance, Train_means)
       Test_mse = self.calculate_mse(Test_variance,Test_means)
       
       return IS_bias, Train_bias, Test_bias, IS_variance, Train_variance, Test_variance, IS_mse, Train_mse, Test_mse

    '''
    Load model
    '''

    def load_model(self, epoch = None):
      filename = self.experiment_instance.generate_file_name()
      file_path = os.path.join(self.folder_path, f"{filename}.pt")
      loaded_data = torch.load(file_path)
      # loaded_data = torch.load(f"{self.folder_name}/{self.filename}.pt")
      experiment_parameters = loaded_data["Experiment Parameters"]
      _, model_weights = self.load_experiment_metrics()

      if epoch is not None:
          # Find the index of the model weights corresponding to the specified epoch
          index = next((i for i, item in enumerate(model_weights) if item["epoch"] == epoch), None)
          if index is not None:
              chosen_model_state = model_weights[index]['model_state']
          else:
              raise ValueError(f"No weights found for epoch {epoch}")
      else:
          # If epoch is not specified, load the final weights
          chosen_model_state = model_weights[-1]['model_state']

      model = NN_l1_l2_reg(input_dim=2,
                           hidden_dims=experiment_parameters["hidden_dims"],
                           output_dim=1, dtype = experiment_parameters["dtype"],
                           l1_lambda=experiment_parameters["l1_reg"],
                           l2_lambda = experiment_parameters["l2_reg"])

      # Load the final weights into the model
      model.load_state_dict(chosen_model_state)

      return model


    '''
    Heatmap for model output
    '''

    # -----------------------
    # Heatmaps for lifegate
    # -----------------------
    def get_model_output_dict(self, epoch):
      model = self.load_model(epoch)
      model.eval()

      # Initialize an empty dictionary to store data
      data = {}

      # Loop through all combinations from [0,0] to [9,9]
      for i in range(10):
        for j in range(10):
            # Prepare input data
            input_data = torch.tensor([i, j], dtype=torch.float64)

            # Pass input through the self.model
            output = model(input_data)

            # Store data in the dictionary
            data[(i, j)] = output.item()

      return data

    def plot_heatmap(self, data):
      values = np.zeros((10, 10))
      for (x, y), value in data.items():
          values[y, x] = value

      # Create the heatmap
      fig = go.Figure(data=go.Heatmap(z=values, colorscale='viridis'))

      # Add colorbar
      fig.update_layout(coloraxis_colorbar=dict(title='Values',
                                                ticks='outside',
                                                tickvals=[np.min(values), np.max(values)],
                                                ticktext=[np.min(values), np.max(values)]))

      # Add labels and title
      fig.update_layout(xaxis=dict(tickvals=np.arange(10), ticktext=list(range(10)), title='X'),
                        yaxis=dict(tickvals=np.arange(9, -1, -1), ticktext=list(range(9, -1, -1)), title='Y', autorange="reversed"),
                        title='Heatmap')

      fig.show()

    def get_heatmap(self, epoch = None):
      data = self.get_model_output_dict(epoch)
      self.plot_heatmap(data)

    
    def plot_heatmap_save(self, data, save = True):
      values = np.zeros((10, 10))
      for (x, y), value in data.items():
          values[y, x] = value

      # Create the heatmap using Matplotlib
      plt.figure(figsize=(8, 6))
      plt.imshow(values, cmap='viridis', origin='lower')
      plt.colorbar(label='Values')

      # Add labels and title
      plt.xlabel('X')
      plt.ylabel('Y')
      plt.title('Heatmap')
      plt.xticks(np.arange(10))
      plt.yticks(np.arange(10))
      if save:
          filename = self.experiment_instance.generate_file_name()
          # filename = self.generate_file_name()
          # generate file path with folder and filename
          file_path = os.path.join(self.folder_path, f"shaping_heatmap_predictions_{filename}.png")
          plt.savefig(file_path)

      plt.show()



    # ---------------------
    # State Visitation Heatmap
    # ---------------------

    def count_state_visits(self, pi_b):
      state_visit_counts = {}
      for trajectory in pi_b:
          for data_point in trajectory:
              state = tuple(data_point['state'])
              if state not in state_visit_counts:
                  state_visit_counts[state] = 0
              state_visit_counts[state] += 1

          # Include last state_next of the trajectory
          last_state_next = tuple(trajectory[-1]['state_next'])
          if last_state_next not in state_visit_counts:
              state_visit_counts[last_state_next] = 0
          state_visit_counts[last_state_next] += 1

      return state_visit_counts

    def create_state_visit_dict(self):
        state_visit_dict = {}
        for i in range(10):
            for j in range(10):
                state_visit_dict[(i, j)] = 0
        return state_visit_dict

    def fill_state_visit_dict(self,state_visit_counts):
        state_visit_dict = self.create_state_visit_dict()
        for state, count in state_visit_counts.items():
            state_visit_dict[state] = count
        return state_visit_dict


    def plot_state_visitations_heatmap(self, state_visit_dict):
      # Create lists to store x, y, and z values
      x = []
      y = []
      z = []

      # Iterate through the state visit dictionary
      for state, count in state_visit_dict.items():
          x.append(state[0])
          y.append(9 - state[1])  # Flip y-axis to have (0, 0) at the bottom-left
          z.append(count)

      # Create the heatmap trace
      trace = go.Heatmap(
          x=x,
          y=y,
          z=z,
          colorscale='Viridis',  # Choose a colorscale
          colorbar=dict(title='Visits'),
          zmin=0,
          zmax=max(z)  # Set maximum value for the color scale
      )

      # Create layout
      layout = go.Layout(
          title='State Visitations Heatmap',
          xaxis=dict(title='X-axis'),
          yaxis=dict(title='Y-axis', tickvals=list(range(10)), ticktext=list(range(9, -1, -1))),
      )

      # Create figure
      fig = go.Figure(data=[trace], layout=layout)
      fig.show()


    def get_state_visitation_heatmap(self):
      pi_b = self.load_pi_b()
      # Count state visits
      state_visit_counts = self.count_state_visits(pi_b)

      # Fill state visit dictionary
      state_visit_dict = self.fill_state_visit_dict(state_visit_counts)

      # Assuming state_visit_dict is your dictionary with state visitations
      self.plot_state_visitations_heatmap(state_visit_dict)

    def epoch_specific_values(self, epoch=None):
        Train_mean, Train_variance, Train_mse_loss, total_loss, Test_mean, Test_variance = self.preprocess_epoch_metrics()
        IS_mean, IS_variance = self.load_IS_estimate()

        IS_bias = self.calculate_bias(IS_mean)
        Train_bias = self.calculate_bias(Train_mean)
        Test_bias = self.calculate_bias(Test_mean)

        # Calculate MSE for IS
        IS_mse = self.calculate_mse(IS_variance, IS_mean)

        # Calculate MSE for Train and Test separately
        Train_mse = self.calculate_mse(Train_variance, Train_mean)
        Test_mse = self.calculate_mse(Test_variance, Test_mean)

        if epoch is None:
            # If no epoch is specified, use the values of the final epoch for Train and Test
            return IS_bias, Train_bias[-1], Test_bias[-1], IS_variance, Train_variance[-1], Test_variance[-1], IS_mse, Train_mse[-1], Test_mse[-1]
        else:
            # If epoch is specified, ensure it's within the range of available epochs
            if epoch < 1 or epoch > len(Train_mse):
                raise ValueError("Epoch out of range.")

            # Return values for the specified epoch for Train and Test
            return IS_bias, Train_bias[epoch-1], Test_bias[epoch-1], IS_variance, Train_variance[epoch-1], Test_variance[epoch-1], IS_mse, Train_mse[epoch-1], Test_mse[epoch-1]

    # def choose_mse_by_epoch(self):



    def plot_mse(self):
        Train_mean, Train_variance, Train_mse_loss, total_loss, Test_mean, Test_variance = self.preprocess_epoch_metrics()
        IS_mean, IS_variance = self.load_IS_estimate()

        IS_mse = self.calculate_mse(IS_variance, IS_mean)
        Train_mse = self.calculate_mse(Train_variance, Train_mean)
        Test_mse = self.calculate_mse(Test_variance, Test_mean)

        epochs = np.arange(1, len(Train_mean) + 1)
        fig = make_subplots(rows=1, cols=1, subplot_titles=("MSE over Epochs"))

        # Add traces for IS, Train, and Test MSE
        # fig.add_trace(go.Scatter(x=epochs, y=IS_mse, mode='lines+markers', name='IS MSE'), row=1, col=1)
        IS_line = [IS_mse] * len(epochs)
        fig.add_trace(go.Scatter(x=epochs, y=IS_line, mode='lines', name='IS MSE'), row=1, col=1)

        fig.add_trace(go.Scatter(x=epochs, y=Train_mse, mode='lines+markers', name='Train MSE'), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=Test_mse, mode='lines+markers', name='Test MSE'), row=1, col=1)

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="MSE", row=1, col=1)

        fig.update_layout(title_text="MSE over Epochs", showlegend=True)
        fig.show()

    def plot_metrics(self):
    # def plot_metrics_test(self, IS_mean, Train_mean, Train_variance, Train_mse_loss, total_loss, Test_mean, Test_variance):
      '''
      Plot metrics over epochs
      '''
      Train_mean, Train_variance, Train_mse_loss, total_loss, Test_mean, Test_variance = self.preprocess_epoch_metrics()

      epochs = np.arange(1, len(Train_mean) + 1)

      on_policy_estimate = self.load_on_policy_estimate()
      # Create a list representing on-policy estimate for each epoch
      on_policy_line = [on_policy_estimate] * len(epochs)

      IS_mean, IS_variance = self.load_IS_estimate()

      IS_mse = self.calculate_mse(IS_variance, IS_mean)
      Train_mse = self.calculate_mse(Train_variance, Train_mean)
      Test_mse = self.calculate_mse(Test_variance, Test_mean)



      fig = make_subplots(rows=2, cols=2, subplot_titles=("Estimate over Epochs",
                                                          "Variance over Epochs", "Shaping Train MSE Loss over Epochs",
                                                          "Total MSE over Epochs"))
      IS_mean_line = [IS_mean] * len(epochs)
      fig.add_trace(go.Scatter(x=epochs, y=IS_mean_line, mode='lines', name='IS Estimate'), row=1, col=1)
      # fig.add_trace(go.Scatter(x=epochs, y=IS_mean, mode='lines+markers', name='IS Estimate'), row=1, col=1)
      fig.add_trace(go.Scatter(x=epochs, y=Train_mean, mode='lines', name='Train Estimate'), row=1, col=1)
      fig.add_trace(go.Scatter(x=epochs, y=Test_mean, mode='lines', name='Test Estimate'), row=1, col=1)
      fig.add_trace(go.Scatter(x=epochs, y=on_policy_line, mode='lines', name='On-policy Estimate'), row=1, col=1)

      IS_var_line = [IS_variance] * len(epochs)
      fig.add_trace(go.Scatter(x=epochs, y=IS_var_line, mode='lines', name='IS Variance'), row=1, col=2)
      fig.add_trace(go.Scatter(x=epochs, y=Train_variance, mode='lines', name='Train Variance'), row=1, col=2)
      fig.add_trace(go.Scatter(x=epochs, y=Test_variance, mode='lines', name='Test Variance'), row=1, col=2)
      
      fig.add_trace(go.Scatter(x=epochs, y=Train_mse_loss, mode='lines', name='Train MSE Loss'), row=2, col=1)
      # fig.add_trace(go.Scatter(x=epochs, y=total_loss, mode='lines+markers', name='Total Loss'), row=2, col=2)
      IS_line = [IS_mse] * len(epochs)
      fig.add_trace(go.Scatter(x=epochs, y=IS_line, mode='lines', name='IS MSE'), row=2, col=2)
      fig.add_trace(go.Scatter(x=epochs, y=Train_mse, mode='lines', name='Train MSE'), row=2, col=2)
      fig.add_trace(go.Scatter(x=epochs, y=Test_mse, mode='lines', name='Test MSE'), row=2, col=2)

      

      fig.update_xaxes(title_text="Epoch", row=1, col=1)
      fig.update_xaxes(title_text="Epoch", row=1, col=2)
      fig.update_xaxes(title_text="Epoch", row=2, col=1)
      fig.update_xaxes(title_text="Epoch", row=2, col=2)

      fig.update_yaxes(title_text="Estimate", row=1, col=1)
      fig.update_yaxes(title_text="Variance", row=1, col=2)
      fig.update_yaxes(title_text="MSE Loss", row=2, col=1)
      fig.update_yaxes(title_text="MSE", row=2, col=2)
      

      fig.update_layout(title_text="Metrics over Epochs", showlegend=True)
      fig.show()


    # def plot_metrics_save(self, save=True):
    #     '''
    #     Plot metrics over epochs and optionally save the plot
    #     '''
    #     Train_mean, Train_variance, Train_mse_loss, total_loss, Test_mean, Test_variance = self.preprocess_epoch_metrics()

    #     epochs = np.arange(1, len(Train_mean) + 1)

    #     on_policy_estimate = self.load_on_policy_estimate()
    #     # Create a list representing on-policy estimate for each epoch
    #     on_policy_line = [on_policy_estimate] * len(epochs)

    #     IS_mean, IS_variance = self.load_IS_estimate()

    #     IS_mse = self.calculate_mse(IS_variance, IS_mean)
    #     Train_mse = self.calculate_mse(Train_variance, Train_mean)
    #     Test_mse = self.calculate_mse(Test_variance, Test_mean)

    #     fig = make_subplots(rows=2, cols=2, subplot_titles=("Estimate over Epochs",
    #                                                         "Variance over Epochs", "Shaping Train MSE Loss over Epochs",
    #                                                         "Total MSE over Epochs"))
    #     IS_mean_line = [IS_mean] * len(epochs)
    #     fig.add_trace(go.Scatter(x=epochs, y=IS_mean_line, mode='lines', name='IS Estimate'), row=1, col=1)
    #     fig.add_trace(go.Scatter(x=epochs, y=Train_mean, mode='lines', name='Train Estimate'), row=1, col=1)
    #     fig.add_trace(go.Scatter(x=epochs, y=Test_mean, mode='lines', name='Test Estimate'), row=1, col=1)
    #     fig.add_trace(go.Scatter(x=epochs, y=on_policy_line, mode='lines', name='On-policy Estimate'), row=1, col=1)

    #     IS_var_line = [IS_variance] * len(epochs)
    #     fig.add_trace(go.Scatter(x=epochs, y=IS_var_line, mode='lines', name='IS Variance'), row=1, col=2)
    #     fig.add_trace(go.Scatter(x=epochs, y=Train_variance, mode='lines', name='Train Variance'), row=1, col=2)
    #     fig.add_trace(go.Scatter(x=epochs, y=Test_variance, mode='lines', name='Test Variance'), row=1, col=2)
        
    #     fig.add_trace(go.Scatter(x=epochs, y=Train_mse_loss, mode='lines', name='Train MSE Loss'), row=2, col=1)
    #     IS_line = [IS_mse] * len(epochs)
    #     fig.add_trace(go.Scatter(x=epochs, y=IS_line, mode='lines', name='IS MSE'), row=2, col=2)
    #     fig.add_trace(go.Scatter(x=epochs, y=Train_mse, mode='lines', name='Train MSE'), row=2, col=2)
    #     fig.add_trace(go.Scatter(x=epochs, y=Test_mse, mode='lines', name='Test MSE'), row=2, col=2)

    #     fig.update_xaxes(title_text="Epoch", row=1, col=1)
    #     fig.update_xaxes(title_text="Epoch", row=1, col=2)
    #     fig.update_xaxes(title_text="Epoch", row=2, col=1)
    #     fig.update_xaxes(title_text="Epoch", row=2, col=2)

    #     fig.update_yaxes(title_text="Estimate", row=1, col=1)
    #     fig.update_yaxes(title_text="Variance", row=1, col=2)
    #     fig.update_yaxes(title_text="MSE Loss", row=2, col=1)
    #     fig.update_yaxes(title_text="MSE", row=2, col=2)

    #     fig.update_layout(title_text="Metrics over Epochs", showlegend=True)

    #     filename = self.experiment_instance.generate_file_name()
    #     # filename = self.generate_file_name()

    #     # generate file path with folder and filename
    #     file_path = os.path.join(self.folder_path, f"{filename}.png")

    #     if save:
    #         fig.write_image(file_path)
    #         fig.show()
    #     else:
    #         fig.show()

    def plot_metrics_save(self, save=True):
        '''
        Plot metrics over epochs and optionally save the plot
        '''
        Train_mean, Train_variance, Train_mse_loss, total_loss, Test_mean, Test_variance = self.preprocess_epoch_metrics()

        epochs = np.arange(1, len(Train_mean) + 1)

        on_policy_estimate = self.load_on_policy_estimate()
        # Create a list representing on-policy estimate for each epoch
        on_policy_line = [on_policy_estimate] * len(epochs)

        IS_mean, IS_variance = self.load_IS_estimate()

        IS_mse = self.calculate_mse(IS_variance, IS_mean)
        Train_mse = self.calculate_mse(Train_variance, Train_mean)
        Test_mse = self.calculate_mse(Test_variance, Test_mean)

        IS_mean_line = [IS_mean] * len(epochs)
        IS_var_line = [IS_variance] * len(epochs)
        IS_line = [IS_mse] * len(epochs)

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        axs[0, 0].plot(epochs, IS_mean_line, label='IS Estimate')
        axs[0, 0].plot(epochs, Train_mean, label='Train Estimate')
        axs[0, 0].plot(epochs, Test_mean, label='Test Estimate')
        axs[0, 0].plot(epochs, on_policy_line, label='On-policy Estimate')
        axs[0, 0].set_title('Estimate over Epochs')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Estimate')
        axs[0, 0].legend()

        axs[0, 1].plot(epochs, IS_var_line, label='IS Variance')
        axs[0, 1].plot(epochs, Train_variance, label='Train Variance')
        axs[0, 1].plot(epochs, Test_variance, label='Test Variance')
        axs[0, 1].set_title('Variance over Epochs')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Variance')
        axs[0, 1].legend()
        
        axs[1, 0].plot(epochs, Train_mse_loss, label='Train MSE Loss')
        axs[1, 0].set_title('Shaping Train MSE Loss over Epochs')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('MSE Loss')
        
        axs[1, 1].plot(epochs, IS_line, label='IS MSE')
        axs[1, 1].plot(epochs, Train_mse, label='Train MSE')
        axs[1, 1].plot(epochs, Test_mse, label='Test MSE')
        axs[1, 1].set_title('Total MSE over Epochs')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('MSE')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.suptitle("Metrics over Epochs", y=1.02)
        
        if save:
            filename = self.experiment_instance.generate_file_name()
            # filename = self.generate_file_name()
            # generate file path with folder and filename
            file_path = os.path.join(self.folder_path, f"{filename}.png")
            plt.savefig(file_path)
        plt.show()


    

    # def loss_plotting(self):


    '''
      Multi-Experiment loading and visualization
    '''


    '''
    Visualizing experiments
    Over trajectories
    Over train/test/splits
    Heatmaps for each and over epochs within experiment
    State Visitation for each
    Varying features

    '''

    # def visualize_experiment(self):
    #   '''
    #   first load data
    #   visualize all experiment information
    #   heatmaps, loss plots, state_visitation frequencies
    #   variance plots over trajectories
    #   plots over varying train/test splits

    #   '''
