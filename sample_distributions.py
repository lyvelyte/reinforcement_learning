import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pylab as plt

def find_closest_value_index(values, target):
    # Initialize the closest value to the first element and its index
    closest_value = values[0]
    closest_value_index = 0
    
    # Loop through the list to find the index of the closest value
    for index, value in enumerate(values):
        if abs(value - target) < abs(closest_value - target):
            closest_value = value
            closest_value_index = index
    
    return closest_value_index

def min_angle_degrees(angle1, angle2):
    """
    Calculate the minimum angle between two angles in degrees.

    Parameters:
    angle1 (float): The first angle in degrees.
    angle2 (float): The second angle in degrees.

    Returns:
    float: The minimum angle between the two angles in degrees.
    """
    # Calculate the absolute difference
    delta = abs(angle1 - angle2) % 360
    # Find the minimum angle
    min_angle = min(delta, 360 - delta)
    return min_angle

class ObjectDistributions():
    def __init__(self, difficulty, ideal_look_angle, plot_distributions_flag=False):
        self.n_x_vals = 100
        self.x_vals = np.linspace(0, 1, self.n_x_vals)   # Example x values from 0 to 1
        self.ideal_look_angle = ideal_look_angle

        # Generate relevant distances (meters)
        self.n_dist = 10    # Number of discrete distance values to use. 
        self.distances = np.linspace(0, 100, self.n_dist)

        if difficulty.lower() == 'easy':
            # Generate target distributions
            self.target_mu_vals = np.linspace(1, 0, self.n_dist)
            self.target_sigma_vals = np.linspace(0.25, 0.05, self.n_dist)
            self.target_pdf_y_vals, self.target_cdf_y_vals = self.gaussian(self.target_mu_vals, self.target_sigma_vals)

            # Generate false alarm distributions
            self.fa_mu_vals = np.linspace(0, 0, self.n_dist)
            self.fa_sigma_vals = np.linspace(0.25, 0.05, self.n_dist)
            self.fa_pdf_y_vals, self.fa_cdf_y_vals = self.gaussian(self.fa_mu_vals, self.fa_sigma_vals)
        elif difficulty.lower() == 'medium':
            # Generate target distributions
            self.target_mu_vals = np.linspace(1, 0, self.n_dist)
            self.target_sigma_vals = np.linspace(0.5, 0.15, self.n_dist)
            self.target_pdf_y_vals, self.target_cdf_y_vals = self.gaussian(self.target_mu_vals, self.target_sigma_vals)

            # Generate false alarm distributions
            self.fa_mu_vals = np.linspace(0, 0, self.n_dist)
            self.fa_sigma_vals = np.linspace(0.5, 0.15, self.n_dist)
            self.fa_pdf_y_vals, self.fa_cdf_y_vals = self.gaussian(self.fa_mu_vals, self.fa_sigma_vals)
        elif difficulty.lower() == 'hard':
            # Generate target distributions
            self.target_mu_vals = np.linspace(1, 0, self.n_dist)
            self.target_sigma_vals = np.linspace(0.75, 0.25, self.n_dist)
            self.target_pdf_y_vals, self.target_cdf_y_vals = self.gaussian(self.target_mu_vals, self.target_sigma_vals)

            # Generate false alarm distributions
            self.fa_mu_vals = np.linspace(0, 0, self.n_dist)
            self.fa_sigma_vals = np.linspace(0.75, 0.25, self.n_dist)
            self.fa_pdf_y_vals, self.fa_cdf_y_vals = self.gaussian(self.fa_mu_vals, self.fa_sigma_vals)
        else:
            raise(ValueError(f"Error: Unrecognized difficulty \"{difficulty}\""))

        if plot_distributions_flag:
            self.plot_3d_gaussian_filled()

    def gaussian(self, mus, sigmas):
        pdf_y_values = []
        cdf_y_vals = []
        for mu, sigma in zip(mus, sigmas):
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((self.x_vals - mu) / sigma) ** 2)
            cdf_y = np.cumsum(y)
            max_val = cdf_y[-1]
            pdf_y_values.append(y / max_val)
            cdf_y_vals.append(cdf_y / max_val)
        
        return pdf_y_values, cdf_y_vals
    
    def plot_3d_gaussian_filled(self):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        # Plot Targets
        for i, distance in enumerate(self.distances):
            # Generate the Gaussian distribution for the current distance
            y = self.target_pdf_y_vals[i]
            z = np.zeros(self.x_vals.shape) + distance  # Offset along the Y-axis
            
            # Plot the line for the Gaussian distribution
            ax.plot(self.x_vals, z, y, label=f"Target - Distance: {distance}")
            
            # Create the vertices for the filled polygon under the curve
            verts = list(zip(self.x_vals, z, y)) + [(self.x_vals[-1], z[-1], 0), (self.x_vals[0], z[0], 0)]
            poly = Poly3DCollection([verts], color='green', alpha=0.5)
            ax.add_collection3d(poly)

        # Plot False Alarms
        for i, distance in enumerate(self.distances):
            # Generate the Gaussian distribution for the current distance
            y = self.fa_pdf_y_vals[i]
            z = np.zeros(self.x_vals.shape) + distance  # Offset along the Y-axis
            
            # Plot the line for the Gaussian distribution
            ax.plot(self.x_vals, z, y, label=f"False Alarm - Distance: {distance}")
            
            # Create the vertices for the filled polygon under the curve
            verts = list(zip(self.x_vals, z, y)) + [(self.x_vals[-1], z[-1], 0), (self.x_vals[0], z[0], 0)]
            poly = Poly3DCollection([verts], color='red', alpha=0.5)
            ax.add_collection3d(poly)

        ax.set_xlabel('Confidece')
        ax.set_ylabel('Distance')
        ax.set_zlabel('PDF')
        ax.set_title('Target vs False Alarm Distributions')
        plt.show()

    def sample(self, distance, is_false_alarm, relative_look_angle = None):
        dist_idx = find_closest_value_index(self.distances, distance)
        
        r = np.random.uniform()
        
        if is_false_alarm:
            idx = np.searchsorted(self.fa_cdf_y_vals[dist_idx], r)    
        else:
            idx = np.searchsorted(self.target_cdf_y_vals[dist_idx], r)
        
        
        if self.ideal_look_angle:
            angle_dif = min_angle_degrees(self.ideal_look_angle, relative_look_angle)
        else: 
            angle_dif = 0

        return self.x_vals[idx]*abs(np.cos(np.radians(angle_dif)))

    def get_probability_of_target(self, samples, prior_p_target = 0.5):
        """
        Calcilates the probability of the object being a target given the samples

        Parameters:
        samples: A list of samples. Each sample should be a dictionary strucured as
            sample = {confidence: conf_value, 
                        distance: dist_value, 
                        relative_look_angle: look_angle_value}
        prior_p_target: float representing the prior probability of being a target (default 0.5))

        Returns:
        float: The probability of the obejct being a target given the samples
        """     

        # Assume an object either is a target or a false alarm (Law of total probability, p_target + p_false_alarm = 1)
        prior_p_false_alarm = 1 - prior_p_target

        # Compute the total likelihood ratio.
        total_log_likelihood_ratio = 0
        for i in range(len(samples)):
            dist_idx = find_closest_value_index(self.distances, samples[i]['distance'])
            idx = np.searchsorted(self.x_vals, samples[i]['confidence']) 

            target_likelihood = self.target_pdf_y_vals[dist_idx][idx]
            fa_likelihood = self.fa_pdf_y_vals[dist_idx][idx]

            # Keep track of the sum of the log likelihoods. 
            total_log_likelihood_ratio = total_log_likelihood_ratio + np.log(target_likelihood / fa_likelihood)

        # Compute the final probability of being a target
        total_likelihood_ratio =  np.exp(total_log_likelihood_ratio)
        p_target = (total_likelihood_ratio * prior_p_target) / (total_likelihood_ratio * prior_p_target + prior_p_false_alarm) 

        return p_target
    
    def get_expected_change_in_probability_of_target(self, samples, new_sample_distance, prior_p_target = 0.5):
        """
        Calcilates the probability of the object being a target given the samples

        Parameters:
        samples: A list of samples. Each sample should be a dictionary strucured as
            sample = {confidence: conf_value, 
                        distance: dist_value, 
                        relative_look_angle: look_angle_value}
        prior_p_target: float representing the prior probability of being a target (default 0.5))
        new_sample_distance: float representing the distance from an object to draw a new sample from. 

        Returns:
        float: The expected change in probability from sampling an object at the given new_sample_distance.
        """    

        # ===== Copied from get_probability_of_target ====
        # Assume an object either is a target or a false alarm (Law of total probability, p_target + p_false_alarm = 1)
        prior_p_false_alarm = 1 - prior_p_target

        # Compute the total log likelihood ratio for the existing samples.
        total_log_likelihood_ratio = 0
        for i in range(len(samples)):
            dist_idx = find_closest_value_index(self.distances, samples[i]['distance'])
            idx = np.searchsorted(self.x_vals, samples[i]['confidence']) 

            target_likelihood = self.target_pdf_y_vals[dist_idx][idx]
            fa_likelihood = self.fa_pdf_y_vals[dist_idx][idx]

            # Keep track of the sum of the log likelihoods. 
            total_log_likelihood_ratio += np.log(target_likelihood / fa_likelihood)

        # Compute the final probability of being a target
        total_likelihood_ratio = np.exp(total_log_likelihood_ratio)
        p_target = (total_likelihood_ratio * prior_p_target) / (total_likelihood_ratio * prior_p_target + prior_p_false_alarm) 
        # =====================================================

        p_false_alarm = 1-p_target

        # Determine the distributions the new sample will be pulled from. 
        dist_idx = find_closest_value_index(self.distances, new_sample_distance)

        # Calculate the average log likelihood change for each confidence possible
        target_likelihood_sum = 0
        false_alarm_likelihood_sum = 0
        n_likelihood_bins = len(self.target_pdf_y_vals[dist_idx])
        for i in range(n_likelihood_bins):
            target_likelihood_sum += p_target*self.target_pdf_y_vals[dist_idx][i]
            false_alarm_likelihood_sum += p_false_alarm*self.fa_pdf_y_vals[dist_idx][i]
            
        expected_log_likelihood_ratio_change = np.log(target_likelihood_sum / false_alarm_likelihood_sum) # n_likelihood bins drops out as it would have been in numerator and denomenator
        new_total_likelihood_ratio = np.exp(total_log_likelihood_ratio + expected_log_likelihood_ratio_change)
        new_p_target = (new_total_likelihood_ratio * p_target) / (new_total_likelihood_ratio * p_target + p_false_alarm)

        delta_p_target = new_p_target - p_target

        return delta_p_target


if __name__ == '__main__':
    distros = ObjectDistributions(difficulty = "hard", ideal_look_angle = None, plot_distributions_flag=False)

    samples = []
    for i in range(5):
        conf_value = distros.sample(distance = 5, is_false_alarm = False, relative_look_angle=None)
        print(f"Sampled confidence value = {conf_value}")

        samples.append({
            'confidence': conf_value,
            'distance': 1,
            'relative_look_angle': None
        })   

    p_target = distros.get_probability_of_target(samples)

    print(f"The probability of being a target is {p_target:.3f}")

    new_distance = 10
    delta_p_target = distros.get_expected_change_in_probability_of_target(samples, new_distance)

    print(f"The expected chagne in probability of being a target if we sampled from a distance of {new_distance} is {delta_p_target:.3f}")
