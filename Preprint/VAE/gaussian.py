from libraries import *

class MultiDimensionalGaussian:
    def __init__(self, means, variances):
        self.means = means
        self.variances = variances

    def sample(self, num_samples=1):
        """
        Generate random samples from the multi-dimensional Gaussian distribution.
        Args:
            num_samples (int): Number of samples to generate.
        Returns:
            np.ndarray: An array of shape (num_samples, 32) containing the samples.
        """
        samples = np.random.normal(self.means, np.sqrt(self.variances), size=(num_samples, len(self.means)))
        return samples
    
def get_distribution_labels(d, train_set, vae, device):
    distribution_dict = {}
    latent_dict = {}

    for label in train_set['labels'].unique():
        class_df = train_set[train_set['labels']==label]
        X_class = torch.tensor(class_df[class_df.columns.difference(['labels'])].values)
        
        vae.eval()
        
        latent = np.zeros((len(X_class), d))

        with torch.no_grad(): # No need to track the gradients
            for i in range(len(X_class)):
                # Move tensor to the proper device
                x = X_class[i].clone().detach().float().to(device)
                
                # Decode data
                x_test = x.unsqueeze(0)
                x_latent = vae.encoder(x_test)
                latent[i, :] = x_latent.detach().cpu().numpy()[0]
        #latent_means = np.mean(latent, axis=0)
        latent_variances = np.var(latent, axis=0)
        
        distribution_dict[label] = (np.zeros(d), latent_variances)
        latent_dict[label] = latent

    return distribution_dict, latent_dict
