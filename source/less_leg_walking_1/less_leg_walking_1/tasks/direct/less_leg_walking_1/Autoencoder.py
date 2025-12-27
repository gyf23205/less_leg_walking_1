import torch
import torch.nn as nn

class Encoder_walk(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim): 
        super(Encoder_walk, self).__init__()

        print('Tanh ver')
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim*4),
            nn.Tanh(),
            nn.Linear(hidden_dim*4, hidden_dim*3),
            nn.Tanh(),
            nn.Linear(hidden_dim*3, hidden_dim*2),
            nn.Tanh(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, observable_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, observable_dim, state_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(observable_dim, state_dim, bias=False)

    def forward(self, x):
        return self.linear(x)
    

class KoopmanAutoencoder_walk(nn.Module):
    def __init__(self, state_dim, hidden_dim, observable_dim,device):
        super().__init__()
        self.encoder = Encoder_walk(state_dim, hidden_dim, observable_dim)
        self.decoder = Decoder(observable_dim, state_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.observable_dim = observable_dim
        self.K = torch.randn(observable_dim, observable_dim).to(device)  
    
    def forward(self, x):

        z = self.encoder(x)  

        # if self.K is not None:
        z_next = torch.matmul(z, self.K.T)  # Apply computed Koopman operator
        # else:
        #     z_next = z  

        y_hat = self.decoder(z_next)
        x_hat = self.decoder(z)  
        return x_hat, z, y_hat
        
    def compute_koopman_operator(self, latent_X, latent_Y,device):
        # # print(latent_X.shape, latent_Y.shape)
        # latent_X = latent_X.view(-1, latent_X.size(-1))  # [N, d]
        # latent_Y = latent_Y.view(-1, latent_Y.size(-1))  # [N, d]

        # X_pseudo_inv = torch.linalg.pinv(latent_X.T)  # Compute pseudo-inverse of latent_X
        # # self.K = torch.matmul(latent_Y.T, X_pseudo_inv.T).to(device)  # K = Y * X^+
        # self.K = (latent_Y.T @ X_pseudo_inv).to(device)
        # # print(self.K.shape)

        latent_X = latent_X.view(-1, latent_X.size(-1))  # [N, d]
        latent_X = latent_X.T # [d, N]
        latent_Y = latent_Y.view(-1, latent_Y.size(-1))  # [N, d]
        latent_Y = latent_Y.T
        # X_pseudo_inv = torch.linalg.pinv(latent_X.T)  # Compute pseudo-inverse of latent_X
        # self.K = latent_Y.T @ X_pseudo_inv
        self.K = latent_Y @ torch.linalg.pinv(latent_X, rcond=1e-5)