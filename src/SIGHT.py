import math

import torch
import torch.nn.functional as F

from src.cnn import SimpleHARCNN


class SIGHT:
    def __init__(self, model, device, num_classes, eta_mu=0.01, beta=1.0, eta_h=0.05, tau=0.2, eps=1e-8):
        self.model = model
        self.device = device
        self.K = num_classes
        self.eta_mu = eta_mu
        self.beta = beta
        self.eta_h = eta_h
        self.tau = tau
        self.eps = eps

        with torch.no_grad():
            W = model.classifier.weight.data.clone().to(device)
            self.mu_source = F.normalize(W, p=2, dim=1)
            self.mu = self.mu_source.clone()

        self.h = torch.ones(self.K, device=device) / self.K
        self.q_prev = None

    @torch.no_grad()
    def step(self, x):
        self.model.eval()

        z = self.model.encode(x)
        logits = self.model.classifier(z)
        z_vec = z[0]
        p = F.softmax(logits[0], dim=0)

        if self.q_prev is None:
            q = p.clone()
        else:
            z_hat = self.mu.T @ self.q_prev
            rho = F.cosine_similarity(z_vec.unsqueeze(0), z_hat.unsqueeze(0)).item()
            lam = 1.0 - math.exp(-self.beta * (1.0 - rho) ** 2)

            delta_z_obs = z_vec - z_hat
            delta_mu = self.mu - z_hat.unsqueeze(0)
            cos_sim = F.cosine_similarity(delta_z_obs.unsqueeze(0), delta_mu, dim=1)
            A_t = F.softmax(cos_sim / self.tau, dim=0)

            pi_fb = A_t * torch.sqrt(self.h)
            pi_fb = pi_fb / pi_fb.sum().clamp(min=self.eps)

            pi = (1.0 - lam) * self.q_prev + lam * pi_fb
            pi = pi.clamp(min=self.eps)

            q = p * pi
            q = q / q.sum().clamp(min=self.eps)

        if self.q_prev is not None:
            self.h = (1.0 - self.eta_h) * self.h + self.eta_h * self.q_prev

        anchor = 0.01
        for k in range(self.K):
            updated = (1.0 - self.eta_mu * q[k]) * self.mu[k] + self.eta_mu * q[k] * z_vec
            updated = (1.0 - anchor) * updated + anchor * self.mu_source[k]
            self.mu[k] = F.normalize(updated, p=2, dim=0)

        self.q_prev = q.clone()
        return q.unsqueeze(0)
