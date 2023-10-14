import torch
import itertools

class StabilizedPermInvSISDRMetric(torch.nn.Module):
    """!
    Class for SISDR computation between reconstructed and target signals."""

    def __init__(self,
                 zero_mean=False,
                 single_source=False,
                 n_estimated_sources=None,
                 n_actual_sources=None,
                 backward_loss=True,
                 improvement=False,
                 return_individual_results=False):
        """
        Initialization for the results and torch tensors that might
        be used afterwards
        :param batch_size: The number of the samples in each batch
        :param zero_mean: If you want to perform zero-mean across
        last dimension (time dim) of the signals before SDR computation
        """
        super().__init__()
        self.perform_zero_mean = zero_mean
        self.backward_loss = backward_loss
        self.improvement = improvement
        self.n_estimated_sources = n_estimated_sources
        self.n_actual_sources = n_actual_sources
        assert self.n_estimated_sources >= self.n_actual_sources, (
            'Estimates need to be at least: {} but got: {}'.format(
                self.n_actual_sources, self.n_estimated_sources))
        self.permutations = list(itertools.permutations(
            torch.arange(self.n_estimated_sources),
            r=self.n_actual_sources))
        self.permutations_tensor = torch.LongTensor(self.permutations)
        self.return_individual_results = return_individual_results
        self.single_source = single_source
        if self.single_source:
            assert self.n_actual_sources == 1

    def normalize_input(self, input_tensor):
        if self.perform_zero_mean:
            return input_tensor - torch.mean(input_tensor, dim=-1, keepdim=True)
        else:
            return input_tensor

    @staticmethod
    def dot(x, y):
        return torch.sum(x * y, dim=-1, keepdim=True)

    def compute_stabilized_sisnr(self,
                                 permuted_pr_batch,
                                 t_batch,
                                 t_signal_powers, eps=1e-8):
        pr_signal_powers = self.dot(permuted_pr_batch, permuted_pr_batch)
        inner_prod_sq = self.dot(permuted_pr_batch, t_batch) ** 2
        rho_sq = inner_prod_sq / (pr_signal_powers * t_signal_powers + eps)
        return 10 * torch.log10((rho_sq + eps) / (1. - rho_sq + eps))

    def compute_sisnr(self,
                      pr_batch,
                      t_batch,
                      eps=1e-8):

        assert t_batch.shape[-2] == self.n_actual_sources
        # The actual number of sources might be less than the estimated ones
        t_signal_powers = self.dot(t_batch, t_batch)

        sisnr_l = []
        for perm in self.permutations:
            permuted_pr_batch = pr_batch[:, perm, :]
            sisnr = self.compute_stabilized_sisnr(
                permuted_pr_batch, t_batch, t_signal_powers, eps=eps)
            sisnr_l.append(sisnr)
        all_sisnrs = torch.cat(sisnr_l, -1)
        best_sisdr, best_perm_ind = torch.max(all_sisnrs.mean(-2), -1)

        if self.improvement:
            initial_mixture = torch.sum(t_batch, -2, keepdim=True)
            initial_mixture = self.normalize_input(initial_mixture)
            initial_mix = initial_mixture.repeat(1, self.n_actual_sources, 1)
            base_sisdr = self.compute_stabilized_sisnr(
                initial_mix, t_batch, t_signal_powers, eps=eps)
            best_sisdr -= base_sisdr.mean()

        if not self.return_individual_results:
            best_sisdr = best_sisdr.mean()

        if self.backward_loss:
            return -best_sisdr, best_perm_ind
        return best_sisdr, best_perm_ind

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-9,
                return_best_permutation=False):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensors of size:
                         batch_size x self.n_sources x length_of_wavs
        :param t_batch: Target wavs: Torch Tensors of size:
                        batch_size x self.n_sources x length_of_wavs
        :param eps: Numerical stability constant.
        :returns results_buffer Buffer for loading the results directly
                 to gpu and not having to reconstruct the results matrix: Torch
                 Tensor of size: batch_size x 1
        """
        if self.single_source:
            pr_batch = torch.sum(pr_batch, -2, keepdim=True)

        pr_batch = self.normalize_input(pr_batch)
        t_batch = self.normalize_input(t_batch)

        sisnr_l, best_perm_ind = self.compute_sisnr(
            pr_batch, t_batch, eps=eps)
        if return_best_permutation:
            best_permutations = self.permutations_tensor[best_perm_ind]
            return sisnr_l, best_permutations
        else:
            return sisnr_l
def get_mask(acc, vad):
    '''
    1. 
    noise -> inactivity -> mask = 1 (determined by ~vad)
    the others (unlabelled) -> mask = 2
    2. 
    acc -> activity -> mask of 1
    others (include nothing, noise, speech) -> inactivity -> all 0
    '''
    mask = torch.zeros_like(acc)
    mask = torch.masked_fill(mask, ~vad.bool(), 1)
    ratio = 1 - torch.mean(vad)
    # threshold = filters.threshold_otsu(acc.numpy())
    # mask = (acc > threshold).to(dtype=torch.float32)
    return mask, ratio
def Spectral_Loss(x_mag, y_mag, vad=1):
    """Calculate forward propagation.
          Args:
              x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
              y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
              vad (Tensor): VAD of groundtruth signal (B, #frames, #freq_bins).
          Returns:
              Tensor: Spectral convergence loss value.
          """
    x_mag = torch.clamp(x_mag, min=1e-7)
    y_mag = torch.clamp(y_mag, min=1e-7)
    spectral_convergenge_loss =  torch.norm(vad * (y_mag - x_mag), p="fro") / torch.norm(y_mag, p="fro")
    log_stft_magnitude = (vad * (torch.log(y_mag) - torch.log(x_mag))).abs().mean()
    return 0.5 * spectral_convergenge_loss + 0.5 * log_stft_magnitude
def sisnr(x, s, eps=1e-8, vad=1):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return -20 * torch.log10(eps + l2norm(t * vad) / (l2norm((x_zm - t)*vad) + eps)).mean()