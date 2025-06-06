import torch
from torch import optim
from speech_enhancement import SpeechEnhancementLightningModule



class SelfSupervisionLightningModule(SpeechEnhancementLightningModule):
    def __init__(self, config):
        super(SelfSupervisionLightningModule, self).__init__(config)
        self.teach_model = self.model # Create a copy of the model for the teacher
        self.teach_model.eval()  # Set teacher model to evaluation mode

    def update_teacher(self, alpha=0.999):
        for teacher_param, student_param in zip(self.teach_model.parameters(), self.model.parameters()):
            teacher_param.data.mul_(alpha).add_((1 - alpha) * student_param.data)

    def compute_snr(self, batch):
        snr_mode = self.config['personalization']['snr_mode']
        # batch: {noisy_audio, vibration, audio, snr}
        batch_size = batch['noisy_audio'].shape[0]; device = batch['noisy_audio'].device
        if snr_mode == None:
            weights = torch.ones(batch_size, device=device)
        elif snr_mode == 'vibration_volume': # weights based on vibration volume (rms) / the whole batch
            vibration_rms = torch.sqrt(torch.mean(batch['vibration'] ** 2, dim=1))
            weights = vibration_rms / torch.sum(vibration_rms)
        elif snr_mode == 'vibration_audio':  # weights based on the proportion of vibration compared to audio
            vibration_rms = torch.sqrt(torch.mean(batch['vibration'] ** 2, dim=1))
            audio_rms = torch.sqrt(torch.mean(batch['noisy_audio'] ** 2, dim=1))
            scale = torch.clamp(vibration_rms / audio_rms, min=0.1, max=10.0)  # Avoid division by zero
            weights = scale / torch.sum(scale)
        elif snr_mode == 'snr_estimator':
            snr = batch['snr']  # Assuming snr is provided in the batch
            weights = snr / torch.sum(snr)
        else:
            raise ValueError(f"Unknown SNR mode: {snr_mode}")
        return weights
    
    def training_step(self, batch, batch_idx):
        # Ensure the student model is in training mode
        self.model.train()

        # Extract inputs and sources
        input_data = [batch[key] for key in self.config['input']]
        ref_output = batch[self.config['output']]

        # Forward pass through the teacher model (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teach_model(*input_data)
            teacher_outputs = teacher_outputs.squeeze(1)  # Shape: (batch_size, samples)
            teacher_noise = input_data[0] - teacher_outputs  # Assuming first input is the mixture

            random_indices = torch.randperm(ref_output.shape[0])
            random_noise = teacher_noise[random_indices]  # Randomly permute noise for diversity

        # Bootstrapped mixture creation
        batch['mixture'] = teacher_outputs + random_noise  # Replace original mixture
        batch[self.config['output']] = teacher_outputs  # Keep the original ground truth
        student_input_data = [batch[key] for key in self.config['input']]
        student_ref_output = batch[self.config['output']]  # Ground truth for student model

        # Forward pass through the student model (gradients enabled)
        student_outputs = self.model(*student_input_data)
        student_outputs = student_outputs.squeeze(1)  # Shape: (batch_size, samples)
        student_noise = student_input_data[0] - student_outputs  # Assuming first input is the mixture

        # Compute losses
        loss1 = self.loss(student_outputs, student_ref_output)
        loss2 = self.loss(student_noise, random_noise)

        # Apply weights based on SNR
        weights = self.compute_snr(batch)
        loss1 = (loss1 * weights).mean()
        loss2 = (loss2 * weights).mean()

        # Log losses
        self.log('train/loss1', loss1, on_step=True, prog_bar=True, logger=True)
        self.log('train/loss2', loss2, on_step=True, prog_bar=True, logger=True)

        # Update teacher model periodically
        if batch_idx % 1000 == 0:
            self.update_teacher()

        return loss1 + loss2

    def validation_step(self, batch, batch_idx):
        input_data = [batch[key] for key in self.config['input']]
        ref_output = batch[self.config['output']]  # Ground truth for validation
        outputs = self.model(*input_data)
        outputs = outputs.squeeze(1)
        loss = self.loss(outputs, ref_output).mean()
        mixture_loss = self.loss(input_data[0], ref_output).mean()
        self.log('val/mixture', mixture_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/sisnr', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/sisnr_i', loss - mixture_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.config.get('lr', 0.001))