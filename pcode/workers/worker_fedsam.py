# -*- coding: utf-8 -*-
import torch

from pcode.workers.worker import Worker


class WorkerFedSam(Worker):
    """FedSam client: standard local training followed by DP(w) upload."""

    def _apply_fedsam_dp_to_model(self):
        if not getattr(self.conf, "fedsam_dp", True):
            return

        clip_norm = float(self.conf.fedsam_dp_clip)
        noise_multiplier = float(self.conf.fedsam_dp_noise_multiplier)
        if clip_norm <= 0 and noise_multiplier <= 0:
            return

        global_params = {
            name: param.detach()
            for name, param in self.init_model.named_parameters()
        }
        local_params = dict(self.model.named_parameters())

        deltas = []
        for name, local_param in local_params.items():
            if name in global_params:
                global_param = global_params[name].to(local_param.device)
                deltas.append((local_param.detach() - global_param).reshape(-1))

        if len(deltas) == 0:
            return

        flat_delta = torch.cat(deltas)
        delta_norm = torch.norm(flat_delta, p=2)
        if clip_norm > 0:
            clip_coef = min(1.0, clip_norm / (delta_norm.item() + 1e-12))
        else:
            clip_coef = 1.0

        with torch.no_grad():
            for name, local_param in local_params.items():
                if name not in global_params:
                    continue
                global_param = global_params[name].to(local_param.device)
                clipped_delta = (local_param.detach() - global_param) * clip_coef
                if noise_multiplier > 0 and clip_norm > 0:
                    noise = torch.randn_like(clipped_delta) * (noise_multiplier * clip_norm)
                    clipped_delta = clipped_delta + noise
                local_param.copy_(global_param + clipped_delta)

        self.conf.logger.log(
            f"FedSam DP(w): clip_norm={clip_norm}, noise_multiplier={noise_multiplier}, "
            f"delta_norm={delta_norm.item()}, clip_coef={clip_coef}."
        )

    def _send_model_to_master(self):
        self._apply_fedsam_dp_to_model()
        super(WorkerFedSam, self)._send_model_to_master()