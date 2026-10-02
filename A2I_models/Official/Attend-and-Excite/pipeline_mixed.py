import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from utils.ptp_utils import AttentionStore, aggregate_attention

# --- 1. Helper for Smoothing (Required for A-Star Masks) ---
class GaussianSmoothing(nn.Module):
    def __init__(self, channels=1, kernel_size=3, sigma=0.5):
        super(GaussianSmoothing, self).__init__()
        x_cord = torch.arange(kernel_size).float()
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        mean = (kernel_size - 1) / 2.
        variance = sigma**2.

        gaussian_kernel = torch.exp(-((x_grid - mean)**2 + (y_grid - mean)**2) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

        self.register_buffer('weight', gaussian_kernel)
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, input):
        return F.conv2d(input, self.weight, groups=self.groups, padding=self.padding)


class AttendAndExcitePipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]
    
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds

    
    # --- A-STAR LOSS HELPER ---
    def _compute_astar_loss(self, attention_store, indices_to_alter, step_idx, prev_soft_masks):
        """ Computes Segregation and Retention losses based on spatial attention maps. """
        # Get full spatial attention maps: [Heads, Res, Res] -> [Res, Res, Tokens]
        # We assume 16x16 resolution for the loss calculation
        attn_maps = aggregate_attention(attention_store, res=16, from_where=("up", "down", "mid"), is_cross=True, select=0)
        
        device = attn_maps.device
        attn = attn_maps.permute(2, 0, 1) # [Tokens, H, W]
        
        smoother = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5).to(device)
        object_data = []
        temp = 20.0 # Temperature for soft mask

        # Generate soft masks for current objects
        for tok_idx in indices_to_alter:
            raw_map = attn[tok_idx].unsqueeze(0).unsqueeze(0)
            smoothed_map = smoother(raw_map)
            thr = smoothed_map.mean()
            soft_mask = torch.sigmoid((smoothed_map - thr) * temp).reshape(-1)
            object_data.append({'mask': soft_mask})

        # 1. Segregation Loss (Spatial Overlap)
        loss_seg = torch.tensor(0.0).to(device)
        num_objs = len(object_data)
        if num_objs > 1:
            for i in range(num_objs):
                for j in range(i + 1, num_objs):
                    inter = (object_data[i]['mask'] * object_data[j]['mask']).sum()
                    union = (object_data[i]['mask'] + object_data[j]['mask']).sum()
                    loss_seg += inter / (union + 1e-6)
            loss_seg /= (num_objs * (num_objs - 1) / 2)

        # 2. Retention Loss (Temporal Consistency)
        loss_ret = torch.tensor(0.0).to(device)
        if step_idx > 0 and len(prev_soft_masks) == num_objs:
            for i in range(num_objs):
                old_mask = prev_soft_masks[i]
                curr_mask = object_data[i]['mask']
                # We want to minimize the difference/maximize overlap with previous step
                inter = torch.min(old_mask, curr_mask).sum()
                union = torch.max(old_mask, curr_mask).sum()
                loss_ret += (1.0 - inter / (union + 1e-6))
            loss_ret /= num_objs

        current_masks = [obj['mask'].detach() for obj in object_data]
        return loss_seg, loss_ret, current_masks

    @staticmethod
    def _compute_loss(max_attention_per_index: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        if return_losses:
            return loss, losses
        else:
            return loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    
    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore, indices_to_alter: List[int], attention_res: int = 16, smooth_attentions: bool = False, sigma: float = 0.5, kernel_size: int = 3, normalize_eot: bool = False):
        """ Aggregates the attention for each token and calculates the max activation value. """
        attention_maps = aggregate_attention(
            attention_store, attention_res, ("up", "down", "mid"), True, 0
        )
        max_attention_per_index = []
        for i in indices_to_alter:
            image = attention_maps[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma).to(image.device)
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            max_attention_per_index.append(image.max())
        return max_attention_per_index

    def _perform_iterative_refinement_step(self,
                                        latents: torch.Tensor,
                                        indices_to_alter: List[int],
                                        loss: torch.Tensor,
                                        threshold: float,
                                        text_embeddings: torch.Tensor,
                                        text_input,
                                        attention_store: AttentionStore,
                                        step_size: float,
                                        t: int,
                                        step_idx: int,
                                        prev_masks: List,
                                        attention_res: int = 16,
                                        smooth_attentions: bool = True,
                                        sigma: float = 0.5,
                                        kernel_size: int = 3,
                                        max_refinement_steps: int = 20,
                                        normalize_eot: bool = False):
        
        iteration = 0
        target_loss = max(0, 1. - threshold)
        updated_masks = prev_masks 

        # Loop until the standard excitation loss is below threshold
        while loss > target_loss:
            iteration += 1
            
            latents = latents.clone().detach().requires_grad_(True)
            # Forward pass to get attention maps
            _ = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            # 1. Calculate Standard Excitation Loss
            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                attention_store, indices_to_alter, attention_res, smooth_attentions, sigma, kernel_size, normalize_eot
            )
            loss_excite, losses = self._compute_loss(max_attention_per_index, return_losses=True)

            # 2. Calculate A-Star Loss (Segregation + Retention)
            loss_seg, loss_ret, current_iter_masks = self._compute_astar_loss(
                attention_store, indices_to_alter, step_idx, prev_masks
            )
            updated_masks = current_iter_masks 

            # 3. Combine Losses
            # Weights: Excite=1.0, Segregation=2.0, Retention=0.5
            total_loss = loss_excite + (2.0 * loss_seg) + (0.5 * loss_ret)

            if loss_excite != 0:
                latents = self._update_latent(latents, total_loss, step_size)

            # Check if we should stop (based on standard loss)
            if iteration >= max_refinement_steps:
                break
            
            # Update 'loss' variable for the while loop condition
            loss = loss_excite

        # Run one final time to update attention store and masks for the next main step
        latents = latents.clone().detach().requires_grad_(True)
        _ = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()
        
        # Get final values to return
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store, indices_to_alter, attention_res, smooth_attentions, sigma, kernel_size, normalize_eot
        )
        loss_excite = self._compute_loss(max_attention_per_index)
        _, _, final_masks = self._compute_astar_loss(attention_store, indices_to_alter, step_idx, prev_masks)

        return loss_excite, latents, max_attention_per_index, final_masks

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
    ):
        # 0. Default height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        # --- FIX IS HERE: Use a different variable name for channel count ---
        num_channels_latents = self.unet.in_channels 
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents, # Pass the count here
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents, # Pass the actual latents (or None) here
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))
        
        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        prev_bin_mask_lst = [] # A-Star mask tracking

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                # Reset masks if we are past the alteration window
                if i >= max_iter_to_alter:
                    prev_bin_mask_lst = []

                with torch.enable_grad():
                    latents = latents.clone().detach().requires_grad_(True)
                    
                    # Forward pass
                    _ = self.unet(latents, t, encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                    self.unet.zero_grad()

                    # Get Max Attention
                    max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                        attention_store=attention_store,
                        indices_to_alter=indices_to_alter,
                        attention_res=attention_res,
                        smooth_attentions=smooth_attentions,
                        sigma=sigma,
                        kernel_size=kernel_size,
                        normalize_eot=sd_2_1)

                    if not run_standard_sd:
                        loss = self._compute_loss(max_attention_per_index=max_attention_per_index)

                        # --- ITERATIVE REFINEMENT ---
                        if i in thresholds.keys() and loss > 1. - thresholds[i]:
                            torch.cuda.empty_cache()
                            loss, latents, max_attention_per_index, updated_masks = self._perform_iterative_refinement_step(
                                latents=latents,
                                indices_to_alter=indices_to_alter,
                                loss=loss,
                                threshold=thresholds[i],
                                text_embeddings=prompt_embeds,
                                text_input=text_inputs,
                                attention_store=attention_store,
                                step_size=scale_factor * np.sqrt(scale_range[i]),
                                t=t,
                                step_idx=i,
                                prev_masks=prev_bin_mask_lst,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_eot=sd_2_1)
                            
                            prev_bin_mask_lst = updated_masks

                        # --- SINGLE STEP UPDATE ---
                        if i < max_iter_to_alter:
                            # 1. Standard Loss
                            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                                attention_store, indices_to_alter, attention_res, smooth_attentions, sigma, kernel_size, sd_2_1
                            )
                            loss_excite = self._compute_loss(max_attention_per_index)
                            
                            # 2. A-Star Loss
                            loss_seg, loss_ret, iter_masks = self._compute_astar_loss(
                                attention_store, indices_to_alter, i, prev_bin_mask_lst
                            )
                            
                            # 3. Update
                            total_loss = loss_excite + (2.0 * loss_seg) + (0.5 * loss_ret)
                            
                            if loss_excite != 0:
                                latents = self._update_latent(latents=latents, loss=total_loss,
                                                              step_size=scale_factor * np.sqrt(scale_range[i]))
                                prev_bin_mask_lst = iter_masks

                # Standard Scheduler Step
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        image = self.decode_latents(latents)
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)