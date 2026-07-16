Here's a full walkthrough of the architecture — math, code, and visuals together. I'll go section by section.

---

## The three token streams

The model operates on three separate sequences simultaneously:

- **Main image tokens** $x^m \in \mathbb{R}^{B \times N_m \times d}$ — the noisy latent you want to denoise
- **Canvas / depth cond tokens** $x^c \in \mathbb{R}^{B \times N_c \times d}$ — reference conditioning packed along the sequence (bbox canvas latents first, optional depth-map latents second when `--depth_image_root` is set). Same modulation and masking story as below; **LoRA** uses `n_loras` matching the number of cond blocks (typically 1 canvas, or **2** for canvas + depth).
- **Text tokens** $x^t \in \mathbb{R}^{B \times L \times d}$ — the prompt

With unified resolution, typically $N_m = N_c$ (e.g. 8160 tokens each at 1024×1024).---

## Step 1 — Double-stream block: concat for joint attention

Before attention, each stream is normalized and modulated with AdaLayerNorm. Main and canvas use separate modulation parameters (canvas uses $t=0$-style modulation), then get **concatenated** along the sequence dimension:

$$\tilde{x}^m = (1 + \gamma_m)\,\mathrm{LN}(x^m) + \beta_m$$

$$\tilde{x}^c = (1 + \gamma_c)\,\mathrm{LN}(x^c) + \beta_c$$

$$\tilde{x}^{\mathrm{img}} = \mathrm{concat}(\tilde{x}^m,\, \tilde{x}^c) \in \mathbb{R}^{B \times (N_m + N_c) \times d}$$

Text is normalized separately:

$$\tilde{x}^t = (1 + \gamma_t)\,\mathrm{LN}(x^t) + \beta_t$$

This is exactly what lines 198–209 of `flux2_transformer_cond.py` do — `norm_img_in = torch.cat([norm_hidden_states, norm_cond], dim=1)`.---

## Step 2 — Attention: how features actually mix

Let $S = L + N_m + N_c$ be the full joint sequence length. Q/K/V are computed (text and image projections concatenated), then:

$$\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right) V$$

where $M_{ij}$ is $0$ for allowed pairs and $-10^{20}$ for forbidden ones. After the softmax, the output is split back:

$$[\underbrace{\text{enc}}_L \;|\; \underbrace{\text{main}}_{N_m} \;|\; \underbrace{\text{cond}}_{N_c}] \leftarrow \text{out} \in \mathbb{R}^{B \times S \times d}$$

This is the `split_with_sizes` calls at lines 52–99.

---

## Step 3 — The attention mask $M$

This is the key to canvas isolation. The mask is built as a $(S \times S)$ matrix:

$$M_{ij} = \begin{cases} 0 & \text{if row } i \in \text{text or main tokens} \\ 0 & \text{if row } i \in \text{canvas}_k \text{ AND col } j \in \text{canvas}_k \\ -10^{20} & \text{otherwise (canvas cross-attending globally)} \end{cases}$$

In words: **main image and text can attend to everything** (including canvas). **Each canvas block can only attend to itself** — not to other canvas blocks and not back to main/text in an unrestricted way.---

## Step 4 — LoRA: low-rank delta on Q/K/V

Standard LoRA factorizes a weight update into two small matrices. For a linear $W \in \mathbb{R}^{d \times d}$, LoRA introduces $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$ with $r \ll d$:

$$W_{\text{eff}} = W + \frac{\alpha}{r}\, B A$$

Applied to the hidden states $H \in \mathbb{R}^{B \times S \times d}$:

$$Q = H\,W_Q + \sum_{i=1}^{n_\text{loras}} w_i \cdot \Delta Q_i(H)$$

where $\Delta Q_i(H) = \mathrm{up}_i(\mathrm{down}_i(\mathbf{m}_i \odot H))$ — the **token mask** $\mathbf{m}_i$ zeroes out everything except the $i$-th canvas block. Concretely from `LoRALinearLayerFlux2.forward`:

$$\mathbf{m}_i[b, t, :] = \begin{cases} 1 & \text{if } t \in [\text{block\_size} + i \cdot c_s,\; \text{block\_size} + (i+1) \cdot c_s) \\ 0 & \text{otherwise} \end{cases}$$

where $c_s = \frac{W/8 \cdot H/8 \cdot 16}{64}$ is the canvas token count. So **each LoRA only "sees" its own canvas block**, not the main image or other canvas blocks.---

## Step 5 — Residual updates and FFN

After attention, both streams get gated residual updates:

$$x^m \leftarrow x^m + g_m \cdot \mathrm{Attn}_m$$
$$x^c \leftarrow x^c + g_c \cdot \mathrm{Attn}_c$$

Then each stream has its own FFN branch (both share the same `self.ff` weight matrix, applied independently to each stream's normalized hidden states).

---

## Step 6 — Training signal

The loss is flow-matching on **main tokens only**:

$$\mathcal{L} = \mathbb{E}\!\left[\left\|f_\theta(\cdot) - (z - x^m_\text{clean})\right\|^2_w\right]$$

The gradient $\partial \mathcal{L}/\partial \theta$ flows back through the attention cross-attention path — i.e. through the main token outputs that depended on canvas keys/values. This is what forces the model to *learn* to use canvas information usefully: there's no supervision on canvas tokens at all, so LoRA parameters update purely because using canvas improved the main-stream prediction.

---

## Full forward pass — putting it together---

## Summary table

| Mechanism | Math | Role |
|---|---|---|
| AdaLN modulation | $(1+\gamma)\,\mathrm{LN}(h) + \beta$ | Timestep-conditioned normalization per stream |
| Concat | $[\tilde{x}^m \| \tilde{x}^c]$ | Puts main+canvas in one sequence for joint attention |
| Joint attention | $\mathrm{softmax}(QK^\top/\sqrt{d_h} + M)\,V$ | Actually mixes features; main queries read canvas keys |
| Mask $M$ | $M_{ij} \in \{0, -10^{20}\}$ | Main sees canvas; canvas only sees itself |
| LoRA token mask | $\mathbf{m}_i \odot H$ before $BA$ | Each LoRA specializes to its own canvas block |
| LoRA weight delta | $W_\text{eff} = W + \frac{\alpha}{r}BA$ | Low-rank correction to Q/K/V/proj for canvas-aware attention |
| Flow-matching loss | $\|f_\theta - (z - x^m_\text{clean})\|^2$ | Supervises only main tokens → forces LoRA to learn useful canvas usage |

The core insight is that **LoRA doesn't transport canvas features by itself** — it just reshapes how Q/K/V are computed so that the masked attention (which can already see canvas) does a better job of extracting useful information from canvas tokens into the main stream.
