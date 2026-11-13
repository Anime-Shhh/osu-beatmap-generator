# Our generative pipeline and dreamer's 27× compression

Focus: the non-autoregressive audio/native flow pipeline, the joint dreamer baseline and our adaptations. Historical autoregressive V2 is no longer the development focus.

## What 27× means

It counts time positions, not the total number of learned parameters or a guaranteed speedup. A chart starts as nine signals sampled every 98/16384 seconds, about 5.98 ms. Dreamer's chart encoder projects those channels into an internal feature width and applies three temporal reductions. In `UNetEncoder`, each stage processes features, applies a learned depthwise convolution, and then `AvgPool1d(3)` averages each group of three positions. Thus a sequence with 27,000 positions becomes 9,000, then 3,000, then 1,000. The temporal head projects each remaining position into six latent values, followed by RMS normalization. A separate style head produces 32 values shared across the chart.

The result has shape `[6, 1000]` in that example, not one vector of six values for the entire song. Learned filtering mixes neighboring information before pooling; these are not independent fixed summaries of individual 27-frame chunks.

The decoder projects the six channels back to an internal width and applies three 3× upsampling stages. Its `nn.Upsample(scale_factor=3)` repeats positions by default; convolution and learned processing refine the expanded features. At matching resolutions, the decoder also receives audio-encoder skip features. The final projection produces nine chart signals. Native decoding turns those signals into objects and curves.

This is learned, lossy compression. The decoder does not mathematically recover all discarded information. It uses learned chart regularities, the style code and audio features to reconstruct details. One compressed position spans roughly 161.5 ms, but the decoded timeline returns to about 5.98 ms frames and can represent multiple objects within that interval.

## Which encoder encodes what?

The chart encoder compresses a human map during codec training and latent-target preparation. Independent generation does not require that human map: the generator samples a new temporal latent from noise, conditioned on audio and style.

The audio encoder is a separate network. It extracts useful musical features at multiple resolutions. Its features guide both the latent generator and final decoder. An encoder is any learned transformation into another representation; it need not reduce dimensions. In our earlier flow code, the audio path initially expands 101 feature channels into 128 while preserving the 10 ms timeline.

## What our existing flow architecture does

1. Audio feature extraction supplies a known timestamped timeline. The model does not invent the song's clock.
2. `OnsetFlow` starts with a noisy scalar signal on that timeline. Repeated learned flow updates generate onset strength. Peaks become predicted object starts.
3. Supervised mark heads predict type and duration using audio plus onset context. These are classification/regression heads, not separate diffusion samplers.
4. `NativeGeometryFlow` predicts slider attributes and transports noisy point coordinates into object heads and slider control points, conditioned on events, settings and audio.
5. Native compilation, validation, difficulty calculation and packaging produce the deliverable.

Thus iterative noise-to-output generation is present in both rhythm and geometry. Technically this implementation uses flow matching, a related generative approach rather than the classic diffusion noise-prediction objective. Calling the entire system simply one encoder and one geometry decoder obscures these separate modules. Helper names imported from `src.v2` reuse geometry/format code and do not reintroduce the autoregressive planner into this path.

## Contrast with dreamer v8

| Aspect | Our earlier audio/native flow | Dreamer v8 / joint direction |
|---|---|---|
| Learned output representation | Onset signal and separate event/point geometry | Nine joint event/cursor signals through a learned codec |
| Generated state | Dense onset values, then explicit geometry | Six-channel compressed temporal latent |
| Rhythm/placement relationship | Placement receives already-selected events | Shared latent represents timing and movement together |
| Audio access | Rhythm and geometry both receive audio | Generator receives audio and decoder receives multiscale audio skips |
| Context | 20.48 s rhythm windows and 64-event geometry windows | Compressed whole-song inference |
| Sampling method | Linear-path flow matching | Distance/direction field with eight-step distance marching |
| AE/VAE | No joint chart autoencoder in this pipeline | Ordinary pretrained codec; our optional VAE changes the temporal posterior |

In our VAE trial, the encoder predicts mean and variance for each temporal code value, samples a code during training, and adds KL divergence regularization. This changes the codec, not the role of the separate generator. The latest 2,000-update dreamer fine-tune retained the published codec and added no VAE or KL term.

## Current decision

Use Amarel and focus on the joint dreamer-style architecture. Preserve verified checkpoints and data rather than recreating exact baselines through retraining. A bounded repeat checks migration, while architectural improvements still require validation across songs. The earlier VAE pilot did not demonstrate a downstream generation gain, so it remains an experiment rather than the default.

Source: pinned v8.0 `osu_dreamer/models/latent/unet.py`, `latent/model.py`, `inference/model.py`; our `src/beatmap_flow/rhythm.py`, `geometry.py`, `src/dreamer_adapter/model.py`; completed evaluation records. Upstream source https://github.com/jaswon/osu-dreamer/tree/v8.0.
