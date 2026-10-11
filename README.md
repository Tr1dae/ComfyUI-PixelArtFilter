# ComfyUI Pixel Art Filter

ComfyUI nodes that turn generated images and video into clean pixel art: one perceptual palette, a collapsed logical pixel grid, and outlines that stay one cell wide and do not flicker between frames.

The video nodes are built for image batches from video models. Content that drifts across a fixed grid normally leaves border cells near a 50/50 vote that tips back and forth every frame. These nodes share one palette across the clip, pool each cell's votes over neighboring frames, hold a cell's color until a new one clearly wins, and draw dark outlines as their own layer. They run on the GPU: 120 frames at 640×640 take about 1.5 seconds on an RTX 4090.

All nodes are in the **PixelArtFilter** category.

## Credits

This pack started as a fork of [ComfyUI-Krea2-Pixel-Art-Refiner](https://github.com/envy-ai/ComfyUI-Krea2-Pixel-Art-Refiner) by [envy-ai](https://github.com/envy-ai), which provides the weighted OKLab palette clustering, modal grid collapse, mesh-detecting auto refiner, and animation pose compositor. The original repository was published without a license. The Temporal and Two-Pass refiners, line detection, and GPU processing are new in this pack.

The Pixelize, Outline Expansion, and Pre-Resize nodes are adapted from [PixelOE](https://github.com/KohakuBlueleaf/PixelOE) by Shin-Ying Yeh (KohakuBlueleaf) under the Apache License 2.0. The Auto Refiner's mesh detector is adapted from [Proper Pixel Art](https://github.com/KennethJAllen/proper-pixel-art) by Kenneth Allen under the MIT license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for both licenses and the changes made.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Tr1dae/ComfyUI-PixelArtFilter.git
```

Restart ComfyUI after installation. The only extra dependency is `opencv-python-headless`, used by the Auto Refiner.

## Nodes

| Node | Use it for |
| --- | --- |
| **Pixel Art Two-Pass Refiner** | Video. Stable fills plus one-cell outlines drawn as a separate layer. Start here for animation. |
| **Pixel Art Temporal Refiner** | Video. Temporally pooled, edge- and line-weighted grid votes. |
| **Pixel Art Refiner** | Single images or batches, with palette images, grid offsets, and processing order. |
| **Pixel Art Auto Refiner** | Images whose pixel grid size is unknown or uneven; detects the grid mesh. |
| **Pixel Art Simple Refiner** | Minimal palette reduction and grid collapse. |
| **Pixel Art Pixelize** | Photos and non-pixel-art renders: contrast-aware outline expansion and downscaling. |
| **Pixel Art Outline Expansion** | Outline expansion only, to feed other pixelizers. |
| **Pixel Art Pre-Resize** | Resizes to a target pixel count times a pixel size. |
| **Pixel Art Animation Pose Compositor** | Turns noisy repeated animation cycles into one clean frame per pose. |
| **Pixel Art Palette Reduce** | Palette reduction only. |
| **Pixel Art Grid Collapse** | Grid collapse only. |

The refiners expect input that already looks like pixel art on a grid, such as pixel-art generations from image or video models. For photos and ordinary renders, use Pixel Art Pixelize.

In the refiners, every grid cell takes its most frequent palette color. Ties go to the color perceptually closest to the cell's center pixel in OKLab.

## Pixel Art Two-Pass Refiner

Treats dark outlines as a separate layer instead of letting them compete with fill colors in each cell's vote, so outline contrast and width stay stable while the fill underneath settles independently.

1. Build one perceptual palette from pixels sampled across every frame and map each frame to it.
2. Detect dark outline pixels with a black top-hat on OKLab lightness over a two-cell window: how far a pixel is darker than its surroundings, only for dark features narrower than the window, minus `line_threshold`. Soft shading and large dark areas are ignored.
3. Fill pass: vote each cell's color, weighting pixels by `1 + edge_weight × edge` (OKLab Sobel magnitude normalized by the batch's 95th percentile, soft-thresholded by `edge_threshold`). Outline pixels only count in cells that contain nothing else.
4. Line pass: for every cell, measure the fraction covered by outline pixels and the smoothed outline response at its center. A straight outline straddling two cells is drawn only in the cell whose center is nearest the outline's center, so it stays one cell wide. Corners and small dots are drawn in every cell they cover. The outline color is the palette color of the cell's darkest outline pixels.
5. Pool fill votes and line evidence over `temporal_radius` frames on each side, weighted by `temporal_decay` raised to the frame distance. Apply `hysteresis` to fill color, outline color, and outline presence, then draw outline cells whose coverage reaches `line_coverage` over the fill.

| Input | Default | Description |
| --- | ---: | --- |
| `image` | — | Ordered video frames at a common resolution. |
| `width` | 64 | Logical pixel-grid width. Need not divide the image width. |
| `height` | 64 | Logical pixel-grid height. Need not divide the image height. |
| `colors` | 24 | Size of the palette shared by every frame. |
| `scale_to_original` | On | Expand the logical grid back to the input dimensions. |
| `temporal_radius` | 1 | Frames pooled on each side of the current frame. `0` disables pooling. |
| `temporal_decay` | 0.7 | Weight multiplier per frame of distance. |
| `hysteresis` | 0.15 | Margin new evidence must win by to change a cell's fill, outline color, or outline presence. `0` disables. |
| `edge_weight` | 1.0 | Extra fill vote weight for pixels on edges between fill colors. |
| `edge_threshold` | 0.1 | Normalized edge strength ignored as noise or texture. |
| `center_bias` | 0.0 | Down-weights fill pixels near cell borders. |
| `line_threshold` | 0.2 | OKLab lightness drop below the surroundings that an outline must exceed. |
| `line_coverage` | 0.25 | Fraction of a cell an outline must cover, where its center passes, to be drawn there. Lower keeps faint thin lines; higher removes doubled ones. |

The second output, `line_map`, shows outline strength for tuning `line_threshold`. Outlines wider than about two cells are handled by the fill pass.

## Pixel Art Temporal Refiner

A single-pass alternative where outlines win cells through extra vote weight rather than a separate layer.

1. Build one shared palette as above.
2. Weight each pixel's vote by `1 + edge_weight × edge + line_weight × line`, with `edge` and `line` as in the Two-Pass Refiner. A line color wins a cell from roughly `1 / (1 + weight)` coverage, so outlines survive thinning and fading. When a straight line straddles two cells, only the cell whose center is nearest the line's center gets the line weight; corners and small dots keep it in every cell they touch. `center_bias` lowers the weight of pixels toward a cell's border.
3. Sum the weighted votes per cell and palette color, normalize them per cell, and pool them over `temporal_radius` frames.
4. Choose the color with the most pooled votes. With hysteresis, a cell keeps its previous color while that color's votes are at least `1 − hysteresis` times the winner's.

Inputs match the Two-Pass Refiner, except `temporal_decay` defaults to 0.5, `line_threshold` to 0.15, and `line_coverage` is replaced by:

| Input | Default | Description |
| --- | ---: | --- |
| `line_weight` | 4.0 | Extra vote weight for pixels on thin dark lines such as outlines. |

The second output, `weight_map`, is the per-pixel vote weight scaled by the maximum possible weight. With pooling, hysteresis, edge and line weighting, and center bias all disabled, the output matches the Pixel Art Refiner with `shared_palette` and `allow_uneven_grid` enabled.

Both video nodes run on ComfyUI's GPU, asking ComfyUI to free VRAM from loaded models when the batch needs it, and return their outputs on the usual intermediate device. Their output is RGB; extra channels are dropped.

## Pixel Art Refiner

Palette reduction and modal grid collapse for single images and batches, in a selectable order.

| Input | Default | Description |
| --- | ---: | --- |
| `image` | — | Image, image batch, or video-frame batch. |
| `width` | 64 | Logical pixel-grid width. |
| `height` | 64 | Logical pixel-grid height. |
| `colors` | 24 | Maximum generated or reduced palette size. |
| `color_reduction_first` | On | Reduce colors before collapsing the pixel grid. |
| `scale_to_original` | On | Expand the logical grid back to the input dimensions. |
| `allow_uneven_grid` | Off | Permit grid cells of slightly different sizes when dimensions do not divide evenly. |
| `shared_palette` | Off | Generate and apply one palette across the entire batch. Enable this for video. |
| `x_offset` | 0.0 | Shift the grid right by a fraction of one logical pixel. |
| `y_offset` | 0.0 | Shift the grid down by a fraction of one logical pixel. |
| `auto_offset` | Off | Detect X/Y grid phase for every image and override the manual offsets. Detection runs per image, so leave it off for video. |
| `align_palette_lightness` | On | Compensate for a shared OKLab lightness shift before matching a supplied palette. |
| `preserve_generated_background` | On | Keep the first image's upper-left color instead of forcing it into the supplied palette. |
| `palette_image` | — | Optional image whose colors become the shared palette, reduced to `colors` when needed. |

Offsets wrap at the outer image edges. When a palette image contains no more than `colors` unique RGB values, those exact colors are kept; larger palettes are perceptually reduced first. Lightness alignment finds one OKLab L offset for the whole batch, applies it only while matching, and emits exact palette colors.

## Pixel Art Auto Refiner

Detects the pixel mesh instead of assuming an even grid, across the full frame or independently inside `direction_count` fixed horizontal regions (for example, a four-direction sprite sheet).

1. Reduce the frame to the requested perceptual palette.
2. Detect edges with Canny and morphological closing on a 2× upscale of each region.
3. Find grid lines with a probabilistic Hough transform, estimate the fundamental pixel pitch, merge false sub-pitch cells, and subdivide remaining gaps so local pitch variation is kept. `width` and `height` act as a hint, a minimum pitch, and a maximum grid size.
4. Collapse each mesh cell with modal voting, treat the upper-left color as transparent, optionally fit the result back to the original size, and composite it on white.

| Input | Default | Description |
| --- | ---: | --- |
| `width` / `height` | 64 | Grid size for manual mode, or fallback if detection fails. |
| `colors` | 24 | Maximum palette size. |
| `scale_to_original` | On | Fit and pad the collapsed frame to the input dimensions. |
| `shared_palette` | On | One palette across the batch. |
| `manual_resolution` | Off | Divide the frame evenly into `width × height` cells instead of detecting a mesh. |
| `direction_count` | 1 | Number of horizontal regions with independent meshes. |
| `align_palette_lightness` | On | As in the Pixel Art Refiner. |
| `preserve_generated_background` | On | As in the Pixel Art Refiner. |
| `palette_image` | — | Optional palette for every frame. |

For batches, meshes are detected from the first frame and reused for every frame, keeping sprite size and position stable.

## Pixel Art Simple Refiner

Palette reduction and grid collapse with only `width`, `height`, `colors`, `color_reduction_first`, and `scale_to_original`. Each image gets its own palette, and the image size must divide evenly by the grid.

## Pixel Art Pixelize

Converts images that are not pixel art, such as photos and ordinary renders, into pixel art. Each `pixel_size × pixel_size` block of the (edge-padded) input becomes one output pixel.

1. Outline expansion: compare each area's lightness median, minimum, and maximum, then blend an erosion and a dilation so thin details grow by `thickness` before downscaling. Dark details on bright surroundings get thicker, and bright details on dark surroundings too. Expansion weights are normalized across the whole batch, so video does not flicker.
2. Restore the input's low-frequency color, keeping the expanded details.
3. Downscale with `mode`:
   - `contrast`: per block, take the darkest or brightest lightness when the block is skewed toward it, otherwise the center pixel; chroma is the block median.
   - `k_centroid`: per block, split colors into two clusters and keep the larger cluster's mean.
   - `nearest` / `bilinear`: plain resampling.
4. With `colors` above 0, map every image to one perceptual palette built from the whole batch.

| Input | Default | Description |
| --- | ---: | --- |
| `pixel_size` | 4 | Source pixels per output pixel. |
| `thickness` | 2 | Outline expansion, 0 to 6. `0` disables it. |
| `mode` | contrast | Downscaling method. |
| `colors` | 0 | Shared palette size. `0` keeps all colors. |
| `scale_to_original` | On | Expand the result back to the input dimensions. |

The second output is the expansion weight map: bright areas erode, dark areas dilate.

## Pixel Art Outline Expansion

Runs only the outline expansion step, with `pixel_size` setting the contrast window and `thickness` from 1 to 6. Use it in front of other pixelizers or downscalers. Outputs the expanded image and its weight map.

## Pixel Art Pre-Resize

Resizes images with antialiased bicubic so the result is about `target_pixels × target_pixels` output pixels of `pixel_size` each, keeping the aspect ratio. Both dimensions come out as multiples of `pixel_size`, ready for Pixel Art Pixelize or the grid refiners.

## Pixel Art Animation Pose Compositor

Turns a noisy video batch of repeated animation cycles into one clean image per distinct pose.

1. Split the batch into held poses wherever the mean 8-bit RGB difference between consecutive frames reaches `transition_threshold`, rejoining holds split by a single transition frame.
2. Optionally remove frames that are robust RGB-distance outliers within their hold.
3. Assign holds to poses in cycle order. With `pose_count` at 0, the shortest cycle length where every later hold clearly matches its first-cycle pose is detected; this needs at least two complete cycles.
4. Take the most frequent exact RGB value at every pixel across each pose's frames, breaking ties by lowest luma.
5. Flood from the image borders through near-background pixels and replace them with the exact background color.

| Input | Default | Description |
| --- | ---: | --- |
| `images` | — | Ordered animation frames at a common resolution. |
| `pose_count` | 0 | Poses per cycle. `0` autodetects. |
| `transition_threshold` | 7.0 | Mean 8-bit RGB frame difference that starts a new held pose. |
| `remove_outliers` | On | Drop anomalous frames from each hold. |

## Pixel Art Palette Reduce and Grid Collapse

**Pixel Art Palette Reduce** reduces each image to an automatically selected palette of up to `colors` entries using weighted OKLab clustering. **Pixel Art Grid Collapse** collapses each `logical_width × logical_height` cell to its modal color and expands it back; the image size must divide evenly by the grid.

## Typical wiring

```text
Video decode → Pixel Art Two-Pass Refiner → Create Video
Image → Pixel Art Refiner → Save Image
Photo → Pixel Art Pre-Resize → Pixel Art Pixelize → Save Image
Video decode → Pixel Art Refiner → Animation Pose Compositor → Create Video
```
