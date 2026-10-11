import cv2
import numpy as np
import torch
from PIL import Image

import comfy.model_management
from comfy_api.latest import io


PALETTE_TRAINING_COLORS = 65536
PALETTE_ASSIGNMENT_CHUNK = 16384
EDGE_SCALE_SAMPLES = 1 << 20


def srgb_to_oklab(rgb):
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    red, green, blue = linear.unbind(-1)
    l = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    m = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    s = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    l, m, s = l.pow(1.0 / 3.0), m.pow(1.0 / 3.0), s.pow(1.0 / 3.0)
    return torch.stack((
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ), dim=-1)


def oklab_to_srgb(lab):
    lightness, a, b = lab.unbind(-1)
    l = (lightness + 0.3963377774 * a + 0.2158037573 * b).pow(3)
    m = (lightness - 0.1055613458 * a - 0.0638541728 * b).pow(3)
    s = (lightness - 0.0894841775 * a - 1.2914855480 * b).pow(3)
    red = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    green = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    blue = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    linear = torch.stack((red, green, blue), dim=-1)
    return torch.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear.clamp_min(0.0).pow(1.0 / 2.4) - 0.055)


def nearest_palette_labels(colors, palette):
    labels = torch.empty(colors.shape[0], dtype=torch.long, device=colors.device)
    for start in range(0, colors.shape[0], PALETTE_ASSIGNMENT_CHUNK):
        chunk = colors[start:start + PALETTE_ASSIGNMENT_CHUNK]
        distances = (chunk[:, None, :] - palette[None, :, :]).square().sum(dim=-1)
        labels[start:start + chunk.shape[0]] = distances.argmin(dim=1)
    return labels


def perceptual_palette(colors, counts, color_count):
    color_count = min(color_count, colors.shape[0])
    weights = counts.to(dtype=colors.dtype)
    first = weights.argmax()
    palette = colors[first].unsqueeze(0)
    min_distances = (colors - palette[0]).square().sum(dim=1)
    for _ in range(1, color_count):
        next_color = (min_distances * weights.sqrt()).argmax()
        new_center = colors[next_color]
        palette = torch.cat((palette, new_center.unsqueeze(0)))
        distance = (colors - new_center).square().sum(dim=1)
        min_distances = torch.minimum(min_distances, distance)

    for _ in range(12):
        labels = nearest_palette_labels(colors, palette)
        weighted_colors = colors * weights[:, None]
        sums = torch.zeros_like(palette)
        totals = torch.zeros(color_count, dtype=weights.dtype, device=weights.device)
        sums.scatter_add_(0, labels[:, None].expand(-1, 3), weighted_colors)
        totals.scatter_add_(0, labels, weights)
        palette = torch.where(totals[:, None] > 0, sums / totals.clamp_min(1.0)[:, None], palette)
    return palette


def unique_image_colors(image):
    rgb8 = (image[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[..., 0] << 16) | (rgb8[..., 1] << 8) | rgb8[..., 2]
    unique_codes, inverse, counts = torch.unique(color_codes.flatten(), return_inverse=True, return_counts=True)
    unique_rgb = torch.stack((
        (unique_codes >> 16) & 255,
        (unique_codes >> 8) & 255,
        unique_codes & 255,
    ), dim=1).to(dtype=image.dtype) / 255.0
    unique_oklab = srgb_to_oklab(unique_rgb)
    return unique_codes, inverse, counts, unique_oklab


def map_unique_colors(image, inverse, unique_oklab, palette_oklab, palette_rgb):
    labels = nearest_palette_labels(unique_oklab, palette_oklab)
    reduced_rgb = palette_rgb[labels[inverse]].reshape(image.shape[0], image.shape[1], 3)
    if image.shape[-1] == 3:
        return reduced_rgb
    return torch.cat((reduced_rgb, image[..., 3:]), dim=-1)


def reduce_image_palette(image, color_count):
    unique_codes, inverse, counts, unique_oklab = unique_image_colors(image)
    if unique_codes.shape[0] > PALETTE_TRAINING_COLORS:
        positions = torch.linspace(0, inverse.shape[0] - 1, PALETTE_TRAINING_COLORS, device=image.device).round().long()
        training_indices, training_counts = torch.unique(inverse[positions], return_counts=True)
        training_oklab = unique_oklab[training_indices]
    else:
        training_oklab = unique_oklab
        training_counts = counts
    palette_oklab = perceptual_palette(training_oklab, training_counts, color_count)
    palette_rgb = (oklab_to_srgb(palette_oklab).clamp(0.0, 1.0) * 255.0).round() / 255.0
    return map_unique_colors(image, inverse, unique_oklab, palette_oklab, palette_rgb)


def shared_perceptual_palette(images, color_count):
    colors, counts = sampled_image_colors(images)
    return perceptual_palette(colors, counts, color_count)


def palette_from_image(palette_image, additional_color, color_count, dtype, device):
    palette_pixels = palette_image[..., :3].to(device=device, dtype=dtype).reshape(-1, 3)
    rgb8 = (palette_pixels.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]
    unique_codes, counts = torch.unique(color_codes, return_counts=True)
    palette_rgb = torch.stack((
        (unique_codes >> 16) & 255,
        (unique_codes >> 8) & 255,
        unique_codes & 255,
    ), dim=1).to(dtype=dtype) / 255.0
    if unique_codes.shape[0] > color_count:
        training_rgb = palette_rgb
        training_counts = counts
        if unique_codes.shape[0] > PALETTE_TRAINING_COLORS:
            positions = torch.linspace(0, color_codes.shape[0] - 1, PALETTE_TRAINING_COLORS,
                                       dtype=torch.float64, device=device).round().long()
            training_codes, training_counts = torch.unique(color_codes[positions], return_counts=True)
            training_rgb = torch.stack((
                (training_codes >> 16) & 255,
                (training_codes >> 8) & 255,
                training_codes & 255,
            ), dim=1).to(dtype=dtype) / 255.0
        palette_oklab = perceptual_palette(srgb_to_oklab(training_rgb), training_counts, color_count)
        palette_rgb = (oklab_to_srgb(palette_oklab).clamp(0.0, 1.0) * 255.0).round() / 255.0

    if additional_color is not None:
        palette_rgb = torch.cat((palette_rgb, additional_color.to(device=device, dtype=dtype).reshape(1, 3)))
    rgb8 = (palette_rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]
    unique_codes = torch.unique(color_codes)
    palette_rgb = torch.stack((
        (unique_codes >> 16) & 255,
        (unique_codes >> 8) & 255,
        unique_codes & 255,
    ), dim=1).to(dtype=dtype) / 255.0
    return palette_rgb, srgb_to_oklab(palette_rgb)


def fixed_palette_from_image(palette_image, background_color, color_count, dtype, device, preserve_background):
    reference_rgb, reference_oklab = palette_from_image(palette_image, None, color_count, dtype, device)
    background = (background_color.to(device=device, dtype=dtype).clamp(0.0, 1.0) * 255.0).round() / 255.0 if preserve_background else None
    if background is None:
        return reference_rgb, reference_oklab, reference_oklab, None
    palette_rgb = torch.cat((reference_rgb, background.reshape(1, 3)))
    rgb8 = (palette_rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]
    unique_codes = torch.unique(color_codes)
    palette_rgb = torch.stack(((unique_codes >> 16) & 255, (unique_codes >> 8) & 255, unique_codes & 255), dim=1).to(dtype=dtype) / 255.0
    return palette_rgb, srgb_to_oklab(palette_rgb), reference_oklab, background


def sampled_image_colors(images, excluded_color=None):
    pixels = images[..., :3].reshape(-1, 3)
    sample_count = min(PALETTE_TRAINING_COLORS, pixels.shape[0])
    if sample_count < pixels.shape[0]:
        positions = torch.linspace(0, pixels.shape[0] - 1, sample_count, dtype=torch.float64, device=images.device).round().long()
        pixels = pixels[positions]
    rgb8 = (pixels.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    if excluded_color is not None:
        excluded8 = (excluded_color.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
        included = torch.any(rgb8 != excluded8, dim=1)
        if torch.any(included):
            rgb8 = rgb8[included]
    color_codes = (rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]
    unique_codes, counts = torch.unique(color_codes, return_counts=True)
    unique_rgb = torch.stack(((unique_codes >> 16) & 255, (unique_codes >> 8) & 255, unique_codes & 255), dim=1).to(dtype=images.dtype) / 255.0
    return srgb_to_oklab(unique_rgb), counts


def palette_alignment_error(colors, counts, palette, lightness_offset):
    total = torch.zeros((), dtype=colors.dtype, device=colors.device)
    weights = counts.to(dtype=colors.dtype)
    for start in range(0, colors.shape[0], PALETTE_ASSIGNMENT_CHUNK):
        chunk = colors[start:start + PALETTE_ASSIGNMENT_CHUNK]
        lightness = chunk[:, None, 0] + lightness_offset - palette[None, :, 0]
        chroma = chunk[:, None, 1:] - palette[None, :, 1:]
        distances = lightness.square() + chroma.square().sum(dim=-1)
        total += (distances.amin(dim=1) * weights[start:start + chunk.shape[0]]).sum()
    return total / weights.sum()


def estimate_palette_lightness_offset(images, palette, excluded_color=None):
    colors, counts = sampled_image_colors(images, excluded_color)
    offsets = torch.linspace(-0.2, 0.2, 17, dtype=colors.dtype, device=colors.device)
    errors = torch.stack([palette_alignment_error(colors, counts, palette, offset) for offset in offsets])
    best = offsets[errors.argmin()]
    offsets = torch.linspace(best - 0.025, best + 0.025, 9, dtype=colors.dtype, device=colors.device).clamp(-0.25, 0.25)
    errors = torch.stack([palette_alignment_error(colors, counts, palette, offset) for offset in offsets])
    return offsets[errors.argmin()]


def reduce_image_batch(images, color_count, shared_palette=False, fixed_palette=None, align_palette_lightness=False):
    if fixed_palette is None and not shared_palette:
        return torch.stack([reduce_image_palette(image, color_count) for image in images])

    if fixed_palette is None:
        palette_oklab = shared_perceptual_palette(images, color_count)
        palette_rgb = (oklab_to_srgb(palette_oklab).clamp(0.0, 1.0) * 255.0).round() / 255.0
    else:
        palette_rgb, palette_oklab = fixed_palette[:2]

    lightness_offset = None
    background = None
    if fixed_palette is not None and len(fixed_palette) == 4:
        reference_oklab = fixed_palette[2]
        background = fixed_palette[3]
        if align_palette_lightness:
            lightness_offset = estimate_palette_lightness_offset(images, reference_oklab, background)

    output = torch.empty_like(images)
    for index, image in enumerate(images):
        _, inverse, _, unique_oklab = unique_image_colors(image)
        matching_oklab = unique_oklab
        if lightness_offset is not None:
            matching_oklab = unique_oklab.clone()
            matching_oklab[:, 0] += lightness_offset
        mapped = map_unique_colors(image, inverse, matching_oklab, palette_oklab, palette_rgb)
        if background is not None:
            rgb8 = (image[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
            background8 = (background.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
            mask = torch.all(rgb8 == background8, dim=-1, keepdim=True)
            mapped_rgb = torch.where(mask, background.reshape(1, 1, 3), mapped[..., :3])
            mapped = mapped_rgb if mapped.shape[-1] == 3 else torch.cat((mapped_rgb, mapped[..., 3:]), dim=-1)
        output[index] = mapped
    return output


class PixelArtFilterPaletteReduce(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtFilterPaletteReduce",
            display_name="Pixel Art Palette Reduce",
            description="Reduces each image to an automatically selected palette using perceptual OKLab color distance.",
            category="PixelArtFilter",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("colors", default=24, min=2, max=256, step=1,
                             tooltip="Maximum number of colors in each image's palette."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, colors):
        if image.shape[-1] < 3:
            raise ValueError(f"Perceptual palette reduction requires at least 3 image channels, got {image.shape[-1]}")
        return io.NodeOutput(torch.stack([reduce_image_palette(batch_image, colors) for batch_image in image]))


def select_modal_pixels(pixels, cell_ids, cell_count, center_pixels):
    rgb8 = (pixels[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[..., 0] << 16) | (rgb8[..., 1] << 8) | rgb8[..., 2]
    color_keys = (cell_ids << 24) | color_codes
    unique_keys, key_inverse, counts = torch.unique(color_keys, return_inverse=True, return_counts=True)
    unique_cells = unique_keys >> 24
    unique_colors = unique_keys & 0xFFFFFF

    max_counts = torch.zeros(cell_count, dtype=counts.dtype, device=pixels.device)
    max_counts.scatter_reduce_(0, unique_cells, counts, reduce="amax")
    modal = counts == max_counts[unique_cells]

    candidate_rgb = torch.stack((
        (unique_colors >> 16) & 255,
        (unique_colors >> 8) & 255,
        unique_colors & 255,
    ), dim=1).to(dtype=pixels.dtype) / 255.0
    center_oklab = srgb_to_oklab(center_pixels[..., :3].clamp(0.0, 1.0))
    candidate_oklab = srgb_to_oklab(candidate_rgb)
    distances = (candidate_oklab - center_oklab[unique_cells]).square().sum(dim=1)
    modal_distances = torch.where(modal, distances, torch.inf)
    min_distances = torch.full((cell_count,), torch.inf, dtype=pixels.dtype, device=pixels.device)
    min_distances.scatter_reduce_(0, unique_cells, modal_distances, reduce="amin")
    closest = modal & torch.isclose(distances, min_distances[unique_cells], rtol=1e-5, atol=1e-8)

    eligible = closest[key_inverse]
    positions = torch.arange(pixels.shape[0], device=pixels.device)
    first_positions = torch.full((cell_count,), pixels.shape[0], dtype=torch.long, device=pixels.device)
    first_positions.scatter_reduce_(0, cell_ids, torch.where(eligible, positions, pixels.shape[0]), reduce="amin")
    return pixels[first_positions]


def detect_axis_offsets(edge_energy, size, logical_size):
    candidate_count = max(1, (size + logical_size - 1) // logical_size)
    if logical_size == 1 or candidate_count == 1:
        return torch.zeros(edge_energy.shape[0], dtype=torch.long, device=edge_energy.device)

    period = max(1, round(size / logical_size))
    periodic_energy = torch.zeros_like(edge_energy)
    if period < edge_energy.shape[1]:
        paired_energy = (edge_energy[:, :-period] * edge_energy[:, period:]).sqrt()
        periodic_energy[:, :-period] += paired_energy
        periodic_energy[:, period:] += paired_energy
    else:
        periodic_energy = edge_energy

    boundaries = torch.arange(logical_size, device=edge_energy.device) * size // logical_size
    scores = []
    for offset in range(candidate_count):
        positions = (boundaries + offset) % size
        positions = positions[positions > 0]
        scores.append(periodic_energy[:, positions - 1].mean(dim=1))
    return torch.stack(scores, dim=1).argmax(dim=1)


def detect_pixel_grid_offsets(image, logical_width, logical_height):
    rgb = image[..., :3]
    horizontal_edges = (rgb[:, :, 1:] - rgb[:, :, :-1]).abs().mean(dim=(1, 3))
    vertical_edges = (rgb[:, 1:] - rgb[:, :-1]).abs().mean(dim=(2, 3))
    column_offsets = detect_axis_offsets(horizontal_edges, image.shape[2], logical_width)
    row_offsets = detect_axis_offsets(vertical_edges, image.shape[1], logical_height)
    return column_offsets, row_offsets


def collapse_single_image_grid(image, logical_width, logical_height, scale_to_original, allow_uneven_grid,
                               column_offset, row_offset):
    height, width, channels = image.shape
    if column_offset or row_offset:
        image = torch.roll(image, shifts=(-row_offset, -column_offset), dims=(0, 1))
    if allow_uneven_grid and (height % logical_height != 0 or width % logical_width != 0):
        row_bounds = torch.arange(logical_height + 1, device=image.device) * height // logical_height
        column_bounds = torch.arange(logical_width + 1, device=image.device) * width // logical_width
        row_ids = torch.repeat_interleave(torch.arange(logical_height, device=image.device), row_bounds[1:] - row_bounds[:-1])
        column_ids = torch.repeat_interleave(torch.arange(logical_width, device=image.device), column_bounds[1:] - column_bounds[:-1])
        cell_count = logical_height * logical_width
        cell_ids = (row_ids[:, None] * logical_width + column_ids[None, :]).reshape(-1)
        center_rows = row_bounds[:-1] + (row_bounds[1:] - row_bounds[:-1]) // 2
        center_columns = column_bounds[:-1] + (column_bounds[1:] - column_bounds[:-1]) // 2
        center_pixels = image[center_rows[:, None], center_columns[None, :], :].reshape(-1, channels)
        selected = select_modal_pixels(image.reshape(-1, channels), cell_ids, cell_count, center_pixels)
        logical = selected.reshape(logical_height, logical_width, channels)
        if scale_to_original:
            output = logical.index_select(0, row_ids).index_select(1, column_ids)
    else:
        block_height = height // logical_height
        block_width = width // logical_width
        block_pixels = block_height * block_width
        cells = image.reshape(logical_height, block_height, logical_width, block_width, channels)
        cells = cells.permute(0, 2, 1, 3, 4).reshape(-1, block_pixels, channels)
        cell_count = cells.shape[0]
        cell_ids = torch.arange(cell_count, device=image.device)[:, None].expand(cell_count, block_pixels).reshape(-1)
        center_index = (block_height // 2) * block_width + block_width // 2
        selected = select_modal_pixels(cells.reshape(-1, channels), cell_ids, cell_count, cells[:, center_index])
        logical = selected.reshape(logical_height, logical_width, channels)
        if scale_to_original:
            output = logical.repeat_interleave(block_height, dim=0).repeat_interleave(block_width, dim=1)

    if not scale_to_original:
        return logical
    if column_offset or row_offset:
        output = torch.roll(output, shifts=(row_offset, column_offset), dims=(0, 1))
    return output


def collapse_pixel_grid_mode(image, logical_width, logical_height, scale_to_original=True, allow_uneven_grid=False,
                             x_offset=0.0, y_offset=0.0, detected_offsets=None):
    batch_size, height, width, channels = image.shape
    if detected_offsets is None:
        column_offsets = [int(x_offset * width / logical_width + 0.5)] * batch_size
        row_offsets = [int(y_offset * height / logical_height + 0.5)] * batch_size
    else:
        column_offsets = detected_offsets[0].tolist()
        row_offsets = detected_offsets[1].tolist()

    output_height = height if scale_to_original else logical_height
    output_width = width if scale_to_original else logical_width
    output = torch.empty((batch_size, output_height, output_width, channels), dtype=image.dtype, device=image.device)
    for index, batch_image in enumerate(image):
        output[index] = collapse_single_image_grid(batch_image, logical_width, logical_height, scale_to_original,
                                                   allow_uneven_grid, column_offsets[index], row_offsets[index])
    return output


class PixelArtFilterGridCollapse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtFilterGridCollapse",
            display_name="Pixel Art Grid Collapse",
            description="Collapses each logical pixel to its modal color and expands it back with nearest-neighbor scaling.",
            category="PixelArtFilter",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("logical_width", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels across the result."),
                io.Int.Input("logical_height", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels down the result."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, logical_width, logical_height):
        height, width = image.shape[1:3]
        if logical_width > width or logical_height > height:
            raise ValueError(f"Logical grid {logical_width}x{logical_height} cannot exceed image size {width}x{height}")
        if width % logical_width != 0 or height % logical_height != 0:
            raise ValueError(f"Image size {width}x{height} must be divisible by logical grid {logical_width}x{logical_height}")
        if logical_width == width and logical_height == height:
            return io.NodeOutput(image)

        return io.NodeOutput(collapse_pixel_grid_mode(image, logical_width, logical_height))


def frame_edge_map(frame, scale):
    rgb = (frame[..., :3].detach().to(device="cpu").clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    if frame.shape[-1] > 3:
        alpha = frame[..., 3].detach().to(device="cpu").numpy()
        rgb[alpha < 0.5] = 255
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    if rgb.shape[0] <= 4 or rgb.shape[1] <= 4:
        raise ValueError("Pixel art auto refiner requires images larger than 4x4 pixels")
    grey = np.asarray(Image.fromarray(rgb[2:-2, 2:-2]).convert("L"))
    edges = cv2.Canny(grey, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def cluster_mesh_lines(lines, threshold=4):
    lines = sorted(lines)
    clusters = [[lines[0]]]
    for position in lines[1:]:
        if position - clusters[-1][-1] <= threshold:
            clusters[-1].append(position)
        else:
            clusters.append([position])
    return [int(np.median(cluster)) for cluster in clusters]


def detect_mesh_lines(edges):
    detected = cv2.HoughLinesP(
        edges, 1.0, np.deg2rad(1.0), 100, minLineLength=50, maxLineGap=10
    )
    height, width = edges.shape
    lines_x = [0, width - 1]
    lines_y = [0, height - 1]
    if detected is not None:
        for x1, y1, x2, y2 in detected.reshape(-1, 4):
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            if angle > np.deg2rad(75):
                lines_x.append(round((x1 + x2) / 2))
            elif angle < np.deg2rad(15):
                lines_y.append(round((y1 + y2) / 2))
    return cluster_mesh_lines(lines_x), cluster_mesh_lines(lines_y)


def estimate_mesh_pixel_width(mesh, minimum_pixel_width):
    gap_sets = [np.diff(lines)[1:-1] for lines in mesh if len(lines) > 3]
    if not gap_sets:
        return minimum_pixel_width
    gaps = np.concatenate(gap_sets)
    gaps = gaps[gaps >= minimum_pixel_width * 0.75]
    if not len(gaps):
        return minimum_pixel_width

    maximum_pixel_width = max(minimum_pixel_width, np.percentile(gaps, 80))
    candidate_count = min(4096, max(2, int(np.ceil((maximum_pixel_width - minimum_pixel_width) * 20)) + 1))
    candidates = np.linspace(minimum_pixel_width, maximum_pixel_width, candidate_count)
    scores = []
    for candidate in candidates:
        multiples = np.maximum(1, np.round(gaps / candidate))
        residuals = np.abs(gaps - multiples * candidate)
        scores.append((np.median(residuals), np.mean(np.minimum(residuals, candidate * 0.25)), candidate))
    hint_score = scores[0]
    if hint_score[0] <= minimum_pixel_width * 0.2 and hint_score[1] <= minimum_pixel_width * 0.2:
        return minimum_pixel_width
    return min(scores)[2]


def filter_weak_mesh_lines(lines, edges, pixel_width):
    lines = list(lines)
    radius = max(1, round(pixel_width * 0.2))
    while len(lines) > 2:
        support = [np.count_nonzero(edges[:, max(0, position - radius):position + radius + 1]) for position in lines]
        remove = []
        for index in range(1, len(lines) - 1):
            close = min(lines[index] - lines[index - 1], lines[index + 1] - lines[index]) < pixel_width * 0.75
            surrounding_support = (support[index - 1] + support[index + 1]) / 2
            if close and support[index] < surrounding_support * 0.7:
                remove.append(index)
        if not remove:
            break
        lines = [position for index, position in enumerate(lines) if index not in remove]
    return lines


def merge_half_mesh_cells(lines, pixel_width):
    lines = list(lines)
    while len(lines) > 3:
        gaps = np.diff(lines)
        merged = False
        for index, (first, second) in enumerate(zip(gaps, gaps[1:])):
            combined = first + second
            if first < pixel_width * 0.8 and second < pixel_width * 0.8 and pixel_width * 0.75 <= combined <= pixel_width * 1.25:
                del lines[index + 1]
                merged = True
                break
        if not merged:
            break
    return lines


def filter_short_mesh_cells(lines, edges, pixel_width):
    lines = list(lines)
    radius = max(1, round(pixel_width * 0.2))
    while len(lines) > 2:
        gaps = np.diff(lines)
        short = np.flatnonzero(gaps < pixel_width * 0.8)
        if not len(short):
            break
        index = int(short[0])
        choices = []
        if index > 0:
            merged = lines[index + 1] - lines[index - 1]
            residual = abs(merged - max(1, round(merged / pixel_width)) * pixel_width)
            support = np.count_nonzero(edges[:, max(0, lines[index] - radius):lines[index] + radius + 1])
            choices.append((residual, support, index))
        if index + 1 < len(lines) - 1:
            merged = lines[index + 2] - lines[index]
            residual = abs(merged - max(1, round(merged / pixel_width)) * pixel_width)
            support = np.count_nonzero(edges[:, max(0, lines[index + 1] - radius):lines[index + 1] + radius + 1])
            choices.append((residual, support, index + 1))
        if not choices:
            break
        del lines[min(choices)[2]]
    return lines


def homogenize_mesh_lines(lines, pixel_width):
    completed = []
    for start, end in zip(lines, lines[1:]):
        cell_count = max(1, int(np.round((end - start) / pixel_width)))
        cell_width = (end - start) / cell_count
        completed.extend(start + int(index * cell_width) for index in range(cell_count))
    completed.append(lines[-1])
    return completed


def mesh_from_edges(edges, minimum_pixel_width, maximum_width, maximum_height):
    initial = detect_mesh_lines(edges)
    if len(initial[0]) in (2, 3) and len(initial[1]) in (2, 3):
        return initial
    pixel_width = estimate_mesh_pixel_width(initial, minimum_pixel_width)
    lines_x = filter_weak_mesh_lines(merge_half_mesh_cells(initial[0], pixel_width), edges, pixel_width)
    lines_y = filter_weak_mesh_lines(merge_half_mesh_cells(initial[1], pixel_width), edges.T, pixel_width)
    lines_x = filter_short_mesh_cells(lines_x, edges, pixel_width)
    lines_y = filter_short_mesh_cells(lines_y, edges.T, pixel_width)

    def exceeds_target(candidate):
        cells_x = sum(max(1, round(gap / candidate)) for gap in np.diff(lines_x))
        cells_y = sum(max(1, round(gap / candidate)) for gap in np.diff(lines_y))
        return cells_x > maximum_width or cells_y > maximum_height

    if exceeds_target(pixel_width):
        lower = pixel_width
        upper = max(edges.shape)
        for _ in range(24):
            middle = (lower + upper) / 2
            if exceeds_target(middle):
                lower = middle
            else:
                upper = middle
        pixel_width = upper
    return homogenize_mesh_lines(lines_x, pixel_width), homogenize_mesh_lines(lines_y, pixel_width)


def usable_mesh(mesh, maximum_width, maximum_height):
    lines_x, lines_y = mesh
    return (2 <= len(lines_x) <= maximum_width + 1 and 2 <= len(lines_y) <= maximum_height + 1
            and not (len(lines_x) in (2, 3) and len(lines_y) in (2, 3)))


def regular_mesh(image_width, image_height, logical_width, logical_height):
    lines_x = [round(index * image_width / logical_width) for index in range(logical_width + 1)]
    lines_y = [round(index * image_height / logical_height) for index in range(logical_height + 1)]
    return lines_x, lines_y


def detect_pixel_mesh(frame, fallback_width, fallback_height):
    minimum_pixel_width = min(frame.shape[1] / fallback_width, frame.shape[0] / fallback_height)
    mesh = mesh_from_edges(frame_edge_map(frame, 2), minimum_pixel_width * 2, fallback_width, fallback_height)
    if usable_mesh(mesh, fallback_width, fallback_height):
        return mesh, 2
    mesh = mesh_from_edges(frame_edge_map(frame, 1), minimum_pixel_width, fallback_width, fallback_height)
    if usable_mesh(mesh, fallback_width, fallback_height):
        return mesh, 1
    return regular_mesh(frame.shape[1], frame.shape[0], fallback_width, fallback_height), 1


def select_modal_rgba(pixels, cell_ids, cell_count, center_pixels):
    rgba8 = (pixels[..., :4].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgba8[..., 0] << 24) | (rgba8[..., 1] << 16) | (rgba8[..., 2] << 8) | rgba8[..., 3]
    color_keys = (cell_ids << 32) | color_codes
    unique_keys, key_inverse, counts = torch.unique(color_keys, return_inverse=True, return_counts=True)
    unique_cells = unique_keys >> 32
    unique_colors = unique_keys & 0xFFFFFFFF

    max_counts = torch.zeros(cell_count, dtype=counts.dtype, device=pixels.device)
    max_counts.scatter_reduce_(0, unique_cells, counts, reduce="amax")
    modal = counts == max_counts[unique_cells]

    candidate_rgba = torch.stack((
        (unique_colors >> 24) & 255,
        (unique_colors >> 16) & 255,
        (unique_colors >> 8) & 255,
        unique_colors & 255,
    ), dim=1).to(dtype=pixels.dtype) / 255.0
    center_oklab = srgb_to_oklab(center_pixels[..., :3].clamp(0.0, 1.0))
    candidate_oklab = srgb_to_oklab(candidate_rgba[:, :3])
    distances = (candidate_oklab - center_oklab[unique_cells]).square().sum(dim=1)
    distances += (candidate_rgba[:, 3] - center_pixels[unique_cells, 3]).square()
    modal_distances = torch.where(modal, distances, torch.inf)
    min_distances = torch.full((cell_count,), torch.inf, dtype=pixels.dtype, device=pixels.device)
    min_distances.scatter_reduce_(0, unique_cells, modal_distances, reduce="amin")
    closest = modal & torch.isclose(distances, min_distances[unique_cells], rtol=1e-5, atol=1e-8)

    eligible = closest[key_inverse]
    positions = torch.arange(pixels.shape[0], device=pixels.device)
    first_positions = torch.full((cell_count,), pixels.shape[0], dtype=torch.long, device=pixels.device)
    first_positions.scatter_reduce_(0, cell_ids, torch.where(eligible, positions, pixels.shape[0]), reduce="amin")
    return pixels[first_positions]


def collapse_frame_mesh(frame, mesh, scale):
    rgb8 = (frame[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    background = rgb8[0, 0]
    visible = torch.any(rgb8 != background, dim=-1)
    if frame.shape[-1] > 3:
        visible &= frame[..., 3] > 0
    rgba = torch.zeros((*frame.shape[:2], 4), dtype=frame.dtype, device=frame.device)
    rgba[..., :3] = frame[..., :3]
    rgba[..., 3] = visible.to(dtype=frame.dtype)
    if scale > 1:
        rgba = rgba.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)

    lines_x, lines_y = mesh
    row_lengths = torch.tensor(np.diff(lines_y), device=frame.device)
    column_lengths = torch.tensor(np.diff(lines_x), device=frame.device)
    logical_height = len(lines_y) - 1
    logical_width = len(lines_x) - 1
    row_ids = torch.repeat_interleave(torch.arange(logical_height, device=frame.device), row_lengths)
    column_ids = torch.repeat_interleave(torch.arange(logical_width, device=frame.device), column_lengths)
    pixels = rgba[:lines_y[-1], :lines_x[-1]]
    cell_ids = (row_ids[:, None] * logical_width + column_ids[None, :]).reshape(-1)
    center_rows = torch.tensor(lines_y[:-1], device=frame.device) + row_lengths // 2
    center_columns = torch.tensor(lines_x[:-1], device=frame.device) + column_lengths // 2
    center_pixels = rgba[center_rows[:, None], center_columns[None, :]].reshape(-1, 4)
    selected = select_modal_rgba(pixels.reshape(-1, 4), cell_ids, logical_height * logical_width, center_pixels)
    return selected.reshape(logical_height, logical_width, 4)


def fit_frame_to_size(frame, width, height):
    scale = min(width / frame.shape[1], height / frame.shape[0])
    scaled_width = max(1, min(width, round(frame.shape[1] * scale)))
    scaled_height = max(1, min(height, round(frame.shape[0] * scale)))
    scaled = torch.nn.functional.interpolate(
        frame.permute(2, 0, 1).unsqueeze(0), size=(scaled_height, scaled_width), mode="nearest-exact"
    )[0].permute(1, 2, 0)
    output = torch.zeros((height, width, 4), dtype=frame.dtype, device=frame.device)
    top = (height - scaled_height) // 2
    left = (width - scaled_width) // 2
    output[top:top + scaled_height, left:left + scaled_width] = scaled
    return output


def composite_white(image):
    alpha = image[..., 3:4].clamp(0.0, 1.0)
    return image[..., :3] * alpha + (1.0 - alpha)


def split_animation_holds(images, transition_threshold):
    rgb8 = (images[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int16)
    differences = (rgb8[1:] - rgb8[:-1]).abs().to(torch.float32).mean(dim=(1, 2, 3))
    boundaries = [index + 1 for index, difference in enumerate(differences.tolist()) if difference >= transition_threshold]
    starts = [0] + boundaries
    ends = boundaries + [images.shape[0]]
    return [images[start:end] for start, end in zip(starts, ends)]


def animation_hold_representative(hold):
    frames = (hold[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.float32).reshape(hold.shape[0], -1)
    scores = torch.zeros(hold.shape[0], device=hold.device)
    for frame in frames:
        scores += (frames - frame).abs().mean(dim=1)
    return frames[scores.argmin()]


def filter_animation_hold_outliers(hold):
    while hold.shape[0] > 2:
        representative = animation_hold_representative(hold)
        frames = (hold[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.float32).reshape(hold.shape[0], -1)
        distances = (frames - representative).abs().mean(dim=1)
        median = distances.median()
        deviation = (distances - median).abs().median()
        limit = median + torch.maximum(3 * deviation, torch.ones((), device=hold.device))
        remove = distances > limit
        if not remove.any().item():
            break
        hold = hold[~remove]
    return hold


def merge_interrupted_animation_holds(holds):
    holds = list(holds)
    while len(holds) >= 3:
        representatives = torch.stack([animation_hold_representative(hold) for hold in holds])
        merged = False
        for index in range(1, len(holds) - 1):
            if holds[index].shape[0] != 1:
                continue
            across = (representatives[index - 1] - representatives[index + 1]).abs().mean()
            left = (representatives[index - 1] - representatives[index]).abs().mean()
            right = (representatives[index] - representatives[index + 1]).abs().mean()
            if across * 2 < torch.minimum(left, right):
                holds[index - 1:index + 2] = [torch.cat((holds[index - 1], holds[index + 1]))]
                merged = True
                break
        if not merged:
            break
    return holds


def detect_animation_cycle(holds):
    if len(holds) == 1:
        return 1, 0
    representatives = torch.stack([animation_hold_representative(hold) for hold in holds])
    distances = torch.empty((len(holds), len(holds)), device=representatives.device)
    for index, representative in enumerate(representatives):
        distances[index] = (representatives - representative).abs().mean(dim=1)
    distances = distances.tolist()

    for start in range(len(holds) - 3):
        remaining = len(holds) - start
        for pose_count in range(2, remaining // 2 + 1):
            matches = True
            for index in range(start + pose_count, len(holds)):
                phase = start + (index - start) % pose_count
                wrong_distance = min(
                    distances[index][candidate]
                    for candidate in range(start, start + pose_count)
                    if candidate != phase
                )
                if distances[index][phase] * 2 >= wrong_distance:
                    matches = False
                    break
            if matches:
                return pose_count, start
    raise ValueError(
        f"Could not find a repeated cycle among {len(holds)} held poses. "
        "Set pose_count explicitly, adjust transition_threshold, or provide at least two complete cycles."
    )


def detect_animation_pose_count(holds):
    return detect_animation_cycle(holds)[0]


def modal_rgb_composite(frames):
    frame_count, height, width = frames.shape[:3]
    rgb8 = (frames[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[..., 0] << 16) | (rgb8[..., 1] << 8) | rgb8[..., 2]
    pixel_count = height * width
    pixel_ids = torch.arange(pixel_count, device=frames.device).repeat(frame_count)
    color_keys = (pixel_ids << 24) | color_codes.reshape(-1)
    unique_keys, counts = torch.unique(color_keys, return_counts=True)
    unique_pixels = unique_keys >> 24
    unique_colors = unique_keys & 0xFFFFFF

    max_counts = torch.zeros(pixel_count, dtype=counts.dtype, device=frames.device)
    max_counts.scatter_reduce_(0, unique_pixels, counts, reduce="amax")
    modal = counts == max_counts[unique_pixels]

    red = (unique_colors >> 16) & 255
    green = (unique_colors >> 8) & 255
    blue = unique_colors & 255
    luma = 2126 * red + 7152 * green + 722 * blue
    rank = luma * 0x1000000 + unique_colors
    sentinel = torch.iinfo(torch.int64).max
    selected = torch.full((pixel_count,), sentinel, dtype=torch.int64, device=frames.device)
    selected.scatter_reduce_(0, unique_pixels, torch.where(modal, rank, sentinel), reduce="amin")
    selected &= 0xFFFFFF
    rgb = torch.stack(((selected >> 16) & 255, (selected >> 8) & 255, selected & 255), dim=1)
    return rgb.to(dtype=frames.dtype).reshape(height, width, 3) / 255.0


def remove_connected_background_halo(image, threshold=0.1):
    background = image[0, 0]
    near_background = (image - background).square().mean(dim=-1) < threshold * threshold
    connected = torch.zeros_like(near_background)
    connected[0] = near_background[0]
    connected[-1] = near_background[-1]
    connected[:, 0] = near_background[:, 0]
    connected[:, -1] = near_background[:, -1]
    while True:
        expanded = torch.nn.functional.max_pool2d(
            connected.to(dtype=image.dtype)[None, None], 3, stride=1, padding=1
        )[0, 0].bool() & near_background
        if torch.equal(expanded, connected):
            break
        connected = expanded
    return torch.where(connected[..., None], background, image)


class PixelArtFilterPoseCompositor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtFilterPoseCompositor",
            display_name="Pixel Art Animation Pose Compositor",
            description="Separates held animation poses, combines repeated cycles with an exact RGB mode, and removes connected near-background halos.",
            category="PixelArtFilter",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("pose_count", default=0, min=0, max=256, step=1,
                             tooltip="Number of distinct poses in one animation cycle. Set to 0 to autodetect from repeated cycles."),
                io.Float.Input("transition_threshold", default=7.0, min=0.0, max=255.0, step=0.1,
                               tooltip="Minimum mean absolute 8-bit RGB difference between consecutive frames that starts a new held pose."),
                io.Boolean.Input("remove_outliers", default=True,
                                 tooltip="Remove frames that are robust RGB-distance outliers within their held pose."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, images, pose_count, transition_threshold, remove_outliers=True):
        if images.shape[-1] < 3:
            raise ValueError(f"Animation pose compositing requires at least 3 image channels, got {images.shape[-1]}")
        holds = merge_interrupted_animation_holds(split_animation_holds(images, transition_threshold))
        if remove_outliers:
            holds = [filter_animation_hold_outliers(hold) for hold in holds]
        if pose_count == 0:
            pose_count, cycle_start = detect_animation_cycle(holds)
            holds = holds[cycle_start:]
        if len(holds) < pose_count:
            raise ValueError(
                f"Detected {len(holds)} held poses, fewer than pose_count={pose_count}. "
                "Adjust transition_threshold or trim the input batch."
            )
        composites = [remove_connected_background_halo(modal_rgb_composite(torch.cat(holds[pose::pose_count])))
                      for pose in range(pose_count)]
        return io.NodeOutput(torch.stack(composites))


class PixelArtFilterRefiner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtFilterRefiner",
            display_name="Pixel Art Refiner",
            description="Combines perceptual palette reduction with a modal pixel-grid collapse in a selectable order.",
            category="PixelArtFilter",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels across the refined image."),
                io.Int.Input("height", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels down the refined image."),
                io.Int.Input("colors", default=24, min=2, max=256, step=1,
                             tooltip="Maximum number of colors in each image's palette."),
                io.Boolean.Input("color_reduction_first", default=True,
                                 tooltip="Reduce the palette before grid collapse. Disable for collapse-then-reduce behavior."),
                io.Boolean.Input("scale_to_original", default=True,
                                 tooltip="Expand the logical pixel grid back to the input dimensions."),
                io.Boolean.Input("allow_uneven_grid", default=False,
                                 tooltip="Allow input dimensions that are not evenly divisible by the logical grid."),
                io.Boolean.Input("shared_palette", default=False,
                                 tooltip="Generate one palette from the entire image batch instead of a separate palette for each image."),
                io.Float.Input("x_offset", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Shift the grid right by this fraction of one logical pixel."),
                io.Float.Input("y_offset", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Shift the grid down by this fraction of one logical pixel."),
                io.Boolean.Input("auto_offset", default=False,
                                 tooltip="Detect the X and Y grid phase separately for each image. Overrides x_offset and y_offset."),
                io.Boolean.Input("align_palette_lightness", default=True,
                                 tooltip="Compensate for a shared OKLab lightness shift before matching a supplied palette."),
                io.Boolean.Input("preserve_generated_background", default=True,
                                 tooltip="Keep the first input image's upper-left color instead of forcing it into the supplied palette."),
                io.Image.Input("palette_image", optional=True,
                               tooltip="Use this image's colors as the palette for the entire batch. Palettes above colors are reduced first. Overrides shared_palette."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, colors, color_reduction_first, scale_to_original, allow_uneven_grid=False,
                shared_palette=False, palette_image=None, x_offset=0.0, y_offset=0.0, auto_offset=False,
                align_palette_lightness=True, preserve_generated_background=True):
        image_height, image_width = image.shape[1:3]
        if image.shape[-1] < 3:
            raise ValueError(f"Pixel art refinement requires at least 3 image channels, got {image.shape[-1]}")
        if palette_image is not None and palette_image.shape[-1] < 3:
            raise ValueError(f"Palette image requires at least 3 image channels, got {palette_image.shape[-1]}")
        if width > image_width or height > image_height:
            raise ValueError(f"Logical grid {width}x{height} cannot exceed image size {image_width}x{image_height}")
        if not allow_uneven_grid and (image_width % width != 0 or image_height % height != 0):
            raise ValueError(f"Image size {image_width}x{image_height} must be divisible by logical grid {width}x{height}")

        fixed_palette = None
        if palette_image is not None:
            fixed_palette = fixed_palette_from_image(palette_image, image[0, 0, 0, :3], colors, image.dtype, image.device,
                                                     preserve_generated_background)
        detected_offsets = detect_pixel_grid_offsets(image, width, height) if auto_offset else None

        if color_reduction_first:
            reduced = reduce_image_batch(image, colors, shared_palette, fixed_palette, align_palette_lightness)
            return io.NodeOutput(collapse_pixel_grid_mode(reduced, width, height, scale_to_original, allow_uneven_grid,
                                                          x_offset, y_offset, detected_offsets))

        collapsed = collapse_pixel_grid_mode(image, width, height, scale_to_original, allow_uneven_grid, x_offset, y_offset,
                                             detected_offsets)
        return io.NodeOutput(reduce_image_batch(collapsed, colors, shared_palette, fixed_palette, align_palette_lightness))


def grid_axis(size, logical_size, device):
    bounds = torch.arange(logical_size + 1, device=device) * size // logical_size
    lengths = bounds[1:] - bounds[:-1]
    ids = torch.repeat_interleave(torch.arange(logical_size, device=device), lengths)
    centers = bounds[:-1] + lengths // 2
    offsets = ((torch.arange(size, device=device) + 0.5 - bounds[:-1][ids]) / lengths[ids] * 2.0 - 1.0).abs()
    return ids, centers, offsets


def batch_edge_strength(images, edge_threshold):
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=images.dtype, device=images.device)
    kernels = torch.stack((kernel_x, kernel_x.T)).unsqueeze(1).repeat(3, 1, 1, 1)
    edges = torch.empty(images.shape[:3], dtype=images.dtype, device=images.device)
    for index, frame in enumerate(images):
        lab = srgb_to_oklab(frame[..., :3].clamp(0.0, 1.0)).permute(2, 0, 1).unsqueeze(0)
        gradients = torch.nn.functional.conv2d(torch.nn.functional.pad(lab, (1, 1, 1, 1), mode="replicate"), kernels, groups=3)
        edges[index] = gradients.square().sum(dim=1)[0].sqrt()

    # One scale for the whole batch so the weighting itself does not flicker.
    samples = edges.flatten()
    positions = torch.linspace(0, samples.shape[0] - 1, min(EDGE_SCALE_SAMPLES, samples.shape[0]),
                               dtype=torch.float64, device=images.device).round().long()
    samples = samples[positions]
    samples = samples[samples > 1e-4]
    if samples.shape[0] == 0:
        return edges.zero_()
    scale = torch.quantile(samples.float(), 0.95)
    return edges.div_(scale).sub_(edge_threshold).div_(1.0 - edge_threshold).clamp_(0.0, 1.0)


def video_processing_device(image):
    # Room for the input, the refined output, and the per-pixel weight and line maps.
    device = comfy.model_management.get_torch_device()
    comfy.model_management.free_memory(3 * image.nbytes + comfy.model_management.minimum_inference_memory(), device)
    return device


def separable_max_pool(values, kernel_height, kernel_width):
    values = torch.nn.functional.max_pool2d(values, (1, kernel_width), stride=1, padding=(0, kernel_width // 2))
    return torch.nn.functional.max_pool2d(values, (kernel_height, 1), stride=1, padding=(kernel_height // 2, 0))


def dark_line_strength(frame, kernel_height, kernel_width, line_threshold):
    # Black top-hat on OKLab lightness: high only on dark features narrower than the kernel, such as outlines.
    lightness = srgb_to_oklab(frame[..., :3].clamp(0.0, 1.0))[..., 0][None, None]
    closed = -separable_max_pool(-separable_max_pool(lightness, kernel_height, kernel_width), kernel_height, kernel_width)
    return (closed - lightness)[0, 0].sub_(line_threshold).div_(1.0 - line_threshold).clamp_(0.0, 1.0)


def line_kernel_sizes(height, width, logical_height, logical_width):
    cell_height = round(height / logical_height)
    cell_width = round(width / logical_width)
    return (2 * cell_height + 1, 2 * cell_width + 1), (cell_height // 2 * 2 + 1, cell_width // 2 * 2 + 1)


def line_cell_features(strength, cell_ids, center_positions, cell_count, kernel_height, kernel_width):
    smooth = strength[None, None]
    for _ in range(2):
        smooth = torch.nn.functional.avg_pool2d(smooth, (1, kernel_width), stride=1, padding=(0, kernel_width // 2),
                                                count_include_pad=False)
        smooth = torch.nn.functional.avg_pool2d(smooth, (kernel_height, 1), stride=1, padding=(kernel_height // 2, 0),
                                                count_include_pad=False)
    smooth = smooth[0, 0]
    gradient_x = torch.nn.functional.pad(smooth[:, 1:] - smooth[:, :-1], (0, 1)).square().flatten()
    gradient_y = torch.nn.functional.pad(smooth[1:] - smooth[:-1], (0, 0, 0, 1)).square().flatten()
    energy_x = torch.zeros(cell_count, dtype=smooth.dtype, device=smooth.device).index_add_(0, cell_ids, gradient_x)
    energy_y = torch.zeros(cell_count, dtype=smooth.dtype, device=smooth.device).index_add_(0, cell_ids, gradient_y)
    return smooth.flatten()[center_positions], energy_x, energy_y


def line_owners(centers, energy_x, energy_y, logical_width, logical_height):
    # The line features peak along each line's center. A straight line straddling two cells is owned only by the cell
    # whose center is nearest that peak across the line, instead of both. Corners and blobs are owned by every cell.
    centers = centers.view(logical_height, logical_width)
    padded = torch.nn.functional.pad(centers, (1, 1, 1, 1))
    vertical_peak = (centers >= torch.maximum(padded[:-2, 1:-1], padded[2:, 1:-1])).flatten()
    horizontal_peak = (centers >= torch.maximum(padded[1:-1, :-2], padded[1:-1, 2:])).flatten()
    return torch.where(energy_y > 2.0 * energy_x, vertical_peak, torch.where(energy_x > 2.0 * energy_y, horizontal_peak, True))


def add_dark_line_weights(weights, images, logical_width, logical_height, line_weight, line_threshold):
    height, width = images.shape[1:3]
    row_ids, center_rows, _ = grid_axis(height, logical_height, images.device)
    column_ids, center_columns, _ = grid_axis(width, logical_width, images.device)
    cell_ids = (row_ids[:, None] * logical_width + column_ids[None, :]).reshape(-1)
    center_positions = (center_rows[:, None] * width + center_columns[None, :]).reshape(-1)
    line_kernel, smooth_kernel = line_kernel_sizes(height, width, logical_height, logical_width)
    for index, frame in enumerate(images):
        strength = dark_line_strength(frame, *line_kernel, line_threshold)
        features = line_cell_features(strength, cell_ids, center_positions, logical_height * logical_width, *smooth_kernel)
        owners = line_owners(*features, logical_width, logical_height)
        weights[index] += line_weight * strength * owners[cell_ids].view(height, width)


def frame_cell_votes(frame, weights, palette_oklab, cell_ids, cell_count, center_positions):
    _, inverse, _, unique_oklab = unique_image_colors(frame)
    labels = nearest_palette_labels(unique_oklab, palette_oklab)[inverse]
    color_count = palette_oklab.shape[0]
    votes = torch.zeros(cell_count * color_count, dtype=weights.dtype, device=weights.device)
    votes.index_add_(0, cell_ids * color_count + labels, weights)
    votes = votes.view(cell_count, color_count)
    return votes / votes.sum(dim=1, keepdim=True), labels[center_positions]


def temporal_grid_collapse(images, pixel_weights, palette_oklab, palette_rgb, logical_width, logical_height, scale_to_original,
                           temporal_radius, temporal_decay, hysteresis, center_bias):
    batch_size, height, width = images.shape[:3]
    row_ids, center_rows, row_offsets = grid_axis(height, logical_height, images.device)
    column_ids, center_columns, column_offsets = grid_axis(width, logical_width, images.device)
    cell_count = logical_height * logical_width
    cell_ids = (row_ids[:, None] * logical_width + column_ids[None, :]).reshape(-1)
    center_positions = (center_rows[:, None] * width + center_columns[None, :]).reshape(-1)
    position_weights = (1.0 - center_bias * torch.maximum(row_offsets[:, None], column_offsets[None, :])).reshape(-1)
    palette_distances = (palette_oklab[:, None] - palette_oklab[None]).square().sum(dim=-1)

    output_height = height if scale_to_original else logical_height
    output_width = width if scale_to_original else logical_width
    output = torch.empty((batch_size, output_height, output_width, 3), dtype=images.dtype, device=images.device)
    window = {}
    previous = None
    for index in range(batch_size):
        for frame_index in range(max(0, index - temporal_radius), min(batch_size, index + temporal_radius + 1)):
            if frame_index not in window:
                weights = position_weights * pixel_weights[frame_index].reshape(-1)
                window[frame_index] = frame_cell_votes(images[frame_index], weights, palette_oklab, cell_ids, cell_count,
                                                       center_positions)
        window.pop(index - temporal_radius - 1, None)
        votes = sum(window[frame_index][0] * temporal_decay ** abs(frame_index - index) for frame_index in window)

        best_votes = votes.amax(dim=1, keepdim=True)
        tie_distances = torch.where(votes == best_votes, palette_distances[window[index][1]], torch.inf)
        labels = tie_distances.argmin(dim=1)
        if previous is not None and hysteresis > 0:
            keep = votes.gather(1, previous[:, None]) >= best_votes * (1.0 - hysteresis)
            labels = torch.where(keep[:, 0], previous, labels)
        previous = labels

        logical = palette_rgb[labels].reshape(logical_height, logical_width, 3)
        if scale_to_original:
            logical = logical.index_select(0, row_ids).index_select(1, column_ids)
        output[index] = logical
    return output


class PixelArtFilterTemporal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtFilterTemporal",
            display_name="Pixel Art Temporal Refiner",
            description="Refines video frames with one shared palette and temporally pooled, edge- and line-weighted grid votes with hysteresis to stop cells flickering.",
            category="PixelArtFilter",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels across the refined image. Need not divide the image width."),
                io.Int.Input("height", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels down the refined image. Need not divide the image height."),
                io.Int.Input("colors", default=24, min=2, max=256, step=1,
                             tooltip="Size of the single palette shared by every frame."),
                io.Boolean.Input("scale_to_original", default=True,
                                 tooltip="Expand the logical pixel grid back to the input dimensions."),
                io.Int.Input("temporal_radius", default=1, min=0, max=8, step=1,
                             tooltip="Pool each cell's votes over this many frames before and after. 0 disables pooling."),
                io.Float.Input("temporal_decay", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Vote weight multiplier per frame of distance when pooling."),
                io.Float.Input("hysteresis", default=0.15, min=0.0, max=0.95, step=0.01,
                               tooltip="Keep a cell's previous color unless the new winner beats it by this fraction. 0 disables."),
                io.Float.Input("edge_weight", default=1.0, min=0.0, max=16.0, step=0.1,
                               tooltip="Extra vote weight for pixels on OKLab Sobel edges. Boosts both sides of an edge, so it favors detail over flat areas."),
                io.Float.Input("edge_threshold", default=0.1, min=0.0, max=0.95, step=0.01,
                               tooltip="Ignore normalized edge strength below this level to skip noise and texture."),
                io.Float.Input("center_bias", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Reduce the vote weight of pixels toward each cell's border."),
                io.Float.Input("line_weight", default=4.0, min=0.0, max=32.0, step=0.1,
                               tooltip="Extra vote weight for pixels on thin dark lines such as outlines. Lines claim a cell from roughly 1/(1+weight) coverage, so high values can thicken lines."),
                io.Float.Input("line_threshold", default=0.15, min=0.0, max=0.95, step=0.01,
                               tooltip="Minimum OKLab lightness drop below the surroundings that counts as a dark line, ignoring soft shading."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Image.Output(display_name="weight_map"),
            ],
        )

    @classmethod
    def execute(cls, image, width, height, colors, scale_to_original, temporal_radius, temporal_decay, hysteresis,
                edge_weight, edge_threshold, center_bias, line_weight, line_threshold):
        image_height, image_width = image.shape[1:3]
        if image.shape[-1] < 3:
            raise ValueError(f"Temporal pixel art refinement requires at least 3 image channels, got {image.shape[-1]}")
        if width > image_width or height > image_height:
            raise ValueError(f"Logical grid {width}x{height} cannot exceed image size {image_width}x{image_height}")

        image = image.to(video_processing_device(image))
        palette_oklab = shared_perceptual_palette(image, colors)
        palette_rgb = (oklab_to_srgb(palette_oklab).clamp(0.0, 1.0) * 255.0).round() / 255.0
        weights = batch_edge_strength(image, edge_threshold).mul_(edge_weight).add_(1.0)
        if line_weight > 0:
            add_dark_line_weights(weights, image, width, height, line_weight, line_threshold)
        refined = temporal_grid_collapse(image, weights, palette_oklab, palette_rgb, width, height, scale_to_original,
                                         temporal_radius, temporal_decay, hysteresis, center_bias)
        weights.div_(1.0 + edge_weight + line_weight)
        intermediate = comfy.model_management.intermediate_device()
        return io.NodeOutput(refined.to(intermediate), weights.to(intermediate).unsqueeze(-1).repeat(1, 1, 1, 3))


def two_pass_frame_features(frame, fill_weights, palette_oklab, cell_ids, cell_sizes, center_positions, line_kernel,
                            smooth_kernel, line_threshold):
    _, inverse, _, unique_oklab = unique_image_colors(frame)
    labels = nearest_palette_labels(unique_oklab, palette_oklab)[inverse]
    strength = dark_line_strength(frame, *line_kernel, line_threshold)
    line = strength.flatten() > 0
    cell_count = cell_sizes.shape[0]
    color_count = palette_oklab.shape[0]
    keys = cell_ids * color_count + labels

    # Line pixels only decide the fill of cells that contain nothing else.
    fill_votes = torch.zeros(cell_count * color_count, dtype=fill_weights.dtype, device=frame.device)
    fill_votes.index_add_(0, keys, fill_weights * torch.where(line, 1e-3, 1.0))
    fill_votes = fill_votes.view(cell_count, color_count)
    line_votes = torch.zeros(cell_count * color_count, dtype=strength.dtype, device=frame.device)
    line_votes.index_add_(0, keys, strength.flatten())
    coverage = torch.zeros(cell_count, dtype=strength.dtype, device=frame.device).index_add_(0, cell_ids, line.to(strength.dtype))
    features = (fill_votes / fill_votes.sum(dim=1, keepdim=True), line_votes.view(cell_count, color_count), coverage / cell_sizes,
                *line_cell_features(strength, cell_ids, center_positions, cell_count, *smooth_kernel))
    return features, labels[center_positions], strength


def two_pass_grid_collapse(images, pixel_weights, palette_oklab, palette_rgb, logical_width, logical_height,
                           scale_to_original, temporal_radius, temporal_decay, hysteresis, center_bias, line_threshold,
                           line_coverage):
    batch_size, height, width = images.shape[:3]
    row_ids, center_rows, row_offsets = grid_axis(height, logical_height, images.device)
    column_ids, center_columns, column_offsets = grid_axis(width, logical_width, images.device)
    cell_count = logical_height * logical_width
    cell_ids = (row_ids[:, None] * logical_width + column_ids[None, :]).reshape(-1)
    cell_sizes = torch.bincount(cell_ids, minlength=cell_count).to(dtype=images.dtype)
    center_positions = (center_rows[:, None] * width + center_columns[None, :]).reshape(-1)
    position_weights = (1.0 - center_bias * torch.maximum(row_offsets[:, None], column_offsets[None, :])).reshape(-1)
    palette_distances = (palette_oklab[:, None] - palette_oklab[None]).square().sum(dim=-1)
    line_kernel, smooth_kernel = line_kernel_sizes(height, width, logical_height, logical_width)

    output_height = height if scale_to_original else logical_height
    output_width = width if scale_to_original else logical_width
    output = torch.empty((batch_size, output_height, output_width, 3), dtype=images.dtype, device=images.device)
    line_map = torch.empty(images.shape[:3], dtype=images.dtype, device=images.device)
    window = {}
    previous = None
    for index in range(batch_size):
        for frame_index in range(max(0, index - temporal_radius), min(batch_size, index + temporal_radius + 1)):
            if frame_index not in window:
                fill_weights = position_weights * pixel_weights[frame_index].reshape(-1)
                features, center_labels, line_map[frame_index] = two_pass_frame_features(
                    images[frame_index], fill_weights, palette_oklab, cell_ids, cell_sizes, center_positions, line_kernel,
                    smooth_kernel, line_threshold)
                window[frame_index] = features, center_labels
        window.pop(index - temporal_radius - 1, None)
        decays = {frame_index: temporal_decay ** abs(frame_index - index) for frame_index in window}
        fill_votes, line_votes, coverage, centers, energy_x, energy_y = (
            sum(window[frame_index][0][part] * decay for frame_index, decay in decays.items()) for part in range(6))
        coverage = coverage / sum(decays.values())

        best_votes = fill_votes.amax(dim=1, keepdim=True)
        tie_distances = torch.where(fill_votes == best_votes, palette_distances[window[index][1]], torch.inf)
        fill_labels = tie_distances.argmin(dim=1)
        line_score = coverage * line_owners(centers, energy_x, energy_y, logical_width, logical_height)
        present = line_score >= line_coverage
        line_labels = line_votes.argmax(dim=1)
        if previous is not None and hysteresis > 0:
            previous_fill, previous_line, previous_present = previous
            keep = fill_votes.gather(1, previous_fill[:, None]) >= best_votes * (1.0 - hysteresis)
            fill_labels = torch.where(keep[:, 0], previous_fill, fill_labels)
            keep = line_votes.gather(1, previous_line[:, None]) >= line_votes.amax(dim=1, keepdim=True) * (1.0 - hysteresis)
            line_labels = torch.where(keep[:, 0], previous_line, line_labels)
            present |= previous_present & (line_score >= line_coverage * (1.0 - hysteresis))
        previous = fill_labels, line_labels, present

        logical = palette_rgb[torch.where(present, line_labels, fill_labels)].reshape(logical_height, logical_width, 3)
        if scale_to_original:
            logical = logical.index_select(0, row_ids).index_select(1, column_ids)
        output[index] = logical
    return output, line_map


class PixelArtFilterTwoPass(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtFilterTwoPass",
            display_name="Pixel Art Two-Pass Refiner",
            description="Refines video frames in two passes: fills each cell without its dark outline pixels, then draws outlines one cell wide along their centers in their own darkest color.",
            category="PixelArtFilter",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels across the refined image. Need not divide the image width."),
                io.Int.Input("height", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels down the refined image. Need not divide the image height."),
                io.Int.Input("colors", default=24, min=2, max=256, step=1,
                             tooltip="Size of the single palette shared by every frame."),
                io.Boolean.Input("scale_to_original", default=True,
                                 tooltip="Expand the logical pixel grid back to the input dimensions."),
                io.Int.Input("temporal_radius", default=1, min=0, max=8, step=1,
                             tooltip="Pool fill votes and line evidence over this many frames before and after. 0 disables pooling."),
                io.Float.Input("temporal_decay", default=0.7, min=0.0, max=1.0, step=0.05,
                               tooltip="Weight multiplier per frame of distance when pooling."),
                io.Float.Input("hysteresis", default=0.15, min=0.0, max=0.95, step=0.01,
                               tooltip="Keep a cell's previous fill color, line color, and line presence unless the new evidence beats it by this fraction. 0 disables."),
                io.Float.Input("edge_weight", default=1.0, min=0.0, max=16.0, step=0.1,
                               tooltip="Extra fill vote weight for pixels on OKLab Sobel edges between fill colors."),
                io.Float.Input("edge_threshold", default=0.1, min=0.0, max=0.95, step=0.01,
                               tooltip="Ignore normalized edge strength below this level to skip noise and texture."),
                io.Float.Input("center_bias", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Reduce the fill vote weight of pixels toward each cell's border."),
                io.Float.Input("line_threshold", default=0.2, min=0.0, max=0.95, step=0.01,
                               tooltip="Minimum OKLab lightness drop below the surroundings that counts as a dark outline, ignoring soft shading."),
                io.Float.Input("line_coverage", default=0.25, min=0.0, max=1.0, step=0.01,
                               tooltip="Fraction of a cell an outline must cover, where the outline's center passes, to draw it there. Lower keeps faint thin lines; higher drops them."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Image.Output(display_name="line_map"),
            ],
        )

    @classmethod
    def execute(cls, image, width, height, colors, scale_to_original, temporal_radius, temporal_decay, hysteresis,
                edge_weight, edge_threshold, center_bias, line_threshold, line_coverage):
        image_height, image_width = image.shape[1:3]
        if image.shape[-1] < 3:
            raise ValueError(f"Two-pass pixel art refinement requires at least 3 image channels, got {image.shape[-1]}")
        if width > image_width or height > image_height:
            raise ValueError(f"Logical grid {width}x{height} cannot exceed image size {image_width}x{image_height}")

        image = image.to(video_processing_device(image))
        palette_oklab = shared_perceptual_palette(image, colors)
        palette_rgb = (oklab_to_srgb(palette_oklab).clamp(0.0, 1.0) * 255.0).round() / 255.0
        weights = batch_edge_strength(image, edge_threshold).mul_(edge_weight).add_(1.0)
        refined, line_map = two_pass_grid_collapse(image, weights, palette_oklab, palette_rgb, width, height,
                                                   scale_to_original, temporal_radius, temporal_decay, hysteresis,
                                                   center_bias, line_threshold, line_coverage)
        intermediate = comfy.model_management.intermediate_device()
        return io.NodeOutput(refined.to(intermediate), line_map.to(intermediate).unsqueeze(-1).repeat(1, 1, 1, 3))


# Outline expansion and contrast downscaling below are adapted from PixelOE (Apache-2.0); see THIRD_PARTY_NOTICES.md.
OUTLINE_KERNEL_SHAPES = {1: (1.0, 3), 2: (1.5, 3), 3: (2.0, 3), 4: (2.5, 5), 5: (3.0, 5), 6: (3.5, 7)}


def outline_kernel(thickness, dtype, device):
    radius, size = OUTLINE_KERNEL_SHAPES[thickness]
    full = 2 * int(radius) + 1
    offsets = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5], [0.0, 0.5], [0.0, -0.5], [0.5, 0.0], [-0.5, 0.0]],
                           dtype=dtype, device=device)
    positions = torch.stack(torch.meshgrid(torch.arange(full, dtype=dtype, device=device),
                                           torch.arange(full, dtype=dtype, device=device), indexing="ij"), dim=-1)
    distances = (positions[:, :, None] + offsets - int(radius)).norm(dim=-1)
    near, far = distances.amin(dim=-1), distances.amax(dim=-1)
    kernel = torch.where(far <= radius, 1.0, torch.where(near <= radius, (radius - near) / (far - near), 0.0))
    crop = (full - size) // 2
    return kernel[crop:crop + size, crop:crop + size]


def soft_morphology(image, kernel, dilate):
    batch, channels, height, width = image.shape
    size = kernel.shape[0]
    padded = torch.nn.functional.pad(image, (size // 2,) * 4, mode="replicate")
    patches = torch.nn.functional.unfold(padded, size).view(batch, channels, size * size, height * width)
    offsets = kernel.reshape(-1, 1) - 1.0
    if dilate:
        return (patches + offsets).amax(dim=2).view(batch, channels, height, width).clamp_(0.0, 1.0)
    return (patches - offsets).amin(dim=2).view(batch, channels, height, width)


def local_lightness_stat(lightness, kernel, stride, stat):
    height, width = lightness.shape[-2:]
    patches = torch.nn.functional.unfold(lightness, kernel, stride=stride, padding=kernel // 2)
    values = stat(patches).expand(-1, patches.shape[1], -1).contiguous()
    totals = torch.nn.functional.fold(values, (height, width), kernel, stride=stride, padding=kernel // 2)
    counts = torch.nn.functional.fold(torch.ones_like(values), (height, width), kernel, stride=stride, padding=kernel // 2)
    return totals / counts.clamp_min(1.0)


def outline_expansion(images, pixel_size, thickness):
    # Erode where the surroundings are bright and dilate where they are dark, so thin details grow to survive downscaling.
    stride = max(1, pixel_size // 2)
    weights = torch.empty_like(images[:, :1])
    for index, image in enumerate(images):
        lightness = srgb_to_oklab(image.permute(1, 2, 0).clamp(0.0, 1.0))[..., 0][None, None]
        median = local_lightness_stat(lightness, 2 * pixel_size, stride, lambda patches: patches.median(dim=1, keepdim=True).values)
        minimum = local_lightness_stat(lightness, pixel_size, stride, lambda patches: patches.amin(dim=1, keepdim=True))
        maximum = local_lightness_stat(lightness, pixel_size, stride, lambda patches: patches.amax(dim=1, keepdim=True))
        weights[index] = torch.sigmoid((median - 0.5) * 10.0 + (2.0 * median - minimum - maximum) * 3.0)[0]
    # One scale for the whole batch so the expansion strength does not flicker between frames.
    weights = (weights - weights.amin()) / (weights.amax() - weights.amin()).clamp_min(1e-8)

    kernel = outline_kernel(thickness, images.dtype, images.device)
    cleanup = outline_kernel(max(thickness - 1, 1), images.dtype, images.device)
    expanded = torch.empty_like(images)
    for index in range(images.shape[0]):
        image = images[index:index + 1]
        weight = weights[index:index + 1]
        frame = soft_morphology(image, kernel, False) * weight + soft_morphology(image, kernel, True) * (1.0 - weight)
        frame = soft_morphology(soft_morphology(frame, cleanup, False), cleanup, True)
        expanded[index] = soft_morphology(soft_morphology(frame, cleanup, True), cleanup, False)[0]
    return expanded, weights


def gaussian_low_frequency(images, levels=5):
    channels = images.shape[1]
    for level in range(1, levels + 1):
        radius = 2 ** level
        taps = torch.arange(-radius, radius + 1, dtype=images.dtype, device=images.device)
        kernel = torch.exp(-taps.square() / (2.0 * radius * radius))
        kernel = (kernel / kernel.sum()).expand(channels, 1, -1)
        images = torch.nn.functional.conv2d(torch.nn.functional.pad(images, (radius, radius, 0, 0), mode="replicate"),
                                            kernel[:, :, None, :], groups=channels)
        images = torch.nn.functional.conv2d(torch.nn.functional.pad(images, (0, 0, radius, radius), mode="replicate"),
                                            kernel[:, :, :, None], groups=channels)
    return images


def contrast_downscale(images, pixel_size):
    batch, _, height, width = images.shape
    lab = srgb_to_oklab(images.permute(0, 2, 3, 1).clamp(0.0, 1.0)).permute(0, 3, 1, 2)
    patches = torch.nn.functional.unfold(lab, pixel_size, stride=pixel_size).view(batch, 3, pixel_size * pixel_size, -1)
    lightness = patches[:, 0]
    median = lightness.median(dim=1).values
    mean = lightness.mean(dim=1)
    minimum = lightness.amin(dim=1)
    maximum = lightness.amax(dim=1)
    # Keep the dark or bright extreme when a patch's lightness is skewed toward it, otherwise the center pixel.
    lightness = torch.where((median < mean) & (maximum - median > median - minimum), minimum,
                            torch.where((median > mean) & (maximum - median < median - minimum), maximum,
                                        lightness[:, lightness.shape[1] // 2]))
    lab = torch.cat((lightness[:, None], patches[:, 1:].median(dim=2).values), dim=1)
    lab = lab.view(batch, 3, height // pixel_size, width // pixel_size).permute(0, 2, 3, 1)
    return oklab_to_srgb(lab).clamp(0.0, 1.0).permute(0, 3, 1, 2)


def k_centroid_downscale(images, pixel_size):
    # Two-means per patch, keeping the centroid of the larger cluster.
    batch, channels, height, width = images.shape
    patches = torch.nn.functional.unfold(images, pixel_size, stride=pixel_size).view(batch, channels, pixel_size * pixel_size, -1)
    patches = patches.permute(0, 3, 2, 1).reshape(-1, pixel_size * pixel_size, channels)
    centroids = torch.stack((patches.amax(dim=1), patches.amin(dim=1)), dim=1)
    for _ in range(4):
        labels = (patches[:, :, None] - centroids[:, None]).square().sum(dim=-1).argmin(dim=-1)
        members = torch.nn.functional.one_hot(labels, 2).to(dtype=patches.dtype)
        counts = members.sum(dim=1)
        centroids = torch.where(counts[..., None] > 0, members.transpose(1, 2) @ patches / counts.clamp_min(1.0)[..., None], centroids)
    dominant = centroids[torch.arange(centroids.shape[0], device=images.device), counts.argmax(dim=1)]
    return dominant.view(batch, height // pixel_size, width // pixel_size, channels).permute(0, 3, 1, 2)


class PixelArtFilterOutlineExpansion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtFilterOutlineExpansion",
            display_name="Pixel Art Outline Expansion",
            description="Grows thin high-contrast details so they survive downscaling to pixel art, eroding on bright surroundings and dilating on dark ones.",
            category="PixelArtFilter",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("pixel_size", default=4, min=1, max=64, step=1,
                             tooltip="Source pixels per output pixel. Sets the window used to judge local contrast."),
                io.Int.Input("thickness", default=3, min=1, max=6, step=1,
                             tooltip="How far details are expanded."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Image.Output(display_name="weight_map"),
            ],
        )

    @classmethod
    def execute(cls, image, pixel_size, thickness):
        if image.shape[-1] < 3:
            raise ValueError(f"Outline expansion requires at least 3 image channels, got {image.shape[-1]}")
        images = image[..., :3].to(video_processing_device(image)).permute(0, 3, 1, 2)
        expanded, weights = outline_expansion(images, pixel_size, thickness)
        intermediate = comfy.model_management.intermediate_device()
        return io.NodeOutput(expanded.permute(0, 2, 3, 1).to(intermediate),
                             weights.permute(0, 2, 3, 1).to(intermediate).repeat(1, 1, 1, 3))


class PixelArtFilterPixelize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtFilterPixelize",
            display_name="Pixel Art Pixelize",
            description="Turns photos and non-pixel-art renders into pixel art: contrast-aware outline expansion, then contrast-aware downscaling and an optional palette shared by the whole batch.",
            category="PixelArtFilter",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("pixel_size", default=4, min=1, max=64, step=1,
                             tooltip="Source pixels per output pixel."),
                io.Int.Input("thickness", default=2, min=0, max=6, step=1,
                             tooltip="Outline expansion before downscaling. 0 disables it."),
                io.Combo.Input("mode", options=["contrast", "k_centroid", "nearest", "bilinear"], default="contrast",
                               tooltip="contrast keeps dark or bright detail per pixel; k_centroid takes each pixel's dominant color."),
                io.Int.Input("colors", default=0, min=0, max=256, step=1,
                             tooltip="Reduce to one perceptual palette shared by every image. 0 keeps all colors."),
                io.Boolean.Input("scale_to_original", default=True,
                                 tooltip="Expand the pixel art back to the input dimensions."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Image.Output(display_name="weight_map"),
            ],
        )

    @classmethod
    def execute(cls, image, pixel_size, thickness, mode, colors, scale_to_original):
        if image.shape[-1] < 3:
            raise ValueError(f"Pixelize requires at least 3 image channels, got {image.shape[-1]}")
        height, width = image.shape[1:3]
        pad_height = -height % pixel_size
        pad_width = -width % pixel_size
        top = pad_height // 2
        left = pad_width // 2
        images = image[..., :3].to(video_processing_device(image)).permute(0, 3, 1, 2)
        images = torch.nn.functional.pad(images, (left, pad_width - left, top, pad_height - top), mode="replicate")

        if thickness > 0:
            expanded, weights = outline_expansion(images, pixel_size, thickness)
            expanded = (expanded - gaussian_low_frequency(expanded) + gaussian_low_frequency(images)).clamp_(0.0, 1.0)
        else:
            expanded, weights = images, torch.zeros_like(images[:, :1])

        if mode == "contrast":
            pixels = contrast_downscale(expanded, pixel_size)
        elif mode == "k_centroid":
            pixels = k_centroid_downscale(expanded, pixel_size)
        else:
            pixels = torch.nn.functional.interpolate(expanded, size=(images.shape[2] // pixel_size, images.shape[3] // pixel_size),
                                                     mode={"nearest": "nearest-exact", "bilinear": "bilinear"}[mode])

        pixels = pixels.permute(0, 2, 3, 1)
        if colors > 0:
            palette_oklab = shared_perceptual_palette(pixels, colors)
            palette_rgb = (oklab_to_srgb(palette_oklab).clamp(0.0, 1.0) * 255.0).round() / 255.0
            labels = nearest_palette_labels(srgb_to_oklab(pixels.clamp(0.0, 1.0)).reshape(-1, 3), palette_oklab)
            pixels = palette_rgb[labels].view(pixels.shape)
        weights = weights[:, :, top:top + height, left:left + width].permute(0, 2, 3, 1)
        if scale_to_original:
            pixels = pixels.repeat_interleave(pixel_size, dim=1).repeat_interleave(pixel_size, dim=2)
            pixels = pixels[:, top:top + height, left:left + width]
        intermediate = comfy.model_management.intermediate_device()
        return io.NodeOutput(pixels.to(intermediate), weights.to(intermediate).repeat(1, 1, 1, 3))


class PixelArtFilterPreResize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtFilterPreResize",
            display_name="Pixel Art Pre-Resize",
            description="Resizes images so they divide into roughly target_pixels × target_pixels output pixels of pixel_size each, keeping the aspect ratio.",
            category="PixelArtFilter",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("target_pixels", default=256, min=1, max=4096, step=1,
                             tooltip="Square root of the desired output pixel count, e.g. 256 for about 256×256 pixels."),
                io.Int.Input("pixel_size", default=4, min=1, max=64, step=1,
                             tooltip="Source pixels per output pixel after resizing."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, target_pixels, pixel_size):
        height, width = image.shape[1:3]
        ratio = width / height
        logical_height = (target_pixels * target_pixels / ratio) ** 0.5
        size = (max(1, int(logical_height)) * pixel_size, max(1, int(logical_height * ratio)) * pixel_size)
        resized = torch.nn.functional.interpolate(image.permute(0, 3, 1, 2), size=size, mode="bicubic", antialias=True)
        return io.NodeOutput(resized.clamp(0.0, 1.0).permute(0, 2, 3, 1))


class PixelArtFilterAutoRefiner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtFilterAutoRefiner",
            display_name="Pixel Art Auto Refiner",
            description="Detects a pixel mesh across the full frame or independently within fixed horizontal direction regions.",
            category="PixelArtFilter",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=64, min=1, max=16384, step=1,
                             tooltip="Full-frame logical width for manual mode or automatic detection fallback."),
                io.Int.Input("height", default=64, min=1, max=16384, step=1,
                             tooltip="Full-frame logical height for manual mode or automatic detection fallback."),
                io.Int.Input("colors", default=24, min=2, max=256, step=1,
                             tooltip="Maximum generated or supplied palette size."),
                io.Boolean.Input("scale_to_original", default=True,
                                 tooltip="Scale and pad the collapsed frame to the input dimensions."),
                io.Boolean.Input("shared_palette", default=True,
                                 tooltip="Generate one palette from the entire image batch instead of a separate palette for each frame."),
                io.Boolean.Input("manual_resolution", default=False,
                                 tooltip="Divide the full frame evenly into width × height cells instead of detecting a mesh."),
                io.Int.Input("direction_count", default=1, min=1, max=64, step=1,
                             tooltip="Detect an independent grid in each fixed horizontal region. Use 4 for a four-direction sprite sheet; 1 keeps whole-frame detection."),
                io.Boolean.Input("align_palette_lightness", default=True,
                                 tooltip="Compensate for a shared OKLab lightness shift before matching a supplied palette."),
                io.Boolean.Input("preserve_generated_background", default=True,
                                 tooltip="Keep the first input image's upper-left color instead of forcing it into the supplied palette."),
                io.Image.Input("palette_image", optional=True,
                               tooltip="Use this image's colors as the palette for the entire batch. Overrides shared_palette."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, colors, scale_to_original, shared_palette=False, palette_image=None,
                manual_resolution=False, direction_count=1, align_palette_lightness=True,
                preserve_generated_background=True):
        if image.shape[-1] < 3:
            raise ValueError(f"Pixel art auto refiner requires at least 3 image channels, got {image.shape[-1]}")
        if palette_image is not None and palette_image.shape[-1] < 3:
            raise ValueError(f"Palette image requires at least 3 image channels, got {palette_image.shape[-1]}")
        if width > image.shape[2] or height > image.shape[1]:
            raise ValueError(f"Fallback grid {width}x{height} cannot exceed image size {image.shape[2]}x{image.shape[1]}")
        if not manual_resolution and (direction_count > image.shape[2] or direction_count > width):
            raise ValueError(f"direction_count={direction_count} cannot exceed image width {image.shape[2]} or fallback width {width}")

        fixed_palette = None
        if palette_image is not None:
            fixed_palette = fixed_palette_from_image(palette_image, image[0, 0, 0, :3], colors, image.dtype, image.device,
                                                     preserve_generated_background)
        if manual_resolution or direction_count == 1:
            if manual_resolution:
                mesh = regular_mesh(image.shape[2], image.shape[1], width, height)
                mesh_scale = 1
            else:
                mesh, mesh_scale = detect_pixel_mesh(image[0], width, height)
            meshes = [(0, image.shape[2], mesh, mesh_scale)]
        else:
            region_bounds = [round(index * image.shape[2] / direction_count) for index in range(direction_count + 1)]
            fallback_bounds = [round(index * width / direction_count) for index in range(direction_count + 1)]
            meshes = []
            for index, (left, right) in enumerate(zip(region_bounds, region_bounds[1:])):
                fallback_width = fallback_bounds[index + 1] - fallback_bounds[index]
                mesh, mesh_scale = detect_pixel_mesh(image[0, :, left:right], fallback_width, height)
                meshes.append((left, right, mesh, mesh_scale))

        reduced = reduce_image_batch(image, colors, shared_palette, fixed_palette, align_palette_lightness)
        collapsed = []
        for frame in reduced:
            regions = [collapse_frame_mesh(frame[:, left:right], mesh, mesh_scale)
                       for left, right, mesh, mesh_scale in meshes]
            if scale_to_original:
                regions = [fit_frame_to_size(region, right - left, image.shape[1])
                           for region, (left, right, _, _) in zip(regions, meshes)]
            else:
                output_height = max(region.shape[0] for region in regions)
                padded = []
                for region in regions:
                    output = torch.zeros((output_height, region.shape[1], 4), dtype=region.dtype, device=region.device)
                    top = (output_height - region.shape[0]) // 2
                    output[top:top + region.shape[0]] = region
                    padded.append(output)
                regions = padded
            collapsed.append(torch.cat(regions, dim=1))

        return io.NodeOutput(composite_white(torch.stack(collapsed)))


class PixelArtFilterSimple(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtFilterSimple",
            display_name="Pixel Art Simple Refiner",
            description="Combines perceptual palette reduction with a modal pixel-grid collapse in a selectable order.",
            category="PixelArtFilter",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels across the refined image."),
                io.Int.Input("height", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels down the refined image."),
                io.Int.Input("colors", default=24, min=2, max=256, step=1,
                             tooltip="Maximum number of colors in each image's palette."),
                io.Boolean.Input("color_reduction_first", default=True,
                                 tooltip="Reduce the palette before grid collapse. Disable for collapse-then-reduce behavior."),
                io.Boolean.Input("scale_to_original", default=True,
                                 tooltip="Expand the logical pixel grid back to the input dimensions."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, colors, color_reduction_first, scale_to_original):
        image_height, image_width = image.shape[1:3]
        if image.shape[-1] < 3:
            raise ValueError(f"Pixel art refinement requires at least 3 image channels, got {image.shape[-1]}")
        if width > image_width or height > image_height:
            raise ValueError(f"Logical grid {width}x{height} cannot exceed image size {image_width}x{image_height}")
        if image_width % width != 0 or image_height % height != 0:
            raise ValueError(f"Image size {image_width}x{image_height} must be divisible by logical grid {width}x{height}")

        if color_reduction_first:
            reduced = torch.stack([reduce_image_palette(batch_image, colors) for batch_image in image])
            return io.NodeOutput(collapse_pixel_grid_mode(reduced, width, height, scale_to_original))

        collapsed = collapse_pixel_grid_mode(image, width, height, scale_to_original)
        return io.NodeOutput(torch.stack([reduce_image_palette(batch_image, colors) for batch_image in collapsed]))
