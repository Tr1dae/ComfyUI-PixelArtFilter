# Third-party notices

## Proper Pixel Art

The Pixel Art Auto Refiner mesh detector is adapted from
[Proper Pixel Art](https://github.com/KennethJAllen/proper-pixel-art).

Copyright (c) 2024 Kenneth Allen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## PixelOE

The Pixel Art Outline Expansion, Pixel Art Pixelize, and Pixel Art Pre-Resize
nodes are adapted from [PixelOE](https://github.com/KohakuBlueleaf/PixelOE) by
Shin-Ying Yeh (KohakuBlueleaf), licensed under the Apache License 2.0. A copy of
the license is in [licenses/PixelOE-LICENSE](licenses/PixelOE-LICENSE).

Changes from the original: reimplemented in this pack's PyTorch style using OKLab
instead of CIELAB; outline weights are normalized across the whole image batch;
morphology uses replicate padding; color matching keeps only the wavelet
low-frequency transfer; k-centroid downscaling was rewritten; k-means palette
quantization and dithering were replaced by this pack's shared perceptual
palette; pre-resizing processes every image in the batch.
