#pragma once

constexpr int smooth_tiny_w = 1024 * 32;
constexpr int smooth_tiny_h = 1024 * 1;

constexpr int smooth_small_w = 1024 * 32;
constexpr int smooth_small_h = 1024 * 4;

constexpr int smooth_large_w = 1024 * 32;
constexpr int smooth_large_h = 1024 * 12;

// 8 GB device memory - / 2 buffers in and out = 4 GB / 4 float = 1 GB = sqrt 1GB = 31622
constexpr int smooth_huge_w = 1024 * 32;
constexpr int smooth_huge_h = 1024 * 28;

constexpr int blockSizeX = 128;
constexpr int blockSizeY = 1;

