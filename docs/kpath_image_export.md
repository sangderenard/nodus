# Image export helpers

`kpath_image_export` provides a lightweight utility layer to serialize the
intermediate raster/film representations that kpath produces. This keeps PNG
writing within the kpath module so tests, tools, or user code can dump the
current state without re-implementing `write_png*` or raw canvas iteration.

## Supported conversions

- `PlateTensor2D` → PNG (single-channel reactance map)
- `FilmTensor2D` → PNG (multi-channel exposure folded by channel reactance or
  a custom aggregator)
- `HolographicPlateTensor2D` → PNG (counts or summed sample weight)
- Any `TensorCanvas2D` triple → RGB PNG (`export_canvas_rgb`)

Each helper returns a normalized grayscale image by default (0..255 range).
The exported PNGs use `kpath_raster::write_png_grayscale_u8` under the hood,
so downstream code can easily feed them into `GraphEdit` builders, toolpath
visualizers, or analysis scripts.

## Path export

The module also exposes `export_canvas_to_png` so any `TensorCanvas2D`
(e.g., a rasterized glyph outline or a solver-generated layout) can be written
out as a PNG without reusing the low-level PNG boilerplate.
