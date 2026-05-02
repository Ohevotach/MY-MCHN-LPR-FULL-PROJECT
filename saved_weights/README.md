# saved_weights

Put reusable Kaggle artifacts here:

- `plate_best.pt`: best YOLO plate detector.
- `plate_last.pt`: latest YOLO plate detector checkpoint for continued training.
- `char_best.pt`: best YOLO character detector.
- `char_last.pt`: latest YOLO character detector checkpoint for continued training.
- `template_cache_32x64.pt`: optional MCHN template cache.

The app auto-discovers `plate_best.pt`, `char_best.pt`, and `template_cache_32x64.pt` in this folder.

