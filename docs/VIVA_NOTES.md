# Explaining the project
1. Grayscale is deterministic preprocessing; there are no learned grayscale weights.
2. Plate localization is learned. A CNN produces multi-scale features; heads predict
   plate score, box distances and centerness at grid locations.
3. A single global box regressor would struggle with several plates. Our spatial,
   multi-scale detector can emit several independent boxes and uses NMS.
4. Cropping and segmentation are classical image processing. We test both foreground
   polarities, filter connected components and group glyphs into rows.
5. The character CNN learns from labeled 32x32 masks. It recognizes 36 classes, including
   both I/O and 1/0; no OCR API is involved.
6. Reconstruction orders rows top-to-bottom, then characters left-to-right.
7. Format checks reject implausible strings; they do not establish legal ownership.
8. The minimum character score gates a whole plate. Scores are not calibrated accuracy.
9. A session registry deduplicates recognized text. This is not motion tracking.
10. One frame with two unseen plates creates one event containing both serials.
11. The vehicle detector is a separately labeled pretrained general component.
12. A working training script is not evidence of high accuracy. Report real held-out
    results, detection failures and segmentation failures as well as successes.

## Questions to demonstrate
- Show original image, grayscale, plate crop, binary mask, glyphs, final text.
- Explain how a bounding box maps back through letterbox scale and padding.
- Show that duplicate frames do not add events.
- Explain why two plates with different numeric suffixes must not be fuzzy-merged.
- Explain grouping by vehicle/capture and why random video-frame splits leak data.
- Show the model's dataset provenance, saved configuration, learning curve and confusion matrix.
- Explain why 320 input and two CPU threads were selected for the laptop.
