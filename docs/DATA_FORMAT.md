# Real-data manifest format

Image paths are relative to the source JSON file, or absolute. Boxes are
[x1,y1,x2,y2], zero-based, half-open pixel coordinates in the correctly oriented image.

```json
{
  "source": "Your dataset name, version, source URL",
  "license": "Exact license or permission reference",
  "records": [
    {
      "image": "images/car001.jpg",
      "group": "capture-session-01-vehicle-A",
      "boxes": [[100, 200, 300, 250]],
      "texts": ["MH12AB1234"],
      "characters": [
        {"label": "M", "box": [110, 207, 125, 240], "polarity": "dark"},
        {"label": "H", "box": [128, 207, 143, 240], "polarity": "dark"}
      ]
    }
  ]
}
```

The numbers above illustrate the schema only; replace them with real annotations.
Annotate **every** character when preparing your real character dataset.
Character boxes are in full-image coordinates, not relative to the plate crop.
Use polarity "light" for light text on dark plates; the default is dark text.
Records with no plates use boxes: [] and texts: [] as negative examples.

The real-data preparer requires source, license and group IDs. It validates decoded images
and plate boxes, hashes decoded pixels, and joins shared capture groups, image hashes
and known plate text identities before splitting approximately 70/15/15 by group.
Character extraction inherits these splits. At least three independent groups are needed.
Near duplicates and recurring vehicles with missing text labels still need manual grouping.

VOC group CSV:
```csv
filename,group
car001.jpg,capture-session-01-vehicle-A
car002.jpg,capture-session-01-vehicle-A
car003.jpg,capture-session-02-vehicle-B
```

The output split manifests reference images/ relative to the data directory.
Do not move split files away from that directory. Keep the dataset card with provenance.
Use a new output directory rather than overwriting a prepared split.
