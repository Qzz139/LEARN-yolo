# Dataset notes

This dataset uses YOLO bounding-box labels with the following class IDs:

- `0`: keyboard
- `1`: monitor
- `2`: mouse

## Annotation policy

- Label every clearly recognizable target object, including objects that are
  partially outside the frame or partially occluded.
- Laptop keyboards count as `keyboard`.
- Only standalone display panels count as `monitor`; laptop screens and monitor
  stands without a visible panel do not.
- Images saved with rendered prediction boxes must never be used as source
  images or ground truth.

## Camera capture split (2026-08-31)

The 11 manually annotated camera originals are grouped by capture time so that
short consecutive captures stay in the same split:

- `train`: 3 images
- `val`: 2 images
- `test`: 6 images

All captures show the same desk and device instances. They are useful for local
model checks, but a stronger benchmark should use a separately captured test
set with different locations, devices, lighting, and operators.
