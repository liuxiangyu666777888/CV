# Occlusion / ID Switch Analysis

- Frame window: 307 ~ 310
- Visible track IDs: [31, 427, 630, 667, 702, 741, 787, 801, 827, 833, 876, 877, 879, 892, 893, 907, 908]
- Objects per frame: {307: 14, 308: 10, 309: 10, 310: 11}

## Manual Analysis Notes

1. Observe whether the same object keeps the same track ID across consecutive frames.
2. Check whether dense overlap or partial occlusion causes missed detections.
3. If IDs change unexpectedly, describe the likely reason: detector miss, overlap, or tracker association failure.
