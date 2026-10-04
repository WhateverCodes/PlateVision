# Own CNN update after 100 character reviews

100 reviews saved: 99 usable, one excluded, two labels corrected. Verified examples cover 34 character classes. No reviewed source overlaps either development evaluation set.

Training snapshot character_verified_100 contains 99 verified, 3014 provisional aligned and 1008 synthetic glyphs. Own CNN was initialized from our previous adaptation_v1 checkpoint and trained for three complete epochs using two CPU threads, about 56 seconds. External OCR weights and evaluation images were not used.

| Development set | Previous exact | Updated exact | Previous accepted correct/wrong | Updated accepted correct/wrong |
|---|---:|---:|---:|---:|
| 30 newer plate crops | 6/30 | 7/30 | 4 / 2 | 5 / 2 |
| 15 difficult crops | 6/15 | 6/15 | 2 / 0 | 3 / 0 |

Character error rate decreased from 31.88% to 29.19% on the 30-crop set, and from 36.62% to 33.10% on the 15-crop set. This is a small development improvement, not evidence of reaching 80% full-plate accuracy. The full-photo own-model scooter demo still reads KL07BX7197 correctly. Final reserved photos remain unused.

The candidate is now selected as the own CNN when its checkpoint is available. Previous weights are retained. The main pretrained whole-line OCR is unchanged. Restart the app and select Compare models to try it. No additional character reviews are requested now; remaining segmentation and recognition failures need targeted diagnosis before another labelling batch.
