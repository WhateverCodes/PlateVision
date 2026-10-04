# Check individual characters

1. Double-click REVIEW_CHARACTERS.cmd in the PlateVision folder.
2. Select a clear plate from the left menu. Start with two or three plates.
3. Select character 1 in Character to check.
4. Adjust Left and right edges and Top and bottom edges until the preview contains only the complete selected letter or digit. Leave a small margin; exclude neighbouring characters, IND, logos and screws.
5. Choose whether the letters are dark or light. Click Confirm this character.
6. Select the next character and repeat, reading the top row first on two-line plates.
7. Click Save completed plate before closing the page.

Suggested boxes are only starting points. Even when the number of boxes matches the registration length, inspect each crop. If a character is hidden, cut off or too difficult to isolate, select another plate for now. Fix an incorrect registration in REVIEW_DATASET.cmd first.

Completed plates save in data/kaggle_indian_review/character_reviews. Unfinished confirmations stay only in the current page session. Editing the source plate box, registration or group makes an older character review stale and requires checking again.

This tool does not start training or overwrite models. These are annotations in original-image coordinates. Next, reviewed character annotations must be checked, related images grouped and train/validation/test splits created before extraction. Two or three plates are a workflow check, not a sufficient training dataset. The existing TRAIN_MODELS.cmd still runs synthetic practice training.
