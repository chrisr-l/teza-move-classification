# teza-move-classification

This project studies continuation versus reversal after extreme short-term SPY moves, with a focus on how target construction affects apparent predictability.

## Main idea

Two target constructions are compared under a fixed event-detection backbone and a fixed pre-event feature set:

- **Method I:** continuation/reversal relative to the original event direction
- **Method II:** continuation/reversal relative to the immediate post-event move

The main result is that delayed reference construction induces a stronger continuation bias, but weaker conditional predictability from pre-event features.

## Project structure

- `event-direction-reference/`: Method I pipeline
- `post-bias-reference/`: Method II pipeline
- `paper/`: LaTeX paper and final PDF

## Methods

- SPY intraday 2-minute data from `yfinance`
- event-driven dataset construction
- logistic regression baseline after feature standardization
- chronological train/test split
- evaluation via accuracy, majority baseline, confusion matrix, and ROC-AUC

## Output

The main deliverable is the paper in `paper/`.
