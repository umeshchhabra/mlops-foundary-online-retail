# Retail repeat-purchase project

This directory will contain the customer repeat-purchase project. The first
implemented step is dataset validation in `pipelines/shared/`. No cleaning,
label construction, feature store, or model training is included yet.

The intended target is a documented prediction window such as: whether a
customer makes another purchase within 30 days after a historical cutoff. That
definition and the treatment of cancellations, returns, missing customer IDs,
and duplicate transactions must be agreed before implementation.
