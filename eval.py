
'''Evaluation wrapper used by the pipeline.

Provides a simple `run_eval` helper that returns evaluation metrics
for use in pipeline stages.
'''

from baseline import evaluate


def run_eval(model, test_loader):
    acc = evaluate(model, test_loader)
    return {"accuracy": acc}
