
from baseline import evaluate

def run_eval(model, test_loader):
    acc = evaluate(model, test_loader)
    return {"accuracy": acc}
