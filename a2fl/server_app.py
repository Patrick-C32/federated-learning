import json
import os
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from a2fl.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    # Read config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    seed = context.run_config["seed"]
    run_name = context.run_config["run-name"]
    out_dir = context.run_config["out-dir"]
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)

    # Load global model, seeded
    torch.manual_seed(seed)
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg x num_rounds
    start_time = time.time()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=lambda server_round, arrays: global_evaluate(
            server_round, arrays, context.run_config, start_time
        ),
    )
    strategy_seconds = time.time() - start_time

    # Save final model
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    name = "%s-s%d" % (run_name, seed)
    torch.save(state_dict, os.path.join(out_dir, "models", name + ".pt"))

    # Collect server, client train and evaluate metrics/round
    rounds = []
    for server_round in sorted(result.evaluate_metrics_serverapp):
        row = {"round": server_round}
        row.update(result.evaluate_metrics_serverapp[server_round])
        if server_round in result.train_metrics_clientapp:
            row.update(result.train_metrics_clientapp[server_round])
        if server_round in result.evaluate_metrics_clientapp:
            row.update(result.evaluate_metrics_clientapp[server_round])
        rounds.append(row)

    # Results to JSON
    results = {
        "run_name": run_name,
        "seed": seed,
        "settings": dict(context.run_config),
        "strategy_seconds": strategy_seconds,
        "rounds": rounds,
    }
    with open(os.path.join(out_dir, "results", name + ".json"), "w") as f:
        json.dump(results, f, indent=1)
    print("Strategy time %.1f s" % strategy_seconds)
    print("Final accuracy %.3f" % rounds[-1]["accuracy"])


def global_evaluate(
    server_round: int, arrays: ArrayRecord, run_config, start_time
) -> MetricRecord:
    """Evaluate model on central data."""

    # Mark time - cover training and evaluation 
    seconds = time.time() - start_time

    # Load model and initialize weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate global model on test set
    test_loss, test_acc = test(model, test_dataloader, device)
    metrics = {"time": seconds, "accuracy": test_acc, "loss": test_loss}

    # Save every 10 seconds (laptop unreliable)
    if server_round > 0:
        if server_round % 10 == 0 or server_round == run_config["num-server-rounds"]:
            name = "%s-s%d-r%d.pt" % (
                run_config["run-name"],
                run_config["seed"],
                server_round,
            )
            path = os.path.join(run_config["out-dir"], "models", name)
            torch.save(model.state_dict(), path)

    # Return metrics
    return MetricRecord(metrics)
