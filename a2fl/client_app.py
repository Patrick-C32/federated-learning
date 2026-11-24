import os

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from a2fl.task import Net, ascent, compress_update, load_data
from a2fl.task import test as test_fn
from a2fl.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):

    # Read config - FedAvg adds round number to train config
    dataset = context.run_config["dataset"]
    seed = context.run_config["seed"]
    run_name = context.run_config["run-name"]
    server_round = msg.content["config"]["server-round"]
    partition_id = context.node_config["partition-id"]

    # T-4.1 Exclude forgotten client with zero weight (ascent rounds train only it)
    ascent_round = context.run_config["unlearn-mode"] == "ascent" and server_round == 1
    if partition_id == context.run_config["forget-client"]:
        leave = not ascent_round
    else:
        leave = ascent_round
    if leave:
        metrics = {
            "train_loss": 0.0,
            "num-examples": 0,
            "update_norm": 0.0,
            "upload_bytes": 0,
        }
        content = RecordDict(
            {"arrays": msg.content["arrays"], "metrics": MetricRecord(metrics)}
        )
        return Message(content=content, reply_to=msg)

    # Load model and initialize with weights
    model = Net(1 if dataset == "mnist" else 3)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    marker_share = 0.0
    if partition_id == context.run_config["marker-client"]:
        marker_share = context.run_config["marker-share"]
    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        batch_size,
        dataset,
        context.run_config["alpha"],
        marker_share,
    )

    # Seed shuffle (and stochastic rounding) per seed, round and client
    torch.manual_seed(seed * 1000 + server_round * 100 + partition_id)
    np.random.seed(seed * 1000 + server_round * 100 + partition_id)

    if ascent_round:
        # T-4.1 Bounded gradient ascent on data to be forgotten
        train_loss = ascent(
            model,
            trainloader,
            context.run_config["ascent-steps"],
            msg.content["config"]["lr"],
            context.run_config["ascent-radius"],
            device,
        )
    else:
        # Call training function
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )

    # Update = local weights - received weights
    received = msg.content["arrays"].to_numpy_ndarrays()
    delta = []
    for local, global_array in zip(model.state_dict().values(), received):
        delta.append(local.cpu().numpy() - global_array)
    update_norm = np.sqrt(sum(np.sum(d**2) for d in delta))

    # T-4.2 Save trained local model in final round (curious-server target)
    is_final_round = server_round == context.run_config["num-server-rounds"]
    if is_final_round and partition_id == context.run_config["save-client-model"]:
        name = "%s-s%d-client%d.pt" % (run_name, seed, partition_id)
        models_dir = os.path.join(context.run_config["out-dir"], "models")
        os.makedirs(models_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(models_dir, name))

    # T-4.3 Compress update and send received weights + decoded update
    upload_bytes = 4 * sum(d.size for d in delta)
    compress = context.run_config["compress"]
    if compress != "none":
        # Unsent part of earlier updates is kept in node state
        if "residual" in context.state:
            residual = context.state["residual"].to_numpy_ndarrays()
        else:
            residual = [np.zeros_like(d) for d in delta]
        decoded, upload_bytes, residual = compress_update(delta, compress, residual)
        context.state["residual"] = ArrayRecord(residual)
        new_state = {}
        for key, global_array, d in zip(model.state_dict().keys(), received, decoded):
            new_state[key] = torch.from_numpy(global_array + d)
        model.load_state_dict(new_state)

    # Reply
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "update_norm": float(update_norm),
        "upload_bytes": upload_bytes,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):

    # T-4.1 Forgotten client reports no examples - only retain clients count
    partition_id = context.node_config["partition-id"]
    if partition_id == context.run_config["forget-client"]:
        metrics = {"eval_loss": 0.0, "eval_acc": 0.0, "num-examples": 0}
        content = RecordDict({"metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)

    # Load model and initialize weights
    dataset = context.run_config["dataset"]
    model = Net(1 if dataset == "mnist" else 3)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(
        partition_id, num_partitions, batch_size, dataset, context.run_config["alpha"]
    )

    # Evaluate
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Reply
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
