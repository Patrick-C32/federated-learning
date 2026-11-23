import glob
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from a2fl.task import Net, load_centralized_dataset, load_data, marker_success

results_dir = "results"
if len(sys.argv) > 1:
    results_dir = sys.argv[1]
models_dir = "models"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


# Load saved state into new Net
def load_model(path, in_channels):
    net = Net(in_channels=in_channels)
    net.load_state_dict(torch.load(path, map_location=device))
    net.to(device)
    net.eval()
    return net


# Per-sample cross-entropy and predicted class for every image in loader
def sample_losses(net, loader):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    losses = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            losses.append(criterion(outputs, labels).cpu().numpy())
            preds.append(torch.max(outputs, 1)[1].cpu().numpy())
    return np.concatenate(losses), np.concatenate(preds)


test_loaders = {}
audit_rows = []

for results_path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    if os.path.basename(results_path) == "audit.json":
        continue
    with open(results_path) as f:
        run = json.load(f)
    run_name = run["run_name"]
    seed = run["seed"]
    settings = run["settings"]
    stem = "%s-s%d" % (run_name, seed)

    final_path = os.path.join(models_dir, stem + ".pt")
    if not os.path.exists(final_path):
        print("%s: skipped, %s missing" % (stem, final_path))
        continue

    dataset = settings["dataset"]
    in_channels = 3
    if dataset == "mnist":
        in_channels = 1
    if dataset not in test_loaders:
        test_loaders[dataset] = load_centralized_dataset(dataset)
    testloader = test_loaders[dataset]

    # T-4.2 audited client: the saved client, otherwise the marker client, or failing that client 0
    marker_client = int(settings["marker-client"])
    save_client = int(settings["save-client-model"])
    audit_client = 0
    if save_client >= 0:
        audit_client = save_client
    elif marker_client >= 0:
        audit_client = marker_client
    trainloader, _ = load_data(
        audit_client,
        int(settings["num-clients"]),
        int(settings["batch-size"]),
        dataset=dataset,
        alpha=float(settings["alpha"]),
    )
    # T-4.1 members: unstamped images only - measured by ASR
    if audit_client == marker_client:
        train_data = trainloader.dataset
        num_marked = int(float(settings["marker-share"]) * len(train_data))
        clean = train_data.select(range(num_marked, len(train_data)))
        trainloader = DataLoader(clean, batch_size=int(settings["batch-size"]))

    targets = [("global", final_path)]
    r10_path = os.path.join(models_dir, stem + "-r10.pt")
    if os.path.exists(r10_path) and int(settings["num-server-rounds"]) > 10:
        targets.append(("global-r10", r10_path))
    if save_client >= 0:
        client_path = os.path.join(models_dir, "%s-client%d.pt" % (stem, save_client))
        if os.path.exists(client_path):
            targets.append(("client", client_path))
        else:
            print("%s: client model %s missing" % (stem, client_path))

    # T-4.1 check with the retrained model same seed
    retrain_preds = None
    if run_name.startswith("unlearn-"):
        retrain_path = os.path.join(models_dir, "unlearn-retrain-s%d.pt" % seed)
        if os.path.exists(retrain_path):
            retrain_net = load_model(retrain_path, in_channels)
            _, retrain_preds = sample_losses(retrain_net, testloader)
        else:
            print("%s: no agreement, %s missing" % (stem, retrain_path))

    for target, model_path in targets:
        net = load_model(model_path, in_channels)
        member_losses, _ = sample_losses(net, trainloader)
        test_losses, test_preds = sample_losses(net, testloader)

        # T-4.2 membership inference: equal numbers of members and test images, score = -loss
        n = min(len(member_losses), len(test_losses))
        scores = np.concatenate([-member_losses[:n], -test_losses[:n]])
        # A diverged model gives NaN losses, treated as ties
        scores = np.nan_to_num(scores)
        is_member = np.concatenate([np.ones(n), np.zeros(n)])
        auc = roc_auc_score(is_member, scores)
        fpr, tpr, _ = roc_curve(is_member, scores)
        advantage = np.max(tpr - fpr)
        keep = np.unique(np.linspace(0, len(fpr) - 1, 101).astype(int))
        print(
            "%s %s: client %d, %d members, AUC %.3f, advantage %.3f"
            % (stem, target, audit_client, n, auc, advantage)
        )

        # T-4.1 marker success (ASR)
        asr = None
        if marker_client >= 0:
            asr = float(marker_success(net, testloader, device))
            print("%s %s: ASR %.3f" % (stem, target, asr))

        agree = None
        if retrain_preds is not None:
            agree = float(np.mean(test_preds == retrain_preds))
            print("%s %s: Agree %.3f" % (stem, target, agree))

        row = {
            "run_name": run_name,
            "seed": seed,
            "target": target,
            "auc": float(auc),
            "advantage": float(advantage),
            "asr": asr,
            "agree": agree,
            "roc_fpr": fpr[keep].tolist(),
            "roc_tpr": tpr[keep].tolist(),
        }
        audit_rows.append(row)

audit_path = os.path.join(results_dir, "audit.json")
with open(audit_path, "w") as f:
    json.dump(audit_rows, f)
print("Wrote %d audit rows to %s" % (len(audit_rows), audit_path))
