# This script has been adapted from the file available at the following address:
# https://github.com/MAGICS-LAB/DNABERT_S/blob/main/evaluate/eval_binning.py
import numpy as np
import torch
import os
from scipy.spatial import distance
from src.REVISIT import NonLinearModel as REVISIT_NonLinearModel
from src.OUR import NonLinearModel as OUR_NonLinearModel
from scipy.optimize import linear_sum_assignment


def get_embedding(
    dna_sequences,
    model_name,
    species,
    sample,
    k=4,
    metric=None,
    task_name="clustering",
    test_model_dir=None,
    suffix="",
):
    # Define the embedding directory and path
    embedding_file_dir = os.path.join(
        "embeddings", species, f"{task_name}_{sample}{suffix}"
    )
    embedding_file_path = os.path.join(embedding_file_dir, f"{model_name}.npy")

    # Load the embedding file if it exists
    if os.path.exists(embedding_file_path):
        print(f"Load embedding from file {embedding_file_path}")
        embedding = np.load(embedding_file_path)

    else:
        print(f"Calculate embedding for {model_name} {species} {sample}")

        if model_name == "nonlinear":
            # REVISIT model: linear1 -> batch1 -> sigmoid -> dropout1 -> linear2
            kwargs, model_state_dict = torch.load(
                test_model_dir, map_location=torch.device("cpu")
            )
            kwargs["device"] = "cpu"
            nlm = REVISIT_NonLinearModel(**kwargs)
            nlm.load_state_dict(model_state_dict)
            embedding = nlm.read2emb(dna_sequences)

        elif model_name == "our":
            # OUR model: linear1 -> batch1 -> dropout1 -> linear2 (no activation)
            kwargs, model_state_dict = torch.load(
                test_model_dir, map_location=torch.device("cpu")
            )
            kwargs["device"] = "cpu"
            nlm = OUR_NonLinearModel(**kwargs)
            nlm.load_state_dict(model_state_dict)
            embedding = nlm.read2emb(dna_sequences)

        else:
            raise ValueError(
                f"Unknown model {model_name}. Supported models: 'nonlinear' (REVISIT), 'our' (OUR/SupCon)"
            )

        # Save the embedding file
        os.makedirs(embedding_file_dir, exist_ok=True)
        with open(embedding_file_path, "wb") as f:
            np.save(f, embedding)

    return embedding


def KMedoid(
    features,
    min_similarity=0.8,
    min_bin_size=100,
    max_iter=300,
    metric="dot",
    scalable=True,
):
    # rank nodes by the number of neighbors
    features = features.astype(np.float32)
    if metric == "dot":
        if scalable:
            similarities = np.zeros((features.shape[0], features.shape[0]))
            for i in range(0, features.shape[0], 512):
                similarities[i : i + 512, :] = features[i : i + 512, :] @ features.T
        else:
            similarities = np.dot(features, features.T)

    elif metric == "euclidean" or metric == "l2":
        similarities = np.exp(
            -distance.squareform(distance.pdist(features, "euclidean"))
        )
    elif metric == "l1":
        similarities = np.exp(
            -distance.squareform(distance.pdist(features, "minkowski", p=1.0))
        )
    else:
        raise ValueError("Invalid metric!")

    # set the values below min_similarity to 0
    similarities[similarities < min_similarity] = 0

    p = -np.ones(len(features), dtype=int)
    row_sum = np.sum(similarities, axis=1)

    iter_count = 1
    while np.any(p == -1):
        if iter_count == max_iter:
            break

        # Select the seed index, i.e. medoid index (Line 4)
        s = np.argmax(row_sum)
        # Initialize the current medoid (Line 4)
        current_medoid = features[s]
        selected_idx = None
        # Optimize the current medoid (Line 5-8)
        for t in range(3):
            # For the current medoid, find its similarities
            if metric == "dot":
                similarity = np.dot(features, current_medoid)
            elif metric == "euclidean" or metric == "l2":
                similarity = np.exp(
                    -distance.cdist(
                        features, np.expand_dims(current_medoid, axis=0), "euclidean"
                    )
                ).squeeze()
            elif metric == "l1":
                similarity = np.exp(
                    -distance.cdist(
                        features,
                        np.expand_dims(current_medoid, axis=0),
                        "minkowski",
                        p=1.0,
                    )
                ).squeeze()
            else:
                raise ValueError("Invalid metric!")
            # Determine the indices that are within the similarity threshold
            idx_within = similarity >= min_similarity
            # Determine the available indices, i.e. the indices that have not been assigned to a cluster yet
            idx_available = p == -1
            # Get the indices that are both within the similarity threshold and available
            selected_idx = np.where(np.logical_and(idx_within, idx_available))[0]
            # Determine the new medoid
            current_medoid = np.mean(features[selected_idx], axis=0)

        # Assign the cluster labels and update the row sums (Lines 9-10)
        if selected_idx is not None and len(selected_idx) > 0:
            p[selected_idx] = iter_count
            row_sum -= np.sum(similarities[:, selected_idx], axis=1)
            row_sum[selected_idx] = 0
            print(
                f"Current label: {iter_count}, Number of assigned elements: {len(selected_idx)}"
            )
        elif selected_idx is not None and len(selected_idx) == 0:
            # No elements found for this iteration - all remaining points are below similarity threshold
            # Break the loop as we can't form more clusters
            break
        else:
            raise ValueError("No selected index")

        iter_count += 1

    # remove bins that are too small
    unique, counts = np.unique(p, return_counts=True)
    for label, count in zip(unique, counts):
        if count < min_bin_size:
            p[p == label] = -1

    return p


def align_labels_via_hungarian_algorithm(true_labels, predicted_labels):
    """
    Aligns the predicted labels with the true labels using the Hungarian algorithm.

    Args:
    true_labels (list or array): The true labels of the data.
    predicted_labels (list or array): The labels predicted by a clustering algorithm.

    Returns:
    dict: A dictionary mapping the predicted labels to the aligned true labels.
    """
    # Create a confusion matrix
    max_label = max(max(true_labels), max(predicted_labels)) + 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(confusion_matrix, maximize=True)

    # Create a mapping from predicted labels to true labels
    label_mapping = {
        predicted_label: true_label
        for true_label, predicted_label in zip(row_ind, col_ind)
    }

    return label_mapping


def compute_class_center_medium_similarity(embeddings, labels, metric="dot"):
    idx = np.argsort(labels)
    embeddings = embeddings[idx]
    labels = labels[idx]
    n_sample_per_class = np.bincount(labels)

    all_similarities = np.zeros(len(embeddings))
    count = 0

    for i in range(len(n_sample_per_class)):
        start = count
        end = count + n_sample_per_class[i]
        # Skip empty classes to avoid numpy warnings
        if n_sample_per_class[i] == 0:
            count += n_sample_per_class[i]
            continue
        mean = np.mean(embeddings[start:end], axis=0)
        if metric == "dot":
            similarities = np.dot(mean, embeddings[start:end].T).reshape(-1)
        elif metric == "euclidean" or metric == "l2":
            similarities = np.exp(
                -distance.cdist(
                    np.expand_dims(mean, axis=0),
                    embeddings[start:end],
                    "minkowski",
                    p=2.0,
                ).reshape(-1)
            )
        elif metric == "l1":
            similarities = np.exp(
                -distance.cdist(
                    np.expand_dims(mean, axis=0),
                    embeddings[start:end],
                    "minkowski",
                    p=1.0,
                ).reshape(-1)
            )
        else:
            raise ValueError("Invalid metric!")

        all_similarities[start:end] = similarities

        count += n_sample_per_class[i]

    all_similarities.sort()
    percentile_values = []
    for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        value = all_similarities[int(percentile / 100 * len(embeddings))]
        percentile_values.append(value)
    print("Percentile values:", percentile_values)

    return percentile_values

