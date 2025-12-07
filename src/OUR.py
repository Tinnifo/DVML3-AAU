import argparse
import itertools
import random
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def reverse_complement(seq: str) -> str:
    comp_map = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp_map)[::-1]



def split_read(seq: str, k: int, L_min_useful: int, W_target: int):
    L = len(seq)
    if L < k:
        return []

    if L < L_min_useful:
        return [seq]

    if L < W_target:
        window_size = max(L // 2, L_min_useful)
        if window_size >= L:
            return [seq]
        v1 = seq[0:window_size]
        v2 = seq[-window_size:]
        views = [v1]
        if v2 != v1:
            views.append(v2)
        return views

    window_size = W_target
    stride = window_size // 2
    views = []
    start = 0
    while start + window_size <= L:
        views.append(seq[start : start + window_size])
        start += stride

    tail_len = L - start
    if tail_len >= L_min_useful:
        last = seq[-window_size:]
        if not views or last != views[-1]:
            views.append(last)

    return views


def make_views_for_read(
    seq: str,
    k: int,
    max_views_per_read: int,
    L_min_useful: int,
    W_target: int,
    use_reverse_complement: bool = True,
):
    base_views = split_read(seq, k, L_min_useful=L_min_useful, W_target=W_target)
    views = []

    for v in base_views:
        views.append(v)
        if use_reverse_complement:
            views.append(reverse_complement(v))
        if len(views) >= max_views_per_read:
            break

    if not views and len(seq) >= k:
        # fallback: use the whole read (and maybe RC) if splitting returned nothing
        views = [seq]
        if use_reverse_complement and max_views_per_read > 1:
            rc = reverse_complement(seq)
            if rc != seq:
                views.append(rc)

    return views[:max_views_per_read]


class SupConPairDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        transform_func,
        k: int,
        max_read_num: int = 0,
        max_views_per_read: int = 4,
        L_min_useful: int = 64,
        W_target: int = 256,
        use_reverse_complement: bool = True,
        verbose: bool = True,
        seed: int = 0,
    ):
        self.__transform_func = transform_func
        self.k = k
        self.max_views_per_read = max_views_per_read
        self.L_min_useful = L_min_useful
        self.W_target = W_target
        self.use_reverse_complement = use_reverse_complement

        set_seed(seed)

        # Count total lines first (for reporting)
        if verbose:
            print("Counting lines in file...")
        with open(file_path, "r") as f:
            lines_num = sum(1 for _ in f)

        # Optimized sampling: only read the lines we need
        if max_read_num > 0 and max_read_num < lines_num:
            # Sample line indices without loading entire file
            sample_size = min(max_read_num, lines_num)
            chosen_indices = sorted(random.sample(range(lines_num), sample_size))
            if verbose:
                print(f"Sampling {sample_size} lines from {lines_num} total lines...")
        else:
            chosen_indices = None
            if verbose:
                print(f"Using all {lines_num} lines from file...")

        self.left_reads = []
        self.right_reads = []

        # Read only selected lines
        if chosen_indices is not None:
            chosen_set = set(chosen_indices)
            chosen_idx = 0
            with open(file_path, "r") as f:
                for line_idx, line in enumerate(f):
                    if line_idx == chosen_indices[chosen_idx]:
                        stripped_line = line.strip()
                        if not stripped_line:
                            chosen_idx += 1
                            if chosen_idx >= len(chosen_indices):
                                break
                            continue

                        # Handle both comma-separated and tab-separated formats
                        if "," in stripped_line:
                            parts = stripped_line.split(",")
                        elif "\t" in stripped_line:
                            parts = stripped_line.split("\t")
                        else:
                            if verbose:
                                print(f"Warning: Line {line_idx + 1} has no separator, skipping")
                            chosen_idx += 1
                            if chosen_idx >= len(chosen_indices):
                                break
                            continue

                        if len(parts) < 2:
                            if verbose:
                                print(f"Warning: Line {line_idx + 1} has less than 2 parts, skipping")
                            chosen_idx += 1
                            if chosen_idx >= len(chosen_indices):
                                break
                            continue

                        left_read = parts[0].strip()
                        right_read = parts[1].strip()

                        if not left_read or not right_read:
                            if verbose:
                                print(f"Warning: Line {line_idx + 1} has empty read(s), skipping")
                            chosen_idx += 1
                            if chosen_idx >= len(chosen_indices):
                                break
                            continue

                        self.left_reads.append(left_read)
                        self.right_reads.append(right_read)
                        chosen_idx += 1
                        if chosen_idx >= len(chosen_indices):
                            break
        else:
            # Read all lines (original behavior for small files)
            with open(file_path, "r") as f:
                for line_idx, line in enumerate(f):
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue

                    # Handle both comma-separated and tab-separated formats
                    if "," in stripped_line:
                        parts = stripped_line.split(",")
                    elif "\t" in stripped_line:
                        parts = stripped_line.split("\t")
                    else:
                        if verbose:
                            print(f"Warning: Line {line_idx + 1} has no separator, skipping")
                        continue

                    if len(parts) < 2:
                        if verbose:
                            print(f"Warning: Line {line_idx + 1} has less than 2 parts, skipping")
                        continue

                    left_read = parts[0].strip()
                    right_read = parts[1].strip()

                    if not left_read or not right_read:
                        if verbose:
                            print(f"Warning: Line {line_idx + 1} has empty read(s), skipping")
                        continue

                    self.left_reads.append(left_read)
                    self.right_reads.append(right_read)

        if verbose:
            print("The data file was read successfully!")
            print(f"\t+ Total number of read pairs in file: {lines_num}")
            print(f"\t+ Number of read pairs used: {len(self.left_reads)}")

    def __len__(self):
        return len(self.left_reads)

    def __getitem__(self, idx):
        def _generate_views(seq):
            return make_views_for_read(
                seq,
                k=self.k,
                max_views_per_read=self.max_views_per_read,
                L_min_useful=self.L_min_useful,
                W_target=self.W_target,
                use_reverse_complement=self.use_reverse_complement,
            )

        left_views = _generate_views(self.left_reads[idx])
        right_views = _generate_views(self.right_reads[idx])
        all_seqs = left_views + right_views

        # Convert each view to k-mer profile
        kmer_profiles = torch.from_numpy(
            np.array([self.__transform_func(s) for s in all_seqs])
        ).float()

        # Same fragment id for all views of this pair
        frag_ids = torch.full((len(all_seqs),), idx, dtype=torch.long)

        return kmer_profiles, frag_ids


def supcon_collate_fn(batch):
    kmer_profiles, frag_ids = zip(*batch)
    return torch.cat(kmer_profiles, dim=0), torch.cat(frag_ids, dim=0)


class NonLinearModel(torch.nn.Module):
    def __init__(
        self,
        k: int,
        dim: int = 256,
        device=torch.device("cpu"),
        verbose: bool = False,
        seed: int = 0,
    ):
        super(NonLinearModel, self).__init__()

        self.__device = device
        self.__verbose = verbose
        self.__k = k
        self.__dim = dim
        self.__letters = ["A", "C", "G", "T"]
        self.__kmer2id = {
            "".join(kmer): i
            for i, kmer in enumerate(itertools.product(self.__letters, repeat=self.__k))
        }
        self.__kmers_num = len(self.__kmer2id)

        set_seed(seed)

        self.linear1 = torch.nn.Linear(
            self.__kmers_num, 512, dtype=torch.float, device=self.__device
        )
        self.batch1 = torch.nn.BatchNorm1d(512, dtype=torch.float, device=self.__device)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(
            512, self.__dim, dtype=torch.float, device=self.__device
        )

        if self.__verbose:
            print(f"NonLinearModel initialized with k={k}, dim={dim}, device={device}")
            print(f"Number of k-mers: {self.__kmers_num}")

    def encoder(self, kmer_profile: torch.Tensor) -> torch.Tensor:
        output = self.linear1(kmer_profile)
        output = self.batch1(output)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.linear2(output)
        return output

    @property
    def k(self):
        return self.__k

    @property
    def dim(self):
        return self.__dim

    @property
    def device(self):
        return self.__device

    def read2kmer_profile(self, read: str, normalized: bool = True):
        kmer_indices = [
            self.__kmer2id[read[i : i + self.__k]]
            for i in range(len(read) - self.__k + 1)
        ]
        kmer_profile = np.bincount(kmer_indices, minlength=self.__kmers_num)
        return kmer_profile / kmer_profile.sum() if normalized else kmer_profile

    def read2emb(self, reads, normalized=True):
        with torch.no_grad():
            kmer_profiles = np.array(
                [self.read2kmer_profile(read, normalized=normalized) for read in reads]
            )
            kmer_profiles = torch.from_numpy(kmer_profiles).float()
            embs = self.encoder(kmer_profiles).detach().cpu().numpy()
        return embs


def supcon_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
):
    device = embeddings.device
    z = F.normalize(embeddings, dim=1)
    N = z.size(0)

    sim = torch.mm(z, z.T) / temperature
    self_mask = torch.eye(N, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, -1e9)

    labels = labels.contiguous().view(-1, 1)
    pos_mask = (labels == labels.T) & ~self_mask

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = -(log_prob * pos_mask).sum(dim=1)[valid] / (pos_counts[valid] + 1e-12)
    return loss.mean()


def single_epoch(model, optimizer, training_loader, temperature: float = 0.1):
    # Get device from model (handles DataParallel)
    device = next(model.parameters()).device
    epoch_loss = 0.0

    for kmer_profiles, frag_ids in training_loader:
        optimizer.zero_grad()
        kmer_profiles = kmer_profiles.to(device)
        frag_ids = frag_ids.to(device)

        embeddings = model.encoder(kmer_profiles)
        batch_loss = supcon_loss(embeddings, frag_ids, temperature=temperature)
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()

    return epoch_loss / len(training_loader)


def _save_model(model, save_path, verbose=False):
    """Helper function to save model, handling DataParallel."""
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(
        [{"k": model.k, "dim": model.dim, "device": model.device}, model.state_dict()],
        save_path,
    )
    if verbose:
        print("Model is saving.")
        print(f"\t- Target path: {save_path}")


def run(
    model,
    learning_rate,
    epoch_num,
    training_loader,
    model_save_path=None,
    checkpoint=0,
    verbose=True,
    temperature=0.1,
    loss_file_path=None,
):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model.to(torch.device("cuda:0"))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(loss_file_path) if loss_file_path else None

    if verbose:
        print("Training has just started.")

    for epoch in range(epoch_num):
        if verbose:
            print(f"\t+ Epoch {epoch + 1}.")

        model.train()
        avg_loss = single_epoch(
            model, optimizer, training_loader, temperature=temperature
        )

        if verbose:
            print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")

        if writer:
            writer.add_scalar("Loss/train", avg_loss, epoch + 1)
            writer.flush()

        # Checkpoint saving
        if model_save_path and checkpoint > 0 and (epoch + 1) % checkpoint == 0:
            checkpoint_path = re.sub(
                "epoch.*_LR", f"epoch={epoch + 1}_LR", model_save_path
            )
            _save_model(model, checkpoint_path, verbose)

    if writer:
        writer.close()

    # Final model saving
    if model_save_path:
        _save_model(model, model_save_path, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SupCon multi-view genome representation"
    )
    parser.add_argument(
        "--input", type=str, help="Input sequence file (left,right per line)"
    )
    parser.add_argument("--k", type=int, default=4, help="k-mer size")
    parser.add_argument("--dim", type=int, default=256, help="embedding dimension")
    parser.add_argument(
        "--max_read_num",
        type=int,
        default=10000,
        help="Maximum number of read pairs to get from the file",
    )
    parser.add_argument("--epoch", type=int, default=1000, help="Epoch number")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=0, help="Batch size (0: use full dataset)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--workers_num", type=int, default=1, help="Number of workers for data loader"
    )
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument(
        "--seed", type=int, default=26042024, help="Seed for random number generator"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=0,
        help="Save the model for every checkpoint epoch",
    )
    parser.add_argument(
        "--max_views_per_read",
        type=int,
        default=4,
        help="Maximum number of views generated per read",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature parameter for SupCon loss",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    model = NonLinearModel(
        k=args.k,
        dim=args.dim,
        device=torch.device(args.device),
        verbose=True,
        seed=args.seed,
    )

    # Heuristics for view splitting
    L_min_useful = 4 * (args.k**2)  # e.g. k=4 -> 64
    W_target = 4 * L_min_useful  # e.g. -> 256

    training_dataset = SupConPairDataset(
        file_path=args.input,
        transform_func=model.read2kmer_profile,
        k=args.k,
        max_read_num=args.max_read_num,
        max_views_per_read=args.max_views_per_read,
        L_min_useful=L_min_useful,
        W_target=W_target,
        use_reverse_complement=True,
        seed=args.seed,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size if args.batch_size else len(training_dataset),
        shuffle=True,
        num_workers=args.workers_num,
        collate_fn=supcon_collate_fn,
    )

    run(
        model,
        args.lr,
        args.epoch,
        training_loader,
        args.output,
        args.checkpoint,
        verbose=True,
        temperature=args.temperature,
        loss_file_path=args.output + ".loss" if args.output else None,
    )
