import argparse
import csv
import json
import os
import random
from collections import Counter

import numpy as np


MASK_ARRAY = np.array(
    [
        [False, False, False, True],
        [False, True, False, False],
        [False, False, True, False],
        [True, False, False, False],
        [False, True, False, True],
        [False, True, True, False],
        [True, False, True, False],
        [False, False, True, True],
        [True, False, False, True],
        [True, True, False, False],
        [True, True, True, False],
        [True, False, True, True],
        [True, True, False, True],
        [False, True, True, True],
        [True, True, True, True],
    ],
    dtype=bool,
)

MODALITY_NAMES = ["flair", "t1", "t1ce", "t2"]
FULL_MASK_ID = len(MASK_ARRAY) - 1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-client FedMASS splits with labeled full-modal samples "
            "and unlabeled missing-modal samples."
        )
    )
    parser.add_argument(
        "--client_split_dir",
        type=str,
        required=True,
        help="Directory that stores client txt files such as client_part_1.txt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory used to save generated csv files and summary statistics.",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=4,
        help="Number of clients to process.",
    )
    parser.add_argument(
        "--client_pattern",
        type=str,
        default="client_part_{client_id}.txt",
        help="Filename pattern for each client txt file.",
    )
    parser.add_argument(
        "--labeled_ratio",
        type=float,
        default=0.2,
        help="Ratio of labeled full-modal samples for each client.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Dirichlet alpha used to sample client-specific mask distributions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Random seed.",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default="fedmass_split_summary.txt",
        help="Summary text filename saved under output_dir.",
    )
    parser.add_argument(
        "--json_summary_name",
        type=str,
        default="fedmass_split_summary.json",
        help="Summary json filename saved under output_dir.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def validate_args(args):
    if not 0.0 <= args.labeled_ratio <= 1.0:
        raise ValueError("--labeled_ratio must be in [0, 1].")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive.")
    if args.num_clients <= 0:
        raise ValueError("--num_clients must be positive.")


def load_case_names(txt_path):
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"Client split file not found: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as handle:
        case_names = [line.strip() for line in handle.readlines() if line.strip()]
    if not case_names:
        raise ValueError(f"Client split file is empty: {txt_path}")
    return case_names


def split_labeled_unlabeled(case_names, labeled_ratio, rng):
    case_names = list(case_names)
    rng.shuffle(case_names)

    if len(case_names) == 1:
        labeled_count = 1 if labeled_ratio > 0 else 0
    else:
        labeled_count = int(round(len(case_names) * labeled_ratio))
        if labeled_ratio > 0 and labeled_count == 0:
            labeled_count = 1
        if labeled_ratio < 1.0 and labeled_count == len(case_names):
            labeled_count = len(case_names) - 1

    labeled_cases = sorted(case_names[:labeled_count])
    unlabeled_cases = sorted(case_names[labeled_count:])
    return labeled_cases, unlabeled_cases


def get_positive_mask_ids(mask):
    positive_ids = []
    for mask_id, candidate_mask in enumerate(MASK_ARRAY):
        if np.all(candidate_mask <= mask):
            positive_ids.append(mask_id)
    return positive_ids


def sample_unlabeled_mask_ids(sample_count, alpha, rng):
    if sample_count <= 0:
        return np.empty((0,), dtype=np.int64), np.zeros((len(MASK_ARRAY),), dtype=np.float32)

    client_probs = rng.dirichlet(np.full((len(MASK_ARRAY),), alpha, dtype=np.float64))
    sampled_mask_ids = rng.choice(
        len(MASK_ARRAY),
        size=sample_count,
        replace=True,
        p=client_probs,
    )
    return sampled_mask_ids.astype(np.int64), client_probs.astype(np.float32)


def write_split_csv(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["data_name", "mask_id", "mask", "pos_mask_ids"])
        writer.writerows(rows)


def build_labeled_rows(case_names):
    full_mask = MASK_ARRAY[FULL_MASK_ID]
    pos_mask_ids = get_positive_mask_ids(full_mask)
    rows = []
    for case_name in case_names:
        rows.append(
            [
                case_name,
                FULL_MASK_ID,
                [bool(flag) for flag in full_mask.tolist()],
                pos_mask_ids,
            ]
        )
    return rows


def build_unlabeled_rows(case_names, mask_ids):
    rows = []
    for case_name, mask_id in zip(case_names, mask_ids):
        mask = MASK_ARRAY[int(mask_id)]
        rows.append(
            [
                case_name,
                int(mask_id),
                [bool(flag) for flag in mask.tolist()],
                get_positive_mask_ids(mask),
            ]
        )
    return rows


def compute_mask_stats(mask_ids):
    stats = {
        "sample_count": int(len(mask_ids)),
        "mask_id_distribution": {},
        "mask_id_proportion": {},
        "modality_counts": {name: 0 for name in MODALITY_NAMES},
        "modality_ratio": {name: 0.0 for name in MODALITY_NAMES},
    }

    if len(mask_ids) == 0:
        return stats

    counter = Counter(int(mask_id) for mask_id in mask_ids)
    stats["mask_id_distribution"] = {str(mask_id): int(count) for mask_id, count in sorted(counter.items())}
    stats["mask_id_proportion"] = {
        str(mask_id): float(count / len(mask_ids)) for mask_id, count in sorted(counter.items())
    }

    modality_counts = np.zeros((len(MODALITY_NAMES),), dtype=np.int64)
    for mask_id in mask_ids:
        modality_counts += MASK_ARRAY[int(mask_id)].astype(np.int64)

    stats["modality_counts"] = {
        modality_name: int(modality_counts[idx]) for idx, modality_name in enumerate(MODALITY_NAMES)
    }
    stats["modality_ratio"] = {
        modality_name: float(modality_counts[idx] / len(mask_ids))
        for idx, modality_name in enumerate(MODALITY_NAMES)
    }
    return stats


def summarize_client(client_id, labeled_cases, unlabeled_cases, unlabeled_mask_ids, dirichlet_probs):
    labeled_mask_ids = np.full((len(labeled_cases),), FULL_MASK_ID, dtype=np.int64)
    all_mask_ids = np.concatenate([labeled_mask_ids, unlabeled_mask_ids], axis=0)

    summary = {
        "client_id": int(client_id),
        "labeled_count": int(len(labeled_cases)),
        "unlabeled_count": int(len(unlabeled_cases)),
        "total_count": int(len(labeled_cases) + len(unlabeled_cases)),
        "dirichlet_mask_probs": {
            str(mask_id): float(dirichlet_probs[mask_id]) for mask_id in range(len(MASK_ARRAY))
        },
        "labeled_stats": compute_mask_stats(labeled_mask_ids),
        "unlabeled_stats": compute_mask_stats(unlabeled_mask_ids),
        "overall_stats": compute_mask_stats(all_mask_ids),
    }
    return summary


def format_client_summary(summary):
    lines = []
    lines.append(f"Client {summary['client_id']}")
    lines.append(
        "  counts: "
        f"total={summary['total_count']}, "
        f"labeled_full={summary['labeled_count']}, "
        f"unlabeled_missing={summary['unlabeled_count']}"
    )
    lines.append(f"  labeled mask_id distribution: {summary['labeled_stats']['mask_id_distribution']}")
    lines.append(f"  unlabeled mask_id distribution: {summary['unlabeled_stats']['mask_id_distribution']}")
    lines.append(f"  unlabeled modality ratio: {summary['unlabeled_stats']['modality_ratio']}")
    lines.append(f"  overall modality ratio: {summary['overall_stats']['modality_ratio']}")
    return "\n".join(lines)


def save_summaries(output_dir, text_name, json_name, summaries):
    text_path = os.path.join(output_dir, text_name)
    json_path = os.path.join(output_dir, json_name)

    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write("FedMASS split summary\n")
        handle.write("=" * 80 + "\n")
        for summary in summaries:
            handle.write(format_client_summary(summary))
            handle.write("\n" + "-" * 80 + "\n")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    return text_path, json_path


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    client_summaries = []
    for client_id in range(1, args.num_clients + 1):
        split_name = args.client_pattern.format(client_id=client_id)
        split_path = os.path.join(args.client_split_dir, split_name)
        case_names = load_case_names(split_path)

        labeled_cases, unlabeled_cases = split_labeled_unlabeled(
            case_names=case_names,
            labeled_ratio=args.labeled_ratio,
            rng=rng,
        )
        unlabeled_mask_ids, dirichlet_probs = sample_unlabeled_mask_ids(
            sample_count=len(unlabeled_cases),
            alpha=args.alpha,
            rng=rng,
        )

        labeled_rows = build_labeled_rows(labeled_cases)
        unlabeled_rows = build_unlabeled_rows(unlabeled_cases, unlabeled_mask_ids)

        labeled_csv = os.path.join(args.output_dir, f"client_{client_id}_labeled_full.csv")
        unlabeled_csv = os.path.join(args.output_dir, f"client_{client_id}_unlabeled_missing.csv")
        write_split_csv(labeled_csv, labeled_rows)
        write_split_csv(unlabeled_csv, unlabeled_rows)

        client_summary = summarize_client(
            client_id=client_id,
            labeled_cases=labeled_cases,
            unlabeled_cases=unlabeled_cases,
            unlabeled_mask_ids=unlabeled_mask_ids,
            dirichlet_probs=dirichlet_probs,
        )
        client_summaries.append(client_summary)
        print(format_client_summary(client_summary))

    text_path, json_path = save_summaries(
        output_dir=args.output_dir,
        text_name=args.summary_name,
        json_name=args.json_summary_name,
        summaries=client_summaries,
    )
    print(f"Saved summary text to: {text_path}")
    print(f"Saved summary json to: {json_path}")


if __name__ == "__main__":
    main()
