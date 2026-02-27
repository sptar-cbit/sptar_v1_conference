'''
Train a Bi-Encoder (DPR-style) using MultipleNegativesRankingLoss.
Optionally performs a simple REINFORCE fine-tuning stage.

Example:
python train_sbert.py --dataset_name msmarco --use_basic_rl --rl_steps 1000
'''

import torch
import torch.nn as nn
from sentence_transformers import losses, models, SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
import argparse
from os.path import join, dirname, abspath
import math
import sys
import random
import json

print("Started", flush=True)

zhiyuan_path = dirname(dirname(dirname(dirname(abspath(__file__)))))
if zhiyuan_path not in sys.path:
    sys.path.append(zhiyuan_path)

from weak_data_loader import WeakDataLoader


class BasicRLPolicy(nn.Module):
    """Tiny policy head over similarity scores."""

    def __init__(self, hidden_dim=8):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, sim_q_pos, sim_q_neg):
        features = torch.stack([sim_q_pos, sim_q_neg], dim=0).unsqueeze(0)
        return self.policy(features).squeeze(0)


def _get_doc_text(doc):
    text = doc.get("text", "")
    title = doc.get("title", "")
    return f"{title} {text}".strip()


def main(args):

    data_dir = join(zhiyuan_path, "datasets")
    raw_dir = join(data_dir, "raw")
    beir_dir = join(raw_dir, "beir")
    xuyang_dir = join(dirname(zhiyuan_path), "xuyang", "data")

    model_name = "bert-large-uncased"

    model_save_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "output",
        args.exp_name,
        str(args.train_num),
        f"{model_name}-v1-{args.dataset_name}"
    )
    os.makedirs(model_save_path, exist_ok=True)

    fh = logging.FileHandler(join(model_save_path, "log.txt"))
    ch = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[fh, ch]
    )

    # Load data
    if args.exp_name == "no_aug":
        corpus, queries, qrels = GenericDataLoader(
            corpus_file=join(beir_dir, args.dataset_name,
                             f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
            query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
            qrels_file=join(xuyang_dir,
                            f"{args.dataset_name}_{args.train_num}",
                            f"prompt_tuning_{args.train_num}.tsv")
        ).load_custom()
    else:
        weak_query_file = join(
            xuyang_dir,
            f"{args.dataset_name}_{args.train_num}",
            args.weak_num,
            f"weak_queries_{args.train_num}_{args.exp_name}.jsonl"
        )
        weak_qrels_file = join(
            xuyang_dir,
            f"{args.dataset_name}_{args.train_num}",
            args.weak_num,
            f"weak_train_{args.train_num}_{args.exp_name}.tsv"
        )
        corpus, queries, qrels = WeakDataLoader(
            corpus_file=join(beir_dir, args.dataset_name,
                             f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
            query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
            qrels_file=join(xuyang_dir,
                            f"{args.dataset_name}_{args.train_num}",
                            f"prompt_tuning_{args.train_num}.tsv"),
            weak_query_file=weak_query_file,
            weak_qrels_file=weak_qrels_file
        ).load_weak_custom()

    dev_corpus, dev_queries, dev_qrels = GenericDataLoader(
        corpus_file=join(beir_dir, args.dataset_name,
                         f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
        query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
        qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")
    ).load_custom()

    # Model
    word_embedding_model = models.Transformer(model_name, max_seq_length=350)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension()
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        device=device
    )

    retriever = TrainRetriever(model=model, batch_size=16)

    train_samples = retriever.load_train(corpus, queries, qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

    ir_evaluator = retriever.load_ir_evaluator(
        dev_corpus, dev_queries, dev_qrels, name="dev"
    )

    num_epochs = args.num_epochs
    evaluation_steps = -1
    warmup_steps = int(
        len(train_samples) * num_epochs / retriever.batch_size * 0.1
    )

    print(">>> Starting supervised DPR training...", flush=True)

    retriever.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=num_epochs,
        output_path=model_save_path,
        warmup_steps=warmup_steps,
        evaluation_steps=evaluation_steps,
        use_amp=True,
        callback=lambda score, epoch, steps:
            print(f"[Epoch {epoch} | Step {steps}] Eval score: {score}", flush=True)
    )

    # -------------------------
    # RL Fine-Tuning Stage
    # -------------------------

    if args.use_basic_rl and args.rl_steps > 0:
        print(f">>> Starting RL fine-tuning ({args.rl_steps} steps)", flush=True)

        model.train()
        policy_head = BasicRLPolicy(
            hidden_dim=args.rl_policy_hidden_dim
        ).to(model.device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(policy_head.parameters()),
            lr=args.rl_lr
        )

        query_ids = [qid for qid, rels in qrels.items() if len(rels) > 0]
        doc_ids = list(corpus.keys())

        for step in range(args.rl_steps):

            qid = random.choice(query_ids)
            pos_ids = list(qrels[qid].keys())
            pos_id = random.choice(pos_ids)

            neg_id = random.choice(doc_ids)
            while neg_id in qrels[qid]:
                neg_id = random.choice(doc_ids)

            query_text = queries[qid]
            pos_text = _get_doc_text(corpus[pos_id])
            neg_text = _get_doc_text(corpus[neg_id])

            features = model.tokenize([query_text, pos_text, neg_text])
            features = {k: v.to(model.device) for k, v in features.items()}
            embeddings = model(features)["sentence_embedding"]

            sim_q_pos = torch.cosine_similarity(
                embeddings[0:1], embeddings[1:2]
            ).squeeze(0)

            sim_q_neg = torch.cosine_similarity(
                embeddings[0:1], embeddings[2:3]
            ).squeeze(0)

            logits = policy_head(sim_q_pos, sim_q_neg)

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            reward = 1.0 if action.item() == 0 else -1.0
            loss = -dist.log_prob(action) * reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0 or step == 0:
                probs = torch.softmax(logits, dim=0)
                print(
                    f"[RL step {step+1}/{args.rl_steps}] "
                    f"sim_pos={sim_q_pos.item():.4f}, "
                    f"sim_neg={sim_q_neg.item():.4f}, "
                    f"p_pos={probs[0].item():.4f}, "
                    f"p_neg={probs[1].item():.4f}, "
                    f"reward={reward:+.1f}, "
                    f"loss={loss.item():.6f}",
                    flush=True
                )

        model.save(model_save_path)
        torch.save(
            policy_head.state_dict(),
            join(model_save_path, "rl_policy_head.pt")
        )

        print(">>> RL fine-tuning complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default="msmarco", type=str)
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument('--train_num', default=50, type=int)
    parser.add_argument('--weak_num', default="5000", type=str)
    parser.add_argument('--exp_name', default="no_aug", type=str)
    parser.add_argument('--use_basic_rl', action='store_true')
    parser.add_argument('--rl_steps', default=0, type=int)
    parser.add_argument('--rl_lr', default=1e-6, type=float)
    parser.add_argument('--rl_policy_hidden_dim', default=8, type=int)

    args = parser.parse_args()
    main(args)
