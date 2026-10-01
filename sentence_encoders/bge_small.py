import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from src.configs import DATASETS
from src.dataset_adapters import DATASET_ADAPTERS
from src.utils import eval_, print_results,save_result

# MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_NAME = "BAAI/bge-m3"


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def cls_pool(last_hidden_states):
    return last_hidden_states[:, 0]


def make_encoder(tokenizer, model):
    def encode(texts, mode):
        batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        batch = {k: v.to(model.device) for k, v in batch.items()} # Comment this for bge-small

        with torch.no_grad():
            output = model(**batch)
        
        emb = cls_pool(output.last_hidden_state)
        return F.normalize(emb.float(), p=2, dim=1)
        # return average_pool(output.last_hidden_state, batch["attention_mask"])
        
    return encode


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()

    for dataset_name, config in DATASETS.items():
        adapter = DATASET_ADAPTERS.get(dataset_name)

        if adapter is None:
            print(f"Skipping {dataset_name}: no dataset adapter is configured")
            continue

        dataset = load_dataset(config["hf_name"], cache_dir="/home/tichar/Documents/ee798/project/data")

        queries, candidates, labels = adapter(dataset, config)
        results = eval_(queries, candidates, labels, make_encoder(tokenizer, model), model, config["shared_candidates"],device="cuda")
        print_results(dataset_name, results)
        save_result(MODEL_NAME, dataset_name, results,path="results/sentence_encoders.json")


if __name__ == "__main__":
    main()