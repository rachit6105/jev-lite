import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from src.configs import DATASETS
from src.dataset_adapters import DATASET_ADAPTERS
from src.utils import eval_, print_results,save_result

MODEL_NAME = "intfloat/e5-small-v2"


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(),0.0)
    return (last_hidden.sum(dim=1)/ attention_mask.sum(dim=1)[..., None])


def make_encoder(tokenizer, model):
    def encode(texts, mode):
        prefix = "query: " if mode == "query" else "passage: "
        e5_texts = [prefix + text for text in texts]
        batch = tokenizer(e5_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            output = model(**batch)
        return average_pool(output.last_hidden_state, batch["attention_mask"])

    return encode


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    for dataset_name, config in DATASETS.items():
        adapter = DATASET_ADAPTERS.get(dataset_name)
        if adapter is None:
            print(f"Skipping {dataset_name}: no dataset adapter is configured")
            continue

        dataset = load_dataset(
            config["hf_name"],
            cache_dir="/home/tichar/Documents/ee798/project/data",
        )
        queries, candidates, labels = adapter(dataset, config)
        results = eval_(queries, candidates, labels, make_encoder(tokenizer, model), model, config["shared_candidates"])
        print_results(dataset_name, results)
        save_result(MODEL_NAME, dataset_name, results,path="results/sentence_encoders.json")



if __name__ == "__main__":
    main()