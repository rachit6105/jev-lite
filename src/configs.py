DATASETS = {
    "emotion": {
        "hf_name": "dair-ai/emotion",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "shared_candidates": True,
    },
    "banking77": {
        "hf_name": "mteb/banking77",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "shared_candidates": True,
    },
    "commonsense_qa": {
        "hf_name": "tau/commonsense_qa",
        "split": "validation",
        "shared_candidates": False,
    },
    "ag_news": {
        "hf_name": "fancyzhx/ag_news",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "shared_candidates": True,
    },
    
}
'''
CORE
1. Emotion
2. AG News
3. Banking77

TRANSFER
4. CLINC150 or HWU64
5. UTCD held-out subsets

STRESS
6. SST-2
7. CommonsenseQA
'''