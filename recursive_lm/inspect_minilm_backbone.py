from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# print(model)

embeddings = model.embeddings
layers = model.encoder.layer
print(model.encoder.layer[0])
print(model.config)
print(type(embeddings))
print(type(layers))
print(len(layers))

for i, layer in enumerate(layers):
    print(i, type(layer))