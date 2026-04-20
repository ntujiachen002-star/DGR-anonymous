import torch
from lang2comp import Lang2Comp, generate_training_texts
from torch.utils.data import DataLoader, TensorDataset

print("Generating 10K training samples...")
samples = generate_training_texts(10000)
texts = [s["text"] for s in samples]
labels = torch.tensor([s["weights"] for s in samples])  # weights is already [sym, smo, com] list

model = Lang2Comp()
optimizer = torch.optim.Adam(model.composition_head.parameters(), lr=1e-3, weight_decay=1e-4)

print("Encoding texts...")
with torch.no_grad():
    embeddings = model.encode_text(texts)

dataset = TensorDataset(embeddings, labels)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

print("Training 50 epochs...")
best_loss = float("inf")
for epoch in range(50):
    total_loss = 0
    for emb_batch, label_batch in loader:
        optimizer.zero_grad()
        logits = model.composition_head(emb_batch)
        w_pred = torch.softmax(logits[:, :3], dim=1)
        loss = torch.nn.functional.mse_loss(w_pred, label_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "checkpoints/lang2comp_retrained.pt")
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50, loss={avg_loss:.6f}, best={best_loss:.6f}")

print("Saved to checkpoints/lang2comp_retrained.pt")
model.eval()
for p in ["a symmetric vase", "a smooth sphere", "a compact stone",
          "a polished marble", "a balanced chess piece", "a dense bowling ball"]:
    pred = model.predict(p)
    w = pred["weights"]
    print(f"  {p:35s}: sym={w['symmetry']:.2f} smo={w['smoothness']:.2f} com={w['compactness']:.2f} -> {pred['dominant_property']}")
