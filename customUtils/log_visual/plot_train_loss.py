import json
import matplotlib.pyplot as plt
import os

# METRIC_LIST = ["train_loss", "train_rec_loss", "train_embed_loss", "test_acc"]
METRIC_LIST = ["train_loss", "test_acc"]

TRAIN_INFO = "250723_0831"

# 로그 파일 경로
log_file = f"output/mpsc/train/{TRAIN_INFO}/log.txt"
output_dir = f"output/mpsc/train/{TRAIN_INFO}/plots"
os.makedirs(output_dir, exist_ok=True)

def run(mode):

   

    if mode == 'dig':
        # METRIC_LIST = ["train_loss", "test_acc"]

        epochs = []
        total_losses = []
        test_acc = []

        with open(log_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    epochs.append(data["epoch"])
                    total_losses.append(data.get("train_loss", None))
                    
                    # if (i + 1) % 5 == 0:
                    test_acc.append(data.get("test_acc", None))
                except (json.JSONDecodeError, KeyError):
                    print("무시된 줄:", line.strip())
                    continue

        save_plot(epochs, total_losses, "Train Total Loss", "Loss", "train_total_loss.png", color="blue")
        save_plot(epochs, test_acc, "Test Accuracy (Every Epochs)", "Accuracy", "test_acc.png", color="red")




    else: # dig-seed
        # METRIC_LIST = ["train_loss", "train_rec_loss", "train_embed_loss", "test_acc"]
        epochs = []
        total_losses = []
        test_acc = []
        rec_losses = []
        embed_losses = []       

        with open(log_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    epochs.append(data["epoch"])
                    total_losses.append(data.get("train_loss", None))
                    rec_losses.append(data.get("train_rec_loss", None))
                    embed_losses.append(data.get("train_embed_loss", None))
                    # if (i + 1) % 5 == 0:
                    test_acc.append(data.get("test_acc", None))
                except (json.JSONDecodeError, KeyError):
                    print("무시된 줄:", line.strip())
                    continue
        save_plot(epochs, total_losses, "Train Total Loss", "Loss", "train_total_loss.png", color="blue")
        save_plot(epochs, rec_losses, "Train Recognition Loss", "Loss", "train_rec_loss.png", color="green")
        save_plot(epochs, embed_losses, "Train Embedding Loss", "Loss", "train_embed_loss.png", color="orange")
        save_plot(epochs, test_acc, "Test Accuracy (Every Epochs)", "Accuracy", "test_acc.png", color="red")


    # with open(log_file, 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         if not line.strip():
    #             continue
    #         try:
    #             data = json.loads(line)
    #             epochs.append(data["epoch"])
    #             total_losses.append(data.get("train_loss", None))
    #             rec_losses.append(data.get("train_rec_loss", None))
    #             embed_losses.append(data.get("train_embed_loss", None))
    #             # if (i + 1) % 5 == 0:
    #             test_acc.append(data.get("test_acc", None))
    #         except (json.JSONDecodeError, KeyError):
    #             print("무시된 줄:", line.strip())
    #             continue
    # save_plot(epochs, total_losses, "Train Total Loss", "Loss", "train_total_loss.png", color="blue")
    # # save_plot(epochs, rec_losses, "Train Recognition Loss", "Loss", "train_rec_loss.png", color="green")
    # # save_plot(epochs, embed_losses, "Train Embedding Loss", "Loss", "train_embed_loss.png", color="orange")
    # save_plot(epochs, test_acc, "Test Accuracy (Every Epochs)", "Accuracy", "test_acc.png", color="red")

# test_acc는 5번째 줄마다만 존재
# test_epochs = [e for i, e in enumerate(epochs) if (i + 1) % 5 == 0]

# 각 그래프 시각화 & 저장
def save_plot(x, y, title, ylabel, filename, color="blue"):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', color=color, label=title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"저장 완료 → {path}")

# 각 항목별 그래프 저장


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "dig-seed"
    run(mode)