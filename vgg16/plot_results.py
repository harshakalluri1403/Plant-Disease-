import pandas as pd
import matplotlib.pyplot as plt

def plot_results(epochs_list):
    plt.figure()
    for epochs in epochs_list:
        try:
            history_df = pd.read_csv(f'history_{epochs}_epochs.csv')
            plt.plot(history_df['accuracy'], label=f'Training Accuracy {epochs} epochs')
            plt.plot(history_df['val_accuracy'], label=f'Validation Accuracy {epochs} epochs')
        except FileNotFoundError:
            print(f"History file for {epochs} epochs not found.")
    
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_comparison.png')
    plt.close()
    
    plt.figure()
    for epochs in epochs_list:
        try:
            history_df = pd.read_csv(f'history_{epochs}_epochs.csv')
            plt.plot(history_df['loss'], label=f'Training Loss {epochs} epochs')
            plt.plot(history_df['val_loss'], label=f'Validation Loss {epochs} epochs')
        except FileNotFoundError:
            print(f"History file for {epochs} epochs not found.")
    
    plt.title('Model Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_comparison.png')
    plt.close()

if __name__ == "__main__":
    epochs_list = [100, 200, 500, 1000]
    plot_results(epochs_list)
    print("Plots have been saved.")

