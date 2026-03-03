import pickle
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "./results/evaluation_results/"
MODEL_DIR = "./results/trained_models/"

def load_model_results(timestamp):
    metrics = {}
    for dataset in ['train', 'val', 'test']:
        metrics_path = RESULTS_DIR + f"res_{timestamp}/metrics_{dataset}.json"
        with open(metrics_path, 'rb') as f:
            metrics[dataset] = json.load(f)

    history_path = MODEL_DIR + f"history_{timestamp}.pkl"
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    return metrics, history

def extract_metrics_for_bar_plot(metrics):
    accuracy = metrics['accuracy']
    precision = metrics['precision_class_1']
    recall = metrics['recall_class_1']
    f1 = metrics['f1_score_class_1']
    return [accuracy, precision, recall, f1]

def make_metrics_bar_plot(metrics, graph_dir):
    training_metrics = extract_metrics_for_bar_plot(metrics['train'])
    val_metrics = extract_metrics_for_bar_plot(metrics['val'])
    test_metrics = extract_metrics_for_bar_plot(metrics['test'])

    metrics_name = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics_name))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))
    # Barres
    bars1 = ax.bar(x - width, training_metrics, width, label='Training',
                   color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, val_metrics, width, label='Validation',
                   color='coral', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, test_metrics, width, label='Test',
                   color='seagreen', alpha=0.8, edgecolor='black')

    # Annotations sur les barres
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Configuration
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Training vs Validation vs Test Metrics',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_name, rotation=0)
    ax.legend(loc='lower left', fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = graph_dir / f'train_vs_val_vs_test.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def make_history_plot(history, graph_dir):
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    x = np.arange(1 , len(train_loss)+1, 1)

    plt.figure(figsize=(12,7))
    plt.plot(x, train_loss, label='Training loss')
    plt.plot(x, val_loss, label='Validation loss')
    plt.title(f'Train & Validation Loss', fontsize=14, fontweight='bold')
    plt.ylabel('Loss')
    plt.xlabel('Époques')
    plt.legend()

    plt.tight_layout()
    output_file = graph_dir / f'history.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

if __name__=='__main__':
    
    timestamp = '20260227_1909'

    graph_dir = Path(RESULTS_DIR) / f'graphs_{timestamp}'
    graph_dir.mkdir(parents=True, exist_ok=True)

    metrics, history = load_model_results(timestamp)

    make_metrics_bar_plot(metrics, graph_dir)
    make_history_plot(history, graph_dir)

    print('Graphs created')