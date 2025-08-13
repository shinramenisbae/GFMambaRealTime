import os
import torch
import numpy as np
import random
import torch.nn as nn
import matplotlib.pyplot as plt

def save_model(save_path, epoch, model, optimizer):
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_best_results(results, best_results, epoch, model, optimizer, ckpt_root, seed, save_best_model):
    if epoch == 1:
        for key, value in results.items():
            best_results[key] = value
    else:
        for key, value in results.items():
            if (key == 'Has0_acc_2') and (value > best_results[key]):
                best_results[key] = value
                best_results['Has0_F1_score'] = results['Has0_F1_score']

                if save_best_model:
                    key_eval = 'Has0_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
            
            elif (key == 'Non0_acc_2') and (value > best_results[key]):
                best_results[key] = value
                best_results['Non0_F1_score'] = results['Non0_F1_score']

                if save_best_model:
                    key_eval = 'Non0_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
            
            elif key == 'MAE' and value < best_results[key]:
                best_results[key] = value
                # best_results['Corr'] = results['Corr']

                if save_best_model:
                    key_eval = 'MAE'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)

            elif key == 'Mult_acc_2' and (value > best_results[key]):
                best_results[key] = value
                best_results['F1_score'] = results['F1_score']

                if save_best_model:
                    key_eval = 'Mult_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)

            elif key == 'Mult_acc_3' or key == 'Mult_acc_5' or key == 'Mult_acc_7' or key == 'Corr':
                if value > best_results[key]:
                    best_results[key] = value

                    if save_best_model:
                        key_eval = key
                        ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                        save_model(ckpt_path, epoch, model, optimizer)
            
            else:
                pass
    
    return best_results

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.scale * (x / (rms + self.eps))
    

def plot_metrics(train_losses, valid_losses, valid_maes=None):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    if valid_losses:
        plt.plot(epochs, valid_losses, label='Valid Loss', marker='s')
    if valid_maes:
        plt.plot(epochs, valid_maes, label='Valid MAE', marker='^')

    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()