import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 设置使用的GPU设备
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml
import argparse
from core.dataset import MMDataLoader
from core.losses import MultimodalLoss
from core.scheduler import get_scheduler
from core.utils import setup_seed, get_best_results
from models.GFMamba import GFMamba
from core.metric import MetricsTop 
from core.utils import plot_metrics
import warnings
warnings.filterwarnings("ignore")


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='') 
parser.add_argument('--seed', type=int, default=-1) 
opt = parser.parse_args()
print(opt)



def main():
    best_valid_results, best_test_results = {}, {}
    train_losses, valid_losses, valid_maes = [], [], []
    best_valid_loss = float('inf')
    patience_counter = 0

    config_file = 'configs/mosi_train.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)

    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)
    print("seed is fixed to {}".format(seed))

    ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
    print("ckpt root :", ckpt_root)

    model = GFMamba(args).to(device)

    dataLoader = MMDataLoader(args)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args['base']['lr'],
                                 weight_decay=args['base']['weight_decay'])
    
    # 添加学习率调度器
    if args['base'].get('lr_scheduler') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args['base']['n_epochs'],
            eta_min=1e-6
        )
    else:
        scheduler = get_scheduler(optimizer, args)

    loss_fn = MultimodalLoss(args)

    metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(args['dataset']['datasetName'])

    # 早停参数
    early_stopping_patience = args['base'].get('early_stopping_patience', 10)

    for epoch in range(1, args['base']['n_epochs']+1):
        train_loss, train_results = train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)
        train_losses.append(train_loss)

        if args['base']['do_validation']:
            valid_results, valid_loss = evaluate(model, dataLoader['valid'], loss_fn, epoch, metrics)
            valid_losses.append(valid_loss)
            valid_maes.append(valid_results['MAE'])

            best_valid_results = get_best_results(valid_results, best_valid_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=False)
            print(f'Current Best Valid Results: {best_valid_results}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'valid_results': valid_results
                }, os.path.join(ckpt_root, f'best_valid_model_seed_{seed}.pth'))
            else:
                patience_counter += 1
                print(f'Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}')
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break

        test_results, test_loss = evaluate(model, dataLoader['test'], loss_fn, epoch, metrics)
        best_test_results = get_best_results(test_results, best_test_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=True)
        print(f'Current Best Test Results: {best_test_results}\n')

        # 更新学习率
        if args['base'].get('lr_scheduler') == 'cosine':
            scheduler.step()
        else:
            if 'MAE' in valid_results:
                scheduler.step(metrics=valid_results['MAE'])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}, Learning Rate: {current_lr:.6f}')
        plot_metrics(train_losses, valid_losses, valid_maes)


def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    y_pred = []
    y_true =[]
    loss_dict = {}
    total_loss = 0.0
    
    model.train()
    for cur_iter, data in enumerate(train_loader):
        # 只使用完整模态输入下、
        text_x = data['text'].to(device)
        video_x = data['vision'].to(device)
        audio_x = data['audio'].to(device)
        # 标签
        sentiment_labels = data['labels']['M'].to(device)
        label = {'sentiment_labels': sentiment_labels}

        # 前向传播，只传完整输入
        out = model(text_x, video_x, audio_x)
        # 损失计算，无需 mask
        loss = loss_fn(out, label)
        if isinstance(out, dict) and 'sentiment_preds' in out:
            preds = out['sentiment_preds'].detach()
           # print(f"[DEBUG] Epoch {epoch} Iter {cur_iter}: pred mean = {preds.mean().item():.4f}, std = {preds.std().item():.4f}")
           # print(f"[DEBUG] label mean = {label['sentiment_labels'].float().mean().item():.4f}, std = {label['sentiment_labels'].float().std().item():.4f}")
           # print(f"[DEBUG] loss = {loss['loss'].item():.4f}")

        loss['loss'].backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        # 累加损失
        total_loss += loss['loss'].item()
        if cur_iter == 0:
            for key, value in loss.items():
                loss_dict[key] = value.item()
        else:
            for key, value in loss.items():
                loss_dict[key] += value.item()

    # 聚合结果，计算评估指标
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)
    loss_dict = {key: value / (cur_iter + 1) for key, value in loss_dict.items()}
    avg_loss = total_loss / (cur_iter + 1)

    print(f'Train Loss Epoch {epoch}: {loss_dict}')
    print(f'Train Results Epoch {epoch}: {results}')

    
    return avg_loss, results
    
def evaluate(model, eval_loader, loss_fn, epoch, metrics):
    loss_dict = {}
    y_pred, y_true = [], []
    total_loss = 0.0

    model.eval()

    for cur_iter, data in enumerate(eval_loader):
        text_x = data['text'].to(device)
        video_x = data['vision'].to(device)
        audio_x = data['audio'].to(device)
        sentiment_labels = data['labels']['M'].to(device)
        label = {'sentiment_labels': sentiment_labels}

        with torch.no_grad():
            out = model(text_x, video_x, audio_x)

        loss = loss_fn(out, label)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        total_loss += loss['loss'].item()
        for key, value in loss.items():
            value = value.item() if hasattr(value, 'item') else value
            loss_dict[key] = loss_dict.get(key, 0) + value

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)
    avg_loss = total_loss / (cur_iter + 1)
    

    return results, avg_loss



if __name__ == '__main__':
    main()


