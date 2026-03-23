from util.metric_tool import ConfuseMatrixMeter
import torch
from option import Options
from data.cd_dataset import DataLoader
from model.create_model import create_model
from tqdm import tqdm
import os
import numpy as np
from PIL import Image


def create_visualization(pred, gt):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    error_vis = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    
    correct_change = np.logical_and(pred == 1, gt == 1)
    error_vis[correct_change] = [255, 255, 255]
    
    correct_no_change = np.logical_and(pred == 0, gt == 0)
    error_vis[correct_no_change] = [0, 0, 0]
    
    false_positive = np.logical_and(pred == 1, gt == 0)
    error_vis[false_positive] = [255, 0, 0]
    
    false_negative = np.logical_and(pred == 0, gt == 1)
    error_vis[false_negative] = [0, 255, 0]
    
    return error_vis

if __name__ == '__main__':
    if os.path.exists('test_complete.flag'):
        os.remove('test_complete.flag')
        print("Removed old test flag, starting fresh test")

    opt = Options().parse()
    opt.phase = 'test'
    
    opt.batch_size = 1
    
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)

    vis_dir = os.path.join('results', opt.name, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    tbar = tqdm(test_data, total=test_size)
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()

    model.eval()
    with torch.no_grad():
        for i, _data in enumerate(tbar):
            img1 = _data['img1'].cuda()
            img2 = _data['img2'].cuda()
            gt = _data['cd_label'].cpu()
            fname = _data['fname'][0]
            
            val_pred = model.inference(img1, img2)

            pred = torch.argmax(val_pred.detach(), dim=1).cpu()
            
            _ = running_metric.update_cm(pr=pred.numpy(), gt=gt.numpy())
            
            pred_np = pred[0].numpy()
            gt_np = gt[0].numpy()
            
            cm_vis = create_visualization(pred_np, gt_np)
            vis_path = os.path.join(vis_dir, f"{fname}")
            Image.fromarray(cm_vis).save(vis_path)
            
            tbar.set_description(f"Processing {i+1}/{test_size}")

        val_scores = running_metric.get_scores()
        
        print("\n" + "="*80)
        print(f"Testing Complete")
        print("="*80)
        print(f"Phase: {opt.phase}")
        print(f"Images processed: {test_size}")
        print(f"Model: {opt.name}")
        print("-"*80)
        print("Evaluation Metrics (%):")
        
        message = '(phase: %s) ' % (opt.phase)
        for k, v in val_scores.items():
            metric_name = k.upper()
            metric_value = v * 100
            print(f"  {metric_name:12}: {metric_value:6.2f}%")
            message += '%s: %.3f ' % (k, v * 100)
            
        print("-"*80)
        print("Visualization Color Encoding:")
        print("  White: TP (True Positive)")
        print("  Black: TN (True Negative)")
        print("  Red: FP (False Positive)")
        print("  Green: FN (False Negative)")
        print("="*80)

        metrics_dir = os.path.join('results', opt.name)
        os.makedirs(metrics_dir, exist_ok=True)
        
        report_path = os.path.join(metrics_dir, 'test_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Test Report\n")
            f.write("="*50 + "\n")
            f.write(f"Phase: {opt.phase}\n")
            f.write(f"Images processed: {test_size}\n")
            f.write(f"Model: {opt.name}\n")
            f.write("-"*50 + "\n")
            f.write("Evaluation Metrics (%):\n")
            for k, v in val_scores.items():
                f.write(f"  {k.upper():12}: {v*100:6.2f}%\n")
            f.write("-"*50 + "\n")
            f.write("Visualization Color Encoding:\n")
            f.write("  White: TP (True Positive)\n")
            f.write("  Black: TN (True Negative)\n")
            f.write("  Red: FP (False Positive)\n")
            f.write("  Green: FN (False Negative)\n")
            
        with open(os.path.join(metrics_dir, 'metrics.txt'), 'w') as f:
            f.write(message)
            
        with open('test_complete.flag', 'w') as f:
            f.write('Test completed')
            
        print(f"\nResults saved to:")
        print(f"  Visualizations: {os.path.join('results', opt.name, 'visualization')}")
        print(f"  Report: {report_path}")
        print(f"  Metrics: {os.path.join(metrics_dir, 'metrics.txt')}")
        print("="*80)

