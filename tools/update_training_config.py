import os
import sys
import argparse
from mmengine.config import Config

def update_config(base_config, train_ann, val_ann, output_config=None):
    print(f"🛠️ Updating config {base_config} with new annotations...")
    
    try:
        cfg = Config.fromfile(base_config)
        
        # Update dataset paths, data_root, and data_prefix
        for dl in [cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader]:
            dl.dataset.ann_file = os.path.abspath(train_ann if dl == cfg.train_dataloader else val_ann)
            dl.dataset.data_root = '/app/data'
            dl.dataset.data_prefix = dict(img='images/hq_consolidation_550/')

        # Update evaluators
        if hasattr(cfg, 'val_evaluator'):
            cfg.val_evaluator.ann_file = os.path.abspath(val_ann)
        if hasattr(cfg, 'test_evaluator'):
            cfg.test_evaluator.ann_file = os.path.abspath(val_ann)
            
        # Update work_dir to avoid overwriting legacy data
        cfg.work_dir = './work_dirs/automation_run'

        # Update load_from
        if hasattr(cfg, 'load_from') and cfg.load_from and not cfg.load_from.startswith(('http', '/')):
            cfg.load_from = os.path.join('/app/mmpose', cfg.load_from)

        if output_config is None:
            output_config = base_config # Overwrite or create new? Let's overwrite for simplicity in automation
            
        cfg.dump(output_config)
        print(f"✅ Config saved to {output_config}")
        return True
    except Exception as e:
        print(f"❌ Failed to update config: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update MMPose config with new annotation paths")
    parser.add_argument('--config', type=str, required=True, help="Path to base config file")
    parser.add_argument('--train', type=str, required=True, help="Path to new train JSON")
    parser.add_argument('--val', type=str, required=True, help="Path to new val JSON")
    parser.add_argument('--output', type=str, help="Output path for updated config")
    args = parser.parse_args()

    if update_config(args.config, args.train, args.val, args.output):
        sys.exit(0)
    else:
        sys.exit(1)
