from pathlib import Path
p = Path('app/realtime_infer.py')
s = p.read_text(encoding='utf-8')
s = s.replace("torch.load(ckpt_path, map_location=device)", "torch.load(ckpt_path, map_location=device, weights_only=False)")
p.write_text(s, encoding='utf-8')
print('ok')
