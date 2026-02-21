import sys, json, os
sys.path.insert(0, '.')
from config import system_configs

tests = [
    ('LSTR_CULANE_100_baseline', 'TransformerEncoder', 5),
    ('LSTR_CULANE_100_mamba',    'BidirectionalMambaEncoder', 2.5),
    ('LSTR_CULANE_2k_baseline',  'TransformerEncoder', 5),
    ('LSTR_CULANE_2k_mamba',     'BidirectionalMambaEncoder', 2.5),
]

all_pass = True
for name, expected_enc, expected_lc in tests:
    try:
        with open(f'config/{name}.json') as f:
            cfg = json.load(f)
        cfg['system']['snapshot_name'] = name
        cfg['system']['data_dir'] = '/home/alki/projects/'
        system_configs.update_config(cfg['system'])

        import importlib
        # Reload to avoid caching issues
        mod_name = f'models.{name}'
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        mod = importlib.import_module(mod_name)

        m = mod.model(flag=True)
        l = mod.loss()
        enc_cls = m.transformer.encoder.__class__.__name__
        lc = l.criterion.weight_dict.get('loss_curves', '?')

        enc_ok = enc_cls == expected_enc
        lc_ok  = abs(float(lc) - expected_lc) < 0.01
        status = 'PASS' if (enc_ok and lc_ok) else 'FAIL'
        if not (enc_ok and lc_ok):
            all_pass = False
        print(f'{status} | {name:<35} | enc={enc_cls} (exp:{expected_enc}) | lc={lc} (exp:{expected_lc})')
    except Exception as e:
        all_pass = False
        print(f'FAIL | {name:<35} | ERROR: {e}')

print()
print('ALL PASS' if all_pass else 'SOME FAIL')
