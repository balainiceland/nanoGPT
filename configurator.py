"""
Safe Configurator. Replaces exec()-based config loading with JSON parsing.

Example usage:
$ python train.py config/override_file.json --batch_size=32
this will first load config/override_file.json, then override batch_size to 32

Also supports legacy .py config files by parsing simple `key = value` assignments
(without executing arbitrary code).

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()
"""

import sys
import json
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")

        if config_file.endswith('.json'):
            # JSON config: safe and structured
            with open(config_file) as f:
                config_data = json.load(f)
            for key, val in config_data.items():
                if key in globals():
                    print(f"  {key} = {val}")
                    globals()[key] = val
                else:
                    raise ValueError(f"Unknown config key: {key}")
        else:
            # Legacy .py config: parse simple key=value assignments safely
            # (no exec — only supports `key = literal_value` lines)
            with open(config_file) as f:
                content = f.read()
            print(content)
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, _, val_str = line.partition('=')
                    key = key.strip()
                    val_str = val_str.strip()
                    # Strip inline comments
                    if '#' in val_str:
                        val_str = val_str[:val_str.index('#')].strip()
                    if key in globals():
                        try:
                            val = literal_eval(val_str)
                        except (SyntaxError, ValueError):
                            val = val_str
                        print(f"  Overriding: {key} = {val}")
                        globals()[key] = val
                    else:
                        raise ValueError(f"Unknown config key in {config_file}: {key}")
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
