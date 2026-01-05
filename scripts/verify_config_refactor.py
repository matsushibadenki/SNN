#!/usr/bin/env python3
import sys
from snn_research.utils.config_loader import load_config


def test_config_loader():
    print("Testing config loader...")
    try:
        # 1. Load default unified config
        cfg = load_config(config_name="config")
        print("✅ Correctly loaded 'config' (unified) config.")

        # 2. Check structure
        assert cfg.model.d_model is not None, "d_model should be present"
        print(f"   d_model: {cfg.model.d_model}")

    except Exception as e:
        print(f"❌ Failed during manual config load test: {e}")
        sys.exit(1)


def main():
    test_config_loader()
    print("✅ All verification steps passed!")


if __name__ == "__main__":
    main()
