import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import time
import numpy as np
import pandas as pd
from scipy import signal
from dotenv import load_dotenv
from brain_engine import BrainEngine
from filters import build_passthought_sos

# 1. Load configuration from .env file
load_dotenv()
SERIAL_REAL = os.environ.get("UNICORN_SERIAL")

if not SERIAL_REAL:
    print("❌ ERROR: UNICORN_SERIAL not found in .env file")
    exit()

# 2. Processing Configuration
# Passthought Filter (8-30 Hz) to detect Mu/Beta rhythm
sos_pt = build_passthought_sos()

def collect_data():
    engine = BrainEngine(serial=SERIAL_REAL)
    dataset = []

    try:
        print(f"🔌 Connecting to Unicorn: {SERIAL_REAL}")
        engine.start()
        print("✅ Connection established. RingBuffer is active.")

        # Define training labels in English
        for state in ['Rest', 'Passthought']:
            print(f"\n{'='*40}")
            print(f" CURRENT PHASE: {state.upper()}")
            print(f"{'='*40}")
            print("Get ready... recording starts in 3 seconds.")
            time.sleep(3)

            print(f"🔴 Recording 30 seconds of {state}...")
            # Capture fifteen 2-second windows
            for i in range(15):
                time.sleep(2)
                
                # Read the last 2 seconds (500 samples) from the centralized buffer
                raw = engine.buffer.read_from(engine.buffer.total_written - 500, 500)
                
                if raw is not None:
                    # Apply Second-Order Section (SOS) filter
                    filtered_data = signal.sosfilt(sos_pt, raw, axis=0)
                    
                    # Extract variance from C3 (index 1) and C4 (index 3) based on brain_engine map
                    e3 = np.var(filtered_data[:, 1])
                    e4 = np.var(filtered_data[:, 3])
                    
                    dataset.append([e3, e4, state])
                    print(f"   [OK] Sample {i+1}/15 | Var C3: {e3:.2f} | Var C4: {e4:.2f}")
                else:
                    print("   [!] Error: Buffer does not have enough data yet.")

    except Exception as e:
        print(f"❌ Critical error: {e}")
    finally:
        print("\nStopping acquisition...")
        engine.stop()
        
        if dataset:
            df = pd.DataFrame(dataset, columns=['Variance_C3', 'Variance_C4', 'Class'])
            df.to_csv('Fase2/dataset_phase2.csv', index=False)
            print(f"\n🏁 DONE! 'dataset_phase2.csv' generated with {len(df)} samples.")

if __name__ == "__main__":
    collect_data()