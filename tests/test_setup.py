"""
Test script to verify the environment setup.
Run this to ensure all packages are installed correctly.
"""

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        
        import sentence_transformers
        print(f"✓ Sentence-Transformers {sentence_transformers.__version__}")
        
        from Bio import Entrez
        print("✓ BioPython (Entrez)")
        
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
        
        print("\n✅ All packages installed successfully!")
        print("\nYou're ready to proceed with data collection!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
