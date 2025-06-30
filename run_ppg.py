import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Debug info
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")

# Try multiple import options
try:
    from ppgs.model import PPGExtractor  # Новый путь
    print("Imported PPGExtractor from ppgs.model")
except ImportError:
    try:
        from ppgs import PPGExtractor  # Старый путь
        print("Imported PPGExtractor from ppgs")
    except ImportError:
        try:
            from ppgs.extract import Extractor as PPGExtractor  # Альтернативное имя
            print("Imported Extractor as PPGExtractor")
        except ImportError as e:
            print(f"Failed to import PPGExtractor: {str(e)}")
            sys.exit(1)

def main():
    try:
        # Configuration
        config = {
            "model_url": "https://github.com/interactiveaudiolab/ppgs/releases/download/v1.0/model.pth",
            "audio_path": "sample.wav",
            "output_dir": "results"
        }
        
        # Verify audio file
        if not os.path.exists(config["audio_path"]):
            raise FileNotFoundError(f"Audio file {config['audio_path']} not found")
        
        # Initialize extractor
        print("Initializing PPGExtractor...")
        extractor = PPGExtractor(
            model_path="pretrained/model.pth",
            device='cpu'
        )
        
        # Process audio
        print("Processing audio...")
        ppgs = extractor.extract(config["audio_path"])
        
        # Save results
        os.makedirs(config["output_dir"], exist_ok=True)
        np.save(f"{config['output_dir']}/ppgs.npy", ppgs)
        
        # Visualization
        plt.figure(figsize=(15,5))
        plt.imshow(ppgs.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plt.savefig(f"{config['output_dir']}/ppgs_plot.png")
        
        print("Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
