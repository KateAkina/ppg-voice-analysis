import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from ppgs import PPGExtractor

def check_audio_file(path):
    if not os.path.exists(path):
        print(f"Error: Audio file {path} not found!", file=sys.stderr)
        return False
    return True

def download_model(url, path):
    try:
        import urllib.request
        print(f"Downloading model from {url}...")
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"Model download failed: {str(e)}", file=sys.stderr)
        return False

def main():
    # Configuration
    config = {
        "model_url": "https://github.com/interactiveaudiolab/ppgs/releases/download/v1.0/model.pth",
        "audio_path": "sample.wav",
        "output_dir": "results"
    }
    
    # Verify input file
    if not check_audio_file(config["audio_path"]):
        return 1
    
    # Prepare model
    os.makedirs("pretrained", exist_ok=True)
    model_path = "pretrained/model.pth"
    
    if not os.path.exists(model_path) and not download_model(config["model_url"], model_path):
        return 1
    
    try:
        # Initialize extractor
        print("Initializing PPG extractor...")
        extractor = PPGExtractor(
            model_path=model_path,
            device='cpu'
        )
        
        # Process audio
        print("Processing audio file...")
        ppgs = extractor.extract(config["audio_path"])
        
        # Save results
        os.makedirs(config["output_dir"], exist_ok=True)
        np.save(f"{config['output_dir']}/ppgs.npy", ppgs)
        
        # Create visualization
        plt.figure(figsize=(15,5))
        plt.imshow(ppgs.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plt.savefig(f"{config['output_dir']}/ppgs_plot.png")
        
        print("Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
