import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from ppgs import PPGExtractor

def main():
    try:
        print("Starting PPG analysis...")
        
        # Config
        config = {
            "model_url": "https://github.com/interactiveaudiolab/ppgs/releases/download/v1.0/model.pth",
            "audio_path": "sample.wav",
            "output_dir": "results"
        }

        # Verify input file
        if not os.path.exists(config["audio_path"]):
            raise FileNotFoundError(f"Audio file {config['audio_path']} not found!")

        # Download model
        os.makedirs("pretrained", exist_ok=True)
        model_path = "pretrained/model.pth"
        
        if not os.path.exists(model_path):
            print("Downloading model...")
            import urllib.request
            urllib.request.urlretrieve(config["model_url"], model_path)

        # Initialize
        print("Initializing PPG extractor...")
        extractor = PPGExtractor(
            model_path=model_path,
            device='cpu'
        )

        # Process
        print("Processing audio...")
        ppgs = extractor.extract(config["audio_path"])
        
        # Save results
        os.makedirs(config["output_dir"], exist_ok=True)
        output_path = f"{config['output_dir']}/ppgs.npy"
        np.save(output_path, ppgs)
        print(f"Saved results to {output_path}")
        
        # Visualization
        plt.figure(figsize=(15,5))
        plt.imshow(ppgs.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plot_path = f"{config['output_dir']}/ppgs_plot.png"
        plt.savefig(plot_path)
        print(f"Saved visualization to {plot_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
