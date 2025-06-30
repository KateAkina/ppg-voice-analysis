import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from ppgs import PPGExtractor

def main():
    # Настройки
    config = {
        "model_url": "https://github.com/interactiveaudiolab/ppgs/releases/download/v1.0/model.pth",
        "audio_path": "sample.wav",
        "output_dir": "results"
    }

    # Скачивание модели
    os.makedirs("pretrained", exist_ok=True)
    model_path = "pretrained/model.pth"
    
    if not os.path.exists(model_path):
        import urllib.request
        urllib.request.urlretrieve(config["model_url"], model_path)

    # Инициализация
    extractor = PPGExtractor(
        model_path=model_path,
        device='cpu'  # GitHub Actions не поддерживает GPU
    )

    # Обработка
    ppgs = extractor.extract(config["audio_path"])
    
    # Сохранение
    os.makedirs(config["output_dir"], exist_ok=True)
    np.save(f"{config['output_dir']}/ppgs.npy", ppgs)
    
    # Визуализация
    plt.figure(figsize=(15,5))
    plt.imshow(ppgs.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.savefig(f"{config['output_dir']}/ppgs_plot.png")
    
    print("Анализ завершён!")

if __name__ == "__main__":
    main()
