# convert_paths.py
import numpy as np

# Загружаем пути
paths = np.load('clip_image_paths.npy', allow_pickle=True)

# Преобразуем все пути в строки Python
paths_as_strings = np.array([str(p) for p in paths])

# Сохраняем обратно
np.save('clip_image_paths.npy', paths_as_strings)

print(f"✅ Преобразовано {len(paths)} путей в строки Python")
print("Примеры после преобразования:")
for i in range(min(3, len(paths_as_strings))):
    print(f"  {i}: {type(paths_as_strings[i])} -> {paths_as_strings[i][:50]}...")