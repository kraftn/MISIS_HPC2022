# Запуск
```
python main.py ./data/Lenna.png
```

# Результаты
Исходное изображение:

![image](data/Lenna.png)

Результат применения шаблона `conv1 = cp.ones((11, 11), dtype=cp.float32)`:

![image](data/result_1.png)

Результат применения шаблона `conv2 = cp.full((3, 3), -1, dtype=cp.float32); conv2[1, 1] = 9`:

![image](data/result_2.png)
