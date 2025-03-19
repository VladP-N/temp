import os
import keras as k
import numpy as np
import matplotlib.pyplot as plt

dataset = open('Dataset.txt', 'r').read().split(', ')
dataset = [float(x) for x in dataset]

train_x = np.array([dataset[0:108]])
train_y = np.array([dataset[108:120]])

future_x = np.array([])

model = k.Sequential()
model.add(k.layers.LSTM(1, return_sequences=True, input_shape=(108, 1)))
model.add(k.layers.Dense(32, activation='relu'))
model.add(k.layers.Dense(12))
model.compile(optimizer='adam', loss='mse')

needLoad = False

if os.path.exists('model.weight.h5'):
    if input('Загрузить веса? Введите <y/n>: ').lower() == 'y':
        needLoad = True
        model.load_weights('model.weight.h5')

if needLoad == False:
    history = model.fit(train_x, train_y, epochs=1000, batch_size=16, verbose=2)

    plt.semilogy(history.history['loss'])
    plt.title('Loss during training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    if input('Сохранить веса? Введите <y/n>: ').lower() == 'y':
        model.save('model.weight.h5')


future_x = open('Dataset_predict.txt', 'r').read().split(', ')
future_x = [float(x) for x in dataset]
future_x = np.array([dataset[12:120]])

dataset_test = open('Dataset_test.txt', 'r').read().split(', ')
dataset_test = [float(x) for x in dataset_test]

test_x = np.array([dataset[0:108]])
test_y = np.array([dataset[108:120]])

score = model.evaluate(test_x, test_y, batch_size=4)
print(f'Оценка точности: {round(score, 3)} MSE ({round(np.sqrt(score), 3)} RMSE)')

future_y = model.predict(future_x, verbose=0)[0][0]

months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
colors = ['blue' if value <= 0 else 'red' for value in future_y]

plt.bar(months, future_y, color=colors)
plt.xlabel('Месяцы')
plt.ylabel('Температура')
plt.title('Температура по месяцам')
plt.axhline(0, color='black', linewidth=0.8)
plt.show()

