import numpy as np
import tensorflow, os, json, pickle, yaml
from tensorflow.keras.utils import to_categorical
from collections import Counter
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
fpath = os.path.join(OUTPUT_DIR, "data.pkl")
with open(fpath, "rb") as fd:
    data = pickle.load(fd)
(x_train, y_train),(x_test, y_test) = data
labels = y_test.astype(int)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
input_size = image_size * image_size

# Model specific code
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test.astype('float32') / 255
# End of Model specific code

model_file = os.path.join(OUTPUT_DIR, "model.h5")
model = tensorflow.keras.models.load_model(model_file)

metrics_dict = model.evaluate(x_test, y_test, return_dict=True)
print(metrics_dict)

METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.json")
with open(METRICS_FILE, "w") as f:
    f.write(json.dumps(metrics_dict))

pred_probabilities = model.predict(x_test)
predictions = np.argmax(pred_probabilities, axis=1)
all_predictions = [{"actual": int(actual), "predicted": int(predicted)} for actual, predicted in zip(labels, predictions)]

os.makedirs(os.path.dirname("output/test/samples_of_mispredicted_images/"), exist_ok=True)
with open("output/test/predictions.json", "w") as f:
    json.dump(all_predictions, f)

predicted_classes = Counter(predictions)
predicted_classes = [{"class": int(k), "count": int(v)} for k,v in predicted_classes.items()]
with open("output/test/predicted_classes.json", "w") as f:
    json.dump(predicted_classes, f)

# Generate and samples of mislabeled images
all_mislabels = []
for i in range(len(all_predictions)):
    if all_predictions[i]['actual'] != all_predictions[i]['predicted']:
        all_mislabels.append({'index':i, 'actual': all_predictions[i]['actual'], 'predicted': all_predictions[i]['predicted']})
mislabels = {}
for label in range(10):
    mislabels[label] = [(ml['index'], ml['predicted']) for ml in all_mislabels if ml['actual'] == label]

label_map = {
            0: "T-shirt or top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
            }

x_test = np.reshape(x_test, [-1, image_size, image_size])
for label in range(10):
    images = mislabels[label]
    if images:
        num_images = len(images)
        num_rows = 1 if num_images < 3 else 2
        num_cols = 1 if num_images == 1 else 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.75*num_cols, 2.5*num_rows))
        fig.suptitle("Mispredicted images of " + label_map[label], fontsize=12, fontweight='bold', color='red')
        for i in range(min(num_rows*num_cols,num_images)):
            ax = axes[i//num_cols, i%num_cols] if num_rows > 1 else axes[i] if num_images > 1 else axes
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(x_test[images[i][0]], cmap='gray')
            ax.set_title('Prediction:\n{}'.format(label_map[images[i][1]]), color='blue')
        plt.savefig("output/test/samples_of_mispredicted_images/"+label_map[label]+".png")
