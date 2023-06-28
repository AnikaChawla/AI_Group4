import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow import keras
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix, f1_score, recall_score, precision_recall_curve, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import BinaryAccuracy, Recall, Precision, AUC
from tensorflow.keras.models import load_model


df = pd.read_csv("AI_Group4-main\Phishing_Mitre_Dataset_Summer_of_AI.csv")

train_df, test_df = train_test_split(df, test_size=0.3, random_state=80)
test_df, val_df = train_test_split(test_df, test_size=0.1, random_state=80)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['URL'])

train_sequences = tokenizer.texts_to_sequences(train_df['URL'])
test_sequences = tokenizer.texts_to_sequences(test_df['URL'])
val_sequences = tokenizer.texts_to_sequences(val_df['URL'])

max_sequence_length = max(len(seq) for seq in train_sequences)
train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)
val_data = pad_sequences(val_sequences, maxlen=max_sequence_length)

train_labels = train_df['Label'].values
test_labels = test_df['Label'].values
val_labels = val_df['Label'].values

# Define the parameter grid for GridSearchCV
param_grid = {
    'dropout_rate': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.01, 0.1],
    'lstm_units': [16, 32, 64, 128, 256],
    'optimizer': ['adam', 'rmsprop'],
}

# Create the RNN model
def create_model(dropout_rate=0.2, learning_rate=0.001, lstm_units=64):
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, 128, input_length=max_sequence_length))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'AUC'])

    return model

# Compile the model
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=1)
# Create GridSearchCV object
# grid_search = GridSearchCV(model, param_grid, cv=3)

# grid_search.fit(train_data, train_labels)

# # Get the best model
# best_model = grid_search.best_estimator_

# Train the model
# model.fit(train_data, train_labels, epochs=10, batch_size=32)


######## RNN MODEL Training
'''
model.fit(train_data, train_labels)
model.model.save("rnn_model.h5")
loss, accuracy, precision, recall, auc = model.model.evaluate(test_data, test_labels)
predicted_rnn = model.predict(test_data)
f1_score_rnn = f1_score(test_labels, predicted_rnn)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')
print(f'Test precision: {precision:.4f}')
print(f'Test recall: {recall:.4f}')
print(f'Test AUC: {auc:.4f}')
print(f'Test F1 score: {f1_score_rnn:.4f}')
'''
## 
model = load_model('rnn_model.h5')
predicted_y_rnn = model.predict(train_data)
predicted_y_test = model.predict(test_data)
predicted_y_val = model.predict(val_data)

predicted_y_rnn_reshaped = np.reshape(predicted_y_rnn, [1, 3359])

train_df["predictions"] = predicted_y_rnn
test_df["predictions"] = predicted_y_test
val_df["predictions"] = predicted_y_val
#train_df["URL"] = train_df["URL"].map(model.predict)
#train_df["URL"] = train_df["URL"].map(lambda x: x + "hello")

train_df = train_df.drop(["URL"], axis=1)
test_df = test_df.drop(["URL"], axis=1)
val_df = val_df.drop(["URL"], axis=1)

xgb_model = XGBClassifier()
xgb_model.fit(train_df, train_labels)
predicted_y_xgb = xgb_model.predict(test_df)
predicted_y_xgb_val = xgb_model.predict(val_df)

accuracy_xgb = accuracy_score(test_labels, predicted_y_xgb)
auc_xgb = roc_auc_score(test_labels, predicted_y_xgb)
log_loss_xgb = log_loss(test_labels, predicted_y_xgb)
f1_score_xgb = f1_score(test_labels, predicted_y_xgb)
recall_score_xgb = recall_score(test_labels, predicted_y_xgb)
precision_score_xgb = precision_score(test_labels, predicted_y_xgb)
print("Test DF Results")
print(f'GB accuracy: {accuracy_xgb:.4f}')
print(f'GB AUC: {auc_xgb:.4f}')
print(f'GB Log Loss: {log_loss_xgb:.4f}')
print(f'F1 Score: {f1_score_xgb:.4f}')
print(f'Recall Score: {recall_score_xgb:.4f}')
print(f'Precision Score: {precision_score_xgb:.4f}')


accuracy_xgb_val = accuracy_score(val_labels, predicted_y_xgb_val)
auc_xgb_val = roc_auc_score(val_labels, predicted_y_xgb_val)
log_loss_xgb_val = log_loss(val_labels, predicted_y_xgb_val)
f1_score_xgb_val = f1_score(val_labels, predicted_y_xgb_val)
recall_score_xgb_val = recall_score(val_labels, predicted_y_xgb_val)
precision_score_xgb_val = precision_score(val_labels, predicted_y_xgb_val)
print("Validation DF Results")
print(f'GB accuracy: {accuracy_xgb_val:.4f}')
print(f'GB AUC: {auc_xgb_val:.4f}')
print(f'GB Log Loss: {log_loss_xgb_val:.4f}')
print(f'F1 Score: {f1_score_xgb_val:.4f}')
print(f'Recall Score: {recall_score_xgb_val:.4f}')
print(f'Precision Score: {precision_score_xgb_val:.4f}')

np.savetxt("test_predicted.csv", predicted_y_xgb, delimiter="/n")
np.savetxt("test_actual.csv", test_labels, delimiter="/n")
#confusion_matrix, precision_recall_curve
# gb_model = tf.keras.GradientBoostedTreesModel()

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# gb_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'AUC'])
# gb_model.fit(rnn_output, train_labels)