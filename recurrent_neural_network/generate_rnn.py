from keras.models import load_model
from keras.utils import to_categorical
import numpy as np

LINE_CHAR_LENGTH = 128

model = load_model('anna.h5')

string_array = np.fromstring(
    "\"“Well, Prince, so Genoa and Lucca are now just family estates of the Buonapartes. But I warn you, if you don’t tell me that this means war, if you still try to defend the infamies and horrors perpetrated by that Antichrist—I really believe he is Antichrist—I will have nothing more to do with you and you are no longer my friend, no longer my 'faithful slave,' as you call yourself! But how do you do? I see I have frightened you—sit down and tell me all the news.\"",
    np.uint8,
)

string_array = string_array[:LINE_CHAR_LENGTH].reshape((1, LINE_CHAR_LENGTH))
string_matrix = to_categorical(string_array, 256)

print(
    model.predict(string_matrix)
)
