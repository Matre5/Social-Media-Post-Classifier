import joblib
import pandas as pd

model = joblib.load('../Model/random_forest_classifier.pkl')

sample = pd.DataFrame([{
    'caption_length': 120,
    'hashtags_count': 5,
    'time_of_day_evening': 1,
    'time_of_day_morning': 0,
    'time_of_day_night': 0,
    'post_type_image': 0,
    'post_type_text': 1,
    'shares': 50,
    'comments': 10
}])

pred = model.predict(sample)
print("Prediction (1=Popular, 0=Not Popular):", pred[0])
