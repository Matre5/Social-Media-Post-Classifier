import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of posts
n_posts = 1000

# Generate features
caption_length = np.random.randint(20, 500, size=n_posts)
hashtags_count = np.random.randint(0, 30, size=n_posts)
time_of_day = np.random.choice(['morning', 'afternoon', 'evening', 'night'], size=n_posts)
post_type = np.random.choice(['image', 'video', 'text'], size=n_posts)
shares = np.random.randint(0, 500, size=n_posts)
comments = np.random.randint(0, 200, size=n_posts)

# Simulate likes based on features
likes = (
    caption_length * 0.5 +
    hashtags_count * 10 +
    np.where(time_of_day == 'evening', 200, 0) +
    np.where(post_type == 'video', 300, 100) +
    shares * 2 +
    comments * 3 +
    np.random.normal(0, 50, size=n_posts)  # noise
)

# Create binary label: popular if likes > 1000
popular = (likes > 1000).astype(int)

# Assemble DataFrame
df = pd.DataFrame({
    'caption_length': caption_length,
    'hashtags_count': hashtags_count,
    'time_of_day': time_of_day,
    'post_type': post_type,
    'shares': shares,
    'comments': comments,
    'likes': likes.astype(int),
    'popular': popular
})

# Save to CSV
df.to_csv('data/social_media_posts.csv', index=False)

print("✅ Dataset created successfully! Shape:", df.shape)
print(df.head())
