import pandas as pd

# Load your data
try:
    df = pd.read_csv('labeled_data.csv')
except FileNotFoundError:
    print("Error: 'labeled_data.csv' not found. Make sure it's in the right directory.")
    # Create a dummy dataframe for demonstration purposes if the file doesn't exist
    data = {'text': ['This is about finance and money.',
                     'The artist painted a masterpiece.',
                     'A new rocket launched to the moon.', # This will be our "new" topic
                     'The economy is growing this quarter.',
                     'Exploring the history of ancient Rome.',
                     'The sun is a star in our solar system.'], # Another "new" topic
            'label': ['economics_finance', 'art_culture', None, 'economics_finance', 'history', None]}
    df = pd.DataFrame(data)
    print("Created a dummy dataframe for demonstration.")


# Assuming your columns are named 'text' and 'label'
# Change these if your column names are different
text_column = 'text'
label_column = 'label'

# Separate the labeled and unlabeled data
labeled_df = df[df[label_column].notna()].copy()
unlabeled_df = df[df[label_column].isna()].copy()

print(f"Found {len(labeled_df)} labeled rows.")
print(f"Found {len(unlabeled_df)} unlabeled rows that need labels.")