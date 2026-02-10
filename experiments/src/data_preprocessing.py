import os
import logging 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the log directory exists
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
# Setting up logger
logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    ps=PorterStemmer()
    text=text.lower()
    text=nltk.word_tokenize(text)
    # Remove non-alphanumeric characters and punctuation
    text=[word for word in text if word.isalnum()]
    # Remove stop words and punctuation
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text=[ps.stem(word) for word in text]
    return " ".join(text)

def prepross_df(df,text_column='text',target_column='target'):
    try:
        logger.debug('Starting preprocessing for DataFrame')
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug('Label encoding completed successfully')

        # remove the duplicate
        df.drop_duplicates(inplace=True)
        logger.debug('Duplicate removal completed successfully')

        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text transformation completed successfully')
        return df
    except KeyError as e:
        logger.error(f"Column not found in DataFrame. Error: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error during DataFrame preprocessing. Error: {e}")
        raise e

def main(text_column='text',target_column='target'):
    try:
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')    
        logger.debug('Data loaded successfully for preprocessing')

        train_data=prepross_df(train_data,text_column,target_column)
        test_data=prepross_df(test_data,text_column,target_column)

        # store data inside data/processed
        data_path=os.path.join('./data','processed')
        os.makedirs(data_path,exist_ok=True)

        train_data.to_csv(os.path.join(data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(data_path,'test.csv'),index=False)
        logger.debug('Preprocessed data saved successfully')
    except Exception as e:
        logger.error(f"Failed to complete the data preprocessing process. Error: {e}")
        raise e
if __name__=='__main__':
    main()

