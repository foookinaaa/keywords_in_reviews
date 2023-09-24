import joblib
import numpy as np
from matplotlib import pyplot as plt
from lime.lime_text import LimeTextExplainer
import pandas as pd


def preprocess_review(review):
    review = review.lower()
    review = np.array([review])
    return review


class ReviewClf:
    def __init__(self, model_path='./train/model.pkl', lime_num_features=10, tfidf_num_features=20):
        self.model = joblib.load(model_path)
        self.lime_num_features = lime_num_features
        self.tfidf_num_features = tfidf_num_features

    def explain_review(self, review):
        explainer = LimeTextExplainer(class_names=['sad', 'happy'])
        return explainer.explain_instance(str(review), self.model.predict_proba, num_features=self.lime_num_features)

    def predict_review(self, review):
        exp = self.explain_review(preprocess_review(review))

        feature_names = self.model.named_steps['tf_idf'].get_feature_names_out()
        feats = pd.DataFrame({i: [j] for i, j in zip(feature_names, self.model['logit'].coef_[0])}).T
        feats = feats.sort_values(0, ascending=False, key=lambda x: abs(x))
        transform_review = self.model['tf_idf'].transform(preprocess_review(review))
        words_imp = [i for i in feature_names[transform_review.nonzero()[1]] if len(i.split()) >= 3]

        if feats[feats.index.isin(words_imp)].shape[0] > 0:
            fig = plt.figure()
            plt.style.use('ggplot')
            plt.tight_layout()
            plt.barh(feats[feats.index.isin(words_imp)].sort_values(0, ascending=True, key=lambda x: abs(x)).head(self.tfidf_num_features).index,
                     feats[feats.index.isin(words_imp)].sort_values(0, ascending=True, key=lambda x: abs(x)).head(self.tfidf_num_features).values.flatten())

            search_words = feats[feats.index.isin(words_imp)].sort_values(0, ascending=True, key=lambda x: abs(x)).head(self.tfidf_num_features).index
            for search_word in search_words:
                if search_word in review:
                    review = review.replace(search_word, f'<span style="background-color: #44aaff; color: #ffaa33;">{search_word}</span>')

        else:
            fig = None

        return review, fig, exp
