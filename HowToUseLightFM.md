# LightFMをどのように使ったらよいかのドキュメント

## 概要
モデルの学習に必要な変数は以下の3個
interactions
user_features
item_features

そして上記の変数を作るために必要な変数は以下の7個。
all_user_ids = [0, 1, 2]
all_item_ids = [3, 4, 5]
all_user_features = ['M', 'F', 'M']
all_item_features = ['sf', 'anime']

data = [(3, 43), (4, 5)]
user_features_data  = [(0, {"M":1, "F":0})] 
item_features_data  = [(4, {"sf":1, "anime":0})] 


カテゴリ変数は0.5。連続変数は0から１で正規化します。

## lightfm/data.py
def fit(self, users, items, user_features=None, item_features=None):
    """
    Fit the user/item id and feature name mappings.

    Calling fit the second time will reset existing mappings.

    Parameters
    ----------

    users: iterable of user ids
    items: iterable of item ids
    user_features: iterable of user features, optional
    item_features: iterable of item features, optional
    """

def build_user_features(self, data, normalize=True):
    """
    Build a user features matrix out of an iterable of the form
    (user id, [list of feature names]) or (user id, {feature name: feature weight}).

    Parameters
    ----------

    data: iterable of the form
        (user id, [list of feature names]) or (user id,
        {feature name: feature weight}).
        User and feature ids will be translated to internal indices
        constructed during the fit call.
    normalize: bool, optional
        If true, will ensure that feature weights sum to 1 in every row.

    Returns
    -------

    feature matrix: CSR matrix (num users, num features)
        Matrix of user features.
    """

def build_item_features(self, data, normalize=True):
    """
    Build a item features matrix out of an iterable of the form
    (item id, [list of feature names]) or (item id, {feature name: feature weight}).

    Parameters
    ----------

    data: iterable of the form
        (item id, [list of feature names]) or (item id,
        {feature name: feature weight}).
        Item and feature ids will be translated to internal indices
        constructed during the fit call.
    normalize: bool, optional
        If true, will ensure that feature weights sum to 1 in every row.

    Returns
    -------

    feature matrix: CSR matrix (num items, num features)
        Matrix of item features.
    """
