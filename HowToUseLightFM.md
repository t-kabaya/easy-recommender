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
# evaluate.py
def precision_at_k(
    model,
    test_interactions,
    train_interactions=None,
    k=10,
    user_features=None,
    item_features=None,
    preserve_rows=False,
    num_threads=1,
    check_intersections=True,
):
    """
    Measure the precision at k metric for a model: the fraction of known
    positives in the first k positions of the ranked list of results.
    A perfect score is 1.0.

    Parameters
    ----------

    model: LightFM instance
         the fitted model to be evaluated
    test_interactions: np.float32 csr_matrix of shape [n_users, n_items]
         Non-zero entries representing known positives in the evaluation set.
    train_interactions: np.float32 csr_matrix of shape [n_users, n_items], optional
         Non-zero entries representing known positives in the train set. These
         will be omitted from the score calculations to avoid re-recommending
         known positives.
    k: integer, optional
         The k parameter.
    user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
         Each row contains that user's weights over features.
    item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
         Each row contains that item's weights over features.
    preserve_rows: boolean, optional
         When False (default), the number of rows in the output will be equal
         to the number of users with interactions in the evaluation set.
         When True, the number of rows in the output will be equal to the
         number of users.
    num_threads: int, optional
         Number of parallel computation threads to use. Should
         not be higher than the number of physical cores.
    check_intersections: bool, optional, True by default,
        Only relevant when train_interactions are supplied.
        A flag that signals whether the test and train matrices should be checked
        for intersections to prevent optimistic ranks / wrong evaluation / bad data split.

    Returns
    -------

    np.array of shape [n_users with interactions or n_users,]
         Numpy array containing precision@k scores for each user. If there are
         no interactions for a given user the returned precision will be 0.
    """
