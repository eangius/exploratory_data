#!usr/bin/env python
# AUTHOR: Elian Angius

# ABOUT:
# Prototype to build an anomaly detector of independent events.
# corresponding to an anonymized manufacturing process. Goal is
# to detect timestamps when sensors values are suspicious.


from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LeakyReLU, Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mae
from tensorflow.keras.models import load_model
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from sklego.meta.outlier_classifier import OutlierClassifier
from sklego.preprocessing import IdentityTransformer
from scikitlab.samplers.balancing import StrataBalancer
from scikitlab.vectorizers.temporal import DateTimeVectorizer
from typing import *
from functools import *
import joblib
from joblib import wrap_non_picklable_objects
import random
import pandas as pd
import numpy as np


# config to not truncate display
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
random_state = 42


# returns dataset loaded into learn/eval parts.
def load_data(
    filename: str,                       # path to data file
    split_fraction: float = 0.8,         # test train split ratio
    is_time_series: bool = False,        # honor chronology?
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # data preparation
    df = pd.read_csv(filename, compression='zip', sep=',')
    df.rename(columns={
        'a00': 'index',
        'a01': 'timestamp',
        'a02': 'lotnum',
        'a03': 'recipe',
    }, inplace=True)
    df.set_index('index', inplace=True)

    # set types to save memory
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df['lotnum'] = df['lotnum'].astype('category')
    df['recipe'] = df['recipe'].astype('category')

    if is_time_series:
        # chronologically split (prevent data leakage)
        df = df.sort_values('timestamp', ascending=True)
    else:
        # random shuffle everything
        df = df.sample(frac=1, random_state=random_state)

    split_idx = int(df.shape[0] * split_fraction)
    return df[:split_idx], df[split_idx:]


# returns altered dataset features with anomaly labels
def simulate_anomalies(
    df: pd.DataFrame,                    # data to modify
    defect_rate: float = 0.1,            # rows deemed defective
    defect_max_severity: int = 5,        # max number of deviations to shift values
    defect_max_dimensions: float = 0.2,  # max number of signals to modify per defect
    random_state: Optional[int] = None,  # reproducible results
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(random_state)

    # output labels flagging index of synthetic anomalies
    n_samples = df.shape[0]
    y = pd.DataFrame(
        [0] * n_samples,
        columns=['anomaly'],
        index=df.index.copy()
    )

    # distribution statistics of each column to fake realistic data from.
    stats = df.describe().transpose()[['mean', 'std']]

    # set of modifiable columns & fraction of those that could be modified
    cols = df.columns.drop(["timestamp", "lotnum", "recipe"]).to_list()
    n_cols = int(len(cols) * defect_max_dimensions)

    # helper to shift a value at a (row,col) in the dataframe away from
    # its distribution average by a random number of standard deviations
    def random_value(row_idx, col_name):
        sigma_offset = random.choice(list(range(2, defect_max_severity + 1)) + [None])
        if sigma_offset:
            mean = stats['mean'][col_name]
            std = stats['std'][col_name]
            val = df[col_name][row_idx]

            if val > mean:
                mod_val = val + (sigma_offset * std)
            elif val < mean:
                mod_val = val - (sigma_offset * std)
            else:
                mod_val = val  # don't shift
        else:
            mod_val = np.nan  # corrupt
        return mod_val

    # randomly choose rows to be anomalous then randomly choose some columns
    # to be modified by a random value.
    for row_idx in random.sample(df.index.to_list(), int(n_samples * defect_rate)):
        mod_cols = random.sample(cols, random.randint(1, n_cols))
        for col_name in mod_cols:
            df.loc[row_idx, col_name] = random_value(row_idx, col_name)
        y['anomaly'][row_idx] = 1

    return df, y


# TODO: un-hack need to hardcode column names & quantities
def get_sensor_column_names() -> list[str]:
    return [f"a{c:02d}" for c in range(4, 88 + 1)]


# feature eng & data prep for learning
def build_vectorizer() -> Pipeline:

    # % of missing column values per row.
    def row_nullity_fraction(X: np.array) -> np.array:
        df = 100 * pd.DataFrame(X).isnull().sum(axis=1) / X.shape[1]
        return df.to_numpy().reshape(-1, 1)

    # some sort of value per row combining signals. domain knowledge helps here.
    def row_fingerprint(X: np.array) -> np.array:
        df = pd.DataFrame(X).sum(axis=1, numeric_only=True)
        df = 100 * df / df.sum()
        return df.to_numpy().reshape(-1, 1)

    return Pipeline(
        verbose=True,
        steps=[
            # We are looking for anomalies in the combination of process: (1) recipe
            # of parameters, (2) sensor measurements & (3) relative time of day.
            ('preproc', ColumnTransformer(
                verbose=True,
                transformers=[
                    ('lotnum', 'drop', ['lotnum']),  # <<dbg useful beyond ids?
                    ('recipe', OneHotEncoder(
                        min_frequency=0.01,          # <<dbg should be fn of categories
                        handle_unknown='infrequent_if_exist',
                    ), ['recipe']),
                    ('times', DateTimeVectorizer(), ['timestamp']),
                ],
                remainder='passthrough',  # other process sensors
            )),

            # Scale by quantiles ranges to preserve outliers & to objectively compare
            # between different signals. Mostly useful for algorithms that are trained
            # on nominal & anomalous data.
            ('scaler', RobustScaler(
                with_centering=True,
                with_scaling=True,
                unit_variance=True
            )),

            # Derive basic features
            ('enricher', FeatureUnion([
                ("identity", IdentityTransformer()),
                ('nullity', FunctionTransformer(
                    func=wrap_non_picklable_objects(row_nullity_fraction)
                )),
                ('fingerprint', FunctionTransformer(
                    func=wrap_non_picklable_objects(row_fingerprint)
                )),  # unorthodox!
            ])),

            # NOTE: impute after scaling to not pollute original distributions. This
            # may skew scaling a bit. Also fill with -1 to distinguish from 0 results.
            ('imputer', SimpleImputer(
                strategy='median',  # not influenced by outliers
                fill_value=-1,      # <<dbg learn hyper-param
            )),
        ]
    )


# autoencoder architecture
def build_arch(
    n_features: int,            # dimensionality of input
    latent_frac: float = 0.05,  # % to compress bottleneck layer to
) -> Model:

    scale_rate = 2
    activ_fn = LeakyReLU  # minimize vanishing gradient

    signal = Input(name="features", shape=(n_features,))

    # Compress signal by a factor with as many layers necessary to get to
    # the desired latent size. Layer depth & size does not matter that much.
    n_bottleneck = int(n_features * latent_frac)
    size = n_features // scale_rate
    encoder = Sequential(name="encoder")
    while size > n_bottleneck:
        encoder.add(Dense(size, activation=activ_fn()))
        size //= scale_rate
    encoder.add(Dense(n_bottleneck,  activation=activ_fn()))

    # Reconstruct signal with similar reverse sequence of layers. Notice that we add
    # an extra layer & expand sizes by +1 since construction is harder than destruction.
    decoder = Sequential(name="decoder")
    size *= scale_rate
    while size < n_features:
        decoder.add(Dense(size + 1, activation=activ_fn()))
        size *= scale_rate
    decoder.add(Dense(n_features, activation='sigmoid'))

    autoencoder = Model(
        inputs=signal,
        outputs=decoder(encoder(signal))
    )
    autoencoder.compile(
        optimizer='adam',
        loss='mae',
    )
    return autoencoder


PROJ_DIR = '.'

# Helper to ensure learn & eval consistence. Training with less severe
# synthetic anomalies makes model more robust to real world detection.
defect_rate = 0.05   # <<dbg eval hyper-param
anomalizer = partial(
    simulate_anomalies,
    defect_rate=defect_rate,
    defect_max_dimensions=0.15,
    defect_max_severity=4,
)

print("\nPreparing datasets ..")
X_learn, X_eval = load_data(
    filename=f'{PROJ_DIR}/data/learn.csv',
    split_fraction=0.8,
    random_state=random_state
)
X_learn_samples = X_learn.shape[0]

# To avoid fooling model on rare lot-num/recipe combos being anomalous
# events, we randomly over sample these groups (only at fit time) to
# ensure fair representation.
print("Recipe Distribution:")
print(100 * X_learn['recipe'].value_counts(normalize=True))
X_learn, *_ = StrataBalancer(
   sampling_mode='over',
   columns=['recipe'],
   random_state=random_state,
).fit_resample(X_learn)
print("\nDataset Sizes")
print(f"learning:   {X_learn_samples} -> {X_learn.shape[0]}")
print(f"evaluating: {X_eval.shape[0]}")
print()

# Prepare data for learning.
print("\nFeature engineering ..")
vectorizer = build_vectorizer()
X_learn1 = vectorizer.fit_transform(X_learn, y=None)
X_learn2, y_learn2 = anomalizer(X_learn)
X_learn2 = vectorizer.fit_transform(X_learn2, y=y_learn2)
print(f"vector dimensionality: {X_learn1.shape[1]}")

# Unsupervised baseline anomaly detection model. We train with fake anomalies
# assumed to be representative of the domain. Wrap around outlier classifier
# to ensure predictions() are binary rather than 1 = anomalous & -1 = anomalous
print("\nTraining baseline model ..")
base_model = OutlierClassifier(IsolationForest(
    n_estimators=100,  # <<dbg learn hyper-param
    contamination=defect_rate,
    verbose=True,
    random_state=random_state,
)).fit(
    X=X_learn2,
    y=y_learn2.to_numpy().ravel()
)


# Unsupervised target anomaly detection model. We train exclusively with
# nominal data to ensure reconstruction.
print("\nTraining target model ..")
target_model = build_arch(
    n_features=X_learn1.shape[1],
    latent_frac=0.05,  # <<dbg learn hyper-param
)
target_model.summary(expand_nested=True)
history = target_model.fit(
    X_learn1, X_learn1,
    epochs=200,  # generous due to early-stopping
    batch_size=64,
    validation_split=0.10,
    shuffle=True,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)

print("\nPersisting models to disk ..")
joblib.dump(vectorizer, f'{PROJ_DIR}/model_vectorizer.zip')
joblib.dump(base_model, f'{PROJ_DIR}/model_isolationforest.zip')
target_model.save(f'{PROJ_DIR}/model_autoencoder.zip')

# For evaluation we synthesize some fake anomalies & see how well
# models can classify them out.
print("\nPreparing evaluation anomalies ..")
vectorizer = joblib.load(f'{PROJ_DIR}/model_vectorizer.zip')
X_eval, y_eval = anomalizer(X_eval)
X_eval = vectorizer.transform(X_eval)

# for the isolation forest model we just evaluate like a
# classification problem.
base_model = joblib.load(f'{PROJ_DIR}/model_isolationforest.zip')
y_pred = base_model.predict(X_eval)
print("\nbaseline model:")
print(classification_report(y_eval, y_pred))
print()

# for the autoencoder model we find the error threshold
# where defect_rate number of anomalies are detected.
target_model = load_model(f'{PROJ_DIR}/model_autoencoder.zip')
X_pred = target_model.predict(X_eval)
metric = mae(X_eval, X_pred)
threshold = np.percentile(metric, 100*(1 - defect_rate))
y_pred = [int(not (i < threshold)) for i in metric]
print("\ntarget model:")
print(classification_report(y_eval, y_pred))
print(f"{threshold=}")


# TODO:
#  [1] parameter hyper-tuning via cross validation
#  [2] enhanced feature engineering:
#     -- cross features?
#     -- lot#, lifetime?
#     -- embed DBSCAN as anomaly signal?
#     -- better NaN imputation strategy?
#  [3] fix model.get_feature_names_out() from
#  [4] feature selection
