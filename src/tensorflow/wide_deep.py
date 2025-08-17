import tensorflow as tf

def build_wide_deep(n_users, n_items, emb_dim=32):
    user_in = tf.keras.Input(shape=(), dtype=tf.int32, name="user_id")
    item_in = tf.keras.Input(shape=(), dtype=tf.int32, name="item_id")

    u_emb = tf.keras.layers.Embedding(n_users, emb_dim, name="user_emb")(user_in)
    i_emb = tf.keras.layers.Embedding(n_items, emb_dim, name="item_emb")(item_in)
    deep = tf.keras.layers.Concatenate(name="deep_concat")([u_emb, i_emb])
    deep = tf.keras.layers.Flatten(name="deep_flatten")(deep)
    deep = tf.keras.layers.Dense(128, activation="relu")(deep)
    deep = tf.keras.layers.Dense(64, activation="relu")(deep)

    user_oh = tf.keras.layers.CategoryEncoding(
        num_tokens=n_users, output_mode="one_hot", name="user_one_hot"
    )(user_in)
    item_oh = tf.keras.layers.CategoryEncoding(
        num_tokens=n_items, output_mode="one_hot", name="item_one_hot"
    )(item_in)
    wide = tf.keras.layers.Concatenate(name="wide_concat")([user_oh, item_oh])

    both = tf.keras.layers.Concatenate(name="merge_wide_deep")([deep, wide])
    out = tf.keras.layers.Dense(1, activation=None, name="logit")(both)

    model = tf.keras.Model(inputs=[user_in, item_in], outputs=out, name="wide_deep")
    return model
