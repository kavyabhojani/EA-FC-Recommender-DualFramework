import tensorflow as tf

def build_wide_deep(n_users, n_items, emb_dim=32):
    user_in = tf.keras.Input(shape=(), dtype=tf.int32, name="user_id")
    item_in = tf.keras.Input(shape=(), dtype=tf.int32, name="item_id")

    u_emb = tf.keras.layers.Embedding(n_users, emb_dim)(user_in)
    i_emb = tf.keras.layers.Embedding(n_items, emb_dim)(item_in)
    deep = tf.keras.layers.Concatenate()([u_emb, i_emb])
    deep = tf.keras.layers.Flatten()(deep)
    deep = tf.keras.layers.Dense(128, activation="relu")(deep)
    deep = tf.keras.layers.Dense(64, activation="relu")(deep)

    #"wide" part: simple crossed one-hots approximated by concatenating ids
    wide = tf.keras.layers.Concatenate()([
        tf.cast(tf.expand_dims(user_in, -1), tf.float32),
        tf.cast(tf.expand_dims(item_in, -1), tf.float32)
    ])
    wide = tf.keras.layers.Flatten()(wide)

    x = tf.keras.layers.Concatenate()([deep, wide])
    out = tf.keras.layers.Dense(1, activation=None)(x)
    model = tf.keras.Model(inputs=[user_in, item_in], outputs=out)
    return model
