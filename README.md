# MovieRecomendations
Recommendation-System-Using-Neural-Collaborative-Filtering

In this project the aim is to recommend a user movies based on their previous rating on movies which was watched by the user

Neural Collaborative Filtering (NCF) is a type of collaborative filtering algorithm that uses neural networks to model the latent features of users and items for recommendation. Collaborative filtering is a technique commonly used in recommendation systems to make predictions about the preferences of a user by leveraging the preferences and behavior of other users.

from keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from keras.models import Model

# Assume num_users and num_items are the total number of users and items, respectively
num_users = 1000
num_items = 2000
embed_size = 50

user_input = Input(shape=[1])
item_input = Input(shape=[1])

user_embedding = Embedding(input_dim=num_users, output_dim=embed_size)(user_input)
item_embedding = Embedding(input_dim=num_items, output_dim=embed_size)(item_input)

user_flat = Flatten()(user_embedding)
item_flat = Flatten()(item_embedding)

concat = Concatenate()([user_flat, item_flat])

dense1 = Dense(64, activation='relu')(concat)
dense2 = Dense(32, activation='relu')(dense1)

output = Dense(1, activation='sigmoid')(dense2)

model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


This is a basic NCF model using embeddings and neural network layers for collaborative filtering. The exact architecture and hyperparameters would need to be adjusted based on the specific characteristics of your data and task.
