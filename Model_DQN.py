import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

from Evaluation import evaluation


class ClassificationEnv:
    def __init__(self, X, y, num_classes):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self.state_size = X.shape[1]
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self.X[self.current_index]

    def step(self, action):
        correct = action == self.y[self.current_index]
        reward = 1 if correct else -1
        self.current_index += 1

        done = self.current_index >= len(self.X)
        next_state = self.X[self.current_index] if not done else np.zeros(self.state_size)

        return next_state, reward, done


class DQNAgent:
    def __init__(self, state_size, num_classes):
        self.state_size = state_size
        self.num_classes = num_classes
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='linear')  # Q-values for each class
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_classes)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])  # Choose the action with the highest Q-value

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.model.predict(np.array([next_state]), verbose=0)[0])

            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def Model_DQN(X_train, y_train, X_test, y_test, steps_per_epoch):
    # Generate Dummy Data (Replace with real data)
    num_classes = 3
    num_features = 10
    num_samples = 500

    # Create Environment and Agent
    env = ClassificationEnv(X_train, y_train, num_classes)
    agent = DQNAgent(state_size=num_features, num_classes=num_classes)

    # Train Agent
    episodes = steps_per_epoch
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.train(batch_size)
        print(f"Episode {e + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Evaluate on Test Set
    correct = 0
    for i in range(len(X_test)):
        action = agent.act(X_test[i])
        correct += (action == y_test[i])
    Predict = agent.predict(X_test)
    Eval = evaluation(Predict, X_test)
    return Eval, Predict
