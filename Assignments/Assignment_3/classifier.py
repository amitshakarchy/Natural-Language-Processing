import pickle
from abc import ABC, abstractmethod
import numpy as np
from joblib import dump, load
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn as nn
from metrics import evaluate_metrics


class Classifier(ABC):
    """
    This Class is an abstract class that represents a classifier interface and abstract methods
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_file_name = "best_model"

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def get_cls(self):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def predict_proba(self, X_test):
        pass

    @abstractmethod
    def to_string(self):
        pass

    def save(self):
        """
        Saves a trained model as a pickle file
        Returns:

        """
        pickle.dump(self.model, open(self.model_file_name, 'wb'))

    def load(self):
        """
        Loads a trained model from a pickle file
        Returns:
            Classifier instance with the loaded model

        """
        self.model = pickle.load(open(self.model_file_name, 'rb'))
        return self


class LRClassifier(Classifier):
    """
    This Class implements the Classifier class, implementing Logistic Regression Classifier
    """

    def __init__(self):
        super().__init__()
        self.model = LogisticRegression()

    def get_cls(self):
        """
        The method returns the model
        Returns:
            model: Logistic Regression model
        """
        return self.model

    def train(self, X_train, y_train):
        """
        The method trains the LR model
        Args:
            X_train: word embeddings
            y_train: true labels

        Returns:
            None

        """
        self.model = self.model.fit(X_train, y_train)
        return

    def predict(self, X_test):
        """
        The method classifies the input data to trump (0) or staffer (1)
        Args:
            X_test: Tweets to classify
        Returns:
            list of predictions
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Returns the probability of each prediction's option
        Args:
            X_test:  Tweets to classify

        Returns:
            list of probabilities
        """
        return self.model.predict_proba(X_test)[:, 1]

    def to_string(self):
        """

        Returns: The name of the algorithm

        """
        return "Logistic_Regression"


class SVMClassifier(Classifier):
    """
    This Class implements the Classifier class, implementing SVM Classifier
    """

    def __init__(self, kernel='liner', gamma='scale'):
        super().__init__()
        self.kernel = kernel
        self.model = SVC(kernel=kernel, gamma=gamma, probability=True)

    def get_cls(self):
        """
        The method returns the model
        Returns:
            model: SVM model
        """
        return self.model

    def train(self, X_train, y_train):
        """
        The method trains the LR model
        Args:
            X_train: word embeddings
            y_train: true labels

        Returns:
            The classifier

        """
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        """
        The method classifies the input data to trump (0) or staffer (1)
        Args:
            X_test: Tweets to classify
        Returns:
            list of predictions
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Returns the probability of each prediction's option
        Args:
            X_test:  Tweets to classify

        Returns:
            list of probabilities
        """
        return self.model.predict_proba(X_test)[:, 1]

    def save(self):
        """
        Saves the model as a pickle file

        """
        pickle.dump(self.model, open(self.model_file_name, 'wb'))

    def to_string(self):
        """

        Returns: The name of the algorithm

        """
        return "SVM_{}".format(self.kernel)


class BasicNN(Classifier):
    """
    This Class implements the Classifier class, implementing Dense DNN Classifier
    """
    def __init__(self, input_size, n_epochs=50, batch_size=32, criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.losses = []

    def get_cls(self):
        """
        The method returns the model
        Returns:
            model: Dense DNN model
        """
        return self.model

    def train(self, X_train, y_train):
        """
        The method trains the dense DNN model
        Args:
            X_train: word embeddings
            y_train: true labels

        Returns:
            The classifier

        """
        X_train = torch.Tensor(X_train)
        y_train = torch.LongTensor(y_train)

        train = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=self.batch_size, shuffle=True, drop_last=True)
        for i in range(self.n_epochs):
            for j, (X_t_loader, y_t_loader) in enumerate(train_loader):
                y_pred_tensor = self.model.forward(X_t_loader)
                loss = self.criterion(y_pred_tensor, y_t_loader)
                self.losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                y_pred = self.model.forward(X_train)
        return None

    def predict(self, X_test):
        """
        The method classifies the input data to trump (0) or staffer (1)
        Args:
            X_test: Tweets to classify
        Returns:
            list of predictions
        """
        tensor_pred = self.model(torch.Tensor(X_test))
        return tensor_pred.max(1).indices

    def predict_proba(self, X_test):
        """
        Returns the probability of each prediction's option
        Args:
            X_test:  Tweets to classify

        Returns:
            list of probabilities
        """
        tensor_pred = self.model(torch.Tensor(X_test))
        return tensor_pred.detach().numpy()[:, 1]

    def to_string(self):
        """

        Returns: The name of the algorithm

        """
        return "DNN"


class LSTMNET(nn.Module):
    """
    This class implements an LSTM neural network
    """
    def __init__(self, input_size, n_layers, linear_dim, dropout=0.5):
        super(LSTMNET, self).__init__()
        self.n_layers = n_layers
        self.linear_dim = linear_dim
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, linear_dim, n_layers,
                            dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(linear_dim, 16), nn.ReLU(),
                                nn.Linear(16, 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        if (len(x.size()) == 2):
            x = x.reshape(batch_size, -1, self.input_size)
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.n_layers, batch_size,
                         self.linear_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.n_layers, batch_size,
                         self.linear_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        lstm_out = lstm_out[:, -1, :]

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out


class LSTMClassifier(Classifier):
    """
    This Class implements the Classifier class, implementing LSTM NN Classifier
    """
    def __init__(self, input_size, n_layers, linear_dim, dropout=0.5, n_epochs=50, batch_size=32,
                 criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.model = LSTMNET(input_size, n_layers, linear_dim, dropout)
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.n_epochs = n_epochs
        self.losses = []

    def get_cls(self):
        """
        The method returns the model
        Returns:
            model: LSTM model
        """
        return self.model

    def train(self, X_train, y_train):
        """
        The method trains the LSTM model
        Args:
            X_train: word embeddings
            y_train: true labels

        Returns:
            The classifier

        """
        for i in range(self.n_epochs):
            self.model.train()
            X_train, y_train = shuffle(X_train, y_train)
            X_train_tensor = torch.Tensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            for j in range(int(X_train_tensor.size()[0] // self.batch_size)):
                y_pred_tensor = self.model.forward(
                    X_train_tensor[j * self.batch_size:(j + 1) * self.batch_size])
                loss = self.criterion(
                    y_pred_tensor, y_train_tensor[j * self.batch_size:(j + 1) * self.batch_size])
                self.losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def predict(self, X_test):
        """
        The method classifies the input data to trump (0) or staffer (1)
        Args:
            X_test: Tweets to classify
        Returns:
            list of predictions
        """
        tensor_pred = self.model(torch.Tensor(X_test))
        return tensor_pred.max(1).indices

    def predict_proba(self, X_test):
        """
        Returns the probability of each prediction's option
        Args:
            X_test:  Tweets to classify

        Returns:
            list of probabilities
        """
        tensor_pred = self.model(torch.Tensor(X_test))
        return tensor_pred.detach().numpy()[:, 1]

    def to_string(self):
        """

        Returns: The name of the algorithm

        """
        return "LSTM"


class LSTMTextNN(nn.Module):
    """ This Class implements LSTM with concatenation of meta-features neural network"""
    def __init__(self, input_size, n_layers, linear_dim, dense_size, numeric_feature_size, dropout=0.5):
        super(LSTMTextNN, self).__init__()
        self.n_layers = n_layers
        self.linear_dim = linear_dim
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, linear_dim, n_layers,
                            dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Sequential(nn.Linear(linear_dim, dense_size), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(dense_size + numeric_feature_size, 16), nn.ReLU(), nn.BatchNorm1d(16), nn.Linear(16, 2))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, nn_input_meta):
        batch_size = x.size(0)
        if (len(x.size()) == 2):
            x = x.reshape(batch_size, -1, self.input_size)

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.n_layers, batch_size,
                         self.linear_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.n_layers, batch_size,
                         self.linear_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        lstm_out = lstm_out[:, -1, :]

        out = self.dropout(lstm_out)
        out = self.fc1(out)

        concat_layer = torch.cat((out, nn_input_meta.float()), 1)
        out = self.fc2(concat_layer)

        out = self.sigmoid(out)

        return out


class TextNumericalInputsClassifier(Classifier):
    """
    This Class implements the Classifier class, implementing LSTM with meta-features Classifier
    """
    def __init__(self, vector_size, n_layers, linear_dim, dense_size, numeric_feature_size, dropout=0.5, n_epochs=50,
                 batch_size=32,
                 criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.model = LSTMTextNN(
            vector_size, n_layers, linear_dim, dense_size, numeric_feature_size, dropout=0.5)
        self.numeric_feature_size = numeric_feature_size
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.n_epochs = n_epochs
        self.losses = []

    def train(self, X_train, y_train):
        """
        The method trains the LSTM with meta-features model
        Args:
            X_train: word embeddings
            y_train: true labels

        Returns:
            The classifier

        """
        for i in range(self.n_epochs):
            self.model.train()
            X_train, y_train = shuffle(X_train, y_train)
            X_train_tensor = torch.Tensor(
                X_train[:, self.numeric_feature_size:])
            meta_data_tensor = torch.LongTensor(
                X_train[:, :self.numeric_feature_size])
            y_train_tensor = torch.LongTensor(y_train)

            train = torch.utils.data.TensorDataset(
                X_train_tensor, meta_data_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(
                train, batch_size=self.batch_size, shuffle=True, drop_last=True)

            for j, (X_t_loader, meta_t_loader, y_t_loader) in enumerate(train_loader):
                y_pred_tensor = self.model.forward(X_t_loader, meta_t_loader)
                loss = self.criterion(y_pred_tensor, y_t_loader)
                self.losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    y_pred = self.model.forward(
                        X_train_tensor, meta_data_tensor)

    def predict(self, X_test):
        """
        The method classifies the input data to trump (0) or staffer (1)
        Args:
            X_test: Tweets to classify
        Returns:
            list of predictions
        """
        X_test_tensor = torch.Tensor(X_test[:, self.numeric_feature_size:])
        meta_data_tensor = torch.LongTensor(
            X_test[:, :self.numeric_feature_size])
        tensor_pred = self.model(X_test_tensor, meta_data_tensor)
        return tensor_pred.max(1).indices

    def predict_proba(self, X_test):
        """
        Returns the probability of each prediction's option
        Args:
            X_test:  Tweets to classify

        Returns:
            list of probabilities
        """
        X_test_tensor = torch.Tensor(X_test[:, self.numeric_feature_size:])
        meta_data_tensor = torch.LongTensor(
            X_test[:, :self.numeric_feature_size])
        tensor_pred = self.model(X_test_tensor, meta_data_tensor)
        return tensor_pred.detach().numpy()[:, 1]

    def get_cls(self):
        """
        The method returns the model
        Returns:
            model: LSTM model
        """
        return self.model

    def to_string(self):
        """

        Returns: The name of the algorithm

        """
        return "LSTM_TEXT"
