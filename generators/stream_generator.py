import math
import numpy as np
import logging

from skmultiflow.utils import check_random_state
from skmultiflow.data import ConceptDriftStream

class RecurrentDriftStream(ConceptDriftStream):

    def __init__(self,
                 generator='agrawal',
                 stable_period=20000,
                 position=20000,
                 width=1,
                 stable_period_logger=None,
                 random_state=0):

        super().__init__()

        self.streams = []
        self.cur_stream = None
        self.stream_idx = 0
        self.drift_stream_idx = 0
        self.sample_idx = 0

        self.generator = generator
        self.stable_period=stable_period
        self.position = position
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)
        self.width = width
        self.stable_period_logger=stable_period_logger

        self.total_sample_idx = 0

    def next_sample(self, batch_size=1):

        """ Returns the next `batch_size` samples.

        Parameters
        ----------
        batch_size: int
            The number of samples to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix
            for the batch_size samples that were requested.

        """

        self.current_sample_x = np.zeros((batch_size, self.n_features))
        self.current_sample_y = np.zeros((batch_size, self.n_targets))

        for j in range(batch_size):
            self.sample_idx += 1
            x = -4.0 * float(self.sample_idx - self.position) / float(self.width)
            probability_drift = 1.0 / (1.0 + np.exp(x))

            if self._random_state.rand() > probability_drift:
                X, y = self.cur_stream.next_sample()
            else:
                X, y = self.drift_stream.next_sample()
            self.current_sample_x[j, :] = X
            self.current_sample_y[j, :] = y

        if self.sample_idx >= self.stable_period + self.width:
            self.sample_idx = 0

            # sequence: strict cyclic
            self.stream_idx = (self.stream_idx + 1) % len(self.streams)
            self.drift_stream_idx = (self.stream_idx + 1) % len(self.streams)

            self.cur_stream = self.streams[self.stream_idx]
            self.drift_stream = self.streams[self.drift_stream_idx]

            if self.stable_period_logger is not None:
                self.stable_period_logger.info(str(self.total_sample_idx))

        self.total_sample_idx += batch_size

        return self.current_sample_x, self.current_sample_y.flatten()

    def get_data_info(self):
        return self.cur_stream.get_data_info()

    def prepare_for_use(self, streams):
        for stream in streams:
            stream.prepare_for_use()
            self.streams.append(stream)

        self.cur_stream = self.streams[0]
        self.drift_stream = self.streams[1]

        stream = self.cur_stream
        self.n_samples = stream.n_samples
        self.n_targets = stream.n_targets
        self.n_features = stream.n_features
        self.n_num_features = stream.n_num_features
        self.n_cat_features = stream.n_cat_features
        self.n_classes = stream.n_classes
        self.cat_features_idx = stream.cat_features_idx
        self.feature_names = stream.feature_names
        self.target_names = stream.target_names
        self.target_values = stream.target_values
        self.n_targets = stream.n_targets
        self.name = 'drifting' + stream.name

    def get_arff_header(self):
        header = []

        for i in range(self.n_features):
            if self.generator == "led":
                header.append(f"@attribute a{i} {0.0, 1.0}")
            elif self.generator == "stagger":
                header.append(f"@attribute a{i} {0.0, 1.0, 2.0}")
            else:
                header.append(f"@attribute a{i} numeric")

        print(f"num_features: {self.n_features}")

        class_str = "@attribute class {"
        for i in range(self.n_classes):
            class_str += f"{i:.1f}"
            if i == self.n_classes - 1:
                class_str += "}"
            else:
                class_str += ", "
        header.append(class_str)
        print(class_str)

        header.append("@data")
        header.append("\n")

        return "\n".join(header)
